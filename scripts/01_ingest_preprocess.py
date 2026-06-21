#!/usr/bin/env python
"""Stage 1: ingest, preprocess, OCR and chunk.

Streams pages from the configured source, crops regions from the ground-truth
boxes, OCRs the text-bearing regions, optionally captions visual regions, and
writes per-document region files plus a global chunk file. The stage is
resumable: documents whose region file already exists are skipped, which matters
on the full dataset where OCR is a long job.

Heavy models (OCR, captioner) are loaded here and explicitly unloaded at the end
so the indexing stage starts with a free GPU.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from publaynet_mmrag.config import Config  # noqa: E402
from publaynet_mmrag.embed.caption import Captioner  # noqa: E402
from publaynet_mmrag.ingest import build_source  # noqa: E402
from publaynet_mmrag.preprocess.chunk import chunk_page_regions  # noqa: E402
from publaynet_mmrag.preprocess.ocr import build_ocr_engine  # noqa: E402
from publaynet_mmrag.preprocess.regions import crop_region, extract_regions  # noqa: E402
from publaynet_mmrag.timing import format_duration  # noqa: E402
from publaynet_mmrag.types import (  # noqa: E402
    Category,
    Chunk,
    Region,
    read_jsonl,
    write_jsonl,
)
from scripts._common import add_config_args, resolve_config  # noqa: E402


def run(config: Config, caption_figures: bool, verbose_ocr: bool = False) -> None:
    """Runs Stage 1 over the configured dataset.

    Resumable: each page's regions are written to its own file
    (``<regions_dir>/<page_key>.jsonl``) atomically, and a page whose file
    already exists is skipped. A crash therefore loses at most the in-flight
    page, and re-running the same command continues from where it stopped. The
    chunk file is regenerated from all per-page region files at the end (cheap
    and deterministic), so it is always consistent with what was OCR'd. Heavy
    models are loaded lazily, so a fully-completed resume loads nothing.

    Args:
        config: The active run configuration.
        caption_figures: Whether to generate VLM captions for visual regions.
        verbose_ocr: Show Surya's per-page progress bars (off by default).
    """
    os.makedirs(config.paths.regions_dir, exist_ok=True)
    os.makedirs(config.paths.crops_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config.paths.chunks_path) or ".", exist_ok=True)

    import time

    start = time.perf_counter()

    source = build_source(config)
    ocr = build_ocr_engine(device=config.models.device, verbose=verbose_ocr)
    ocr_loaded = False
    captioner = (
        Captioner(model_name=config.models.caption_model, device=config.models.device)
        if caption_figures
        else None
    )

    from tqdm import tqdm

    total = config.ingest.max_pages or source.expected_len()
    if total is None:
        print(
            "Stage 1: page total unknown while streaming (bar shows count + "
            "rate). Set ingest.streaming: false for an ETA over the full subset."
        )

    processed = 0
    skipped = 0
    bar = tqdm(source, total=total, desc="Stage 1: pages", unit="page")
    for page in bar:
        page_file = os.path.join(config.paths.regions_dir, f"{page.key}.jsonl")
        if os.path.exists(page_file):
            skipped += 1
            bar.set_postfix(ocr=processed, skip=skipped)
            continue

        regions = extract_regions(page, config.paths.crops_dir)

        # OCR text-bearing regions in one batch per page (load OCR on first use).
        text_regions = [r for r in regions if r.category in Category.textual()]
        if text_regions:
            if not ocr_loaded:
                ocr.load()
                ocr_loaded = True
            crops = [crop_region(page, r) for r in text_regions]
            texts = ocr.recognise(crops)
            for region, text in zip(text_regions, texts):
                region.text = text

        # Optional captioning of visual regions.
        if captioner is not None:
            for region in regions:
                if region.category in Category.visual() and region.crop_path:
                    region.caption = captioner.caption(crop_region(page, region))

        _write_regions_atomic(page_file, regions)
        processed += 1
        bar.set_postfix(ocr=processed, skip=skipped)

    if ocr_loaded:
        ocr.unload()
    if captioner is not None:
        captioner.unload()

    # Regenerate the chunk file from all per-page region files. This is cheap,
    # deterministic and always reflects exactly what has been OCR'd so far.
    num_chunks = _regenerate_chunks(config)
    print(
        f"Stage 1 complete: {processed} pages processed, {skipped} skipped, "
        f"{num_chunks} chunks -> {config.paths.chunks_path}"
    )
    print(f"Stage 1 finished in {format_duration(time.perf_counter() - start)}.")


def _write_regions_atomic(path: str, regions: list[Region]) -> None:
    """Writes a page's regions to its file atomically.

    Writes to a temporary file and renames it into place so a crash mid-write
    never leaves a partial file that would be mistaken for completed work.

    Args:
        path: The per-page region file.
        regions: Regions for that page.
    """
    import json

    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        for region in regions:
            handle.write(json.dumps(region.to_dict(), ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _regenerate_chunks(config: Config) -> int:
    """Rebuilds the chunk file from every per-page region file.

    Args:
        config: The active run configuration.

    Returns:
        The number of chunks written.
    """
    import glob

    chunks: list[Chunk] = []
    for path in sorted(glob.glob(os.path.join(config.paths.regions_dir, "*.jsonl"))):
        regions = [Region.from_dict(row) for row in read_jsonl(path)]
        chunks.extend(
            chunk_page_regions(
                regions,
                max_chars=config.chunk.max_chars,
                overlap_chars=config.chunk.overlap_chars,
                min_chars=config.chunk.min_chars,
            )
        )
    write_jsonl(config.paths.chunks_path, [c.to_dict() for c in chunks])
    return len(chunks)


def main() -> None:
    """Parses arguments and runs Stage 1."""
    parser = argparse.ArgumentParser(description="Stage 1: ingest + preprocess.")
    add_config_args(parser)
    parser.add_argument(
        "--caption-figures",
        action="store_true",
        help="Generate VLM captions for figure/table regions.",
    )
    parser.add_argument(
        "--verbose-ocr",
        action="store_true",
        help="Show Surya's per-page progress bars (off by default).",
    )
    args = parser.parse_args()
    run(
        resolve_config(args),
        caption_figures=args.caption_figures,
        verbose_ocr=args.verbose_ocr,
    )


if __name__ == "__main__":
    main()
