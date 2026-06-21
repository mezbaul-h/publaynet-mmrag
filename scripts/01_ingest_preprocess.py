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
from publaynet_mmrag.types import Category, Chunk, Region, write_jsonl  # noqa: E402
from scripts._common import add_config_args, resolve_config  # noqa: E402


def run(config: Config, caption_figures: bool) -> None:
    """Runs Stage 1 over the configured dataset.

    Args:
        config: The active run configuration.
        caption_figures: Whether to generate VLM captions for visual regions.
    """
    os.makedirs(config.paths.regions_dir, exist_ok=True)
    os.makedirs(config.paths.crops_dir, exist_ok=True)

    import time

    start = time.perf_counter()

    source = build_source(config)
    ocr = build_ocr_engine(device=config.models.device)
    ocr.load()
    captioner = None
    if caption_figures:
        captioner = Captioner(
            model_name=config.models.caption_model, device=config.models.device
        )

    all_chunks: list[Chunk] = []
    processed_docs: set[str] = set()

    from tqdm import tqdm

    total = config.ingest.max_pages or source.expected_len()
    if total is None:
        print(
            "Stage 1: page total unknown while streaming (bar shows count + "
            "rate). Set ingest.streaming: false for an ETA over the full subset."
        )
    for page in tqdm(source, total=total, desc="Stage 1: pages", unit="page"):
        region_file = os.path.join(config.paths.regions_dir, f"{page.doc_id}.jsonl")
        regions = extract_regions(page, config.paths.crops_dir)

        # OCR text-bearing regions in one batch per page.
        text_regions = [r for r in regions if r.category in Category.textual()]
        if text_regions:
            crops = [crop_region(page, r) for r in text_regions]
            texts = ocr.recognise(crops)
            for region, text in zip(text_regions, texts):
                region.text = text

        # Optional captioning of visual regions.
        if captioner is not None:
            for region in regions:
                if region.category in Category.visual() and region.crop_path:
                    region.caption = captioner.caption(crop_region(page, region))

        _append_regions(region_file, regions, fresh=page.doc_id not in processed_docs)
        processed_docs.add(page.doc_id)

        all_chunks.extend(
            chunk_page_regions(
                regions,
                max_chars=config.chunk.max_chars,
                overlap_chars=config.chunk.overlap_chars,
                min_chars=config.chunk.min_chars,
            )
        )

    write_jsonl(config.paths.chunks_path, [c.to_dict() for c in all_chunks])
    print(
        f"Stage 1 complete: {len(processed_docs)} documents, "
        f"{len(all_chunks)} chunks -> {config.paths.chunks_path}"
    )

    ocr.unload()
    if captioner is not None:
        captioner.unload()

    print(f"Stage 1 finished in {format_duration(time.perf_counter() - start)}.")


def _append_regions(path: str, regions: list[Region], fresh: bool) -> None:
    """Appends serialised regions to a per-document file.

    Args:
        path: The per-document region file.
        regions: Regions to append.
        fresh: If ``True``, truncates the file first (new document this run).
    """
    mode = "w" if fresh else "a"
    import json

    with open(path, mode, encoding="utf-8") as handle:
        for region in regions:
            handle.write(json.dumps(region.to_dict(), ensure_ascii=False) + "\n")


def main() -> None:
    """Parses arguments and runs Stage 1."""
    parser = argparse.ArgumentParser(description="Stage 1: ingest + preprocess.")
    add_config_args(parser)
    parser.add_argument(
        "--caption-figures",
        action="store_true",
        help="Generate VLM captions for figure/table regions.",
    )
    args = parser.parse_args()
    run(resolve_config(args), caption_figures=args.caption_figures)


if __name__ == "__main__":
    main()
