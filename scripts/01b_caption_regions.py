#!/usr/bin/env python
"""Stage 1b: caption a subset of already-cropped visual regions.

Stage 1 crops every figure/table but only captions them when run with
``--caption-figures``; on a resumed corpus that flag is a no-op because finished
pages are skipped. This standalone pass adds captions *without* re-running OCR or
re-cropping: it reads the saved crop PNGs directly, captions a deterministic
sample, writes the captions back into the per-page region files, and sets them on
the existing ``visual_regions`` index points (no re-embedding -- the SigLIP image
vectors are unchanged).

Captions are what let the evaluation author the *visual* QA split (questions
answerable from a specific figure/table) and give the generator a text handle on
retrieved figures. The pass is resumable: regions that already have a caption are
skipped.

Example:
    python scripts/01b_caption_regions.py            # caption ~500 crops
    python scripts/01b_caption_regions.py --limit 0  # caption all crops
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from tqdm import tqdm  # noqa: E402

from publaynet_mmrag.config import Config  # noqa: E402
from publaynet_mmrag.embed.caption import Captioner  # noqa: E402
from publaynet_mmrag.index import schema  # noqa: E402
from publaynet_mmrag.index.store import VectorStore, _point_id  # noqa: E402
from publaynet_mmrag.timing import format_duration  # noqa: E402
from publaynet_mmrag.types import Category, Region, read_jsonl  # noqa: E402
from scripts._common import add_config_args, resolve_config  # noqa: E402


def _visual_uncaptioned(config: Config) -> list[tuple[str, int, Region]]:
    """Collects visual regions that have a crop but no caption yet.

    Args:
        config: The active run configuration.

    Returns:
        ``(page_file, line_index, region)`` for each uncaptioned visual region
        whose crop file exists on disk.
    """
    out: list[tuple[str, int, Region]] = []
    for page_file in sorted(
        glob.glob(os.path.join(config.paths.regions_dir, "*.jsonl"))
    ):
        rows = read_jsonl(page_file)
        for idx, row in enumerate(rows):
            region = Region.from_dict(row)
            if (
                region.category in Category.visual()
                and region.crop_path
                and not region.caption
                and os.path.exists(region.crop_path)
            ):
                out.append((page_file, idx, region))
    return out


def _write_captions(page_file: str, captions: dict[int, str]) -> None:
    """Rewrites a page file, injecting captions at the given line indices.

    Args:
        page_file: The per-page region ``.jsonl`` file.
        captions: Mapping of line index -> caption text to set.
    """
    rows = read_jsonl(page_file)
    for idx, caption in captions.items():
        rows[idx]["caption"] = caption
    tmp = f"{page_file}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, page_file)


def run(config: Config, limit: int) -> None:
    """Captions a sample of visual region crops.

    Args:
        config: The active run configuration.
        limit: Maximum regions to caption (0 = all uncaptioned regions).
    """
    import time

    start = time.perf_counter()

    targets = _visual_uncaptioned(config)
    if not targets:
        print("No uncaptioned visual regions with crops found; nothing to do.")
        return

    rng = random.Random(config.eval.seed)
    rng.shuffle(targets)
    if limit > 0:
        targets = targets[:limit]
    print(f"Captioning {len(targets)} visual regions.")

    captioner = Captioner(
        model_name=config.models.caption_model, device=config.models.device
    )

    store = VectorStore(
        path=config.paths.qdrant_path,
        text_dim=config.models.text_embed_dim,
        image_dim=config.models.image_embed_dim,
    )
    has_index = store.client.collection_exists(schema.IMAGE_COLLECTION)

    from PIL import Image

    from publaynet_mmrag.shutdown import graceful_shutdown

    # Group writes per page file so each file is rewritten once at the end of its
    # regions, keeping the pass resumable if interrupted between pages.
    pending: dict[str, dict[int, str]] = {}

    def _flush(page_file: str) -> None:
        if page_file in pending:
            _write_captions(page_file, pending.pop(page_file))

    captioned = 0
    with graceful_shutdown(message="Stage 1b interrupted; saving captions..."):
        prev_file: str | None = None
        for page_file, line_idx, region in tqdm(
            targets, desc="Captioning", unit="region"
        ):
            if prev_file is not None and page_file != prev_file:
                _flush(prev_file)
            prev_file = page_file

            with Image.open(region.crop_path) as image:
                caption = captioner.caption(image.convert("RGB"))
            pending.setdefault(page_file, {})[line_idx] = caption

            if has_index and caption:
                store.client.set_payload(
                    collection_name=schema.IMAGE_COLLECTION,
                    payload={"caption": caption},
                    points=[_point_id(region.region_id)],
                )
            captioned += 1
        if prev_file is not None:
            _flush(prev_file)

    captioner.unload()
    print(
        f"Stage 1b complete: captioned {captioned} regions "
        f"(index payloads updated: {has_index})."
    )
    print(f"Stage 1b finished in {format_duration(time.perf_counter() - start)}.")


def main() -> None:
    """Parses arguments and runs Stage 1b."""
    parser = argparse.ArgumentParser(description="Stage 1b: caption visual regions.")
    add_config_args(parser)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max regions to caption (default: eval.caption_sample_size; 0 = all).",
    )
    args = parser.parse_args()
    config = resolve_config(args)
    limit = config.eval.caption_sample_size if args.limit is None else args.limit
    run(config, limit=limit)


if __name__ == "__main__":
    main()
