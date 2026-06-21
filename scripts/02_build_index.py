#!/usr/bin/env python
"""Stage 2: build the vector index.

Loads the chunks and region files from Stage 1, embeds chunk text with BGE-M3
(dense + sparse) and visual-region crops with SigLIP2, and upserts both into the
embedded Qdrant store. Embedders are loaded, used, and unloaded in turn so they
do not co-reside unnecessarily.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from PIL import Image  # noqa: E402

from publaynet_mmrag.config import Config  # noqa: E402
from publaynet_mmrag.embed.image import ImageEmbedder  # noqa: E402
from publaynet_mmrag.embed.text import TextEmbedder  # noqa: E402
from publaynet_mmrag.index.store import VectorStore  # noqa: E402
from publaynet_mmrag.preprocess.regions import MIN_CROP_PX as _MIN_IMG_PX  # noqa: E402
from publaynet_mmrag.timing import format_duration  # noqa: E402
from publaynet_mmrag.types import Category, Chunk, Region, read_jsonl  # noqa: E402
from scripts._common import add_config_args, resolve_config  # noqa: E402


def _load_chunks(path: str) -> list[Chunk]:
    """Loads chunks from the Stage 1 chunk file.

    Args:
        path: The chunk JSONL path.

    Returns:
        The parsed chunks.
    """
    return [Chunk.from_dict(row) for row in read_jsonl(path)]


def _load_visual_regions(regions_dir: str) -> list[Region]:
    """Loads visual regions that have an on-disk crop.

    Args:
        regions_dir: Directory of per-document region files.

    Returns:
        Visual regions (figures, tables) with a crop path.
    """
    regions: list[Region] = []
    for path in glob.glob(os.path.join(regions_dir, "*.jsonl")):
        for row in read_jsonl(path):
            region = Region.from_dict(row)
            if region.category in Category.visual() and region.crop_path:
                regions.append(region)
    return regions


def run(config: Config, with_image: bool) -> None:
    """Runs Stage 2 indexing.

    Args:
        config: The active run configuration.
        with_image: Whether to index the visual modality.
    """
    chunks = _load_chunks(config.paths.chunks_path)
    import time

    _t0 = time.perf_counter()
    store = VectorStore(
        path=config.paths.qdrant_path,
        text_dim=config.models.text_embed_dim,
        image_dim=config.models.image_embed_dim,
    )
    store.create_collections(with_image=with_image)

    text_embedder = TextEmbedder(model_name=config.models.text_embed_model)
    pending = store.new_text_chunks(chunks)
    skipped = len(chunks) - len(pending)
    if not pending:
        print(f"All {len(chunks)} text chunks already indexed; nothing to do.")
    else:
        if skipped:
            print(
                f"Resuming: {skipped} text chunks already indexed, {len(pending)} to go."
            )
        text_embedder.load()
        batch = 64
        from tqdm import tqdm

        for start in tqdm(
            range(0, len(pending), batch), desc="Stage 2: text", unit="batch"
        ):
            part = pending[start : start + batch]
            embeddings = text_embedder.embed([c.text for c in part])
            store.upsert_text(part, embeddings)
        text_embedder.unload()
        print(f"Indexed {len(pending)} text chunks.")

    if with_image:
        regions = _load_visual_regions(config.paths.regions_dir)
        regions = store.new_image_regions(regions)
        if regions:
            image_embedder = ImageEmbedder(
                model_name=config.models.image_embed_model,
                device=config.models.device,
            )
            image_embedder.load()
            from tqdm import tqdm

            img_batch = 32
            indexed = 0
            for start in tqdm(
                range(0, len(regions), img_batch), desc="Stage 2: images", unit="batch"
            ):
                part = regions[start : start + img_batch]
                images = []
                kept = []
                for region in part:
                    try:
                        img = Image.open(region.crop_path).convert("RGB")
                    except Exception:
                        continue
                    # Skip crops too small for the vision processor (degenerate
                    # annotations); these would break SigLIP's preprocessing.
                    if img.width < _MIN_IMG_PX or img.height < _MIN_IMG_PX:
                        continue
                    images.append(img)
                    kept.append(region)
                if not images:
                    continue
                vectors = image_embedder.embed_images(images)
                store.upsert_images(kept, vectors)
                indexed += len(kept)
            image_embedder.unload()
            print(f"Indexed {indexed} visual regions.")
        else:
            print("No new visual regions to index.")

    print(f"Stage 2 finished in {format_duration(time.perf_counter() - _t0)}.")


def main() -> None:
    """Parses arguments and runs Stage 2."""
    parser = argparse.ArgumentParser(description="Stage 2: build the index.")
    add_config_args(parser)
    parser.add_argument(
        "--no-image",
        action="store_true",
        help="Skip indexing the visual modality.",
    )
    args = parser.parse_args()
    from publaynet_mmrag.shutdown import graceful_shutdown

    with graceful_shutdown(message="Stage 2 interrupted; re-run to resume."):
        run(resolve_config(args), with_image=not args.no_image)


if __name__ == "__main__":
    main()
