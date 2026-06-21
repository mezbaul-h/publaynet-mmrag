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
    store = VectorStore(
        path=config.paths.qdrant_path,
        text_dim=config.models.text_embed_dim,
        image_dim=config.models.image_embed_dim,
    )
    store.create_collections(with_image=with_image)

    text_embedder = TextEmbedder(model_name=config.models.text_embed_model)
    text_embedder.load()
    batch = 64
    from tqdm import tqdm

    for start in tqdm(
        range(0, len(chunks), batch), desc="Stage 2: text", unit="batch"
    ):
        part = chunks[start : start + batch]
        embeddings = text_embedder.embed([c.text for c in part])
        store.upsert_text(part, embeddings)
    text_embedder.unload()
    print(f"Indexed {len(chunks)} text chunks.")

    if with_image:
        regions = _load_visual_regions(config.paths.regions_dir)
        if regions:
            image_embedder = ImageEmbedder(
                model_name=config.models.image_embed_model,
                device=config.models.device,
            )
            image_embedder.load()
            from tqdm import tqdm

            img_batch = 32
            for start in tqdm(
                range(0, len(regions), img_batch), desc="Stage 2: images", unit="batch"
            ):
                part = regions[start : start + img_batch]
                images = [Image.open(r.crop_path).convert("RGB") for r in part]
                vectors = image_embedder.embed_images(images)
                store.upsert_images(part, vectors)
            image_embedder.unload()
            print(f"Indexed {len(regions)} visual regions.")
        else:
            print("No visual regions with crops found; skipping image index.")


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
    run(resolve_config(args), with_image=not args.no_image)


if __name__ == "__main__":
    main()
