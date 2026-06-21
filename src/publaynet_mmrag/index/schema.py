"""Collection names, vector names and payload schema for the index.

Two collections are used: a text collection with named dense and sparse vectors
(for hybrid search), and an image collection with a single dense vector for
SigLIP2 figure/table embeddings. Payloads carry full provenance (document, page,
category, source region ids) so retrieval results can be filtered and grounded.
"""

from __future__ import annotations

TEXT_COLLECTION = "text_chunks"
IMAGE_COLLECTION = "visual_regions"

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
IMAGE_VECTOR = "image"


def text_payload(chunk) -> dict:
    """Builds the Qdrant payload for a text chunk.

    Args:
        chunk: The :class:`~publaynet_mmrag.types.Chunk` being indexed.

    Returns:
        A JSON-serialisable payload dictionary.
    """
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "page_index": chunk.page_index,
        "text": chunk.text,
        "region_ids": chunk.region_ids,
        "category": int(chunk.category),
        "modality": "text",
    }


def image_payload(region) -> dict:
    """Builds the Qdrant payload for a visual region.

    Args:
        region: The :class:`~publaynet_mmrag.types.Region` being indexed.

    Returns:
        A JSON-serialisable payload dictionary.
    """
    return {
        "region_id": region.region_id,
        "doc_id": region.doc_id,
        "page_index": region.page_index,
        "category": int(region.category),
        "caption": region.caption or "",
        "crop_path": region.crop_path or "",
        "modality": "image",
    }
