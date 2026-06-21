"""Chunk assembly from OCR'd regions.

Chunks respect reading order and page boundaries, and every chunk records the
region identifiers it was built from. That provenance is what lets a retrieval
hit be traced back to specific bounding boxes for the explainability overlay.
"""

from __future__ import annotations

from publaynet_mmrag.types import Category, Chunk, Region


def chunk_page_regions(
    regions: list[Region],
    max_chars: int,
    overlap_chars: int,
    min_chars: int,
) -> list[Chunk]:
    """Packs a single page's text regions into overlapping chunks.

    Regions are concatenated in reading order; a new chunk is started when the
    accumulated length would exceed ``max_chars``. A trailing overlap is carried
    into the next chunk to preserve context across boundaries.

    Args:
        regions: All regions for one page, in reading order.
        max_chars: Soft maximum chunk length in characters.
        overlap_chars: Characters of trailing context carried to the next chunk.
        min_chars: Minimum length for a region's text to be included.

    Returns:
        The page's chunks, each carrying its source region identifiers.
    """
    text_regions = [
        r
        for r in regions
        if r.category in Category.textual() and r.text and len(r.text) >= min_chars
    ]
    if not text_regions:
        return []

    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_ids: list[str] = []
    buffer_len = 0
    doc_id = text_regions[0].doc_id
    page_index = text_regions[0].page_index
    seq = 0

    def flush() -> str:
        """Emits the current buffer as a chunk and returns the overlap tail."""
        nonlocal seq, buffer, buffer_ids, buffer_len
        text = "\n".join(buffer).strip()
        if not text:
            return ""
        chunk = Chunk(
            chunk_id=f"{doc_id}:{page_index}:c{seq}",
            doc_id=doc_id,
            page_index=page_index,
            text=text,
            region_ids=list(buffer_ids),
            category=Category.TEXT,
        )
        chunks.append(chunk)
        seq += 1
        tail = text[-overlap_chars:] if overlap_chars else ""
        return tail

    for region in text_regions:
        region_text = region.text or ""
        if buffer_len + len(region_text) > max_chars and buffer:
            tail = flush()
            buffer = [tail] if tail else []
            buffer_ids = []
            buffer_len = len(tail)
        buffer.append(region_text)
        buffer_ids.append(region.region_id)
        buffer_len += len(region_text)

    flush()
    return chunks
