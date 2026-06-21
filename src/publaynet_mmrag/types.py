"""Data contracts shared across pipeline stages.

These lightweight, serialisable structures are the only objects that cross
stage boundaries (each stage writes them to disk and the next reads them back).
Keeping them framework-free avoids coupling the contract to any one model
library and keeps the pipeline resumable.
"""

from __future__ import annotations

import dataclasses
import enum
import json
from dataclasses import dataclass, field
from typing import Any, Optional


class Category(enum.IntEnum):
    """PubLayNet COCO layout categories.

    The integer values match the ``category_id`` field in the dataset
    annotations. PubLayNet defines exactly these five classes.
    """

    TEXT = 1
    TITLE = 2
    LIST = 3
    TABLE = 4
    FIGURE = 5

    @classmethod
    def textual(cls) -> set["Category"]:
        """Returns categories whose content is read with OCR.

        Returns:
            The set of categories treated as text-bearing (text, title, list).
        """
        return {cls.TEXT, cls.TITLE, cls.LIST}

    @classmethod
    def visual(cls) -> set["Category"]:
        """Returns categories embedded with the vision model.

        Returns:
            The set of categories treated as visual (figure, table).
        """
        return {cls.FIGURE, cls.TABLE}


@dataclass
class BBox:
    """Axis-aligned bounding box in COCO ``[x, y, w, h]`` pixel format.

    Attributes:
        x: Left edge in pixels.
        y: Top edge in pixels.
        w: Width in pixels.
        h: Height in pixels.
    """

    x: float
    y: float
    w: float
    h: float

    def xyxy(self) -> tuple[int, int, int, int]:
        """Converts to integer ``(x1, y1, x2, y2)`` corner format.

        Returns:
            The top-left and bottom-right corners as integers, suitable for
            ``PIL.Image.crop``.
        """
        return (int(self.x), int(self.y), int(self.x + self.w), int(self.y + self.h))

    @classmethod
    def from_coco(cls, bbox: list[float]) -> "BBox":
        """Builds a ``BBox`` from a COCO ``[x, y, w, h]`` list.

        Args:
            bbox: A four-element list in COCO bounding-box format.

        Returns:
            The corresponding ``BBox`` instance.
        """
        return cls(x=bbox[0], y=bbox[1], w=bbox[2], h=bbox[3])


@dataclass
class Region:
    """A single annotated layout region on a page.

    Attributes:
        region_id: Stable unique identifier (``{doc_id}:{page_index}:{ann_id}``).
        doc_id: Source document identifier (e.g. ``PMC4991227``).
        page_key: Dataset sample key (e.g. ``PMC4991227_00003``).
        page_index: Zero-based page ordinal within the document.
        category: Layout category of the region.
        bbox: Bounding box of the region in page pixel coordinates.
        reading_order: Index of the region in top-to-bottom reading order.
        text: OCR-extracted text (text-bearing regions only).
        caption: Optional model-generated caption (visual regions only).
        crop_path: On-disk path to the saved crop (visual regions only).
    """

    region_id: str
    doc_id: str
    page_key: str
    page_index: int
    category: Category
    bbox: BBox
    reading_order: int = 0
    text: Optional[str] = None
    caption: Optional[str] = None
    crop_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialises the region to a JSON-compatible dictionary.

        Returns:
            A dictionary with enums and nested dataclasses flattened.
        """
        out = dataclasses.asdict(self)
        out["category"] = int(self.category)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Region":
        """Reconstructs a region from a serialised dictionary.

        Args:
            data: A dictionary previously produced by :meth:`to_dict`.

        Returns:
            The reconstructed ``Region`` instance.
        """
        return cls(
            region_id=data["region_id"],
            doc_id=data["doc_id"],
            page_key=data["page_key"],
            page_index=data["page_index"],
            category=Category(data["category"]),
            bbox=BBox(**data["bbox"]),
            reading_order=data.get("reading_order", 0),
            text=data.get("text"),
            caption=data.get("caption"),
            crop_path=data.get("crop_path"),
        )


@dataclass
class Chunk:
    """A retrievable text unit assembled from one or more regions.

    A chunk is the granularity at which text is embedded and retrieved. It
    carries full provenance back to its source regions so retrieval results can
    be visually grounded for the explainability stage.

    Attributes:
        chunk_id: Stable unique identifier for the chunk.
        doc_id: Source document identifier.
        page_index: Page the chunk originates from.
        text: Concatenated, reading-order text content.
        region_ids: Identifiers of the regions composing the chunk.
        category: Dominant category of the source regions.
    """

    chunk_id: str
    doc_id: str
    page_index: int
    text: str
    region_ids: list[str] = field(default_factory=list)
    category: Category = Category.TEXT

    def to_dict(self) -> dict[str, Any]:
        """Serialises the chunk to a JSON-compatible dictionary.

        Returns:
            A dictionary with the category enum flattened to its integer value.
        """
        out = dataclasses.asdict(self)
        out["category"] = int(self.category)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        """Reconstructs a chunk from a serialised dictionary.

        Args:
            data: A dictionary previously produced by :meth:`to_dict`.

        Returns:
            The reconstructed ``Chunk`` instance.
        """
        return cls(
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            page_index=data["page_index"],
            text=data["text"],
            region_ids=data.get("region_ids", []),
            category=Category(data.get("category", Category.TEXT)),
        )


@dataclass
class RetrievedItem:
    """A single retrieval hit with its provenance and score.

    Attributes:
        chunk_id: Identifier of the retrieved chunk (text hits).
        region_id: Identifier of the retrieved region (image hits).
        doc_id: Source document identifier.
        page_index: Source page index.
        score: Retrieval or rerank score (higher is more relevant).
        modality: Source modality, either ``"text"`` or ``"image"``.
        text: Chunk text or figure caption used as context.
        crop_path: On-disk crop path for image hits, if available.
        source: Retrieval channel that surfaced the item (for explainability).
    """

    doc_id: str
    page_index: int
    score: float
    modality: str
    text: str = ""
    chunk_id: Optional[str] = None
    region_id: Optional[str] = None
    crop_path: Optional[str] = None
    source: str = ""


@dataclass
class Answer:
    """A generated answer together with its supporting evidence.

    Attributes:
        question: The original natural-language query.
        text: The generated answer text.
        reasoning: Optional chain-of-thought trace, when surfaced.
        citations: Region/chunk identifiers cited in the answer.
        evidence: The retrieved items passed to the generator.
        graph_paths: Knowledge-graph paths used during retrieval, if any.
    """

    question: str
    text: str
    reasoning: str = ""
    citations: list[str] = field(default_factory=list)
    evidence: list[RetrievedItem] = field(default_factory=list)
    graph_paths: list[str] = field(default_factory=list)


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    """Writes an iterable of dictionaries to a JSON Lines file.

    Args:
        path: Destination file path.
        rows: Rows to serialise, one JSON object per line.
    """
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict[str, Any]]:
    """Reads a JSON Lines file into a list of dictionaries.

    Args:
        path: Source file path.

    Returns:
        The parsed rows in file order.
    """
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
