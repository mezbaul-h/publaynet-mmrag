"""Source-agnostic ingestion interface.

The whole pipeline consumes :class:`PageSample` objects and never sees the
underlying storage format. The subset ships as WebDataset shards while the full
PubLayNet release ships as a monolithic COCO annotation file plus image folders;
both are exposed here behind one iterator, so scaling from the proof-of-concept
to the full dataset is a single configuration change.
"""

from __future__ import annotations

import abc
import re
from dataclasses import dataclass
from typing import Any, Iterator, Optional

# ``PIL.Image.Image`` is only needed for typing; import lazily where heavy.
try:  # pragma: no cover - import guard for type checkers without Pillow.
    from PIL.Image import Image as PILImage
except Exception:  # pragma: no cover
    PILImage = Any  # type: ignore[assignment,misc]


_KEY_RE = re.compile(r"^(?P<doc>.+?)_(?P<page>\d+)$")


def parse_key(key: str) -> tuple[str, int]:
    """Splits a dataset sample key into document id and page index.

    Keys follow the pattern ``{doc_id}_{page:05d}`` (e.g. ``PMC4991227_00003``).

    Args:
        key: The raw sample key.

    Returns:
        A ``(doc_id, page_index)`` tuple. If the key does not match the
        expected pattern, the whole key is returned as the document id with a
        page index of ``0``.
    """
    match = _KEY_RE.match(key)
    if not match:
        return key, 0
    return match.group("doc"), int(match.group("page"))


@dataclass
class PageSample:
    """A single page image with its COCO layout annotations.

    Attributes:
        key: The dataset sample key (e.g. ``PMC4991227_00003``).
        doc_id: Document identifier parsed from the key.
        page_index: Page ordinal parsed from the key.
        image: The page image as a Pillow image.
        width: Image width in pixels.
        height: Image height in pixels.
        annotations: COCO-style annotation dicts, each with ``category_id``,
            ``bbox`` and an annotation ``id``.
    """

    key: str
    doc_id: str
    page_index: int
    image: "PILImage"
    width: int
    height: int
    annotations: list[dict[str, Any]]


class DocumentSource(abc.ABC):
    """Abstract iterator over annotated pages.

    Implementations yield :class:`PageSample` objects in any order. Downstream
    stages depend only on this contract.
    """

    @abc.abstractmethod
    def __iter__(self) -> Iterator[PageSample]:
        """Yields page samples.

        Yields:
            One :class:`PageSample` per page.
        """
        raise NotImplementedError

    def expected_len(self) -> Optional[int]:
        """Returns the page count if cheaply knowable, else ``None``.

        Used only to give progress bars a total (and therefore an ETA). A
        ``None`` result means the bar shows a running count and rate instead.

        Returns:
            The number of pages, or ``None`` for streaming/unknown sources.
        """
        return None


def build_source(config: Any) -> DocumentSource:
    """Constructs the configured ingestion backend.

    Args:
        config: The validated run configuration.

    Returns:
        A concrete :class:`DocumentSource`.

    Raises:
        ValueError: If ``config.ingest.source`` is not recognised.
    """
    source = config.ingest.source
    if source == "webdataset":
        from publaynet_mmrag.ingest.webdataset_source import WebDatasetSource

        return WebDatasetSource(
            shard_urls=config.ingest.shard_urls,
            max_pages=config.ingest.max_pages,
            streaming=config.ingest.streaming,
        )
    if source == "coco":
        from publaynet_mmrag.ingest.coco_source import CocoSource

        return CocoSource(
            annotations_path=config.ingest.coco_annotations,
            image_dir=config.ingest.coco_image_dir,
            max_pages=config.ingest.max_pages,
        )
    raise ValueError(f"Unknown ingestion source: {source!r}")
