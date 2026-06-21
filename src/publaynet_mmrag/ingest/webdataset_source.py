"""WebDataset ingestion for the ``lhoestq/small-publaynet-wds`` subset.

Each sample in the shards carries a ``__key__``, a ``png`` page image and a
``json`` dictionary holding COCO ``annotations`` for that page. The shards are
streamed so the whole 1.22 GB subset never has to be materialised at once.
"""

from __future__ import annotations

from typing import Any, Iterator

from publaynet_mmrag.ingest.base import DocumentSource, PageSample, parse_key


class WebDatasetSource(DocumentSource):
    """Streams pages from the PubLayNet WebDataset subset.

    Attributes:
        shard_urls: Remote or local ``.tar`` shard locations.
        max_pages: Optional cap on the number of pages yielded (``0`` = all).
    """

    def __init__(
        self,
        shard_urls: list[str],
        max_pages: int = 0,
        streaming: bool = True,
    ) -> None:
        """Initialises the source.

        Args:
            shard_urls: The WebDataset shard URLs to read.
            max_pages: Maximum pages to yield; ``0`` yields every page.
            streaming: If ``True``, stream shards on the fly (no up-front
                download). If ``False``, download the shards first -- which
                shows a progress bar and is fine for the small subset -- then
                iterate from the local cache.
        """
        self.shard_urls = shard_urls
        self.max_pages = max_pages
        self.streaming = streaming
        self._dataset = None

    def _load(self):
        """Loads (and caches) the underlying Hugging Face dataset.

        Returns:
            The streaming or in-memory dataset over the shards.
        """
        if self._dataset is None:
            from datasets import load_dataset

            self._dataset = load_dataset(
                "webdataset",
                data_files={"train": self.shard_urls},
                split="train",
                streaming=self.streaming,
            )
        return self._dataset

    def expected_len(self):
        """Returns the page count when not streaming, else ``None``.

        Returns:
            The (optionally capped) number of pages, or ``None`` while
            streaming, where the length is not known up front.
        """
        if self.streaming:
            return None
        n = len(self._load())
        return min(n, self.max_pages) if self.max_pages else n

    def __iter__(self) -> Iterator[PageSample]:
        """Streams page samples from the configured shards.

        Yields:
            One :class:`PageSample` per page in the subset.
        """
        dataset = self._load()

        count = 0
        for sample in dataset:
            if self.max_pages and count >= self.max_pages:
                break
            page = self._to_page_sample(sample)
            if page is None:
                continue
            yield page
            count += 1

    @staticmethod
    def _to_page_sample(sample: dict[str, Any]) -> PageSample | None:
        """Maps a raw WebDataset sample to a :class:`PageSample`.

        Args:
            sample: A decoded WebDataset record.

        Returns:
            The mapped page sample, or ``None`` if the record is malformed.
        """
        key = sample.get("__key__")
        image = sample.get("png")
        meta = sample.get("json")
        if key is None or image is None or meta is None:
            return None
        if image.mode != "RGB":
            image = image.convert("RGB")

        annotations = meta.get("annotations", []) if isinstance(meta, dict) else []
        doc_id, page_index = parse_key(key)
        return PageSample(
            key=key,
            doc_id=doc_id,
            page_index=page_index,
            image=image,
            width=image.width,
            height=image.height,
            annotations=annotations,
        )
