"""COCO ingestion for the full PubLayNet release.

The full dataset is distributed as a single COCO-format annotation file
(``train.json`` / ``val.json``) keyed by ``image_id``, plus image folders. This
loader joins annotations to images by ``image_id`` and yields the same
:class:`PageSample` contract as the subset loader, so no downstream code
changes when moving from the proof-of-concept to the ~96 GB full set.

The annotation file is large; it is parsed once and indexed by image id. For
very large runs prefer the streaming ``ijson`` path (left as a documented
extension) over loading the entire JSON into memory.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Iterator

from publaynet_mmrag.ingest.base import DocumentSource, PageSample, parse_key


class CocoSource(DocumentSource):
    """Iterates pages from a COCO annotation file and an image directory.

    Attributes:
        annotations_path: Path to the COCO JSON annotation file.
        image_dir: Directory containing the page images.
        max_pages: Optional cap on the number of pages yielded (``0`` = all).
    """

    def __init__(
        self,
        annotations_path: str,
        image_dir: str,
        max_pages: int = 0,
    ) -> None:
        """Initialises the source.

        Args:
            annotations_path: Path to the COCO JSON annotation file.
            image_dir: Directory containing the page images.
            max_pages: Maximum pages to yield; ``0`` yields every page.
        """
        self.annotations_path = annotations_path
        self.image_dir = image_dir
        self.max_pages = max_pages

    def __iter__(self) -> Iterator[PageSample]:
        """Streams page samples joined from annotations and images.

        Yields:
            One :class:`PageSample` per image in the annotation file.
        """
        from PIL import Image

        with open(self.annotations_path, "r", encoding="utf-8") as handle:
            coco = json.load(handle)

        anns_by_image: dict[int, list[dict]] = defaultdict(list)
        for ann in coco.get("annotations", []):
            anns_by_image[ann["image_id"]].append(ann)

        cat_name = {c["id"]: c["name"] for c in coco.get("categories", [])}

        count = 0
        for image_meta in coco.get("images", []):
            if self.max_pages and count >= self.max_pages:
                break
            file_name = image_meta["file_name"]
            image_path = os.path.join(self.image_dir, file_name)
            if not os.path.exists(image_path):
                continue
            image = Image.open(image_path).convert("RGB")

            annotations = anns_by_image.get(image_meta["id"], [])
            for ann in annotations:
                ann.setdefault("category_name", cat_name.get(ann["category_id"], ""))

            key = os.path.splitext(file_name)[0]
            doc_id, page_index = parse_key(key)
            yield PageSample(
                key=key,
                doc_id=doc_id,
                page_index=page_index,
                image=image,
                width=image_meta.get("width", image.width),
                height=image_meta.get("height", image.height),
                annotations=annotations,
            )
            count += 1
