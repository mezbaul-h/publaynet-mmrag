"""Region extraction from page annotations.

Because the dataset ships ground-truth layout boxes, regions are produced
directly from the annotations rather than by running a layout detector. Visual
regions (figures, tables) are cropped and saved to disk for the vision encoder,
captioner and explainability overlay; text-bearing regions are left for OCR.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from publaynet_mmrag.types import BBox, Category, Region

if TYPE_CHECKING:  # pragma: no cover
    from publaynet_mmrag.ingest.base import PageSample


def _reading_order(annotations: list[dict]) -> list[int]:
    """Computes a top-to-bottom, left-to-right reading order.

    Args:
        annotations: COCO annotations for a single page.

    Returns:
        Indices into ``annotations`` sorted by vertical then horizontal
        position, using a coarse row band to group side-by-side regions.
    """

    def sort_key(index: int) -> tuple[float, float]:
        bbox = annotations[index]["bbox"]
        # Quantise the y coordinate into 20 px bands so columns read sensibly.
        return (round(bbox[1] / 20.0), bbox[0])

    return sorted(range(len(annotations)), key=sort_key)


def extract_regions(
    page: "PageSample",
    crops_dir: str,
    save_visual_crops: bool = True,
) -> list[Region]:
    """Builds :class:`Region` objects for a page.

    Args:
        page: The source page sample.
        crops_dir: Directory in which to save visual-region crops.
        save_visual_crops: Whether to crop and persist figure/table regions.

    Returns:
        The page's regions in reading order, with crop paths populated for
        visual regions.
    """
    os.makedirs(crops_dir, exist_ok=True)
    order = _reading_order(page.annotations)
    order_lookup = {idx: rank for rank, idx in enumerate(order)}

    regions: list[Region] = []
    for index, ann in enumerate(page.annotations):
        try:
            category = Category(ann["category_id"])
        except ValueError:
            continue
        bbox = BBox.from_coco(ann["bbox"])
        ann_id = ann.get("id", index)
        region_id = f"{page.doc_id}:{page.page_index}:{ann_id}"

        region = Region(
            region_id=region_id,
            doc_id=page.doc_id,
            page_key=page.key,
            page_index=page.page_index,
            category=category,
            bbox=bbox,
            reading_order=order_lookup.get(index, index),
        )

        if save_visual_crops and category in Category.visual():
            crop = page.image.crop(bbox.xyxy())
            crop_path = os.path.join(crops_dir, region_id.replace(":", "_") + ".png")
            crop.save(crop_path)
            region.crop_path = crop_path

        regions.append(region)

    regions.sort(key=lambda r: r.reading_order)
    return regions


def crop_region(page: "PageSample", region: Region):
    """Crops the image patch for a region.

    Args:
        page: The page the region belongs to.
        region: The region to crop.

    Returns:
        A Pillow image of the cropped region.
    """
    return page.image.crop(region.bbox.xyxy())
