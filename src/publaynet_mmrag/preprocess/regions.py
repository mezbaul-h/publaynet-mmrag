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

# Minimum crop edge (px). Smaller visual crops (degenerate annotations) confuse
# the SigLIP image processor's channel-dimension inference, so they are skipped.
MIN_CROP_PX = 8


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


def _clamp_box(bbox: BBox, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Clamps a region box to the image bounds.

    Dataset annotations occasionally extend past the page edge (or are
    degenerate). PIL allows a crop box outside the image but then fails to save
    the partially-materialised result, so the box is clamped here and zero-area
    boxes are rejected.

    Args:
        bbox: The region bounding box.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        A clamped ``(x0, y0, x1, y1)`` tuple, or ``None`` if the box has no
        area inside the image.
    """
    x0, y0, x1, y1 = bbox.xyxy()
    x0 = max(0, min(int(x0), width))
    y0 = max(0, min(int(y0), height))
    x1 = max(0, min(int(x1), width))
    y1 = max(0, min(int(y1), height))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


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
            box = _clamp_box(bbox, page.image.width, page.image.height)
            if (
                box is not None
                and (box[2] - box[0]) >= MIN_CROP_PX
                and (box[3] - box[1]) >= MIN_CROP_PX
            ):
                crop = page.image.crop(box)
                crop.load()  # Materialise before save to avoid lazy-crop errors.
                crop_path = os.path.join(
                    crops_dir, region_id.replace(":", "_") + ".png"
                )
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
        A Pillow image of the cropped region; a 1x1 white image if the region
        box has no area inside the page.
    """
    box = _clamp_box(region.bbox, page.image.width, page.image.height)
    if box is None:
        from PIL import Image

        return Image.new("RGB", (1, 1), (255, 255, 255))
    crop = page.image.crop(box)
    crop.load()
    return crop
