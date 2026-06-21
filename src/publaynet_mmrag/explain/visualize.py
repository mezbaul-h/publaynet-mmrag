"""Visual grounding for explanations.

Two renderers: one draws cited region bounding boxes onto the page image; the
other renders the knowledge-graph subgraph used during retrieval. Both are
optional conveniences for the demo and are import-guarded so the core pipeline
does not depend on matplotlib.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from publaynet_mmrag.types import BBox, Region

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image as PILImage

_CITED_COLOUR = (215, 90, 48)  # Coral.
_OTHER_COLOUR = (136, 135, 128)  # Grey.


def highlight_regions(
    page_image: "PILImage",
    regions: Iterable[Region],
    cited_ids: set[str],
    width: int = 4,
) -> "PILImage":
    """Draws region boxes on a copy of the page image.

    Cited regions are outlined in coral; other retrieved regions in grey.

    Args:
        page_image: The full page image.
        regions: Regions to outline.
        cited_ids: Identifiers of regions the answer cited.
        width: Outline stroke width in pixels.

    Returns:
        A new image with the boxes drawn on.
    """
    from PIL import ImageDraw

    canvas = page_image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for region in regions:
        colour = _CITED_COLOUR if region.region_id in cited_ids else _OTHER_COLOUR
        box: BBox = region.bbox
        draw.rectangle(box.xyxy(), outline=colour, width=width)
    return canvas


def render_subgraph(graph, entity_names: list[str], out_path: str, hops: int = 1):
    """Renders the local subgraph around the given entities to a PNG.

    Args:
        graph: The full knowledge graph.
        entity_names: Entity names to centre the subgraph on.
        out_path: Destination PNG path.
        hops: Neighbourhood radius to draw.

    Returns:
        The output path on success, or ``None`` if rendering is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
    except Exception:  # pragma: no cover
        return None

    from publaynet_mmrag.kg.build import ENTITY

    anchors = [
        node
        for node, data in graph.nodes(data=True)
        if data.get("ntype") == ENTITY
        and str(data.get("name", "")).lower() in {name.lower() for name in entity_names}
    ]
    if not anchors:
        return None

    undirected = graph.to_undirected(as_view=True)
    nodes: set[str] = set()
    for anchor in anchors:
        nodes |= set(nx.ego_graph(undirected, anchor, radius=hops).nodes())
    sub = graph.subgraph(nodes)

    labels = {
        n: graph.nodes[n].get("name", str(n))[:24]
        for n in sub.nodes()
        if graph.nodes[n].get("ntype") == ENTITY
    }
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(sub, seed=42)
    nx.draw_networkx_nodes(sub, pos, node_size=400, node_color="#AFA9EC")
    nx.draw_networkx_edges(sub, pos, alpha=0.4)
    nx.draw_networkx_labels(sub, pos, labels=labels, font_size=8)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    return out_path
