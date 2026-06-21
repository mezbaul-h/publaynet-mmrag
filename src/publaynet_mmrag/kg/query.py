"""Graph-augmented retrieval over the knowledge graph.

A query is linked to entity nodes by name matching; the neighbourhood is then
expanded up to ``hops`` edges to surface related entities and the documents that
mention them. The expansion returns both the document ids (used to pull
additional chunks) and a readable path trace (shown in the explainability view).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from publaynet_mmrag.kg.build import CHUNK, DOC, ENTITY


@dataclass
class GraphExpansion:
    """The result of expanding the graph around a query.

    Attributes:
        doc_ids: Documents reachable from the matched entities.
        entity_names: Names of the entities that anchored the expansion.
        paths: Readable relation paths for explainability.
    """

    doc_ids: list[str] = field(default_factory=list)
    entity_names: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)


def _match_entities(graph: nx.MultiDiGraph, query: str) -> list[str]:
    """Links a query to entity nodes by case-insensitive substring match.

    Args:
        graph: The knowledge graph.
        query: The query text.

    Returns:
        Matching entity node ids.
    """
    query_lower = query.lower()
    matches: list[str] = []
    seen_names: set[str] = set()
    for node, data in graph.nodes(data=True):
        if data.get("ntype") != ENTITY:
            continue
        name = str(data.get("name", "")).lower().strip()
        if name and name in query_lower and name not in seen_names:
            matches.append(node)
            seen_names.add(name)
    return matches


def expand(graph: nx.MultiDiGraph, query: str, hops: int = 1) -> GraphExpansion:
    """Expands the graph neighbourhood around query-linked entities.

    Args:
        graph: The knowledge graph.
        query: The natural-language query.
        hops: Number of edges to traverse from each matched entity.

    Returns:
        A :class:`GraphExpansion` with reachable documents and path traces.
    """
    anchors = _match_entities(graph, query)
    if not anchors:
        return GraphExpansion()

    undirected = graph.to_undirected(as_view=True)
    reached_entities: set[str] = set(anchors)
    for anchor in anchors:
        ego = nx.ego_graph(undirected, anchor, radius=hops)
        for node, data in ego.nodes(data=True):
            if data.get("ntype") == ENTITY:
                reached_entities.add(node)

    doc_ids: set[str] = set()
    paths: list[str] = []
    for entity in reached_entities:
        for predecessor in graph.predecessors(entity):
            pdata = graph.nodes[predecessor]
            if pdata.get("ntype") == CHUNK:
                doc_id = pdata.get("doc_id")
                if doc_id:
                    doc_ids.add(doc_id)
            elif pdata.get("ntype") == DOC:
                doc_ids.add(predecessor)

    for anchor in anchors:
        anchor_name = graph.nodes[anchor].get("name", anchor)
        for _, target, data in graph.out_edges(anchor, data=True):
            if data.get("key") == "RELATES" or "relation" in data:
                target_name = graph.nodes[target].get("name", target)
                rel = data.get("relation", "relates_to")
                paths.append(f"{anchor_name} --{rel}--> {target_name}")

    return GraphExpansion(
        doc_ids=sorted(doc_ids),
        entity_names=[graph.nodes[a].get("name", a) for a in anchors],
        paths=paths[:10],
    )
