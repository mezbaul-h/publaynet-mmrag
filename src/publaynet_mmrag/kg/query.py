"""Graph-augmented retrieval over the knowledge graph.

A query is linked to entity nodes by name matching; the neighbourhood is then
expanded up to ``hops`` edges to surface related entities and the chunks that
mention them. The expansion returns a *proximity-ranked* list of chunk ids (used
as a retrieval channel that can promote a specific bridge chunk vector search
ranked poorly), the documents reached, and a readable path trace (shown in the
explainability view).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import networkx as nx

from publaynet_mmrag.kg.build import CHUNK, DOC, ENTITY

_MIN_ENTITY_CHARS = 4
_MAX_ANCHORS = 8  # Keep only the most specific matched entities.
_FREQ_CAP = 40  # Skip entities mentioned in more chunks than this (too generic).


def _chunk_frequency(graph: nx.MultiDiGraph, node: str) -> int:
    """Counts how many distinct chunks mention an entity (its document freq).

    Args:
        graph: The knowledge graph.
        node: An entity node id.

    Returns:
        The number of distinct chunk nodes with a ``MENTIONS`` edge to ``node``.
    """
    return sum(
        1 for p in graph.predecessors(node) if graph.nodes[p].get("ntype") == CHUNK
    )


def _anchor_index(graph: nx.MultiDiGraph) -> list[tuple[str, str, int]]:
    """Returns (and caches) the candidate-anchor index for a graph.

    Built once per loaded graph: every entity long enough and specific enough
    (chunk frequency in ``(0, _FREQ_CAP]``) as ``(name_lower, node, freq)``.
    Caching this on the graph turns each query's entity linking into a cheap
    substring scan instead of recomputing frequencies over all ~46k entities.

    Args:
        graph: The knowledge graph.

    Returns:
        The cached list of candidate anchors.
    """
    index = graph.graph.get("_anchor_index")
    if index is None:
        index = []
        for node, data in graph.nodes(data=True):
            if data.get("ntype") != ENTITY:
                continue
            name = str(data.get("name", "")).lower().strip()
            if len(name) < _MIN_ENTITY_CHARS:
                continue
            freq = _chunk_frequency(graph, node)
            if 0 < freq <= _FREQ_CAP:
                index.append((name, node, freq))
        graph.graph["_anchor_index"] = index
    return index


@dataclass
class GraphExpansion:
    """The result of expanding the graph around a query.

    Attributes:
        chunk_ids: Chunks mentioning reached entities, in graph-proximity order
            (closest/most-connected first). This is the graph retrieval channel.
        doc_ids: Documents reachable from the matched entities.
        entity_names: Names of the entities that anchored the expansion.
        paths: Readable relation paths for explainability.
    """

    chunk_ids: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    entity_names: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)


def _match_entities(graph: nx.MultiDiGraph, query: str) -> list[str]:
    """Links a query to its most *specific* entity nodes.

    Matching is by word boundary rather than bare substring, and entity names
    shorter than ``_MIN_ENTITY_CHARS`` are ignored, so short or noisy surface
    forms do not match inside unrelated words. Matched entities that appear in
    more than ``_FREQ_CAP`` chunks are dropped as too generic (e.g. "study",
    "data"); of those that remain the rarest ``_MAX_ANCHORS`` are kept, since a
    low-frequency entity is a far more informative anchor for graph expansion.

    Args:
        graph: The knowledge graph.
        query: The query text.

    Returns:
        Up to ``_MAX_ANCHORS`` entity node ids, most specific first.
    """
    query_lower = query.lower()
    matches: list[tuple[int, str]] = []
    seen_names: set[str] = set()
    for name, node, freq in _anchor_index(graph):
        if name in seen_names or name not in query_lower:
            continue
        if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", query_lower):
            seen_names.add(name)
            matches.append((freq, node))
    matches.sort(key=lambda pair: pair[0])
    return [node for _, node in matches[:_MAX_ANCHORS]]


def expand(graph: nx.MultiDiGraph, query: str, hops: int = 1) -> GraphExpansion:
    """Expands the graph neighbourhood around query-linked entities.

    From each anchored entity the entity neighbourhood is explored up to ``hops``
    edges. Every chunk that mentions a reached entity is scored by proximity --
    closer entities and chunks touching several reached entities score higher --
    and the chunks are returned in that order, so a multi-hop bridge chunk that
    vector search ranks poorly can be promoted by the graph channel.

    Args:
        graph: The knowledge graph.
        query: The natural-language query.
        hops: Number of edges to traverse from each matched entity.

    Returns:
        A :class:`GraphExpansion` with proximity-ranked chunks, reached
        documents and path traces.
    """
    anchors = _match_entities(graph, query)
    if not anchors:
        return GraphExpansion()

    undirected = graph.to_undirected(as_view=True)

    # Distance of each reached entity from the nearest anchor (0 = an anchor).
    entity_dist: dict[str, int] = {}
    for anchor in anchors:
        lengths = nx.single_source_shortest_path_length(undirected, anchor, cutoff=hops)
        for node, dist in lengths.items():
            if graph.nodes[node].get("ntype") != ENTITY:
                continue
            if node not in entity_dist or dist < entity_dist[node]:
                entity_dist[node] = dist

    # Score chunks that mention reached entities: closer entities contribute more
    # (1 / (dist + 1)), and chunks touching several reached entities accumulate.
    chunk_score: dict[str, float] = {}
    doc_ids: set[str] = set()
    for entity, dist in entity_dist.items():
        weight = 1.0 / (dist + 1)
        for predecessor in graph.predecessors(entity):
            pdata = graph.nodes[predecessor]
            if pdata.get("ntype") == CHUNK:
                chunk_score[predecessor] = chunk_score.get(predecessor, 0.0) + weight
                doc_id = pdata.get("doc_id")
                if doc_id:
                    doc_ids.add(doc_id)
            elif pdata.get("ntype") == DOC:
                doc_ids.add(predecessor)

    ranked_chunks = sorted(chunk_score, key=lambda cid: chunk_score[cid], reverse=True)

    paths: list[str] = []
    for anchor in anchors:
        anchor_name = graph.nodes[anchor].get("name", anchor)
        for _, target, data in graph.out_edges(anchor, data=True):
            if data.get("key") == "RELATES" or "relation" in data:
                target_name = graph.nodes[target].get("name", target)
                rel = data.get("relation", "relates_to")
                paths.append(f"{anchor_name} --{rel}--> {target_name}")

    return GraphExpansion(
        chunk_ids=ranked_chunks,
        doc_ids=sorted(doc_ids),
        entity_names=[graph.nodes[a].get("name", a) for a in anchors],
        paths=paths[:10],
    )
