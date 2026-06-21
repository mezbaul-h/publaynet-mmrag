"""Knowledge-graph construction over extracted entities and relations.

The graph is a NetworkX ``MultiDiGraph`` with three node types -- documents,
chunks and entities -- linked by ``HAS_CHUNK``, ``MENTIONS``, ``RELATES`` and
(optionally) ``CO_OCCURS`` edges. It is small enough for the proof-of-concept to
hold in memory and persist to GraphML; the same construction logic backs an
embedded graph database (e.g. Kuzu) at full scale.
"""

from __future__ import annotations

from typing import Iterable

import networkx as nx

from publaynet_mmrag.kg.extract import Entity, Triple
from publaynet_mmrag.types import Chunk

DOC = "document"
CHUNK = "chunk"
ENTITY = "entity"


def _entity_node(text: str, label: str) -> str:
    """Builds a canonical entity node id.

    Args:
        text: The entity surface form.
        label: The entity type.

    Returns:
        A normalised ``entity::{label}::{text}`` node identifier.
    """
    return f"entity::{label}::{text.lower().strip()}"


class KnowledgeGraphBuilder:
    """Incrementally builds the knowledge graph.

    Attributes:
        graph: The underlying NetworkX multigraph.
        cooccurrence: Whether to add entity co-occurrence edges per chunk.
    """

    def __init__(self, cooccurrence: bool = True) -> None:
        """Initialises an empty graph.

        Args:
            cooccurrence: Whether to add co-occurrence edges between entities
                appearing in the same chunk.
        """
        self.graph = nx.MultiDiGraph()
        self.cooccurrence = cooccurrence
        self._name_index: dict[str, str] = {}

    def _resolve_entity(self, surface: str, label: str) -> str:
        """Returns the node id for a surface form, reusing any existing node.

        Triple endpoints arrive as bare surface strings without a type. To
        avoid coining a duplicate node when the same surface form was already
        added as a typed entity, an existing node with the same (normalised)
        name is reused; otherwise a new node is created under ``label``.

        Args:
            surface: The entity surface form.
            label: The label to assign if a new node is created.

        Returns:
            The canonical entity node id.
        """
        key = surface.lower().strip()
        existing = self._name_index.get(key)
        if existing is not None:
            return existing
        node = _entity_node(surface, label)
        self.graph.add_node(node, ntype=ENTITY, label=label, name=surface.strip())
        self._name_index[key] = node
        return node

    def add_chunk(
        self,
        chunk: Chunk,
        entities: Iterable[Entity],
        triples: Iterable[Triple] | None = None,
    ) -> None:
        """Adds a chunk and its extracted knowledge to the graph.

        Args:
            chunk: The chunk being ingested.
            entities: Entities mentioned in the chunk.
            triples: Optional relation triples extracted from the chunk.
        """
        self.graph.add_node(chunk.doc_id, ntype=DOC)
        self.graph.add_node(
            chunk.chunk_id,
            ntype=CHUNK,
            doc_id=chunk.doc_id,
            page_index=chunk.page_index,
        )
        self.graph.add_edge(chunk.doc_id, chunk.chunk_id, key="HAS_CHUNK")

        entity_nodes: list[str] = []
        for entity in entities:
            node = self._resolve_entity(entity.text, entity.label)
            self.graph.add_edge(chunk.chunk_id, node, key="MENTIONS")
            entity_nodes.append(node)

        if self.cooccurrence:
            for i in range(len(entity_nodes)):
                for j in range(i + 1, len(entity_nodes)):
                    self.graph.add_edge(
                        entity_nodes[i], entity_nodes[j], key="CO_OCCURS"
                    )

        for triple in triples or []:
            subject = self._resolve_entity(triple.subject, "concept")
            obj = self._resolve_entity(triple.object, "concept")
            self.graph.add_edge(
                subject, obj, key="RELATES", relation=triple.relation,
                chunk_id=chunk.chunk_id,
            )

    def save(self, path: str) -> None:
        """Persists the graph to GraphML.

        Args:
            path: Destination ``.graphml`` path.
        """
        nx.write_graphml(self.graph, path)


def load_graph(path: str) -> nx.MultiDiGraph:
    """Loads a graph previously saved with :meth:`KnowledgeGraphBuilder.save`.

    Args:
        path: Source ``.graphml`` path.

    Returns:
        The loaded multigraph.
    """
    return nx.read_graphml(path)
