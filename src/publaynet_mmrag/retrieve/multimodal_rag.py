"""Configurable retriever covering both the baseline and enhanced pipelines.

A single retriever implements both arms of the comparison; the
:class:`~publaynet_mmrag.config.RetrievalConfig` flags decide which channels are
active:

* ``use_sparse``  -- dense-only vs dense+sparse hybrid text search.
* ``use_image``   -- add SigLIP2 figure/table retrieval (text-to-image).
* ``use_graph``   -- add knowledge-graph expansion to pull connected chunks.
* ``use_rerank``  -- apply the cross-encoder over the fused candidate pool.

The baseline config disables all four, so the only differences between the two
arms are modality and structured knowledge -- the variables under study.

**Fusion.** Each active channel produces a *ranked* candidate list. The lists
are combined with weighted Reciprocal Rank Fusion (RRF) -- by rank, not by raw
score -- because the channels' raw scores are on incomparable scales (BGE cosine
~0.5-0.7, SigLIP cosine ~0.0-0.1, graph proximity). RRF lets a top image or
graph hit reach the final top-k instead of being buried below every text hit, so
the visual and graph channels actually influence the ranking. When reranking is
enabled the cross-encoder then rescores the fused pool, providing a calibrated
cross-modal final ordering.
"""

from __future__ import annotations

from typing import Optional

import networkx as nx

from publaynet_mmrag.config import Config
from publaynet_mmrag.embed.image import ImageEmbedder
from publaynet_mmrag.embed.text import TextEmbedder
from publaynet_mmrag.index.store import VectorStore
from publaynet_mmrag.kg import query as kg_query
from publaynet_mmrag.retrieve.rerank import Reranker
from publaynet_mmrag.types import Category, RetrievedItem


def _item_key(item: RetrievedItem) -> str:
    """Returns the fusion/dedup key for a retrieved item.

    Args:
        item: The retrieved item.

    Returns:
        Its chunk id (text/graph) or region id (image); both namespaces are
        disjoint so the same chunk surfaced by two channels merges correctly.
    """
    return item.chunk_id or item.region_id or f"anon:{id(item)}"


def fuse_channels(
    channels: list[tuple[float, list[RetrievedItem]]], rrf_k: int
) -> list[RetrievedItem]:
    """Fuses ranked per-channel candidate lists with weighted RRF.

    Each item's fused score is ``sum_channel weight / (rrf_k + rank + 1)`` over
    the channels that returned it, so an item retrieved near the top of any
    channel -- or by several channels -- ranks highly. Items are deduplicated by
    :func:`_item_key`; the first occurrence (channels are passed text-first)
    supplies the kept item object and its ``source`` label.

    Args:
        channels: ``(weight, ranked_items)`` per channel, in priority order.
        rrf_k: The RRF damping constant (larger flattens rank differences).

    Returns:
        The fused items sorted by descending fused score, with ``score`` set to
        that fused value.
    """
    scores: dict[str, float] = {}
    kept: dict[str, RetrievedItem] = {}
    for weight, items in channels:
        for rank, item in enumerate(items):
            key = _item_key(item)
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank + 1)
            kept.setdefault(key, item)
    for key, item in kept.items():
        item.score = scores[key]
    return sorted(kept.values(), key=lambda it: it.score, reverse=True)


class Retriever:
    """Retrieves and ranks evidence for a query under one configuration.

    Attributes:
        config: The active run configuration.
    """

    def __init__(
        self,
        config: Config,
        store: VectorStore,
        text_embedder: TextEmbedder,
        image_embedder: Optional[ImageEmbedder] = None,
        reranker: Optional[Reranker] = None,
        graph: Optional[nx.MultiDiGraph] = None,
    ) -> None:
        """Wires the retriever to its components.

        Args:
            config: The active run configuration.
            store: The vector store.
            text_embedder: The text embedder (always required).
            image_embedder: The image embedder (enhanced only).
            reranker: The cross-encoder reranker (enhanced only).
            graph: The knowledge graph (enhanced only).
        """
        self.config = config
        self.store = store
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.reranker = reranker
        self.graph = graph

    def _text_channel(self, embedding) -> list[RetrievedItem]:
        """Builds the text channel: dense-only or dense+sparse hybrid."""
        cfg = self.config.retrieval
        if cfg.use_sparse:
            hits = self.store.search_hybrid(
                dense=embedding.dense,
                sparse=embedding.sparse,
                limit=cfg.candidate_k,
                candidate_k=cfg.candidate_k,
            )
            source = "text_hybrid"
        else:
            hits = self.store.search_dense(dense=embedding.dense, limit=cfg.candidate_k)
            source = "text_dense"
        return [
            RetrievedItem(
                doc_id=hit["doc_id"],
                page_index=hit["page_index"],
                score=hit["score"],
                modality="text",
                text=hit.get("text", ""),
                chunk_id=hit.get("chunk_id"),
                source=source,
            )
            for hit in hits
        ]

    def _graph_channel(self, query: str) -> tuple[list[RetrievedItem], list[str]]:
        """Builds the graph channel: proximity-ranked chunks from KG expansion."""
        cfg = self.config.retrieval
        expansion = kg_query.expand(self.graph, query, hops=cfg.graph_hops)
        if not expansion.chunk_ids:
            return [], expansion.paths
        items = [
            RetrievedItem(
                doc_id=hit["doc_id"],
                page_index=hit["page_index"],
                score=0.0,
                modality="text",
                text=hit.get("text", ""),
                chunk_id=hit.get("chunk_id"),
                source="graph",
            )
            for hit in self.store.fetch_text_by_chunk_ids(
                expansion.chunk_ids[: cfg.candidate_k]
            )
        ]
        return items, expansion.paths

    def _image_channel(self, query: str) -> list[RetrievedItem]:
        """Builds the image channel: SigLIP2 text-to-image figure/table search."""
        cfg = self.config.retrieval
        query_vec = self.image_embedder.embed_text([query])[0]
        items: list[RetrievedItem] = []
        for hit in self.store.search_image(query_vec, limit=cfg.candidate_k):
            label = Category(hit.get("category", Category.FIGURE)).name.lower()
            caption = hit.get("caption", "")
            items.append(
                RetrievedItem(
                    doc_id=hit["doc_id"],
                    page_index=hit["page_index"],
                    score=hit["score"],
                    modality="image",
                    text=caption or f"[{label} region]",
                    region_id=hit.get("region_id"),
                    crop_path=hit.get("crop_path"),
                    source="image",
                )
            )
        return items

    def retrieve(self, query: str) -> tuple[list[RetrievedItem], list[str]]:
        """Retrieves ranked evidence for a query.

        Args:
            query: The natural-language query.

        Returns:
            A tuple of the top-k retrieved items and the knowledge-graph path
            traces used (empty when graph expansion is disabled).
        """
        cfg = self.config.retrieval
        embedding = self.text_embedder.embed_query(query)

        # Build each active channel as a ranked list, text first so it wins ties.
        channels: list[tuple[float, list[RetrievedItem]]] = [
            (cfg.text_weight, self._text_channel(embedding))
        ]

        graph_paths: list[str] = []
        if cfg.use_graph and self.graph is not None:
            graph_items, graph_paths = self._graph_channel(query)
            if graph_items:
                channels.append((cfg.graph_weight, graph_items))

        if cfg.use_image and self.image_embedder is not None:
            image_items = self._image_channel(query)
            if image_items:
                channels.append((cfg.image_weight, image_items))

        items = fuse_channels(channels, rrf_k=cfg.rrf_k)

        # Reranking rescores the whole fused pool, so a strong image/graph hit the
        # fusion surfaced can be promoted (or demoted) on calibrated relevance.
        if cfg.use_rerank and self.reranker is not None and items:
            scores = self.reranker.score(query, [it.text for it in items])
            for item, score in zip(items, scores):
                item.score = score
            items.sort(key=lambda it: it.score, reverse=True)

        return items[: cfg.top_k], graph_paths
