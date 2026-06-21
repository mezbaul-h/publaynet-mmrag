"""Configurable retriever covering both the baseline and enhanced pipelines.

A single retriever implements both arms of the comparison; the
:class:`~publaynet_mmrag.config.RetrievalConfig` flags decide which channels are
active:

* ``use_sparse``  -- dense-only vs dense+sparse hybrid text search.
* ``use_image``   -- add SigLIP2 figure/table retrieval (text-to-image).
* ``use_graph``   -- add knowledge-graph expansion to pull connected chunks.
* ``use_rerank``  -- apply the cross-encoder over merged candidates.

The baseline config disables all four, so the only differences between the two
arms are modality and structured knowledge -- the variables under study.
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

        # Text channel: dense-only (baseline) or dense+sparse hybrid (enhanced).
        if cfg.use_sparse:
            text_hits = self.store.search_hybrid(
                dense=embedding.dense,
                sparse=embedding.sparse,
                limit=cfg.candidate_k,
                candidate_k=cfg.candidate_k,
            )
        else:
            text_hits = self.store.search_dense(
                dense=embedding.dense, limit=cfg.candidate_k
            )

        candidates: dict[str, RetrievedItem] = {}
        for hit in text_hits:
            item = RetrievedItem(
                doc_id=hit["doc_id"],
                page_index=hit["page_index"],
                score=hit["score"],
                modality="text",
                text=hit.get("text", ""),
                chunk_id=hit.get("chunk_id"),
                source="text_hybrid" if cfg.use_sparse else "text_dense",
            )
            candidates[item.chunk_id or f"t{len(candidates)}"] = item

        graph_paths: list[str] = []
        if cfg.use_graph and self.graph is not None:
            expansion = kg_query.expand(self.graph, query, hops=cfg.graph_hops)
            graph_paths = expansion.paths
            if expansion.doc_ids:
                for hit in self.store.fetch_text_by_doc(
                    expansion.doc_ids, limit=cfg.candidate_k
                ):
                    cid = hit.get("chunk_id")
                    if cid and cid not in candidates:
                        candidates[cid] = RetrievedItem(
                            doc_id=hit["doc_id"],
                            page_index=hit["page_index"],
                            score=0.0,
                            modality="text",
                            text=hit.get("text", ""),
                            chunk_id=cid,
                            source="graph",
                        )

        if cfg.use_image and self.image_embedder is not None:
            query_vec = self.image_embedder.embed_text([query])[0]
            for hit in self.store.search_image(query_vec, limit=cfg.top_k):
                rid = hit.get("region_id")
                caption = hit.get("caption", "")
                label = Category(hit.get("category", Category.FIGURE)).name.lower()
                candidates[rid or f"i{len(candidates)}"] = RetrievedItem(
                    doc_id=hit["doc_id"],
                    page_index=hit["page_index"],
                    score=hit["score"] * cfg.image_weight,
                    modality="image",
                    text=caption or f"[{label} region]",
                    region_id=rid,
                    crop_path=hit.get("crop_path"),
                    source="image",
                )

        items = list(candidates.values())

        if cfg.use_rerank and self.reranker is not None and items:
            scores = self.reranker.score(query, [it.text for it in items])
            for item, score in zip(items, scores):
                item.score = score

        items.sort(key=lambda it: it.score, reverse=True)
        return items[: cfg.top_k], graph_paths
