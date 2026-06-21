"""Embedded Qdrant vector store wrapper.

Qdrant runs in embedded mode (``QdrantClient(path=...)``) with no server or
Docker, so the proof-of-concept is fully local; pointing the same client at a
server URL scales the identical code to the full dataset. The text collection
holds named dense and sparse vectors and hybrid search fuses them with
Reciprocal Rank Fusion. The image collection holds SigLIP2 vectors.

Hybrid search uses the ``query_points`` + ``Prefetch`` + ``FusionQuery`` API
introduced in qdrant-client 1.10; pin ``qdrant-client>=1.10``.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any, Optional

from publaynet_mmrag.embed.text import SparseVector, TextEmbedding
from publaynet_mmrag.index import schema
from publaynet_mmrag.types import Chunk, Region


def _point_id(raw: str) -> str:
    """Derives a stable UUID point id from an arbitrary string.

    Args:
        raw: The natural identifier (chunk or region id).

    Returns:
        A deterministic UUID string accepted by Qdrant as a point id.
    """
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:32]
    return str(uuid.UUID(digest))


class VectorStore:
    """Manages the text and image collections in embedded Qdrant.

    Attributes:
        path: On-disk storage directory for the embedded instance.
        text_dim: Dense text-embedding dimensionality.
        image_dim: Image-embedding dimensionality.
    """

    def __init__(self, path: str, text_dim: int, image_dim: int) -> None:
        """Opens (or creates) the embedded store.

        Args:
            path: On-disk storage directory.
            text_dim: Dense text-embedding dimensionality.
            image_dim: Image-embedding dimensionality.
        """
        from qdrant_client import QdrantClient

        self.path = path
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.client = QdrantClient(path=path)

    def create_collections(self, with_image: bool = True) -> None:
        """Creates the text and (optionally) image collections.

        Existing collections are recreated so a rebuild starts clean.

        Args:
            with_image: Whether to create the image collection.
        """
        from qdrant_client import models

        self.client.recreate_collection(
            collection_name=schema.TEXT_COLLECTION,
            vectors_config={
                schema.DENSE_VECTOR: models.VectorParams(
                    size=self.text_dim, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={schema.SPARSE_VECTOR: models.SparseVectorParams()},
        )
        if with_image:
            self.client.recreate_collection(
                collection_name=schema.IMAGE_COLLECTION,
                vectors_config={
                    schema.IMAGE_VECTOR: models.VectorParams(
                        size=self.image_dim, distance=models.Distance.COSINE
                    )
                },
            )

    def upsert_text(self, chunks: list[Chunk], embeddings: list[TextEmbedding]) -> None:
        """Upserts text chunks with dense and sparse vectors.

        Args:
            chunks: Chunks to index.
            embeddings: Embeddings aligned with ``chunks``.
        """
        from qdrant_client import models

        points = []
        for chunk, emb in zip(chunks, embeddings):
            points.append(
                models.PointStruct(
                    id=_point_id(chunk.chunk_id),
                    vector={
                        schema.DENSE_VECTOR: emb.dense,
                        schema.SPARSE_VECTOR: models.SparseVector(
                            indices=emb.sparse.indices, values=emb.sparse.values
                        ),
                    },
                    payload=schema.text_payload(chunk),
                )
            )
        self.client.upsert(schema.TEXT_COLLECTION, points=points)

    def upsert_images(self, regions: list[Region], vectors: list[list[float]]) -> None:
        """Upserts visual regions with their SigLIP2 vectors.

        Args:
            regions: Visual regions to index.
            vectors: Image embeddings aligned with ``regions``.
        """
        from qdrant_client import models

        points = [
            models.PointStruct(
                id=_point_id(region.region_id),
                vector={schema.IMAGE_VECTOR: vector},
                payload=schema.image_payload(region),
            )
            for region, vector in zip(regions, vectors)
        ]
        self.client.upsert(schema.IMAGE_COLLECTION, points=points)

    def search_dense(
        self,
        dense: list[float],
        limit: int,
        doc_ids: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Runs dense-only text search (the baseline retrieval channel).

        Args:
            dense: The dense query vector.
            limit: Maximum hits to return.
            doc_ids: Optional document-id filter.

        Returns:
            Payload dictionaries with an added ``score`` field.
        """
        from qdrant_client import models

        result = self.client.query_points(
            collection_name=schema.TEXT_COLLECTION,
            query=dense,
            using=schema.DENSE_VECTOR,
            limit=limit,
            query_filter=_doc_filter(models, doc_ids),
            with_payload=True,
        )
        return _format(result)

    def search_hybrid(
        self,
        dense: list[float],
        sparse: SparseVector,
        limit: int,
        candidate_k: int,
        doc_ids: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Runs dense+sparse hybrid text search fused with RRF.

        Args:
            dense: The dense query vector.
            sparse: The sparse query vector.
            limit: Maximum hits to return after fusion.
            candidate_k: Candidates fetched per channel before fusion.
            doc_ids: Optional document-id filter.

        Returns:
            Payload dictionaries with an added ``score`` field.
        """
        from qdrant_client import models

        prefetch = [
            models.Prefetch(query=dense, using=schema.DENSE_VECTOR, limit=candidate_k),
            models.Prefetch(
                query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                using=schema.SPARSE_VECTOR,
                limit=candidate_k,
            ),
        ]
        result = self.client.query_points(
            collection_name=schema.TEXT_COLLECTION,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            query_filter=_doc_filter(models, doc_ids),
            with_payload=True,
        )
        return _format(result)

    def search_image(self, vector: list[float], limit: int) -> list[dict[str, Any]]:
        """Runs image search against the visual-region collection.

        Args:
            vector: A query vector in the SigLIP2 space (image or text tower).
            limit: Maximum hits to return.

        Returns:
            Payload dictionaries with an added ``score`` field.
        """
        result = self.client.query_points(
            collection_name=schema.IMAGE_COLLECTION,
            query=vector,
            using=schema.IMAGE_VECTOR,
            limit=limit,
            with_payload=True,
        )
        return _format(result)

    def fetch_text_by_doc(self, doc_ids: list[str], limit: int) -> list[dict[str, Any]]:
        """Retrieves text chunks belonging to the given documents.

        Used by graph expansion to pull chunks for documents surfaced via the
        knowledge graph rather than via vector similarity.

        Args:
            doc_ids: Document identifiers to fetch.
            limit: Maximum chunks to return.

        Returns:
            Payload dictionaries (no similarity score).
        """
        from qdrant_client import models

        records, _ = self.client.scroll(
            collection_name=schema.TEXT_COLLECTION,
            scroll_filter=_doc_filter(models, doc_ids),
            limit=limit,
            with_payload=True,
        )
        return [dict(record.payload, score=0.0) for record in records]


def _doc_filter(models, doc_ids: Optional[list[str]]):
    """Builds a document-id ``match-any`` filter, or ``None``.

    Args:
        models: The ``qdrant_client.models`` module.
        doc_ids: Document ids to match, or ``None`` for no filter.

    Returns:
        A Qdrant ``Filter`` or ``None``.
    """
    if not doc_ids:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(
                key="doc_id", match=models.MatchAny(any=list(doc_ids))
            )
        ]
    )


def _format(result) -> list[dict[str, Any]]:
    """Flattens a ``query_points`` result into payload+score dicts.

    Args:
        result: The object returned by ``query_points``.

    Returns:
        One dictionary per hit, the payload augmented with ``score``.
    """
    return [dict(point.payload, score=float(point.score)) for point in result.points]
