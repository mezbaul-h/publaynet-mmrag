"""Retrieval quality metrics against known-gold targets.

All metrics treat a single gold item per query (the chunk a synthetic question
was generated from). Matching is by chunk id, with a fallback to document id so
that retrieving a different chunk of the correct document still counts under the
document-level variants.
"""

from __future__ import annotations

import math
from typing import Optional


def _rank_of_gold(
    retrieved_ids: list[str],
    gold_id: str,
    retrieved_docs: Optional[list[str]] = None,
    gold_doc: Optional[str] = None,
) -> Optional[int]:
    """Finds the 1-based rank of the gold target in a result list.

    Args:
        retrieved_ids: Retrieved chunk ids in rank order.
        gold_id: The gold chunk id.
        retrieved_docs: Retrieved document ids in rank order (doc fallback).
        gold_doc: The gold document id (doc fallback).

    Returns:
        The 1-based rank of the first match, or ``None`` if absent.
    """
    for i, rid in enumerate(retrieved_ids):
        if rid == gold_id:
            return i + 1
    if retrieved_docs and gold_doc:
        for i, did in enumerate(retrieved_docs):
            if did == gold_doc:
                return i + 1
    return None


def recall_at_k(rank: Optional[int], k: int) -> float:
    """Returns 1.0 if the gold target is within the top ``k``, else 0.0.

    Args:
        rank: The 1-based gold rank, or ``None``.
        k: The cut-off.

    Returns:
        The binary recall (equivalently hit@k) for one query.
    """
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(rank: Optional[int]) -> float:
    """Returns the reciprocal rank of the gold target.

    Args:
        rank: The 1-based gold rank, or ``None``.

    Returns:
        ``1 / rank`` if found, else ``0.0``.
    """
    return 1.0 / rank if rank is not None else 0.0


def ndcg_at_k(rank: Optional[int], k: int) -> float:
    """Returns nDCG@k for a single gold target.

    With one relevant item the ideal DCG is 1, so nDCG reduces to the gain at
    the gold rank if it falls within ``k``.

    Args:
        rank: The 1-based gold rank, or ``None``.
        k: The cut-off.

    Returns:
        The nDCG@k for one query.
    """
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def aggregate(
    per_query_ranks: list[Optional[int]], ks: list[int]
) -> dict[str, float]:
    """Aggregates per-query gold ranks into mean metrics.

    Args:
        per_query_ranks: The gold rank for each query (``None`` if missed).
        ks: Cut-offs at which to report Recall and nDCG.

    Returns:
        A dictionary of mean metrics: ``mrr``, ``recall@k`` and ``ndcg@k`` for
        each ``k``.
    """
    n = len(per_query_ranks) or 1
    metrics: dict[str, float] = {
        "mrr": sum(reciprocal_rank(r) for r in per_query_ranks) / n,
    }
    for k in ks:
        metrics[f"recall@{k}"] = (
            sum(recall_at_k(r, k) for r in per_query_ranks) / n
        )
        metrics[f"ndcg@{k}"] = sum(ndcg_at_k(r, k) for r in per_query_ranks) / n
    return metrics


def gold_rank(
    retrieved_ids: list[str],
    retrieved_docs: list[str],
    gold_id: str,
    gold_doc: str,
) -> Optional[int]:
    """Convenience wrapper returning the gold rank for one query.

    Args:
        retrieved_ids: Retrieved chunk ids in rank order.
        retrieved_docs: Retrieved document ids in rank order.
        gold_id: The gold chunk id.
        gold_doc: The gold document id.

    Returns:
        The 1-based gold rank, or ``None`` if not retrieved.
    """
    return _rank_of_gold(retrieved_ids, gold_id, retrieved_docs, gold_doc)
