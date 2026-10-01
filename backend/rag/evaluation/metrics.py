"""Retrieval quality metrics for AURA RAG evaluation.

All metrics are deterministic and LLM-independent.
They compare retrieved identifiers against expected relevant identifiers.

Metrics implemented:
- Recall@K
- Precision@K
- Hit Rate@K (binary: did at least one relevant item appear in top-K?)
- MRR (Mean Reciprocal Rank)
"""

from rag.exceptions import EvaluationError


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Compute Recall@K.

    Recall@K = |relevant ∩ retrieved[:k]| / |relevant|

    Args:
        retrieved_ids: Ordered list of retrieved chunk/document identifiers.
        relevant_ids: Set of expected relevant identifiers.
        k: Number of top results to consider.

    Returns:
        Recall score in [0.0, 1.0].

    Raises:
        EvaluationError: If k <= 0.
    """
    if k <= 0:
        raise EvaluationError("k must be greater than 0.")
    relevant_set = relevant_ids if isinstance(relevant_ids, (set, frozenset)) else set(relevant_ids)
    if not relevant_set:
        # No relevant items defined — recall is undefined; return 0.0
        # by convention (nothing to recall).
        return 0.0

    top_k = retrieved_ids[:k]
    hits = len(set(top_k) & relevant_set)
    return hits / len(relevant_set)


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Compute Precision@K.

    Precision@K = |relevant ∩ retrieved[:k]| / k

    Args:
        retrieved_ids: Ordered list of retrieved chunk/document identifiers.
        relevant_ids: Set of expected relevant identifiers.
        k: Number of top results to consider.

    Returns:
        Precision score in [0.0, 1.0].

    Raises:
        EvaluationError: If k <= 0.
    """
    if k <= 0:
        raise EvaluationError("k must be greater than 0.")
    relevant_set = relevant_ids if isinstance(relevant_ids, (set, frozenset)) else set(relevant_ids)
    if not relevant_set:
        # No relevant items — everything retrieved is irrelevant.
        return 0.0

    top_k = retrieved_ids[:k]
    hits = len(set(top_k) & relevant_set)
    return hits / k


def hit_rate_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Compute Hit Rate@K.

    Binary metric: 1.0 if at least one relevant item appears in
    the top-K retrieved results, 0.0 otherwise.

    Args:
        retrieved_ids: Ordered list of retrieved chunk/document identifiers.
        relevant_ids: Set of expected relevant identifiers.
        k: Number of top results to consider.

    Returns:
        1.0 or 0.0.

    Raises:
        EvaluationError: If k <= 0.
    """
    if k <= 0:
        raise EvaluationError("k must be greater than 0.")
    relevant_set = relevant_ids if isinstance(relevant_ids, (set, frozenset)) else set(relevant_ids)
    if not relevant_set:
        return 0.0

    top_k = retrieved_ids[:k]
    for rid in top_k:
        if rid in relevant_set:
            return 1.0
    return 0.0


def mrr(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """Compute Mean Reciprocal Rank (MRR) for a single query.

    MRR = 1 / rank_of_first_relevant_result

    If no relevant result is found, returns 0.0.

    Args:
        retrieved_ids: Ordered list of retrieved chunk/document identifiers.
        relevant_ids: Set of expected relevant identifiers.

    Returns:
        Reciprocal rank in (0.0, 1.0] or 0.0 if no relevant result found.
    """
    relevant_set = relevant_ids if isinstance(relevant_ids, (set, frozenset)) else set(relevant_ids)
    if not relevant_set:
        return 0.0

    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_set:
            return 1.0 / rank
    return 0.0
