"""Reciprocal Rank Fusion (RRF) for hybrid retrieval in AURA (M13).

Combines candidate sets from dense vector retrieval and lexical full-text search
into a single deduplicated, fused candidate set using Reciprocal Rank Fusion:
    RRF_score(d) = sum( weight / (k + rank(d)) )

Features:
- Deterministic tie-breaking using (-rrf_score, chunk_id).
- Deduplicates identical chunks by chunk_id.
- Retains complete chunk metadata, document titles, sources, pages, and offsets.
"""

from typing import Sequence

from rag.retrieval import RetrievalResult


def reciprocal_rank_fusion(
    dense_results: Sequence[RetrievalResult],
    lexical_results: Sequence[RetrievalResult],
    k: int = 60,
    dense_weight: float = 1.0,
    lexical_weight: float = 1.0,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """Fuse dense and lexical retrieval results using Reciprocal Rank Fusion.

    Args:
        dense_results: Candidate chunks from dense vector similarity search.
        lexical_results: Candidate chunks from lexical full-text search.
        k: Smoothing constant (standard default is 60).
        dense_weight: Multiplier weight for dense rankings.
        lexical_weight: Multiplier weight for lexical rankings.
        top_k: Optional maximum number of fused candidates to return.

    Returns:
        List of fused RetrievalResult sorted by descending RRF score, with
        deterministic tie-breaking by chunk_id ascending.
    """
    if k <= 0:
        k = 60

    rrf_scores: dict[str, float] = {}
    best_candidates: dict[str, RetrievalResult] = {}

    # 1. Accumulate dense rankings
    for rank, res in enumerate(dense_results, start=1):
        cid = res.chunk_id
        score_contrib = dense_weight / (k + rank)
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score_contrib
        if cid not in best_candidates:
            best_candidates[cid] = res

    # 2. Accumulate lexical rankings
    for rank, res in enumerate(lexical_results, start=1):
        cid = res.chunk_id
        score_contrib = lexical_weight / (k + rank)
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score_contrib
        if cid not in best_candidates:
            best_candidates[cid] = res

    if not best_candidates:
        return []

    # 3. Deterministic sort: score DESC, chunk_id ASC for stable tie-breaking
    sorted_chunk_ids = sorted(
        best_candidates.keys(),
        key=lambda cid: (-rrf_scores[cid], cid),
    )

    limit = top_k if top_k is not None and top_k > 0 else len(sorted_chunk_ids)
    fused_results: list[RetrievalResult] = []

    for new_rank, cid in enumerate(sorted_chunk_ids[:limit], start=1):
        original = best_candidates[cid]
        fused_score = round(rrf_scores[cid], 6)
        fused_results.append(
            RetrievalResult(
                chunk_id=original.chunk_id,
                document_id=original.document_id,
                content=original.content,
                score=fused_score,
                rank=new_rank,
                chunk_index=original.chunk_index,
                document_title=original.document_title,
                document_source=original.document_source,
                start_offset=original.start_offset,
                end_offset=original.end_offset,
                chunk_metadata=original.chunk_metadata,
                chunk=original.chunk,
            )
        )

    return fused_results
