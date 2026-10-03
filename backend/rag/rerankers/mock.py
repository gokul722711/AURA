"""Deterministic Mock Reranker for unit testing (M13).

Provides deterministic reranking behavior without external dependencies or model files.
"""

from collections.abc import Callable
from typing import Sequence

from rag.rerankers.base import Reranker
from rag.retrieval import RetrievalResult


class MockReranker(Reranker):
    """Deterministic mock reranker for testing."""

    def __init__(
        self,
        scorer: Callable[[str, RetrievalResult], float] | None = None,
        reverse: bool = False,
        min_score: float = 0.0,
    ) -> None:
        self.scorer = scorer
        self.reverse = reverse
        self.min_score = min_score

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[RetrievalResult]:
        """Rerank candidates using a mock scoring function or existing score.

        Args:
            query: The search query text.
            candidates: Retrieved candidate chunks.
            top_k: Optional maximum number of reranked candidates to return.
            min_score: Optional minimum relevance score threshold. Candidates scoring
                below this threshold are excluded. Defaults to self.min_score.

        Returns:
            List of RetrievalResult ordered by descending reranked score.
        """
        if not candidates:
            return []

        effective_min_score = self.min_score if min_score is None else min_score
        cand_list = list(candidates)

        if self.scorer is not None:
            scored = [(self.scorer(query, c), c) for c in cand_list]
        elif self.reverse:
            # Reverse order of candidates for testing order changes
            scored = [((idx + 1) * 0.1, c) for idx, c in enumerate(cand_list)]
        else:
            # Preserve existing score
            scored = [(c.score, c) for c in cand_list]

        # Sort by score DESC, chunk_id ASC
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))

        # Filter by minimum score threshold
        if effective_min_score > 0.0:
            qualifying = [item for item in scored if item[0] >= effective_min_score]
        else:
            qualifying = scored

        # top_k represents the MAXIMUM number of candidates to return, not a quota
        limit = top_k if top_k is not None and top_k > 0 else len(qualifying)
        results: list[RetrievalResult] = []

        for rank_idx, (score, orig) in enumerate(qualifying[:limit], start=1):
            results.append(
                RetrievalResult(
                    chunk_id=orig.chunk_id,
                    document_id=orig.document_id,
                    content=orig.content,
                    score=round(score, 6),
                    rank=rank_idx,
                    chunk_index=orig.chunk_index,
                    document_title=orig.document_title,
                    document_source=orig.document_source,
                    start_offset=orig.start_offset,
                    end_offset=orig.end_offset,
                    chunk_metadata=orig.chunk_metadata,
                    chunk=orig.chunk,
                )
            )

        return results
