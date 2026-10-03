"""AURA Reranker abstract base interface (M13).

Provides a provider-agnostic contract for passage reranking, isolating
external reranking frameworks behind an AURA-owned abstraction.
"""

from abc import ABC, abstractmethod
from typing import Sequence

from rag.retrieval import RetrievalResult


class Reranker(ABC):
    """Abstract base class for passage rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[RetrievalResult]:
        """Rerank candidate chunks according to relevance to the search query.

        Args:
            query: The search query text.
            candidates: Retrieved candidate chunks.
            top_k: Optional maximum number of reranked candidates to return.
            min_score: Optional minimum relevance score threshold. Candidates scoring
                below this threshold are excluded.

        Returns:
            List of RetrievalResult ordered by descending reranker score.
        """
        pass
