"""FlashRank adapter implementation of AURA Reranker interface (M13).

Wraps the lightweight, CPU-efficient FlashRank library behind the AURA Reranker ABC.
Supports client dependency injection for deterministic offline unit testing.
"""

from typing import Any, Sequence

from rag.rerankers.base import Reranker
from rag.retrieval import RetrievalResult


_SHARED_RANKERS: dict[tuple[str, str, int], Any] = {}


def get_shared_ranker(
    model_name: str = "ms-marco-TinyBERT-L-2-v2",
    cache_dir: str = "/tmp",
    max_length: int = 512,
) -> Any:
    """Retrieve or initialize a process-level shared FlashRank Ranker instance.

    Avoids re-opening ONNX sessions and re-loading tokenizers on every request.
    """
    key = (model_name, cache_dir, max_length)
    if key not in _SHARED_RANKERS:
        from flashrank import Ranker

        _SHARED_RANKERS[key] = Ranker(
            model_name=model_name,
            cache_dir=cache_dir,
            max_length=max_length,
        )
    return _SHARED_RANKERS[key]


class FlashRankReranker(Reranker):
    """Reranker using FlashRank's lightweight ONNX cross-encoders."""

    def __init__(
        self,
        model_name: str = "ms-marco-TinyBERT-L-2-v2",
        cache_dir: str = "/tmp",
        max_length: int = 512,
        client: Any = None,
        min_score: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_length = max_length
        self._client = client
        self.min_score = min_score

    @property
    def ranker(self) -> Any:
        """Lazily retrieve shared FlashRank Ranker if not injected."""
        if self._client is None:
            self._client = get_shared_ranker(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
                max_length=self.max_length,
            )
        return self._client

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[RetrievalResult]:
        """Rerank candidates using FlashRank cross-encoder.

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

        if not query or not query.strip():
            # Return original candidates up to top_k if query is empty, filtering by min_score if set
            filtered = [
                c for c in candidates
                if effective_min_score <= 0.0 or c.score >= effective_min_score
            ]
            limit = top_k if top_k is not None and top_k > 0 else len(filtered)
            return list(filtered[:limit])

        # Prepare passages for FlashRank (store original reference)
        passages: list[dict[str, Any]] = [
            {
                "id": c.chunk_id,
                "text": c.content,
                "_original": c,
            }
            for c in candidates
        ]

        from flashrank import RerankRequest

        request = RerankRequest(query=query.strip(), passages=passages)
        ranked_passages = self.ranker.rerank(request)

        # Relevance filtering: remove candidates below minimum score threshold
        if effective_min_score > 0.0:
            qualifying_passages = [
                p for p in ranked_passages
                if float(p.get("score", 0.0)) >= effective_min_score
            ]
        else:
            qualifying_passages = ranked_passages

        # top_k represents the MAXIMUM number of candidates to return, not a quota
        limit = top_k if top_k is not None and top_k > 0 else len(qualifying_passages)
        reranked: list[RetrievalResult] = []

        for rank_idx, p in enumerate(qualifying_passages[:limit], start=1):
            original: RetrievalResult = p["_original"]
            new_score = round(float(p.get("score", 0.0)), 6)
            reranked.append(
                RetrievalResult(
                    chunk_id=original.chunk_id,
                    document_id=original.document_id,
                    content=original.content,
                    score=new_score,
                    rank=rank_idx,
                    chunk_index=original.chunk_index,
                    document_title=original.document_title,
                    document_source=original.document_source,
                    start_offset=original.start_offset,
                    end_offset=original.end_offset,
                    chunk_metadata=original.chunk_metadata,
                    chunk=original.chunk,
                )
            )

        return reranked
