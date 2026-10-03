"""Unit tests for AURA Reranker abstraction and FlashRank adapter (M13)."""

from unittest import TestCase
from unittest.mock import MagicMock

from rag.rerankers.base import Reranker
from rag.rerankers.flashrank import FlashRankReranker
from rag.rerankers.mock import MockReranker
from rag.retrieval import RetrievalResult


def _create_candidate(chunk_id: str, content: str = "chunk text", score: float = 0.5) -> RetrievalResult:
    chunk_mock = MagicMock()
    chunk_mock.pk = chunk_id
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        content=content,
        score=score,
        rank=1,
        chunk_index=0,
        document_title="Doc Title",
        document_source="source.txt",
        start_offset=0,
        end_offset=len(content),
        chunk_metadata={"page": 1},
        chunk=chunk_mock,
    )


class MockRerankerTests(TestCase):
    """Tests for MockReranker."""

    def test_empty_candidates_returns_empty_list(self) -> None:
        reranker = MockReranker()
        self.assertEqual(reranker.rerank("query", []), [])

    def test_reverse_ordering(self) -> None:
        c1 = _create_candidate("c1")
        c2 = _create_candidate("c2")
        reranker = MockReranker(reverse=True)
        results = reranker.rerank("query", [c1, c2])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].chunk_id, "c2")
        self.assertEqual(results[1].chunk_id, "c1")
        self.assertEqual(results[0].rank, 1)
        self.assertEqual(results[1].rank, 2)

    def test_custom_scorer(self) -> None:
        c1 = _create_candidate("c1", content="alpha")
        c2 = _create_candidate("c2", content="beta")
        # Give higher score if 'beta' in content
        reranker = MockReranker(scorer=lambda q, c: 1.0 if "beta" in c.content else 0.1)
        results = reranker.rerank("test", [c1, c2])
        self.assertEqual(results[0].chunk_id, "c2")
        self.assertEqual(results[1].chunk_id, "c1")

    def test_top_k_limiting(self) -> None:
        candidates = [_create_candidate(f"c{i}") for i in range(5)]
        reranker = MockReranker()
        results = reranker.rerank("query", candidates, top_k=2)
        self.assertEqual(len(results), 2)

    def test_relevance_threshold_filters_low_scores(self) -> None:
        """5 candidates, 3 above threshold, 2 below -> exactly 3 returned."""
        candidates = [
            _create_candidate("c1", score=0.9),
            _create_candidate("c2", score=0.5),
            _create_candidate("c3", score=0.2),
            _create_candidate("c4", score=0.05),
            _create_candidate("c5", score=0.01),
        ]
        reranker = MockReranker(min_score=0.1)
        results = reranker.rerank("query", candidates, top_k=5)
        self.assertEqual(len(results), 3)
        self.assertEqual([r.chunk_id for r in results], ["c1", "c2", "c3"])

    def test_fewer_candidates_than_top_k_returns_only_qualifying(self) -> None:
        """top_k=5, but only 1 candidate above threshold -> returns exactly 1 (not 5)."""
        candidates = [
            _create_candidate("c1", score=0.8),
            _create_candidate("c2", score=0.05),
            _create_candidate("c3", score=0.02),
        ]
        reranker = MockReranker(min_score=0.1)
        results = reranker.rerank("query", candidates, top_k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "c1")

    def test_no_candidates_above_threshold_returns_empty(self) -> None:
        """When 0 candidates meet min_score, returns empty list."""
        candidates = [
            _create_candidate("c1", score=0.05),
            _create_candidate("c2", score=0.02),
        ]
        reranker = MockReranker(min_score=0.1)
        results = reranker.rerank("query", candidates, top_k=5)
        self.assertEqual(len(results), 0)


class FlashRankRerankerTests(TestCase):
    """Tests for FlashRankReranker using client dependency injection (no internet downloads)."""

    def test_empty_candidates_returns_empty_list_without_calling_client(self) -> None:
        mock_client = MagicMock()
        reranker = FlashRankReranker(client=mock_client)
        res = reranker.rerank("query", [])
        self.assertEqual(res, [])
        mock_client.rerank.assert_not_called()

    def test_empty_query_returns_original_candidates(self) -> None:
        mock_client = MagicMock()
        reranker = FlashRankReranker(client=mock_client)
        c1 = _create_candidate("c1")
        res = reranker.rerank("", [c1])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].chunk_id, "c1")
        mock_client.rerank.assert_not_called()

    def test_rerank_translates_passages_and_updates_scores(self) -> None:
        c1 = _create_candidate("c1", content="First passage")
        c2 = _create_candidate("c2", content="Second passage")

        # Mock FlashRank client returning c2 higher than c1
        mock_client = MagicMock()

        def fake_rerank(request: Any) -> list[dict[str, Any]]:
            self.assertEqual(request.query, "test query")
            self.assertEqual(len(request.passages), 2)
            # Return reversed with custom scores
            return [
                {"id": "c2", "score": 0.92, "_original": c2},
                {"id": "c1", "score": 0.41, "_original": c1},
            ]

        mock_client.rerank.side_effect = fake_rerank

        reranker = FlashRankReranker(client=mock_client)
        results = reranker.rerank("test query", [c1, c2])

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].chunk_id, "c2")
        self.assertEqual(results[0].score, 0.92)
        self.assertEqual(results[0].rank, 1)

        self.assertEqual(results[1].chunk_id, "c1")
        self.assertEqual(results[1].score, 0.41)
        self.assertEqual(results[1].rank, 2)

    def test_rerank_preserves_metadata(self) -> None:
        c1 = _create_candidate("c1", content="Metadata test")
        c1 = RetrievalResult(
            chunk_id="c1",
            document_id="doc-42",
            content="Metadata test",
            score=0.1,
            rank=1,
            chunk_index=2,
            document_title="My Doc",
            document_source="https://example.com/page",
            start_offset=50,
            end_offset=63,
            chunk_metadata={"page": 3, "url": "https://example.com/page"},
            chunk=c1.chunk,
        )

        mock_client = MagicMock()
        mock_client.rerank.return_value = [{"id": "c1", "score": 0.99, "_original": c1}]

        reranker = FlashRankReranker(client=mock_client)
        results = reranker.rerank("query", [c1])

        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertEqual(res.chunk_id, "c1")
        self.assertEqual(res.document_id, "doc-42")
        self.assertEqual(res.document_title, "My Doc")
        self.assertEqual(res.document_source, "https://example.com/page")
        self.assertEqual(res.chunk_metadata, {"page": 3, "url": "https://example.com/page"})
        self.assertEqual(res.score, 0.99)

    def test_flashrank_relevance_threshold_filters_low_scores(self) -> None:
        """FlashRank excludes candidates below min_score (e.g. 5 candidates -> 3 qualifying)."""
        c = [_create_candidate(f"c{i}") for i in range(5)]
        mock_client = MagicMock()
        mock_client.rerank.return_value = [
            {"id": "c0", "score": 0.997235, "_original": c[0]},
            {"id": "c1", "score": 0.000432, "_original": c[1]},
            {"id": "c2", "score": 0.000150, "_original": c[2]},
            {"id": "c3", "score": 0.000029, "_original": c[3]},
            {"id": "c4", "score": 0.000024, "_original": c[4]},
        ]
        reranker = FlashRankReranker(client=mock_client, min_score=0.0001)
        results = reranker.rerank("query", c, top_k=5)
        self.assertEqual(len(results), 3)
        self.assertEqual([r.chunk_id for r in results], ["c0", "c1", "c2"])

    def test_flashrank_fewer_candidates_than_top_k_returns_only_qualifying(self) -> None:
        """top_k=5, but only 1 candidate above min_score -> returns 1, not 5."""
        c = [_create_candidate(f"c{i}") for i in range(3)]
        mock_client = MagicMock()
        mock_client.rerank.return_value = [
            {"id": "c0", "score": 0.997235, "_original": c[0]},
            {"id": "c1", "score": 0.000029, "_original": c[1]},
            {"id": "c2", "score": 0.000024, "_original": c[2]},
        ]
        reranker = FlashRankReranker(client=mock_client, min_score=0.0001)
        results = reranker.rerank("query", c, top_k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "c0")

    def test_flashrank_no_candidates_above_threshold_returns_empty(self) -> None:
        """When all candidates score below min_score, returns empty list."""
        c = [_create_candidate(f"c{i}") for i in range(2)]
        mock_client = MagicMock()
        mock_client.rerank.return_value = [
            {"id": "c0", "score": 0.000029, "_original": c[0]},
            {"id": "c1", "score": 0.000024, "_original": c[1]},
        ]
        reranker = FlashRankReranker(client=mock_client, min_score=0.0001)
        results = reranker.rerank("query", c, top_k=5)
        self.assertEqual(len(results), 0)
