"""Unit tests for Reciprocal Rank Fusion (RRF) in AURA (M13)."""

from unittest import TestCase
from unittest.mock import MagicMock

from rag.fusion import reciprocal_rank_fusion
from rag.retrieval import RetrievalResult


def _create_mock_result(chunk_id: str, content: str = "text", score: float = 0.5) -> RetrievalResult:
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
        chunk_metadata={"source_type": "text"},
        chunk=chunk_mock,
    )


class RRFFusionTests(TestCase):
    """Tests for reciprocal_rank_fusion."""

    def test_empty_inputs_returns_empty_list(self) -> None:
        """When both dense and lexical lists are empty, returns empty list."""
        res = reciprocal_rank_fusion([], [])
        self.assertEqual(res, [])

    def test_dense_only_returns_fused_results(self) -> None:
        """When only dense results are present, produces sorted results with RRF scores."""
        c1 = _create_mock_result("c1")
        c2 = _create_mock_result("c2")
        fused = reciprocal_rank_fusion([c1, c2], [], k=60)
        self.assertEqual(len(fused), 2)
        self.assertEqual(fused[0].chunk_id, "c1")
        self.assertEqual(fused[1].chunk_id, "c2")
        self.assertAlmostEqual(fused[0].score, 1.0 / 61.0, places=5)
        self.assertAlmostEqual(fused[1].score, 1.0 / 62.0, places=5)
        self.assertEqual(fused[0].rank, 1)
        self.assertEqual(fused[1].rank, 2)

    def test_lexical_only_returns_fused_results(self) -> None:
        """When only lexical results are present, produces sorted results with RRF scores."""
        c1 = _create_mock_result("c1")
        fused = reciprocal_rank_fusion([], [c1], k=60)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].chunk_id, "c1")
        self.assertAlmostEqual(fused[0].score, 1.0 / 61.0, places=5)

    def test_deduplication_and_score_accumulation(self) -> None:
        """Chunks present in both dense and lexical lists accumulate scores and rank highest."""
        # c1 in both dense and lexical (ranks 1 and 2)
        # c2 in dense only (rank 2)
        # c3 in lexical only (rank 1)
        c1_dense = _create_mock_result("c1", content="Chunk 1")
        c2_dense = _create_mock_result("c2", content="Chunk 2")
        c3_lex = _create_mock_result("c3", content="Chunk 3")
        c1_lex = _create_mock_result("c1", content="Chunk 1")

        fused = reciprocal_rank_fusion(
            dense_results=[c1_dense, c2_dense],
            lexical_results=[c3_lex, c1_lex],
            k=60,
        )

        self.assertEqual(len(fused), 3)
        # c1 appears in both: 1/61 + 1/62 = 0.016393 + 0.016129 = 0.032522
        # c3 is rank 1 in lexical: 1/61 = 0.016393
        # c2 is rank 2 in dense: 1/62 = 0.016129
        self.assertEqual(fused[0].chunk_id, "c1")
        self.assertEqual(fused[1].chunk_id, "c3")
        self.assertEqual(fused[2].chunk_id, "c2")

        expected_c1_score = 1.0 / 61.0 + 1.0 / 62.0
        self.assertAlmostEqual(fused[0].score, round(expected_c1_score, 6), places=5)

    def test_deterministic_tie_breaking(self) -> None:
        """Equal scores are tie-broken deterministically by chunk_id ascending."""
        # c_beta and c_alpha both only appear in lexical at rank 1, but c_alpha < c_beta
        c_beta = _create_mock_result("chunk-b")
        c_alpha = _create_mock_result("chunk-a")

        fused = reciprocal_rank_fusion([], [c_beta, c_alpha], k=60)
        # Here beta is rank 1 (1/61), alpha is rank 2 (1/62).
        # Let's make them equal score by having beta in dense rank 1 and alpha in lexical rank 1:
        fused_equal = reciprocal_rank_fusion([c_beta], [c_alpha], k=60)
        self.assertEqual(len(fused_equal), 2)
        self.assertAlmostEqual(fused_equal[0].score, fused_equal[1].score)
        # Tie break by chunk_id ascending: "chunk-a" comes before "chunk-b"
        self.assertEqual(fused_equal[0].chunk_id, "chunk-a")
        self.assertEqual(fused_equal[1].chunk_id, "chunk-b")

    def test_weighting_multipliers(self) -> None:
        """Custom dense and lexical weights modify score contribution."""
        c1 = _create_mock_result("c1")
        c2 = _create_mock_result("c2")

        # Dense weighted 2.0, lexical weighted 0.5
        fused = reciprocal_rank_fusion(
            dense_results=[c1],
            lexical_results=[c2],
            k=60,
            dense_weight=2.0,
            lexical_weight=0.5,
        )
        self.assertEqual(fused[0].chunk_id, "c1")
        self.assertAlmostEqual(fused[0].score, 2.0 / 61.0, places=5)
        self.assertEqual(fused[1].chunk_id, "c2")
        self.assertAlmostEqual(fused[1].score, 0.5 / 61.0, places=5)

    def test_top_k_limiting(self) -> None:
        """top_k restricts the number of returned fused candidates."""
        candidates = [_create_mock_result(f"c{i}") for i in range(10)]
        fused = reciprocal_rank_fusion(candidates, [], top_k=3)
        self.assertEqual(len(fused), 3)

    def test_preserves_metadata_and_attributes(self) -> None:
        """All chunk attributes and metadata survive fusion intact."""
        orig = _create_mock_result("c1", content="Special Content")
        orig = RetrievalResult(
            chunk_id="c1",
            document_id="doc-99",
            content="Special Content",
            score=0.95,
            rank=1,
            chunk_index=3,
            document_title="Important Doc",
            document_source="https://example.com/doc",
            start_offset=100,
            end_offset=115,
            chunk_metadata={"page": 4, "format": "pdf"},
            chunk=orig.chunk,
        )

        fused = reciprocal_rank_fusion([orig], [])
        self.assertEqual(len(fused), 1)
        res = fused[0]
        self.assertEqual(res.chunk_id, "c1")
        self.assertEqual(res.document_id, "doc-99")
        self.assertEqual(res.content, "Special Content")
        self.assertEqual(res.document_title, "Important Doc")
        self.assertEqual(res.document_source, "https://example.com/doc")
        self.assertEqual(res.start_offset, 100)
        self.assertEqual(res.end_offset, 115)
        self.assertEqual(res.chunk_metadata, {"page": 4, "format": "pdf"})
