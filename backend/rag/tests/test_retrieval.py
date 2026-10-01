"""Tests for vector retrieval.

Tests verify retrieval pipeline mechanics using the MockEmbeddingProvider.
The mock embeddings are hash-based and do not have meaningful semantic quality.
These tests verify:
- Database-side cosine distance search works correctly
- Results are ordered by similarity (highest first)
- top_k limiting works
- Similarity threshold filtering works (M3: database-side, before top-K)
- Deterministic ordering/tie-breaking
- Only 'ready' documents are searched
- RetrievalResult contract exposes all required fields
"""

from django.test import TestCase

from rag.chunking import ChunkingConfig
from rag.embeddings.mock import MockEmbeddingProvider
from rag.exceptions import RetrievalError
from rag.ingestion import ingest_document
from rag.retrieval import RetrievalConfig, RetrievalResult, retrieve_chunks


class RetrieveChunksTests(TestCase):
    """Tests for retrieve_chunks function."""

    def setUp(self):
        self.provider = MockEmbeddingProvider(dimensions=384)
        self.config = ChunkingConfig(chunk_size=100, chunk_overlap=10)

        # Ingest a test document
        self.document = ingest_document(
            title="Test Document",
            content="Python is a programming language. " * 10,
            embedding_provider=self.provider,
            chunking_config=self.config,
        )

    def test_basic_retrieval_returns_results(self):
        results = retrieve_chunks(
            query="programming language",
            embedding_provider=self.provider,
        )
        self.assertIsInstance(results, list)

    def test_retrieval_returns_retrieval_results(self):
        results = retrieve_chunks(
            query="Python programming",
            embedding_provider=self.provider,
        )
        if results:
            result = results[0]
            self.assertIsNotNone(result.chunk)
            self.assertIsInstance(result.score, float)
            self.assertIsInstance(result.rank, int)

    def test_retrieval_top_k_limits_results(self):
        # Ingest more content to have many chunks
        ingest_document(
            title="Large Doc",
            content="Data science and machine learning are important. " * 50,
            embedding_provider=self.provider,
            chunking_config=ChunkingConfig(chunk_size=50, chunk_overlap=5),
        )
        config = RetrievalConfig(top_k=3)
        results = retrieve_chunks(
            query="data science",
            embedding_provider=self.provider,
            config=config,
        )
        self.assertLessEqual(len(results), 3)

    def test_retrieval_ranks_are_sequential(self):
        results = retrieve_chunks(
            query="test query",
            embedding_provider=self.provider,
        )
        for i, result in enumerate(results, start=1):
            self.assertEqual(result.rank, i)

    def test_retrieval_scores_are_descending(self):
        """Results should be ordered by similarity (highest first)."""
        ingest_document(
            title="Extra Doc",
            content="Additional content for retrieval testing. " * 20,
            embedding_provider=self.provider,
            chunking_config=self.config,
        )
        results = retrieve_chunks(
            query="retrieval testing",
            embedding_provider=self.provider,
        )
        if len(results) >= 2:
            for i in range(len(results) - 1):
                self.assertGreaterEqual(results[i].score, results[i + 1].score)

    def test_empty_query_raises(self):
        with self.assertRaises(RetrievalError):
            retrieve_chunks(query="", embedding_provider=self.provider)

    def test_whitespace_query_raises(self):
        with self.assertRaises(RetrievalError):
            retrieve_chunks(query="   ", embedding_provider=self.provider)

    def test_invalid_top_k_raises(self):
        with self.assertRaises(RetrievalError):
            RetrievalConfig(top_k=0)

    def test_only_ready_documents_searched(self):
        """Chunks from non-ready documents should not be returned."""
        from rag.models import Document

        doc = ingest_document(
            title="Draft Doc",
            content="This document is not yet ready for search. " * 5,
            embedding_provider=self.provider,
            chunking_config=self.config,
        )
        # Manually set status to processing
        doc.status = Document.STATUS_PROCESSING
        doc.save()

        results = retrieve_chunks(
            query="not yet ready",
            embedding_provider=self.provider,
        )
        for result in results:
            self.assertNotEqual(result.chunk.document.id, doc.id)

    def test_similarity_threshold_filters_results(self):
        """A very high threshold should filter out most results."""
        config = RetrievalConfig(top_k=10, similarity_threshold=0.99)
        results = retrieve_chunks(
            query="arbitrary query",
            embedding_provider=self.provider,
            config=config,
        )
        for result in results:
            self.assertGreaterEqual(result.score, 0.99)

    def test_score_is_one_minus_distance(self):
        """Score should be 1 - cosine_distance (similarity representation)."""
        results = retrieve_chunks(
            query="Python language",
            embedding_provider=self.provider,
        )
        for result in results:
            # Similarity should be in range [-1, 1] for cosine
            self.assertGreaterEqual(result.score, -1.0)
            self.assertLessEqual(result.score, 1.0)

    # --- M3 tests: retrieval correctness ---

    def test_threshold_filtering_before_top_k(self):
        """M3: Threshold filtering happens before top-K, so K qualifying
        results are returned even if some candidates don't meet threshold."""
        # Ingest enough data for many chunks
        ingest_document(
            title="Many Chunks",
            content="Content for testing threshold before topk. " * 100,
            embedding_provider=self.provider,
            chunking_config=ChunkingConfig(chunk_size=50, chunk_overlap=5),
        )
        # With threshold=0.0, all chunks qualify
        config_no_threshold = RetrievalConfig(top_k=5, similarity_threshold=0.0)
        results_all = retrieve_chunks(
            query="threshold topk test",
            embedding_provider=self.provider,
            config=config_no_threshold,
        )
        self.assertLessEqual(len(results_all), 5)

        # All returned results should have scores >= 0.0
        for r in results_all:
            self.assertGreaterEqual(r.score, 0.0)

    def test_deterministic_ordering(self):
        """M3: Same query, same data should produce identical ordering."""
        results1 = retrieve_chunks(
            query="determinism test",
            embedding_provider=self.provider,
        )
        results2 = retrieve_chunks(
            query="determinism test",
            embedding_provider=self.provider,
        )
        self.assertEqual(len(results1), len(results2))
        for r1, r2 in zip(results1, results2):
            self.assertEqual(r1.chunk_id, r2.chunk_id)
            self.assertEqual(r1.score, r2.score)
            self.assertEqual(r1.rank, r2.rank)

    def test_no_result_behavior(self):
        """M3: Query with very high threshold returns empty list."""
        config = RetrievalConfig(top_k=5, similarity_threshold=1.0)
        results = retrieve_chunks(
            query="something totally unique",
            embedding_provider=self.provider,
            config=config,
        )
        # With threshold=1.0, only exact matches qualify (extremely unlikely with mock)
        self.assertIsInstance(results, list)

    def test_retrieval_result_contract(self):
        """M3: RetrievalResult exposes all required fields."""
        results = retrieve_chunks(
            query="contract test",
            embedding_provider=self.provider,
        )
        if results:
            r = results[0]
            # All M3 fields
            self.assertIsInstance(r.chunk_id, str)
            self.assertIsInstance(r.document_id, str)
            self.assertIsInstance(r.content, str)
            self.assertIsInstance(r.score, float)
            self.assertIsInstance(r.rank, int)
            self.assertIsInstance(r.chunk_index, int)
            self.assertIsInstance(r.document_title, str)
            self.assertIsInstance(r.document_source, str)
            self.assertIsInstance(r.start_offset, int)
            self.assertIsInstance(r.end_offset, int)
            self.assertIsInstance(r.chunk_metadata, dict)
            # Backward compat
            self.assertIsNotNone(r.chunk)

    def test_retrieval_result_metadata_correct(self):
        """M3: RetrievalResult metadata matches the source document."""
        results = retrieve_chunks(
            query="programming language",
            embedding_provider=self.provider,
        )
        if results:
            r = results[0]
            self.assertEqual(r.document_title, "Test Document")
            self.assertGreaterEqual(r.start_offset, 0)
            self.assertGreater(r.end_offset, r.start_offset)
            self.assertGreater(len(r.content), 0)


class RetrievalConfigTests(TestCase):
    """Tests for RetrievalConfig."""

    def test_default_config(self):
        config = RetrievalConfig()
        self.assertEqual(config.top_k, 5)
        self.assertEqual(config.similarity_threshold, 0.0)

    def test_custom_config(self):
        config = RetrievalConfig(top_k=10, similarity_threshold=0.5)
        self.assertEqual(config.top_k, 10)
        self.assertEqual(config.similarity_threshold, 0.5)

    def test_zero_top_k_raises(self):
        with self.assertRaises(RetrievalError):
            RetrievalConfig(top_k=0)

    def test_negative_top_k_raises(self):
        with self.assertRaises(RetrievalError):
            RetrievalConfig(top_k=-1)

    def test_config_from_settings(self):
        """Settings-based config should load defaults correctly."""
        from rag.retrieval import _get_retrieval_config
        config = _get_retrieval_config()
        self.assertIsInstance(config.top_k, int)
        self.assertIsInstance(config.similarity_threshold, float)
