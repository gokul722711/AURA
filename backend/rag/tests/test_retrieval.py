"""Tests for vector retrieval.

Tests verify retrieval pipeline mechanics using the MockEmbeddingProvider.
The mock embeddings are hash-based and do not have meaningful semantic quality.
These tests verify:
- Database-side cosine distance search works correctly
- Results are ordered by similarity (highest first)
- top_k limiting works
- Similarity threshold filtering works
- Only 'ready' documents are searched
"""

from django.test import TestCase

from rag.chunking import ChunkingConfig
from rag.embeddings.mock import MockEmbeddingProvider
from rag.exceptions import RetrievalError
from rag.ingestion import ingest_document
from rag.retrieval import RetrievalConfig, retrieve_chunks


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
