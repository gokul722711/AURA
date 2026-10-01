"""Tests for context assembly.

M3 tests cover:
- Context budget enforcement
- Ranking preservation
- Source/chunk attribution
- Redundancy handling for overlapping chunks
- Empty context behavior
- No-context formatting
"""

from django.test import TestCase

from rag.chunking import ChunkingConfig
from rag.context import ContextConfig, assemble_context, _is_redundant
from rag.embeddings.mock import MockEmbeddingProvider
from rag.exceptions import ContextAssemblyError
from rag.ingestion import ingest_document
from rag.retrieval import RetrievalConfig, RetrievalResult, retrieve_chunks


class AssembleContextTests(TestCase):
    """Tests for assemble_context function."""

    def setUp(self):
        self.provider = MockEmbeddingProvider(dimensions=384)

    def test_no_results_produces_no_context_message(self):
        result = assemble_context([], "What is Python?")
        self.assertIn("No relevant context", result)
        self.assertIn("What is Python?", result)

    def test_context_includes_query(self):
        ingest_document(
            title="Doc",
            content="Python is a programming language. " * 5,
            embedding_provider=self.provider,
        )
        results = retrieve_chunks(
            query="What is Python?",
            embedding_provider=self.provider,
        )
        context = assemble_context(results, "What is Python?")
        self.assertIn("What is Python?", context)

    def test_context_includes_source_references(self):
        ingest_document(
            title="Python Guide",
            content="Python is a programming language used for many tasks. " * 5,
            embedding_provider=self.provider,
            source="/docs/python.txt",
        )
        results = retrieve_chunks(
            query="What is Python?",
            embedding_provider=self.provider,
        )
        if results:
            context = assemble_context(results, "What is Python?")
            self.assertIn("/docs/python.txt", context)

    def test_context_includes_chunk_content(self):
        ingest_document(
            title="Doc",
            content="Django is a web framework for Python.",
            embedding_provider=self.provider,
        )
        results = retrieve_chunks(
            query="Django framework",
            embedding_provider=self.provider,
        )
        if results:
            context = assemble_context(results, "Django framework")
            # The chunk content should be present in the assembled context
            self.assertIn("Django", context)

    def test_context_has_structure(self):
        ingest_document(
            title="Structured Doc",
            content="Machine learning is a subset of AI. " * 5,
            embedding_provider=self.provider,
        )
        results = retrieve_chunks(
            query="machine learning",
            embedding_provider=self.provider,
        )
        if results:
            context = assemble_context(results, "machine learning")
            self.assertIn("Retrieved Context", context)
            self.assertIn("End of Context", context)
            self.assertIn("Question:", context)

    def test_empty_query_raises(self):
        with self.assertRaises(ContextAssemblyError):
            assemble_context([], "")

    def test_whitespace_query_raises(self):
        with self.assertRaises(ContextAssemblyError):
            assemble_context([], "   ")

    def test_context_includes_similarity_score(self):
        ingest_document(
            title="Score Doc",
            content="Testing similarity scores in context assembly. " * 5,
            embedding_provider=self.provider,
        )
        results = retrieve_chunks(
            query="similarity scores",
            embedding_provider=self.provider,
        )
        if results:
            context = assemble_context(results, "similarity scores")
            self.assertIn("Similarity:", context)

    def test_title_used_when_no_source(self):
        ingest_document(
            title="My Title",
            content="Content without a source field set. " * 3,
            embedding_provider=self.provider,
        )
        results = retrieve_chunks(
            query="content without source",
            embedding_provider=self.provider,
        )
        if results:
            context = assemble_context(results, "content without source")
            self.assertIn("My Title", context)

    # --- M3 tests: context optimization ---

    def test_context_budget_limits_content(self):
        """M3: Context assembly should stop when budget is exhausted."""
        ingest_document(
            title="Budget Doc",
            content="A" * 1000,
            embedding_provider=self.provider,
            chunking_config=ChunkingConfig(chunk_size=100, chunk_overlap=10),
        )
        results = retrieve_chunks(
            query="budget test",
            embedding_provider=self.provider,
            config=RetrievalConfig(top_k=10),
        )
        # Set a budget that allows only ~2 chunks (100 chars each)
        config = ContextConfig(max_chars=200)
        context = assemble_context(results, "budget test", config=config)
        # Count how many [Source: sections are present
        source_count = context.count("[Source:")
        # Should include at most 2 chunks of 100 chars each
        self.assertLessEqual(source_count, 2)

    def test_context_budget_none_means_no_limit(self):
        """M3: No budget means all chunks are included."""
        ingest_document(
            title="No Limit Doc",
            content="B" * 500,
            embedding_provider=self.provider,
            chunking_config=ChunkingConfig(chunk_size=100, chunk_overlap=10),
        )
        results = retrieve_chunks(
            query="no limit test",
            embedding_provider=self.provider,
            config=RetrievalConfig(top_k=10),
        )
        config = ContextConfig(max_chars=None)
        context = assemble_context(results, "no limit test", config=config)
        # All retrieved results should be present
        source_count = context.count("[Source:")
        self.assertEqual(source_count, len(results))

    def test_ranking_preserved_in_context(self):
        """M3: Context should preserve retrieval ranking order."""
        ingest_document(
            title="Ranking Doc",
            content="Ranking test content for context assembly. " * 20,
            embedding_provider=self.provider,
            chunking_config=ChunkingConfig(chunk_size=80, chunk_overlap=10),
        )
        results = retrieve_chunks(
            query="ranking preservation",
            embedding_provider=self.provider,
        )
        if len(results) >= 2:
            context = assemble_context(results, "ranking preservation")
            # First chunk should appear before second chunk in context
            first_chunk_pos = context.find(results[0].content[:20])
            second_chunk_pos = context.find(results[1].content[:20])
            if first_chunk_pos >= 0 and second_chunk_pos >= 0:
                self.assertLess(first_chunk_pos, second_chunk_pos)

    def test_empty_context_from_budget_returns_no_context(self):
        """M3: If budget is too small for any chunk, return no-context message."""
        ingest_document(
            title="Tiny Budget Doc",
            content="This content is longer than budget. " * 3,
            embedding_provider=self.provider,
        )
        results = retrieve_chunks(
            query="tiny budget",
            embedding_provider=self.provider,
        )
        # Budget of 1 char — no chunk can fit
        config = ContextConfig(max_chars=1)
        context = assemble_context(results, "tiny budget", config=config)
        self.assertIn("No relevant context", context)

    def test_zero_budget_returns_no_context(self):
        """M3: Budget of 0 chars returns no-context message."""
        ingest_document(
            title="Zero Budget Doc",
            content="Some document content here. " * 3,
            embedding_provider=self.provider,
        )
        results = retrieve_chunks(
            query="zero budget",
            embedding_provider=self.provider,
        )
        config = ContextConfig(max_chars=0)
        context = assemble_context(results, "zero budget", config=config)
        self.assertIn("No relevant context", context)

    def test_negative_budget_returns_no_context(self):
        """M3: Negative budget returns no-context message."""
        ingest_document(
            title="Negative Budget Doc",
            content="Some document content here. " * 3,
            embedding_provider=self.provider,
        )
        results = retrieve_chunks(
            query="negative budget",
            embedding_provider=self.provider,
        )
        config = ContextConfig(max_chars=-10)
        context = assemble_context(results, "negative budget", config=config)
        self.assertIn("No relevant context", context)

    def test_attribution_present(self):
        """M3: Each chunk section has source attribution."""
        ingest_document(
            title="Attribution Doc",
            content="Testing attribution in context. " * 5,
            embedding_provider=self.provider,
            source="test-source.txt",
        )
        results = retrieve_chunks(
            query="attribution test",
            embedding_provider=self.provider,
        )
        if results:
            context = assemble_context(results, "attribution test")
            self.assertIn("Source:", context)
            self.assertIn("Chunk", context)
            self.assertIn("Similarity:", context)


class RedundancyHandlingTests(TestCase):
    """Tests for deterministic redundancy handling."""

    def test_is_redundant_same_doc_fully_contained(self):
        """A chunk fully contained within another from the same doc is redundant."""
        from rag.models import DocumentChunk, Document

        # Create fake RetrievalResult objects for testing _is_redundant
        # We need to construct minimal RetrievalResult instances
        existing = RetrievalResult(
            chunk_id="c1", document_id="d1", content="full content",
            score=0.9, rank=1, chunk_index=0, document_title="Doc",
            document_source="", start_offset=0, end_offset=100,
            chunk_metadata={}, chunk=None,
        )
        candidate = RetrievalResult(
            chunk_id="c2", document_id="d1", content="partial",
            score=0.8, rank=2, chunk_index=1, document_title="Doc",
            document_source="", start_offset=10, end_offset=60,
            chunk_metadata={}, chunk=None,
        )
        self.assertTrue(_is_redundant(candidate, [existing]))

    def test_is_not_redundant_different_doc(self):
        """Chunks from different documents are never redundant."""
        existing = RetrievalResult(
            chunk_id="c1", document_id="d1", content="content",
            score=0.9, rank=1, chunk_index=0, document_title="Doc1",
            document_source="", start_offset=0, end_offset=100,
            chunk_metadata={}, chunk=None,
        )
        candidate = RetrievalResult(
            chunk_id="c2", document_id="d2", content="content",
            score=0.8, rank=2, chunk_index=0, document_title="Doc2",
            document_source="", start_offset=0, end_offset=100,
            chunk_metadata={}, chunk=None,
        )
        self.assertFalse(_is_redundant(candidate, [existing]))

    def test_is_not_redundant_partial_overlap(self):
        """Partially overlapping chunks are NOT redundant (only fully contained)."""
        existing = RetrievalResult(
            chunk_id="c1", document_id="d1", content="content",
            score=0.9, rank=1, chunk_index=0, document_title="Doc",
            document_source="", start_offset=0, end_offset=100,
            chunk_metadata={}, chunk=None,
        )
        candidate = RetrievalResult(
            chunk_id="c2", document_id="d1", content="content",
            score=0.8, rank=2, chunk_index=1, document_title="Doc",
            document_source="", start_offset=50, end_offset=150,
            chunk_metadata={}, chunk=None,
        )
        self.assertFalse(_is_redundant(candidate, [existing]))

    def test_is_not_redundant_empty_accepted(self):
        """No redundancy if no chunks have been accepted yet."""
        candidate = RetrievalResult(
            chunk_id="c1", document_id="d1", content="content",
            score=0.9, rank=1, chunk_index=0, document_title="Doc",
            document_source="", start_offset=0, end_offset=100,
            chunk_metadata={}, chunk=None,
        )
        self.assertFalse(_is_redundant(candidate, []))
