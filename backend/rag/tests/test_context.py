"""Tests for context assembly."""

from django.test import TestCase

from rag.chunking import ChunkingConfig
from rag.context import assemble_context
from rag.embeddings.mock import MockEmbeddingProvider
from rag.exceptions import ContextAssemblyError
from rag.ingestion import ingest_document
from rag.retrieval import RetrievalResult, retrieve_chunks


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
