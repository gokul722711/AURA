"""Tests for the RAG pipeline.

Tests the full RAG flow: query → retrieve → assemble → ModelGateway.generate() → response.
Uses MockLLMProvider and MockEmbeddingProvider — no external APIs required.
"""

from django.test import TestCase

from gateway.gateway import ModelGateway
from gateway.providers.mock import MockLLMProvider

from rag.chunking import ChunkingConfig
from rag.embeddings.mock import MockEmbeddingProvider
from rag.ingestion import ingest_document
from rag.pipeline import RAGConfig, RAGPipeline, RAGResponse
from rag.retrieval import RetrievalConfig


class RAGPipelineTests(TestCase):
    """Tests for RAGPipeline."""

    def setUp(self):
        self.embedding_provider = MockEmbeddingProvider(dimensions=384)
        self.llm_provider = MockLLMProvider()
        self.gateway = ModelGateway(provider=self.llm_provider)
        self.pipeline = RAGPipeline(
            gateway=self.gateway,
            embedding_provider=self.embedding_provider,
        )

        # Ingest a test document
        self.document = ingest_document(
            title="Python Guide",
            content=(
                "Python is a high-level programming language. "
                "It supports multiple programming paradigms. "
                "Python is widely used in data science, web development, and automation. "
            ) * 5,
            embedding_provider=self.embedding_provider,
            source="/docs/python.md",
            chunking_config=ChunkingConfig(chunk_size=100, chunk_overlap=10),
        )

    def test_basic_query(self):
        response = self.pipeline.query("What is Python?")
        self.assertIsInstance(response, RAGResponse)
        self.assertIsInstance(response.answer, str)
        self.assertGreater(len(response.answer), 0)

    def test_response_has_sources(self):
        response = self.pipeline.query("What is Python?")
        self.assertIsInstance(response.sources, list)

    def test_response_has_query(self):
        response = self.pipeline.query("What is Python?")
        self.assertEqual(response.query, "What is Python?")

    def test_response_has_usage(self):
        response = self.pipeline.query("What is Python?")
        self.assertIn("prompt_tokens", response.usage)
        self.assertIn("completion_tokens", response.usage)
        self.assertIn("total_tokens", response.usage)

    def test_response_metadata_has_provider(self):
        response = self.pipeline.query("What is Python?")
        self.assertEqual(response.metadata["provider"], "mock")
        self.assertIn("retrieval_count", response.metadata)

    def test_generation_goes_through_gateway(self):
        """Verify generation uses ModelGateway, not direct provider call."""
        self.assertIsInstance(self.pipeline.gateway, ModelGateway)
        response = self.pipeline.query("test query")
        # MockLLMProvider produces "Mock response to: ..." format
        self.assertIn("Mock response to:", response.answer)

    def test_custom_rag_config(self):
        config = RAGConfig(
            retrieval_config=RetrievalConfig(top_k=2),
            temperature=0.5,
        )
        response = self.pipeline.query("Python language", config=config)
        self.assertIsInstance(response, RAGResponse)
        self.assertLessEqual(len(response.sources), 2)

    def test_query_with_no_documents(self):
        """Query when no documents exist should still produce a response."""
        from rag.models import Document

        Document.objects.all().delete()
        response = self.pipeline.query("nonexistent topic")
        self.assertIsInstance(response, RAGResponse)
        self.assertIsInstance(response.answer, str)
        self.assertEqual(len(response.sources), 0)

    def test_pipeline_properties(self):
        self.assertIsInstance(self.pipeline.gateway, ModelGateway)
        self.assertIsInstance(self.pipeline.embedding_provider, MockEmbeddingProvider)

    def test_multiple_queries(self):
        """Pipeline should handle multiple sequential queries."""
        r1 = self.pipeline.query("first question")
        r2 = self.pipeline.query("second question")
        self.assertIsInstance(r1, RAGResponse)
        self.assertIsInstance(r2, RAGResponse)
        # Different queries should produce different answers
        self.assertNotEqual(r1.answer, r2.answer)

    def test_rag_response_fields(self):
        response = self.pipeline.query("What is Python?")
        self.assertIsInstance(response.answer, str)
        self.assertIsInstance(response.sources, list)
        self.assertIsInstance(response.query, str)
        self.assertIsInstance(response.usage, dict)
        self.assertIsInstance(response.metadata, dict)


class RAGConfigTests(TestCase):
    """Tests for RAGConfig."""

    def test_default_config(self):
        config = RAGConfig()
        self.assertIsNotNone(config.system_prompt)
        self.assertEqual(config.temperature, 0.1)
        self.assertIsNone(config.max_tokens)

    def test_custom_config(self):
        config = RAGConfig(
            retrieval_config=RetrievalConfig(top_k=3),
            temperature=0.7,
            max_tokens=500,
        )
        self.assertEqual(config.retrieval_config.top_k, 3)
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.max_tokens, 500)
