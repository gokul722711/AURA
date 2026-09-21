"""Tests for embedding providers and registry.

Tests verify pipeline behavior and deterministic vector generation.
The mock embeddings are hash-based and do NOT have meaningful semantic quality.
"""

from django.test import TestCase

from rag.embeddings.base import EmbeddingProvider
from rag.embeddings.mock import MockEmbeddingProvider
from rag.embeddings.registry import (
    create_embedding_provider,
    get_embedding_provider_class,
    register_embedding_provider,
)
from rag.exceptions import EmbeddingError


class MockEmbeddingProviderTests(TestCase):
    """Tests for MockEmbeddingProvider."""

    def test_default_dimensions(self):
        provider = MockEmbeddingProvider()
        self.assertEqual(provider.dimensions(), 384)

    def test_custom_dimensions(self):
        provider = MockEmbeddingProvider(dimensions=768)
        self.assertEqual(provider.dimensions(), 768)

    def test_embed_query_returns_correct_dimensions(self):
        provider = MockEmbeddingProvider(dimensions=384)
        vector = provider.embed_query("test query")
        self.assertEqual(len(vector), 384)

    def test_embed_texts_returns_correct_count(self):
        provider = MockEmbeddingProvider()
        texts = ["text one", "text two", "text three"]
        embeddings = provider.embed_texts(texts)
        self.assertEqual(len(embeddings), 3)

    def test_embed_texts_correct_dimensions(self):
        provider = MockEmbeddingProvider(dimensions=384)
        embeddings = provider.embed_texts(["hello", "world"])
        for embedding in embeddings:
            self.assertEqual(len(embedding), 384)

    def test_deterministic_same_input_same_output(self):
        """Same input text must produce the same vector."""
        provider = MockEmbeddingProvider()
        v1 = provider.embed_query("test input")
        v2 = provider.embed_query("test input")
        self.assertEqual(v1, v2)

    def test_deterministic_embed_texts(self):
        """embed_texts must be deterministic."""
        provider = MockEmbeddingProvider()
        e1 = provider.embed_texts(["a", "b"])
        e2 = provider.embed_texts(["a", "b"])
        self.assertEqual(e1, e2)

    def test_different_inputs_different_vectors(self):
        """Different inputs should produce different vectors."""
        provider = MockEmbeddingProvider()
        v1 = provider.embed_query("input one")
        v2 = provider.embed_query("input two")
        self.assertNotEqual(v1, v2)

    def test_embed_query_matches_embed_texts(self):
        """embed_query and embed_texts should produce the same vector for the same text."""
        provider = MockEmbeddingProvider()
        query_vector = provider.embed_query("test text")
        batch_vectors = provider.embed_texts(["test text"])
        self.assertEqual(query_vector, batch_vectors[0])

    def test_vectors_are_unit_normalized(self):
        """Vectors should be approximately unit-length for cosine distance."""
        import math

        provider = MockEmbeddingProvider()
        vector = provider.embed_query("normalization test")
        magnitude = math.sqrt(sum(x * x for x in vector))
        self.assertAlmostEqual(magnitude, 1.0, places=6)

    def test_all_values_are_floats(self):
        provider = MockEmbeddingProvider()
        vector = provider.embed_query("type check")
        for value in vector:
            self.assertIsInstance(value, float)

    def test_empty_text_produces_valid_vector(self):
        provider = MockEmbeddingProvider()
        vector = provider.embed_query("")
        self.assertEqual(len(vector), 384)

    def test_implements_embedding_provider_interface(self):
        provider = MockEmbeddingProvider()
        self.assertIsInstance(provider, EmbeddingProvider)


class EmbeddingRegistryTests(TestCase):
    """Tests for the embedding provider registry."""

    def test_mock_provider_registered_by_default(self):
        cls = get_embedding_provider_class("mock")
        self.assertEqual(cls, MockEmbeddingProvider)

    def test_case_insensitive_lookup(self):
        cls = get_embedding_provider_class("MOCK")
        self.assertEqual(cls, MockEmbeddingProvider)

    def test_unregistered_provider_raises(self):
        with self.assertRaises(EmbeddingError):
            get_embedding_provider_class("nonexistent")

    def test_register_custom_provider(self):
        class CustomEmbedding(MockEmbeddingProvider):
            pass

        register_embedding_provider("custom_test", CustomEmbedding)
        cls = get_embedding_provider_class("custom_test")
        self.assertEqual(cls, CustomEmbedding)

    def test_register_invalid_class_raises(self):
        class NotAnEmbeddingProvider:
            pass

        with self.assertRaises(EmbeddingError):
            register_embedding_provider("invalid", NotAnEmbeddingProvider)

    def test_register_empty_name_raises(self):
        with self.assertRaises(EmbeddingError):
            register_embedding_provider("", MockEmbeddingProvider)

    def test_register_non_string_name_raises(self):
        with self.assertRaises(EmbeddingError):
            register_embedding_provider(None, MockEmbeddingProvider)

    def test_create_embedding_provider(self):
        provider = create_embedding_provider("mock", dimensions=384)
        self.assertIsInstance(provider, MockEmbeddingProvider)
        self.assertEqual(provider.dimensions(), 384)

    def test_create_embedding_provider_custom_dimensions(self):
        provider = create_embedding_provider("mock", dimensions=768)
        self.assertEqual(provider.dimensions(), 768)
