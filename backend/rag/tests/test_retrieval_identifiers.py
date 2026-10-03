"""Regression tests for document identifier resolution in retrieval (UUID and non-UUID)."""

import uuid
from django.test import TestCase

from rag.chunking import ChunkingConfig
from rag.embeddings.mock import MockEmbeddingProvider
from rag.ingestion import ingest_document
from rag.retrieval import (
    RetrievalConfig,
    retrieve_chunks,
    resolve_document_identifiers,
)


class DocumentIdentifierResolutionTests(TestCase):
    """Tests verifying safe handling and resolution of UUID and non-UUID document identifiers."""

    def setUp(self) -> None:
        self.provider = MockEmbeddingProvider(dimensions=384)
        self.config = ChunkingConfig(chunk_size=100, chunk_overlap=10)

        self.doc_a = ingest_document(
            title="Alpha Project Guide",
            content="Autonomous research agents perform structured planning and retrieval. " * 5,
            embedding_provider=self.provider,
            chunking_config=self.config,
            source="docs/alpha.md",
        )
        self.doc_b = ingest_document(
            title="Beta Web Services",
            content="Web services communicate via REST APIs and JSON endpoints. " * 5,
            embedding_provider=self.provider,
            chunking_config=self.config,
            source="https://example.com/beta-service",
            metadata={
                "url": "https://example.com/beta-service",
                "canonical_url": "https://example.com/canonical-beta",
            },
        )

    def test_valid_uuid_document_ids(self) -> None:
        """Retrieval restricted to valid Document.id UUID returns only matching chunks."""
        config = RetrievalConfig(
            top_k=5,
            document_ids=[str(self.doc_a.id)],
        )
        results = retrieve_chunks(
            query="autonomous research",
            embedding_provider=self.provider,
            config=config,
        )
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r.document_id, str(self.doc_a.id))
            self.assertEqual(r.document_title, "Alpha Project Guide")

    def test_document_title_identifier(self) -> None:
        """Retrieval resolves document title (exact and case-insensitive)."""
        # Exact title
        config_exact = RetrievalConfig(
            top_k=5,
            document_ids=["Alpha Project Guide"],
        )
        results_exact = retrieve_chunks(
            query="autonomous research",
            embedding_provider=self.provider,
            config=config_exact,
        )
        self.assertGreater(len(results_exact), 0)
        for r in results_exact:
            self.assertEqual(r.document_id, str(self.doc_a.id))

        # Case-insensitive title
        config_case = RetrievalConfig(
            top_k=5,
            document_ids=["alpha project guide"],
        )
        results_case = retrieve_chunks(
            query="autonomous research",
            embedding_provider=self.provider,
            config=config_case,
        )
        self.assertGreater(len(results_case), 0)
        for r in results_case:
            self.assertEqual(r.document_id, str(self.doc_a.id))

    def test_document_source_identifier(self) -> None:
        """Retrieval resolves document source identifier."""
        config = RetrievalConfig(
            top_k=5,
            document_ids=["docs/alpha.md"],
        )
        results = retrieve_chunks(
            query="autonomous research",
            embedding_provider=self.provider,
            config=config,
        )
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r.document_id, str(self.doc_a.id))

    def test_web_url_identifier(self) -> None:
        """Retrieval resolves web URL identifier, including trailing slash tolerance."""
        # Exact URL
        config_url = RetrievalConfig(
            top_k=5,
            document_ids=["https://example.com/beta-service"],
        )
        results = retrieve_chunks(
            query="web services",
            embedding_provider=self.provider,
            config=config_url,
        )
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r.document_id, str(self.doc_b.id))

        # URL with trailing slash
        config_slash = RetrievalConfig(
            top_k=5,
            document_ids=["https://example.com/beta-service/"],
        )
        results_slash = retrieve_chunks(
            query="web services",
            embedding_provider=self.provider,
            config=config_slash,
        )
        self.assertGreater(len(results_slash), 0)
        for r in results_slash:
            self.assertEqual(r.document_id, str(self.doc_b.id))

    def test_canonical_url_identifier(self) -> None:
        """Retrieval resolves canonical URL from document metadata."""
        config = RetrievalConfig(
            top_k=5,
            document_ids=["https://example.com/canonical-beta"],
        )
        results = retrieve_chunks(
            query="web services",
            embedding_provider=self.provider,
            config=config,
        )
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r.document_id, str(self.doc_b.id))

    def test_unknown_identifier_no_validation_error(self) -> None:
        """Unknown identifier does not raise ValidationError and does not fall back to unrestricted search."""
        config = RetrievalConfig(
            top_k=5,
            document_ids=["completely_nonexistent_document_identifier_12345"],
        )
        # Must not raise ValidationError
        results = retrieve_chunks(
            query="autonomous research web services",
            embedding_provider=self.provider,
            config=config,
        )
        # Must safely produce no results rather than returning all chunks
        self.assertEqual(results, [])

    def test_unknown_uuid_identifier_no_crash(self) -> None:
        """Valid UUID that doesn't exist returns empty results without crashing."""
        random_uuid = str(uuid.uuid4())
        config = RetrievalConfig(
            top_k=5,
            document_ids=[random_uuid],
        )
        results = retrieve_chunks(
            query="autonomous research",
            embedding_provider=self.provider,
            config=config,
        )
        self.assertEqual(results, [])

    def test_mixed_valid_and_unknown_identifiers(self) -> None:
        """Mixed valid and unknown identifiers resolve the valid documents."""
        config = RetrievalConfig(
            top_k=5,
            document_ids=["nonexistent_title", str(self.doc_a.id)],
        )
        results = retrieve_chunks(
            query="autonomous research",
            embedding_provider=self.provider,
            config=config,
        )
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r.document_id, str(self.doc_a.id))

    def test_empty_document_ids_list_returns_empty(self) -> None:
        """Empty list of document_ids restricts to empty set, producing 0 results."""
        config = RetrievalConfig(
            top_k=5,
            document_ids=[],
        )
        results = retrieve_chunks(
            query="autonomous research",
            embedding_provider=self.provider,
            config=config,
        )
        self.assertEqual(results, [])
