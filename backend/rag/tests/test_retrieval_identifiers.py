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

    def test_single_document_identifier_bypasses_max_chunks_per_document(self) -> None:
        """When document_ids restricts to exactly one document, max_chunks_per_document is not applied."""
        doc_large = ingest_document(
            title="Large Topic Document",
            content="Detailed section on agentic retrieval systems and pgvector vector search algorithms. " * 20,
            embedding_provider=self.provider,
            chunking_config=self.config,
        )
        # Search restricted to doc_large with top_k=5 and max_chunks_per_document=2
        # Must return 5 chunks from doc_large rather than being capped at 2.
        config_uuid = RetrievalConfig(
            top_k=5,
            document_ids=[str(doc_large.id)],
            max_chunks_per_document=2,
        )
        results_uuid = retrieve_chunks(
            query="agentic retrieval systems",
            embedding_provider=self.provider,
            config=config_uuid,
        )
        self.assertEqual(len(results_uuid), 5)
        for r in results_uuid:
            self.assertEqual(r.document_id, str(doc_large.id))

        # Also works via title resolution
        config_title = RetrievalConfig(
            top_k=5,
            document_ids=["Large Topic Document"],
            max_chunks_per_document=2,
        )
        results_title = retrieve_chunks(
            query="agentic retrieval systems",
            embedding_provider=self.provider,
            config=config_title,
        )
        self.assertEqual(len(results_title), 5)
        for r in results_title:
            self.assertEqual(r.document_id, str(doc_large.id))

    def test_multi_document_retrieval_still_enforces_max_chunks_per_document(self) -> None:
        """When multiple documents are resolved or search is unrestricted, max_chunks_per_document is enforced."""
        doc_large = ingest_document(
            title="Second Large Document",
            content="Autonomous research agents perform structured planning and retrieval operations. " * 20,
            embedding_provider=self.provider,
            chunking_config=self.config,
        )
        # 1. Multiple resolved documents
        config_multi = RetrievalConfig(
            top_k=6,
            document_ids=[str(self.doc_a.id), str(doc_large.id)],
            max_chunks_per_document=2,
        )
        results_multi = retrieve_chunks(
            query="autonomous research planning",
            embedding_provider=self.provider,
            config=config_multi,
        )
        doc_counts: dict[str, int] = {}
        for r in results_multi:
            doc_counts[r.document_id] = doc_counts.get(r.document_id, 0) + 1
        for did, count in doc_counts.items():
            self.assertLessEqual(count, 2)
        self.assertGreater(len(doc_counts), 1)

        # 2. Unrestricted search across all documents
        config_unrestricted = RetrievalConfig(
            top_k=6,
            max_chunks_per_document=2,
        )
        results_unrestricted = retrieve_chunks(
            query="autonomous research planning",
            embedding_provider=self.provider,
            config=config_unrestricted,
        )
        unrestricted_counts: dict[str, int] = {}
        for r in results_unrestricted:
            unrestricted_counts[r.document_id] = unrestricted_counts.get(r.document_id, 0) + 1
        for did, count in unrestricted_counts.items():
            self.assertLessEqual(count, 2)

    def test_unknown_identifier_with_max_chunks_returns_zero(self) -> None:
        """Unknown identifier with max_chunks_per_document configured fails closed with 0 chunks."""
        config = RetrievalConfig(
            top_k=5,
            document_ids=["completely_nonexistent_document_identifier_99999"],
            max_chunks_per_document=2,
        )
        results = retrieve_chunks(
            query="autonomous research planning",
            embedding_provider=self.provider,
            config=config,
        )
        self.assertEqual(results, [])
