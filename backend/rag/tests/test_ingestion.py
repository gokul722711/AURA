"""Tests for document ingestion."""

from django.test import TestCase

from rag.chunking import ChunkingConfig
from rag.embeddings.mock import MockEmbeddingProvider
from rag.exceptions import DocumentError, EmbeddingError
from rag.ingestion import ingest_document
from rag.models import Document, DocumentChunk


class IngestDocumentTests(TestCase):
    """Tests for the ingest_document function."""

    def setUp(self):
        self.provider = MockEmbeddingProvider(dimensions=384)
        self.chunking_config = ChunkingConfig(chunk_size=50, chunk_overlap=10)

    def test_basic_ingestion(self):
        doc = ingest_document(
            title="Test Doc",
            content="This is a test document with enough content to be ingested properly.",
            embedding_provider=self.provider,
            chunking_config=self.chunking_config,
        )
        self.assertEqual(doc.status, Document.STATUS_READY)
        self.assertEqual(doc.title, "Test Doc")
        self.assertGreater(doc.chunks.count(), 0)

    def test_chunks_have_embeddings(self):
        doc = ingest_document(
            title="Test Doc",
            content="A" * 200,
            embedding_provider=self.provider,
            chunking_config=self.chunking_config,
        )
        for chunk in doc.chunks.all():
            self.assertIsNotNone(chunk.embedding)
            self.assertEqual(len(chunk.embedding), 384)

    def test_chunk_indexes_are_sequential(self):
        doc = ingest_document(
            title="Test Doc",
            content="X" * 200,
            embedding_provider=self.provider,
            chunking_config=self.chunking_config,
        )
        indexes = list(doc.chunks.values_list("chunk_index", flat=True))
        self.assertEqual(indexes, list(range(len(indexes))))

    def test_document_metadata_stored(self):
        doc = ingest_document(
            title="Test Doc",
            content="Content for metadata test.",
            embedding_provider=self.provider,
            metadata={"author": "Test Author", "version": 1},
        )
        self.assertEqual(doc.metadata["author"], "Test Author")
        self.assertEqual(doc.metadata["version"], 1)

    def test_document_source_stored(self):
        doc = ingest_document(
            title="Test Doc",
            content="Content for source test.",
            embedding_provider=self.provider,
            source="/path/to/file.txt",
        )
        self.assertEqual(doc.source, "/path/to/file.txt")

    def test_empty_title_raises(self):
        with self.assertRaises(DocumentError):
            ingest_document(
                title="",
                content="Some content",
                embedding_provider=self.provider,
            )

    def test_whitespace_title_raises(self):
        with self.assertRaises(DocumentError):
            ingest_document(
                title="   ",
                content="Some content",
                embedding_provider=self.provider,
            )

    def test_empty_content_raises(self):
        with self.assertRaises(DocumentError):
            ingest_document(
                title="Test Doc",
                content="",
                embedding_provider=self.provider,
            )

    def test_whitespace_content_raises(self):
        with self.assertRaises(DocumentError):
            ingest_document(
                title="Test Doc",
                content="   \n\t  ",
                embedding_provider=self.provider,
            )

    def test_title_is_stripped(self):
        doc = ingest_document(
            title="  Padded Title  ",
            content="Some content for padding test.",
            embedding_provider=self.provider,
        )
        self.assertEqual(doc.title, "Padded Title")

    def test_failing_embedding_provider_sets_error_status(self):
        class FailingProvider(MockEmbeddingProvider):
            def embed_texts(self, texts):
                raise RuntimeError("Embedding service unavailable")

        with self.assertRaises(EmbeddingError):
            ingest_document(
                title="Fail Doc",
                content="Content that will fail embedding.",
                embedding_provider=FailingProvider(),
            )

        doc = Document.objects.get(title="Fail Doc")
        self.assertEqual(doc.status, Document.STATUS_ERROR)

    def test_short_content_single_chunk(self):
        doc = ingest_document(
            title="Short Doc",
            content="Short.",
            embedding_provider=self.provider,
            chunking_config=ChunkingConfig(chunk_size=100, chunk_overlap=10),
        )
        self.assertEqual(doc.status, Document.STATUS_READY)
        self.assertEqual(doc.chunks.count(), 1)

    def test_multiple_documents_independent(self):
        doc1 = ingest_document(
            title="Doc 1",
            content="First document content here.",
            embedding_provider=self.provider,
        )
        doc2 = ingest_document(
            title="Doc 2",
            content="Second document content here.",
            embedding_provider=self.provider,
        )
        self.assertNotEqual(doc1.id, doc2.id)
        self.assertEqual(doc1.status, Document.STATUS_READY)
        self.assertEqual(doc2.status, Document.STATUS_READY)

    def test_failed_chunk_persistence_rolls_back_chunks_and_sets_error_status(self):
        """Failure during chunk persistence sets status to error and persists no chunks."""
        from unittest.mock import patch

        with patch.object(
            DocumentChunk.objects, "bulk_create", side_effect=RuntimeError("Database failure")
        ):
            with self.assertRaises(RuntimeError):
                ingest_document(
                    title="Rollback Doc",
                    content="Content that will fail chunk persistence.",
                    embedding_provider=self.provider,
                )

        doc = Document.objects.get(title="Rollback Doc")
        self.assertEqual(doc.status, Document.STATUS_ERROR)
        self.assertEqual(doc.chunks.count(), 0)

