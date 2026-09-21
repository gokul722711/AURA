"""Tests for RAG Django models."""

from django.test import TestCase

from rag.embeddings.mock import MockEmbeddingProvider
from rag.models import Document, DocumentChunk


class DocumentModelTests(TestCase):
    """Tests for the Document model."""

    def test_create_document(self):
        doc = Document.objects.create(
            title="Test Document",
            content="This is test content.",
        )
        self.assertIsNotNone(doc.id)
        self.assertEqual(doc.title, "Test Document")
        self.assertEqual(doc.content, "This is test content.")
        self.assertEqual(doc.status, Document.STATUS_PENDING)
        self.assertIsNotNone(doc.created_at)
        self.assertIsNotNone(doc.updated_at)

    def test_default_status_is_pending(self):
        doc = Document.objects.create(title="Doc", content="Content")
        self.assertEqual(doc.status, "pending")

    def test_default_metadata_is_empty_dict(self):
        doc = Document.objects.create(title="Doc", content="Content")
        self.assertEqual(doc.metadata, {})

    def test_default_source_is_empty(self):
        doc = Document.objects.create(title="Doc", content="Content")
        self.assertEqual(doc.source, "")

    def test_status_transition(self):
        doc = Document.objects.create(
            title="Doc", content="Content", status=Document.STATUS_PROCESSING
        )
        doc.status = Document.STATUS_READY
        doc.save()
        doc.refresh_from_db()
        self.assertEqual(doc.status, "ready")

    def test_str_representation(self):
        doc = Document(title="My Doc", status="ready")
        self.assertIn("My Doc", str(doc))
        self.assertIn("ready", str(doc))


class DocumentChunkModelTests(TestCase):
    """Tests for the DocumentChunk model."""

    def setUp(self):
        self.provider = MockEmbeddingProvider(dimensions=384)
        self.document = Document.objects.create(
            title="Test Doc",
            content="Full document content for testing.",
            status=Document.STATUS_READY,
        )

    def test_create_chunk_with_embedding(self):
        embedding = self.provider.embed_query("chunk text")
        chunk = DocumentChunk.objects.create(
            document=self.document,
            content="chunk text",
            chunk_index=0,
            start_offset=0,
            end_offset=10,
            embedding=embedding,
        )
        self.assertIsNotNone(chunk.id)
        self.assertEqual(chunk.document, self.document)
        self.assertEqual(chunk.chunk_index, 0)

    def test_chunk_document_relationship(self):
        embedding = self.provider.embed_query("test")
        DocumentChunk.objects.create(
            document=self.document,
            content="chunk 1",
            chunk_index=0,
            start_offset=0,
            end_offset=7,
            embedding=embedding,
        )
        DocumentChunk.objects.create(
            document=self.document,
            content="chunk 2",
            chunk_index=1,
            start_offset=7,
            end_offset=14,
            embedding=embedding,
        )
        self.assertEqual(self.document.chunks.count(), 2)

    def test_cascade_delete(self):
        embedding = self.provider.embed_query("test")
        DocumentChunk.objects.create(
            document=self.document,
            content="chunk",
            chunk_index=0,
            start_offset=0,
            end_offset=5,
            embedding=embedding,
        )
        doc_id = self.document.id
        self.document.delete()
        self.assertEqual(DocumentChunk.objects.filter(document_id=doc_id).count(), 0)

    def test_str_representation(self):
        embedding = self.provider.embed_query("test")
        chunk = DocumentChunk(
            document=self.document,
            content="test",
            chunk_index=3,
            start_offset=0,
            end_offset=4,
            embedding=embedding,
        )
        self.assertIn("Test Doc", str(chunk))
        self.assertIn("3", str(chunk))

    def test_ordering(self):
        embedding = self.provider.embed_query("test")
        DocumentChunk.objects.create(
            document=self.document, content="c2", chunk_index=2,
            start_offset=10, end_offset=12, embedding=embedding,
        )
        DocumentChunk.objects.create(
            document=self.document, content="c0", chunk_index=0,
            start_offset=0, end_offset=2, embedding=embedding,
        )
        DocumentChunk.objects.create(
            document=self.document, content="c1", chunk_index=1,
            start_offset=2, end_offset=10, embedding=embedding,
        )
        chunks = list(self.document.chunks.all())
        self.assertEqual([c.chunk_index for c in chunks], [0, 1, 2])
