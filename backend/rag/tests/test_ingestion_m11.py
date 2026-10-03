"""Deterministic unit and integration tests for M11 document ingestion."""

from django.test import TestCase

from rag.chunking import ChunkingConfig
from rag.embeddings.mock import MockEmbeddingProvider
from rag.exceptions import DocumentError, EmbeddingError
from rag.ingestion import ingest_document, ingest_file
from rag.models import Document, DocumentChunk
from rag.tests.test_extraction import _make_docx, _make_pdf


class DocumentIngestionM11Tests(TestCase):
    """Tests for multi-format document ingestion pipeline."""

    def setUp(self) -> None:
        self.embedding_provider = MockEmbeddingProvider(dimensions=384)

    def test_ingest_txt_file(self) -> None:
        txt_bytes = b"Artificial intelligence is transforming autonomous engineering workflows."
        doc = ingest_file(
            file_bytes=txt_bytes,
            filename="intro.txt",
            title="AI Workflows",
            embedding_provider=self.embedding_provider,
        )

        self.assertEqual(doc.status, Document.STATUS_READY)
        self.assertEqual(doc.title, "AI Workflows")
        self.assertEqual(doc.source_type, "txt")
        self.assertEqual(doc.filename, "intro.txt")
        self.assertEqual(doc.file_size, len(txt_bytes))
        self.assertGreater(doc.chunks.count(), 0)

        chunk = doc.chunks.first()
        self.assertEqual(chunk.metadata.get("source_type"), "txt")

    def test_ingest_markdown_file(self) -> None:
        md_bytes = b"# System Architecture\n\n- Model Gateway\n- pgvector RAG\n- Agent System"
        doc = ingest_file(
            file_bytes=md_bytes,
            filename="arch.md",
            embedding_provider=self.embedding_provider,
        )

        self.assertEqual(doc.status, Document.STATUS_READY)
        self.assertEqual(doc.title, "arch")  # defaults to basename
        self.assertEqual(doc.source_type, "markdown")
        self.assertEqual(doc.filename, "arch.md")
        self.assertGreater(doc.chunks.count(), 0)

    def test_ingest_pdf_file_preserves_page_metadata_on_chunks(self) -> None:
        pdf_bytes = _make_pdf([
            "Page 1: Fundamentals of Quantum Computing and Qubits in superposition.",
            "Page 2: Entangled states enable quantum teleportation and dense coding.",
            "Page 3: Limitations include environmental thermal noise and decoherence rates.",
        ])

        doc = ingest_file(
            file_bytes=pdf_bytes,
            filename="quantum_paper.pdf",
            title="Quantum Fundamentals",
            embedding_provider=self.embedding_provider,
            chunking_config=ChunkingConfig(chunk_size=120, chunk_overlap=10),
        )

        self.assertEqual(doc.status, Document.STATUS_READY)
        self.assertEqual(doc.source_type, "pdf")
        self.assertEqual(doc.metadata.get("page_count"), 3)

        chunks = list(doc.chunks.order_by("chunk_index"))
        self.assertGreaterEqual(len(chunks), 3)

        # Chunks must preserve page numbers
        chunk_pages = [c.metadata.get("page") for c in chunks]
        self.assertIn(1, chunk_pages)
        self.assertIn(2, chunk_pages)
        self.assertIn(3, chunk_pages)

        # Sequential indexing and valid offsets
        for idx, c in enumerate(chunks):
            self.assertEqual(c.chunk_index, idx)
            self.assertGreater(c.end_offset, c.start_offset)
            self.assertEqual(c.content, doc.content[c.start_offset:c.end_offset])

    def test_ingest_docx_file_preserves_structural_metadata_on_chunks(self) -> None:
        docx_bytes = _make_docx(
            elements=[
                ("heading", "Executive Strategy Overview", "Heading 1"),
                ("paragraph", "Strategic implementation plan for autonomous engineering.", None),
                ("list", "Phase 1: Foundation", "List Bullet"),
            ],
            tables=[
                [["Target", "Timeline"], ["M11", "Q4"]]
            ],
        )

        doc = ingest_file(
            file_bytes=docx_bytes,
            filename="plan.docx",
            embedding_provider=self.embedding_provider,
        )

        self.assertEqual(doc.status, Document.STATUS_READY)
        self.assertEqual(doc.source_type, "docx")

        chunks = list(doc.chunks.all())
        self.assertGreaterEqual(len(chunks), 3)

        block_types = [c.metadata.get("block_type") for c in chunks]
        self.assertIn("heading", block_types)
        self.assertIn("paragraph", block_types)
        self.assertIn("table", block_types)

    def test_ingest_failure_does_not_leave_document_ready(self) -> None:
        class FailingEmbeddingProvider:
            def embed_texts(self, texts):
                raise RuntimeError("Embedding cluster connection refused")

        pdf_bytes = _make_pdf(["Page with some valid text content."])
        with self.assertRaises(EmbeddingError):
            ingest_file(
                file_bytes=pdf_bytes,
                filename="fail.pdf",
                embedding_provider=FailingEmbeddingProvider(),
            )

        failed_doc = Document.objects.filter(source="fail.pdf").first()
        self.assertIsNotNone(failed_doc)
        self.assertEqual(failed_doc.status, Document.STATUS_ERROR)
        self.assertNotEqual(failed_doc.status, Document.STATUS_READY)
        self.assertEqual(failed_doc.chunks.count(), 0)

    def test_empty_file_ingest_fails_cleanly(self) -> None:
        with self.assertRaises(DocumentError):
            ingest_file(
                file_bytes=b"",
                filename="empty.txt",
                embedding_provider=self.embedding_provider,
            )

        self.assertEqual(Document.objects.filter(title="empty").count(), 0)

    def test_unsupported_file_format_fails_cleanly(self) -> None:
        with self.assertRaises(DocumentError) as ctx:
            ingest_file(
                file_bytes=b"raw binary data",
                filename="data.bin",
                embedding_provider=self.embedding_provider,
            )
        self.assertIn(".bin", str(ctx.exception))
