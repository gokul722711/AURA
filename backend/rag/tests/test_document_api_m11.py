"""Deterministic API tests for M11 multi-format file uploads and ingestion."""

import io
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from rest_framework import status
from rest_framework.test import APIClient

from rag.models import Document, DocumentChunk
from rag.tests.test_extraction import _make_docx, _make_pdf


class DocumentAPIM11Tests(TestCase):
    """Tests for POST /api/documents/ multipart file uploads and validations."""

    def setUp(self) -> None:
        self.client = APIClient()
        Document.objects.all().delete()

    def test_upload_txt_file(self) -> None:
        """POST /api/documents/ with .txt file creates document and chunks."""
        txt_content = b"Plain text knowledge file for RAG indexing."
        file_obj = SimpleUploadedFile("knowledge.txt", txt_content, content_type="text/plain")

        response = self.client.post(
            "/api/documents/",
            {"file": file_obj, "title": "Knowledge Text"},
            format="multipart",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        data = response.json()
        self.assertEqual(data["title"], "Knowledge Text")
        self.assertEqual(data["source_type"], "txt")
        self.assertEqual(data["filename"], "knowledge.txt")
        self.assertEqual(data["file_size"], len(txt_content))
        self.assertEqual(data["status"], "ready")
        self.assertGreater(data["chunk_count"], 0)

    def test_upload_markdown_file(self) -> None:
        """POST /api/documents/ with .md file creates document and chunks."""
        md_content = b"# Agent Architecture\n\n- State Machine\n- Tool Registry"
        file_obj = SimpleUploadedFile("architecture.md", md_content, content_type="text/markdown")

        response = self.client.post(
            "/api/documents/",
            {"file": file_obj},
            format="multipart",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        data = response.json()
        self.assertEqual(data["title"], "architecture")  # defaulted from filename
        self.assertEqual(data["source_type"], "markdown")
        self.assertEqual(data["filename"], "architecture.md")

    def test_upload_pdf_file_preserves_page_metadata(self) -> None:
        """POST /api/documents/ with .pdf file creates document and chunks with pages."""
        pdf_bytes = _make_pdf([
            "Page 1: Overview of Quantum Gates and Qubits.",
            "Page 2: Phase error rates and decoherence bounds.",
        ])
        file_obj = SimpleUploadedFile("quantum_gates.pdf", pdf_bytes, content_type="application/pdf")

        response = self.client.post(
            "/api/documents/",
            {"file": file_obj, "title": "Quantum Gates Paper"},
            format="multipart",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        data = response.json()
        self.assertEqual(data["title"], "Quantum Gates Paper")
        self.assertEqual(data["source_type"], "pdf")
        self.assertEqual(data["filename"], "quantum_gates.pdf")
        self.assertEqual(data["status"], "ready")

        # Verify chunks have page metadata in database
        doc = Document.objects.get(id=data["id"])
        chunk_pages = [c.metadata.get("page") for c in doc.chunks.all()]
        self.assertIn(1, chunk_pages)
        self.assertIn(2, chunk_pages)

    def test_upload_docx_file_preserves_structural_metadata(self) -> None:
        """POST /api/documents/ with .docx file creates document and chunks with block_types."""
        docx_bytes = _make_docx(
            elements=[
                ("heading", "System Architecture", "Heading 1"),
                ("paragraph", "Autonomous research runtime orchestrates multi-step execution.", None),
            ],
            tables=[
                [["Component", "Role"], ["Gateway", "Inference"]]
            ],
        )
        file_obj = SimpleUploadedFile("specs.docx", docx_bytes, content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

        response = self.client.post(
            "/api/documents/",
            {"file": file_obj, "title": "System Specs"},
            format="multipart",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        data = response.json()
        self.assertEqual(data["title"], "System Specs")
        self.assertEqual(data["source_type"], "docx")

        doc = Document.objects.get(id=data["id"])
        block_types = [c.metadata.get("block_type") for c in doc.chunks.all()]
        self.assertIn("heading", block_types)
        self.assertIn("paragraph", block_types)
        self.assertIn("table", block_types)

    def test_upload_unsupported_format_fails(self) -> None:
        """Uploading unsupported extensions (.png, .csv, .exe) returns 400."""
        file_obj = SimpleUploadedFile("image.png", b"\x89PNG\r\n\x1a\n", content_type="image/png")
        response = self.client.post(
            "/api/documents/",
            {"file": file_obj},
            format="multipart",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        data = response.json()
        self.assertIn("Unsupported file format", data["error"])

    def test_upload_empty_file_fails(self) -> None:
        """Uploading an empty 0-byte file returns 400."""
        file_obj = SimpleUploadedFile("empty.txt", b"", content_type="text/plain")
        response = self.client.post(
            "/api/documents/",
            {"file": file_obj},
            format="multipart",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        data = response.json()
        self.assertIn("empty", data["error"].lower())

    def test_upload_corrupt_file_fails(self) -> None:
        """Uploading corrupted PDF or DOCX bytes returns 400."""
        file_obj = SimpleUploadedFile("corrupt.pdf", b"%PDF corrupt junk", content_type="application/pdf")
        response = self.client.post(
            "/api/documents/",
            {"file": file_obj},
            format="multipart",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        data = response.json()
        self.assertIn("Failed to parse", data["error"])

    def test_upload_file_size_exceeding_limit_fails(self) -> None:
        """Uploading a file exceeding 20MB returns 400."""
        large_file = SimpleUploadedFile(
            "large.txt",
            b"x" * (21 * 1024 * 1024),
            content_type="text/plain",
        )
        response = self.client.post(
            "/api/documents/",
            {"file": large_file},
            format="multipart",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        data = response.json()
        self.assertIn("exceeds maximum limit", data["error"])

    def test_existing_json_ingestion_regression(self) -> None:
        """Existing JSON pasted-text ingestion continues working 100% identically."""
        response = self.client.post(
            "/api/documents/",
            {
                "title": "Pasted Note",
                "content": "AURA is an LLM-agnostic agentic AI platform.",
                "source": "notes/pasted.txt",
                "metadata": {"tag": "research"},
            },
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        data = response.json()
        self.assertEqual(data["title"], "Pasted Note")
        self.assertEqual(data["content"], "AURA is an LLM-agnostic agentic AI platform.")
        self.assertEqual(data["metadata"], {"tag": "research"})
        self.assertEqual(data["source_type"], "text")
        self.assertEqual(data["status"], "ready")

    def test_document_list_exposes_m11_metadata(self) -> None:
        """GET /api/documents/ exposes source_type, filename, and file_size."""
        txt_bytes = b"Sample knowledge content"
        file_obj = SimpleUploadedFile("sample.txt", txt_bytes, content_type="text/plain")
        self.client.post(
            "/api/documents/",
            {"file": file_obj, "title": "Sample Doc"},
            format="multipart",
        )

        response = self.client.get("/api/documents/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        docs = response.json()
        self.assertGreaterEqual(len(docs), 1)
        doc = docs[0]
        self.assertIn("source_type", doc)
        self.assertIn("filename", doc)
        self.assertIn("file_size", doc)
