"""Deterministic API tests for M12 URL web-page ingestion via POST /api/documents/."""

from unittest.mock import patch
from django.test import TestCase
from rest_framework import status
from rest_framework.test import APIClient

from rag.models import Document, DocumentChunk
from rag.web.mock import MockWebFetcher


class DocumentAPIM12Tests(TestCase):
    """Tests for POST /api/documents/ URL ingestion, SSRF validation, duplicate handling, and serialization."""

    def setUp(self) -> None:
        self.client = APIClient()
        Document.objects.all().delete()
        self.mock_fetcher = MockWebFetcher()

    @patch("rag.ingestion.HTTPXWebFetcher")
    def test_post_url_successful_ingestion(self, mock_fetcher_cls) -> None:
        """POST /api/documents/ with valid URL ingests page and returns 201 Created with metadata."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Superconducting Qubits Research</title>
            <meta name="author" content="Dr. Quantum"/>
            <link rel="canonical" href="https://example.com/superconducting-qubits"/>
        </head>
        <body>
            <article>
                <h1>Superconducting Qubits Research</h1>
                <p>
                    Superconducting transmon circuits demonstrate coherence times exceeding 100 microseconds.
                    These devices operate at dilution refrigerator temperatures near 15 millikelvin.
                </p>
            </article>
        </body>
        </html>
        """
        self.mock_fetcher.register_html("https://example.com/superconducting-qubits", html)
        mock_fetcher_cls.return_value = self.mock_fetcher

        response = self.client.post(
            "/api/documents/",
            {"url": "https://example.com/superconducting-qubits"},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        data = response.json()
        self.assertIn("id", data)
        self.assertEqual(data["title"], "Superconducting Qubits Research")
        self.assertEqual(data["source_type"], "web_page")
        self.assertEqual(data["url"], "https://example.com/superconducting-qubits")
        self.assertEqual(data["domain"], "example.com")
        self.assertEqual(data["canonical_url"], "https://example.com/superconducting-qubits")
        self.assertEqual(data["status"], "ready")
        self.assertGreater(data["chunk_count"], 0)

        # Verify Document exists in DB
        doc = Document.objects.get(id=data["id"])
        self.assertEqual(doc.source_type, "web_page")
        self.assertEqual(doc.url, "https://example.com/superconducting-qubits")

    @patch("rag.ingestion.HTTPXWebFetcher")
    def test_post_url_custom_title_override(self, mock_fetcher_cls) -> None:
        """POST /api/documents/ with custom title overrides the page title."""
        html = "<html><head><title>Default Title</title></head><body><article><p>Some valid text content for ingestion.</p></article></body></html>"
        self.mock_fetcher.register_html("https://example.com/article", html)
        mock_fetcher_cls.return_value = self.mock_fetcher

        response = self.client.post(
            "/api/documents/",
            {
                "url": "https://example.com/article",
                "title": "Custom Overridden Title",
            },
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        data = response.json()
        self.assertEqual(data["title"], "Custom Overridden Title")

    def test_post_missing_url(self) -> None:
        """POST /api/documents/ with empty URL raises HTTP 400."""
        response = self.client.post(
            "/api/documents/",
            {"url": ""},
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        data = response.json()
        self.assertIn("error", data)

    def test_post_unsupported_scheme(self) -> None:
        """POST /api/documents/ with file:// or ftp:// raises HTTP 400."""
        unsupported_schemes = [
            "file:///etc/passwd",
            "ftp://files.example.com/data.txt",
            "javascript:alert(1)",
            "data:text/html,test",
        ]
        for bad_url in unsupported_schemes:
            with self.subTest(url=bad_url):
                response = self.client.post(
                    "/api/documents/",
                    {"url": bad_url},
                    format="json",
                )
                self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
                data = response.json()
                self.assertIn("error", data)
                self.assertNotIn("Traceback", data["error"])

    def test_post_embedded_credentials_blocked(self) -> None:
        """POST /api/documents/ with credentials in URL raises HTTP 400 without leaking credentials."""
        response = self.client.post(
            "/api/documents/",
            {"url": "https://user:supersecretpass@example.com/admin"},
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        data = response.json()
        self.assertIn("error", data)
        self.assertNotIn("supersecretpass", data["error"])

    def test_post_ssrf_internal_destinations_blocked(self) -> None:
        """POST /api/documents/ targeting localhost or private IPs raises HTTP 400."""
        blocked_targets = [
            "http://localhost:8000/internal",
            "http://127.0.0.1/status",
            "http://10.0.0.1/secret",
            "http://192.168.1.1/router",
            "http://169.254.169.254/latest/meta-data/",
            "http://server.local/info",
        ]
        for target in blocked_targets:
            with self.subTest(url=target):
                response = self.client.post(
                    "/api/documents/",
                    {"url": target},
                    format="json",
                )
                self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
                data = response.json()
                self.assertIn("error", data)

    @patch("rag.ingestion.HTTPXWebFetcher")
    def test_post_duplicate_url_returns_409_conflict(self, mock_fetcher_cls) -> None:
        """POST /api/documents/ with an already indexed URL returns HTTP 409 Conflict."""
        html = "<html><head><title>Title</title></head><body><article><p>Article content text.</p></article></body></html>"
        self.mock_fetcher.register_html("https://example.com/dup", html)
        mock_fetcher_cls.return_value = self.mock_fetcher

        # First ingestion
        res1 = self.client.post(
            "/api/documents/",
            {"url": "https://example.com/dup"},
            format="json",
        )
        self.assertEqual(res1.status_code, status.HTTP_201_CREATED)

        # Duplicate ingestion
        res2 = self.client.post(
            "/api/documents/",
            {"url": "https://example.com/dup"},
            format="json",
        )
        self.assertEqual(res2.status_code, status.HTTP_409_CONFLICT)
        data = res2.json()
        self.assertIn("error", data)
        self.assertIn("already been indexed", data["error"])

    @patch("rag.ingestion.HTTPXWebFetcher")
    def test_get_document_list_and_detail_includes_web_metadata(self, mock_fetcher_cls) -> None:
        """GET /api/documents/ and GET /api/documents/{id}/ serialize web fields."""
        html = "<html><head><title>Web Article</title></head><body><article><p>Web article body text.</p></article></body></html>"
        self.mock_fetcher.register_html("https://docs.example.org/guide", html)
        mock_fetcher_cls.return_value = self.mock_fetcher

        # Ingest document
        create_res = self.client.post(
            "/api/documents/",
            {"url": "https://docs.example.org/guide"},
            format="json",
        )
        self.assertEqual(create_res.status_code, status.HTTP_201_CREATED)
        doc_id = create_res.json()["id"]

        # List documents
        list_res = self.client.get("/api/documents/")
        self.assertEqual(list_res.status_code, status.HTTP_200_OK)
        docs = list_res.json()
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["url"], "https://docs.example.org/guide")
        self.assertEqual(docs[0]["domain"], "docs.example.org")
        self.assertEqual(docs[0]["source_type"], "web_page")

        # Get document detail
        detail_res = self.client.get(f"/api/documents/{doc_id}/")
        self.assertEqual(detail_res.status_code, status.HTTP_200_OK)
        detail = detail_res.json()
        self.assertEqual(detail["id"], doc_id)
        self.assertEqual(detail["url"], "https://docs.example.org/guide")
        self.assertEqual(detail["domain"], "docs.example.org")
        self.assertEqual(detail["source_type"], "web_page")

    @patch("rag.ingestion.HTTPXWebFetcher")
    def test_delete_web_document(self, mock_fetcher_cls) -> None:
        """DELETE /api/documents/{id}/ deletes the web document and its chunks."""
        html = "<html><head><title>Deletable</title></head><body><article><p>Deletable content.</p></article></body></html>"
        self.mock_fetcher.register_html("https://example.com/delete-me", html)
        mock_fetcher_cls.return_value = self.mock_fetcher

        create_res = self.client.post(
            "/api/documents/",
            {"url": "https://example.com/delete-me"},
            format="json",
        )
        doc_id = create_res.json()["id"]

        del_res = self.client.delete(f"/api/documents/{doc_id}/")
        self.assertEqual(del_res.status_code, status.HTTP_204_NO_CONTENT)

        self.assertFalse(Document.objects.filter(id=doc_id).exists())
        self.assertFalse(DocumentChunk.objects.filter(document_id=doc_id).exists())
