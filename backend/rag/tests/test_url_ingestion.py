"""Integration tests for URL ingestion pipeline into Knowledge Base."""

from django.test import TestCase

from rag.exceptions import DuplicateURLError, ExtractionError, WebFetchError
from rag.ingestion import ingest_url
from rag.models import Document, DocumentChunk
from rag.web.mock import MockWebFetcher


class URLIngestionTests(TestCase):
    """Deterministic tests for URL -> fetch -> extract -> chunk -> embed -> pgvector pipeline."""

    def setUp(self) -> None:
        Document.objects.all().delete()
        self.fetcher = MockWebFetcher()

    def test_ingest_url_end_to_end(self) -> None:
        """A public URL is fetched, extracted, chunked, and stored with web provenance."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Quantum Teleportation Breakthrough</title>
            <meta name="author" content="Alice Quantum"/>
            <meta name="date" content="2026-01-15"/>
            <link rel="canonical" href="https://science.example/quantum-teleportation"/>
        </head>
        <body>
            <article>
                <h1>Quantum Teleportation Breakthrough</h1>
                <p>
                    Physicists have achieved high-fidelity quantum teleportation across a 50-kilometer
                    fiber optic network with over 99 percent fidelity.
                </p>
                <p>
                    This breakthrough relies on deterministic entanglement swapping protocols and
                    low-noise superconducting nanowire single-photon detectors.
                </p>
            </article>
        </body>
        </html>
        """
        self.fetcher.register_html("https://science.example/quantum-teleportation", html)

        doc = ingest_url(
            url="https://science.example/quantum-teleportation",
            fetcher=self.fetcher,
        )

        self.assertIsNotNone(doc.id)
        self.assertEqual(doc.title, "Quantum Teleportation Breakthrough")
        self.assertEqual(doc.source_type, "web_page")
        self.assertEqual(doc.source, "https://science.example/quantum-teleportation")
        self.assertEqual(doc.url, "https://science.example/quantum-teleportation")
        self.assertEqual(doc.canonical_url, "https://science.example/quantum-teleportation")
        self.assertEqual(doc.domain, "science.example")
        self.assertEqual(doc.status, "ready")
        self.assertGreater(doc.chunk_count, 0)

        # Verify chunks exist in pgvector
        chunks = list(DocumentChunk.objects.filter(document=doc).order_by("chunk_index"))
        self.assertEqual(len(chunks), doc.chunk_count)

        # Verify provenance on chunks
        first_chunk = chunks[0]
        self.assertIn("teleportation", first_chunk.content.lower())
        self.assertEqual(first_chunk.metadata.get("url"), "https://science.example/quantum-teleportation")
        self.assertEqual(first_chunk.metadata.get("domain"), "science.example")
        self.assertEqual(first_chunk.metadata.get("source_type"), "web_page")

    def test_user_title_override(self) -> None:
        """User-specified title overrides the HTML page title."""
        html = "<html><head><title>Default Page Title</title></head><body><article><h1>Heading</h1><p>Body paragraph with enough meaningful text to satisfy extraction requirements.</p></article></body></html>"
        self.fetcher.register_html("https://example.com/custom-title", html)

        doc = ingest_url(
            url="https://example.com/custom-title",
            title="Custom Research Title",
            fetcher=self.fetcher,
        )

        self.assertEqual(doc.title, "Custom Research Title")
        self.assertEqual(doc.url, "https://example.com/custom-title")

    def test_duplicate_url_rejected(self) -> None:
        """Ingesting the same canonical URL twice raises DuplicateURLError."""
        html = "<html><head><title>Title</title></head><body><article><p>Article content for duplicate testing with sufficient text.</p></article></body></html>"
        self.fetcher.register_html("https://example.com/page", html)

        # First ingestion succeeds
        doc1 = ingest_url("https://example.com/page", fetcher=self.fetcher)
        self.assertIsNotNone(doc1.id)

        # Exact duplicate
        with self.assertRaises(DuplicateURLError):
            ingest_url("https://example.com/page", fetcher=self.fetcher)

        # Duplicate with fragment
        with self.assertRaises(DuplicateURLError):
            ingest_url("https://example.com/page#heading-2", fetcher=self.fetcher)

        # Duplicate with scheme/hostname casing
        with self.assertRaises(DuplicateURLError):
            ingest_url("HTTPS://EXAMPLE.COM/page", fetcher=self.fetcher)

    def test_failed_fetch_raises_and_records_status(self) -> None:
        """Fetch failures raise WebFetchError and do not leave orphaned ready documents."""
        initial_doc_count = Document.objects.count()

        with self.assertRaises(WebFetchError):
            ingest_url(
                url="https://example.com/fails",
                fetcher=MockWebFetcher(allow_synthetic=False),
            )

        # No document persisted as ready
        self.assertEqual(Document.objects.filter(status="ready").count(), initial_doc_count)

    def test_spa_extraction_failure(self) -> None:
        """Ingesting a JavaScript SPA shell raises ExtractionError."""
        spa_html = "<!DOCTYPE html><html><head><title>App</title></head><body><div id='root'></div><script src='/app.js'></script></body></html>"
        self.fetcher.register_html("https://spa.example.com", spa_html)

        with self.assertRaises(ExtractionError):
            ingest_url("https://spa.example.com", fetcher=self.fetcher)
