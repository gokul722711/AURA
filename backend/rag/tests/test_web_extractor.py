"""Unit tests for WebPageExtractor using local deterministic HTML fixtures."""

from django.test import TestCase

from rag.exceptions import ExtractionError
from rag.extraction.base import ExtractedDocument
from rag.extraction.registry import get_extractor
from rag.extraction.web import WebPageExtractor


class WebPageExtractorTests(TestCase):
    """Deterministic tests for HTML main content extraction, metadata, and SPA detection."""

    def setUp(self) -> None:
        self.extractor = WebPageExtractor()

    def test_registry_registration(self) -> None:
        """WebPageExtractor is registered for HTML extensions and MIME types."""
        ext_html = get_extractor("document.html")
        self.assertIsInstance(ext_html, WebPageExtractor)

        ext_htm = get_extractor("page.htm")
        self.assertIsInstance(ext_htm, WebPageExtractor)

        mime_html = get_extractor(content_type="text/html")
        self.assertIsInstance(mime_html, WebPageExtractor)

        mime_xhtml = get_extractor(content_type="application/xhtml+xml")
        self.assertIsInstance(mime_xhtml, WebPageExtractor)

    def test_extract_article_with_metadata_and_boilerplate_removal(self) -> None:
        """Main text, title, author, date, and canonical URL are extracted while boilerplate is discarded."""
        html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Autonomous Systems in Modern Computing - Tech Journal</title>
            <meta name="author" content="Dr. Elena Vance">
            <meta name="date" content="2026-02-18">
            <meta name="description" content="An in-depth analysis of autonomous agentic systems.">
            <link rel="canonical" href="https://techjournal.example/articles/autonomous-systems">
        </head>
        <body>
            <header>
                <nav>
                    <ul>
                        <li><a href="/">Home</a></li>
                        <li><a href="/topics">Topics</a></li>
                        <li><a href="/subscribe">Subscribe</a></li>
                    </ul>
                </nav>
                <div class="ad-banner">Sponsored: Buy More Servers Today!</div>
            </header>

            <main>
                <article>
                    <h1>Autonomous Systems in Modern Computing</h1>
                    <p class="byline">By Dr. Elena Vance &bull; February 18, 2026</p>

                    <p>
                        Autonomous agents represent a paradigm shift from simple command-response architectures
                        to goal-oriented, multi-step execution loops. These systems employ explicit tools,
                        grounded reasoning, and persistent memory.
                    </p>

                    <p>
                        In distributed engineering environments, stateful orchestration frameworks such as
                        directed acyclic graphs enable deterministic validation at each step of the pipeline.
                        Furthermore, empirical evidence shows significant reductions in hallucinations when
                        strict citation integrity is enforced.
                    </p>
                </article>
            </main>

            <aside class="sidebar">
                <h3>Related Articles</h3>
                <ul>
                    <li><a href="/rel1">Quantum Computing Basics</a></li>
                    <li><a href="/rel2">Database Vector Indexing</a></li>
                </ul>
                <div class="newsletter-signup">
                    Sign up for our newsletter!
                </div>
            </aside>

            <footer>
                <p>&copy; 2026 Tech Journal Publishing Group. All rights reserved.</p>
                <p><a href="/privacy">Privacy Policy</a> | <a href="/terms">Terms of Service</a></p>
            </footer>
        </body>
        </html>
        """
        doc = self.extractor.extract(
            html.encode("utf-8"),
            source="https://techjournal.example/articles/autonomous-systems?ref=rss",
        )

        self.assertIsInstance(doc, ExtractedDocument)
        self.assertEqual(doc.source_type, "web_page")

        # Title extraction
        self.assertTrue("Autonomous Systems in Modern Computing" in doc.title)

        # Main content preserved
        self.assertIn("Autonomous agents represent a paradigm shift", doc.text)
        self.assertIn("hallucinations when strict citation integrity is enforced", doc.text)

        # Boilerplate removed
        self.assertNotIn("Sponsored: Buy More Servers Today!", doc.text)
        self.assertNotIn("Privacy Policy", doc.text)
        self.assertNotIn("Terms of Service", doc.text)

        # Metadata captured
        meta = doc.metadata
        self.assertEqual(meta["source_type"], "web_page")
        self.assertEqual(meta["url"], "https://techjournal.example/articles/autonomous-systems?ref=rss")
        self.assertEqual(meta["canonical_url"], "https://techjournal.example/articles/autonomous-systems")
        self.assertEqual(meta["domain"], "techjournal.example")
        self.assertIn("Elena Vance", meta["author"])
        self.assertIn("retrieved_at", meta)

    def test_html_entity_decoding(self) -> None:
        """HTML entities such as &amp;, &quot;, &lt;, &gt; are cleanly decoded."""
        html = """
        <html>
        <head><title>Syntax &amp; Entities Guide</title></head>
        <body>
            <article>
                <h1>Syntax &amp; Entities Guide</h1>
                <p>
                    Writing expressions like &quot;x &lt; y &amp;&amp; y &gt; z&quot;
                    requires careful character escaping in raw HTML documents.
                </p>
            </article>
        </body>
        </html>
        """
        doc = self.extractor.extract(html.encode("utf-8"))
        self.assertIn('Syntax & Entities Guide', doc.title)
        self.assertIn('"x < y && y > z"', doc.text)

    def test_empty_html_raises_extraction_error(self) -> None:
        """Empty or whitespace-only HTML raises ExtractionError."""
        with self.assertRaises(ExtractionError):
            self.extractor.extract(b"")

        with self.assertRaises(ExtractionError):
            self.extractor.extract(b"   \n  \t  ")

    def test_javascript_rendered_spa_shell_raises_extraction_error(self) -> None:
        """Client-side JavaScript SPA shells with no readable text raise ExtractionError."""
        spa_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Single Page App</title>
            <script defer src="/static/js/bundle.js"></script>
        </head>
        <body>
            <noscript>You need to enable JavaScript to run this app.</noscript>
            <div id="root"></div>
        </body>
        </html>
        """
        with self.assertRaises(ExtractionError) as ctx:
            self.extractor.extract(spa_html.encode("utf-8"), source="https://app.example.com")
        self.assertIn("JavaScript", str(ctx.exception))

    def test_insufficient_extractable_content_raises_error(self) -> None:
        """Pages with only minimal tags and no substantive text raise ExtractionError."""
        tiny_html = "<html><head><title>Empty</title></head><body><div></div></body></html>"
        with self.assertRaises(ExtractionError):
            self.extractor.extract(tiny_html.encode("utf-8"))
