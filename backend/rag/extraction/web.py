"""Web page content extractor using Trafilatura for AURA (M12).

Extracts meaningful main-body text and structured metadata (title, author,
canonical URL, publication date, description) while removing boilerplate,
navigation menus, scripts, advertisements, and page chrome.
"""

import logging
import re
from typing import Any

import trafilatura

from rag.exceptions import ExtractionError
from rag.extraction.base import DocumentExtractor, ExtractedBlock, ExtractedDocument

logger = logging.getLogger(__name__)


class WebPageExtractor(DocumentExtractor):
    """HTML web-page extractor isolating Trafilatura library dependencies."""

    @property
    def supported_source_type(self) -> str:
        """Return the canonical source type identifier."""
        return "web_page"

    def extract(self, content: bytes, filename: str = "", source: str = "") -> ExtractedDocument:
        """Extract main text content and metadata from HTML bytes.

        Args:
            content: Raw HTML response bytes.
            filename: Source URL or filename (for provenance and fallback).
            source: Optional alias for source URL.

        Returns:
            ExtractedDocument containing normalized blocks and page metadata.

        Raises:
            ExtractionError: If content is empty or contains no extractable text
                (e.g. client-side JavaScript rendered pages).
        """
        if not content or not content.strip():
            raise ExtractionError("Web page response is empty.")

        # Decode content safely
        try:
            html_text = content.decode("utf-8", errors="replace")
        except Exception as exc:
            raise ExtractionError(f"Failed to decode HTML content: {exc}") from exc

        # Extract structured metadata via Trafilatura
        meta = None
        try:
            meta = trafilatura.extract_metadata(html_text)
        except Exception as exc:
            logger.debug("Trafilatura metadata extraction notice: %s", exc)

        # Extract main body text (favor precision, include tables, omit comments/images)
        extracted_text = None
        try:
            extracted_text = trafilatura.extract(
                html_text,
                output_format="txt",
                include_comments=False,
                include_tables=True,
                include_images=False,
                include_links=False,
                favor_precision=True,
            )
        except Exception as exc:
            raise ExtractionError(f"Trafilatura HTML extraction failed: {exc}") from exc

        if not extracted_text or not extracted_text.strip():
            raise ExtractionError(
                "No extractable content found in web page. "
                "The page may be empty or require client-side JavaScript rendering."
            )

        clean_text = extracted_text.strip()

        # Build paragraph/section blocks
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", clean_text) if p.strip()]
        if not paragraphs:
            paragraphs = [clean_text]

        block_metadata = {
            "source_type": "web_page",
        }

        blocks = [
            ExtractedBlock(
                content=para,
                metadata=dict(block_metadata),
            )
            for para in paragraphs
        ]

        from urllib.parse import urlsplit
        from django.utils import timezone

        url_or_name = source or filename or ""
        parsed_domain = ""
        if url_or_name.startswith("http://") or url_or_name.startswith("https://"):
            parsed_domain = urlsplit(url_or_name).netloc
        domain = parsed_domain or (meta.hostname.strip() if (meta and meta.hostname) else "")

        # Assemble metadata
        doc_metadata: dict[str, Any] = {
            "source_type": "web_page",
            "url": url_or_name,
            "domain": domain,
            "title": (meta.title.strip() if (meta and meta.title) else "") or "",
            "author": (meta.author.strip() if (meta and meta.author) else "") or "",
            "date": (meta.date.strip() if (meta and meta.date) else "") or "",
            "description": (meta.description.strip() if (meta and meta.description) else "") or "",
            "canonical_url": (meta.url.strip() if (meta and meta.url) else "") or "",
            "site_name": (meta.sitename.strip() if (meta and meta.sitename) else "") or "",
            "hostname": domain,
            "retrieved_at": timezone.now().isoformat(),
        }

        return ExtractedDocument(
            text=clean_text,
            blocks=blocks,
            source_type="web_page",
            metadata=doc_metadata,
        )
