"""Deterministic MockWebFetcher for offline testing of AURA web ingestion (M12)."""

from typing import Any
from urllib.parse import urlsplit

from rag.exceptions import (
    ContentTooLargeError,
    UnsupportedContentTypeError,
    WebFetchError,
)
from rag.web.base import WebFetcher, WebFetchResult
from rag.web.security import validate_url_security


class MockWebFetcher(WebFetcher):
    """Deterministic, offline-capable WebFetcher for unit and integration testing."""

    def __init__(
        self,
        canned_responses: dict[str, str | bytes | WebFetchResult | Exception] | None = None,
        default_html: str | None = None,
        validate_security: bool = True,
        max_redirects: int = 5,
        max_body_bytes: int = 5 * 1024 * 1024,
        allow_synthetic: bool = True,
    ) -> None:
        self.canned_responses = canned_responses or {}
        self.default_html = default_html
        self.validate_security = validate_security
        self.max_redirects = max_redirects
        self.max_body_bytes = max_body_bytes
        self.allow_synthetic = allow_synthetic
        self.fetch_history: list[str] = []

    @property
    def calls(self) -> list[str]:
        """List of all fetched URLs in order of invocation."""
        return list(self.fetch_history)

    def register_response(
        self,
        url: str,
        response: str | bytes | WebFetchResult | Exception,
    ) -> None:
        """Register a canned response for a specific URL."""
        self.canned_responses[url] = response

    def register_html(self, url: str, html: str) -> None:
        """Convenience method to register an HTML string for a URL."""
        self.register_response(url, html)

    def fetch(self, url: str) -> WebFetchResult:
        """Fetch using canned responses or deterministic synthetic HTML."""
        current_url = url
        redirect_count = 0

        while True:
            self.fetch_history.append(current_url)

            if self.validate_security:
                # Validate current URL against SSRF
                current_url = validate_url_security(current_url, resolve_dns=False)

            canned = (
                self.canned_responses.get(current_url)
                or self.canned_responses.get(current_url.rstrip("/"))
                or self.canned_responses.get(url)
                or self.canned_responses.get(url.rstrip("/"))
            )

            if isinstance(canned, Exception):
                raise canned

            if isinstance(canned, WebFetchResult):
                if canned.status_code in (301, 302, 303, 307, 308) and "location" in canned.headers:
                    redirect_count += 1
                    if redirect_count > self.max_redirects:
                        raise WebFetchError(f"Exceeded maximum redirect limit of {self.max_redirects}.")
                    current_url = canned.headers["location"]
                    continue
                return canned

            if isinstance(canned, (str, bytes)):
                body_bytes = canned.encode("utf-8") if isinstance(canned, str) else canned
                if len(body_bytes) > self.max_body_bytes:
                    raise ContentTooLargeError(
                        f"Response body ({len(body_bytes)} bytes) exceeds limit of {self.max_body_bytes} bytes."
                    )
                return WebFetchResult(
                    requested_url=url,
                    final_url=current_url,
                    status_code=200,
                    content_type="text/html",
                    body=body_bytes,
                    encoding="utf-8",
                    headers={"content-type": "text/html; charset=utf-8"},
                    metadata={"redirect_count": redirect_count},
                )

            # Fallback default HTML if provided
            if self.default_html is not None:
                body_bytes = self.default_html.encode("utf-8")
                return WebFetchResult(
                    requested_url=url,
                    final_url=current_url,
                    status_code=200,
                    content_type="text/html",
                    body=body_bytes,
                    encoding="utf-8",
                    headers={"content-type": "text/html; charset=utf-8"},
                    metadata={"redirect_count": redirect_count},
                )

            # Default synthetic article
            if not self.allow_synthetic:
                raise WebFetchError(f"MockWebFetcher: No registered response for URL '{url}'.")

            parsed = urlsplit(current_url)
            domain = parsed.netloc or "example.com"
            title = f"Article on {domain}"
            synthetic_html = (
                f"<!DOCTYPE html><html><head><title>{title}</title>"
                f"<meta name='description' content='Synthetic test page for {current_url}'/>"
                f"<link rel='canonical' href='{current_url}'/></head>"
                f"<body><header><nav><a href='/'>Home</a></nav></header>"
                f"<main><article><h1>{title}</h1>"
                f"<p>This is deterministic mock content retrieved for testing URL ingestion.</p>"
                f"<p>AURA Knowledge Base supports document-grounded research from public web pages.</p>"
                f"</article></main>"
                f"<footer><p>© 2026 {domain}. All rights reserved.</p></footer></body></html>"
            )
            return WebFetchResult(
                requested_url=url,
                final_url=current_url,
                status_code=200,
                content_type="text/html",
                body=synthetic_html.encode("utf-8"),
                encoding="utf-8",
                headers={"content-type": "text/html; charset=utf-8"},
                metadata={"redirect_count": redirect_count},
            )
