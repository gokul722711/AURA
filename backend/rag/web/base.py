"""Base abstraction and data models for AURA web fetching (M12)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WebFetchResult:
    """The result of fetching a web page via HTTP.

    Attributes:
        requested_url: The initial URL requested by the user.
        final_url: The final URL after following validated redirects.
        status_code: HTTP response status code (e.g. 200).
        content_type: Normalized MIME content type (e.g. 'text/html').
        body: Raw response content bytes.
        encoding: Character encoding detected or reported by the server.
        headers: Case-insensitive or normalized response headers.
        metadata: Additional provenance metadata (e.g. redirect_count, elapsed_seconds).
    """

    requested_url: str
    final_url: str
    status_code: int
    content_type: str
    body: bytes
    encoding: str = "utf-8"
    headers: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Decode response body using reported or fallback encoding."""
        try:
            return self.body.decode(self.encoding or "utf-8", errors="replace")
        except Exception:
            return self.body.decode("utf-8", errors="replace")


class WebFetcher(ABC):
    """Abstract interface for fetching web pages securely."""

    @abstractmethod
    def fetch(self, url: str) -> WebFetchResult:
        """Fetch a web page securely, validating against SSRF and enforcing limits.

        Args:
            url: The public HTTP/HTTPS URL to fetch.

        Returns:
            WebFetchResult containing the response body, final URL, and metadata.

        Raises:
            URLSecurityError: If URL fails security checks.
            SSRFError: If URL or any redirect destination is internal/blocked.
            WebFetchError: If network error, timeout, or HTTP error occurs.
            ContentTooLargeError: If response body exceeds size limits.
            UnsupportedContentTypeError: If response is not HTML.
        """
        ...
