"""HTTPX-based concrete WebFetcher implementation for AURA (M12).

Provides bounded, SSRF-protected, streaming HTTP fetching isolated behind the
WebFetcher interface. Pins validated destinations to eliminate DNS TOCTOU
rebinding vulnerabilities, preserves resource identity on trailing-slash redirects,
and strictly validates Content-Type headers.
"""

import logging
from typing import Any
from urllib.parse import urljoin, urlsplit

from django.conf import settings
import httpx

from agent.security import sanitize_text
from rag.exceptions import (
    ContentTooLargeError,
    SSRFError,
    UnsupportedContentTypeError,
    URLSecurityError,
    WebFetchError,
)
from rag.web.base import WebFetcher, WebFetchResult
from rag.web.security import SSRFSafeTransport, validate_url_security

logger = logging.getLogger(__name__)

DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_READ_TIMEOUT = 10.0
DEFAULT_MAX_REDIRECTS = 5
DEFAULT_MAX_BODY_BYTES = 5 * 1024 * 1024  # 5 MB
DEFAULT_USER_AGENT = "AURA-KnowledgeIngest/1.0 (+https://aura.local)"

ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset({
    "text/html",
    "application/xhtml+xml",
})


def _get_fetcher_config() -> dict[str, Any]:
    """Retrieve WebFetcher configuration from Django settings."""
    rag_settings = getattr(settings, "AI_RAG", {})
    return rag_settings.get("WEB_FETCHER", {})


class HTTPXWebFetcher(WebFetcher):
    """Secure HTTP web fetcher implementing SSRF protection and streaming bounds."""

    def __init__(
        self,
        connect_timeout: float | None = None,
        read_timeout: float | None = None,
        max_redirects: int | None = None,
        max_body_bytes: int | None = None,
        user_agent: str | None = None,
        resolve_dns: bool = True,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        cfg = _get_fetcher_config()
        self.connect_timeout = (
            connect_timeout
            if connect_timeout is not None
            else cfg.get("CONNECT_TIMEOUT", DEFAULT_CONNECT_TIMEOUT)
        )
        self.read_timeout = (
            read_timeout
            if read_timeout is not None
            else cfg.get("READ_TIMEOUT", DEFAULT_READ_TIMEOUT)
        )
        self.max_redirects = (
            max_redirects
            if max_redirects is not None
            else cfg.get("MAX_REDIRECTS", DEFAULT_MAX_REDIRECTS)
        )
        self.max_body_bytes = (
            max_body_bytes
            if max_body_bytes is not None
            else cfg.get("MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES)
        )
        self.user_agent = (
            user_agent
            if user_agent is not None
            else cfg.get("USER_AGENT", DEFAULT_USER_AGENT)
        )
        self.resolve_dns = resolve_dns
        self.transport = transport

    def fetch(self, url: str) -> WebFetchResult:
        """Fetch a public web page securely.

        Validates against SSRF on initial request and every subsequent redirect.
        Pins the validated destination IP to the HTTP connection to eliminate
        DNS TOCTOU rebinding vulnerabilities.
        """
        initial_url = url
        pinned_hosts: dict[str, str] = {}
        current_url = validate_url_security(
            url,
            resolve_dns=self.resolve_dns,
            pinned_hosts=pinned_hosts,
        )
        redirect_count = 0

        timeout = httpx.Timeout(
            connect=self.connect_timeout,
            read=self.read_timeout,
            write=self.connect_timeout,
            pool=self.connect_timeout,
        )

        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
            "Accept-Encoding": "gzip, deflate, br",
        }

        # Use caller-provided transport (e.g. MockTransport in tests) or SSRFSafeTransport
        transport = self.transport
        if transport is None:
            transport = SSRFSafeTransport(
                pinned_hosts=pinned_hosts,
                resolve_dns=self.resolve_dns,
                verify=True,
            )

        with httpx.Client(
            transport=transport,
            timeout=timeout,
            verify=True,
            follow_redirects=False,
        ) as client:
            while True:
                try:
                    request = client.build_request("GET", current_url, headers=headers)
                    response = client.send(request, stream=True)
                except SSRFError:
                    raise
                except httpx.TimeoutException as exc:
                    raise WebFetchError(f"Connection timed out while fetching {sanitize_text(current_url)}.") from exc
                except (httpx.ConnectError, httpx.NetworkError) as exc:
                    raise WebFetchError(f"Network error while connecting to server: {sanitize_text(str(exc))}") from exc
                except httpx.HTTPError as exc:
                    raise WebFetchError(f"HTTP request failed: {sanitize_text(str(exc))}") from exc

                # Check for redirects
                if response.is_redirect:
                    response.close()
                    redirect_count += 1
                    if redirect_count > self.max_redirects:
                        raise WebFetchError(
                            f"Exceeded maximum redirect limit of {self.max_redirects}."
                        )

                    location = response.headers.get("location")
                    if not location:
                        raise WebFetchError("Redirect response missing Location header.")

                    # Resolve relative redirect URLs against the current request URL
                    next_url = urljoin(current_url, location)

                    # Validate redirect target against SSRF and pin destination
                    current_url = validate_url_security(
                        next_url,
                        resolve_dns=self.resolve_dns,
                        pinned_hosts=pinned_hosts,
                    )
                    redirect_host = urlsplit(current_url).hostname
                    if (
                        redirect_host
                        and hasattr(transport, "pin_host")
                        and hasattr(current_url, "resolved_ip")
                        and current_url.resolved_ip
                    ):
                        transport.pin_host(redirect_host, current_url.resolved_ip)
                    continue

                try:
                    if response.status_code >= 400:
                        response.close()
                        raise WebFetchError(
                            f"Server returned HTTP status {response.status_code}."
                        )

                    # Validate Content-Type: reject missing, empty, or unsupported MIME types
                    raw_content_type = response.headers.get("content-type", "")
                    clean_content_type = raw_content_type.split(";")[0].strip().lower()

                    if not clean_content_type or clean_content_type not in ALLOWED_CONTENT_TYPES:
                        response.close()
                        type_desc = f"'{clean_content_type}'" if clean_content_type else "missing or empty"
                        raise UnsupportedContentTypeError(
                            f"Unsupported content type {type_desc}. "
                            "Only HTML web pages (text/html, application/xhtml+xml) are supported for URL ingestion."
                        )

                    # Check Content-Length if reported
                    content_length = response.headers.get("content-length")
                    if content_length and content_length.isdigit():
                        if int(content_length) > self.max_body_bytes:
                            response.close()
                            raise ContentTooLargeError(
                                f"Response size ({content_length} bytes) exceeds limit of "
                                f"{self.max_body_bytes} bytes."
                            )

                    # Stream response body with strict byte accumulation bounds
                    chunks: list[bytes] = []
                    total_bytes = 0
                    for chunk in response.iter_bytes(chunk_size=8192):
                        total_bytes += len(chunk)
                        if total_bytes > self.max_body_bytes:
                            response.close()
                            raise ContentTooLargeError(
                                f"Response body exceeded maximum limit of {self.max_body_bytes} bytes."
                            )
                        chunks.append(chunk)

                    body = b"".join(chunks)
                    encoding = response.encoding or "utf-8"

                    return WebFetchResult(
                        requested_url=initial_url,
                        final_url=current_url,
                        status_code=response.status_code,
                        content_type=clean_content_type,
                        body=body,
                        encoding=encoding,
                        headers=dict(response.headers),
                        metadata={
                            "redirect_count": redirect_count,
                            "final_url": current_url,
                        },
                    )
                finally:
                    response.close()
