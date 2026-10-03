"""Unit tests for WebFetcher implementations (HTTPXWebFetcher & MockWebFetcher)."""

import socket
from unittest.mock import MagicMock, patch

import httpx
from django.test import TestCase

from rag.exceptions import (
    ContentTooLargeError,
    SSRFError,
    UnsupportedContentTypeError,
    WebFetchError,
)
from rag.web.fetcher import HTTPXWebFetcher
from rag.web.mock import MockWebFetcher


class WebFetcherTests(TestCase):
    """Deterministic tests for HTTP fetching, timeouts, limits, and redirect SSRF protections."""

    def test_mock_web_fetcher_basic(self) -> None:
        """MockWebFetcher returns pre-registered results or raises errors."""
        fetcher = MockWebFetcher()
        fetcher.register_html("https://example.com/article", "<html><body>Article content</body></html>")

        res = fetcher.fetch("https://example.com/article")
        self.assertEqual(res.status_code, 200)
        self.assertIn("Article content", res.text)
        self.assertEqual(res.content_type, "text/html")
        self.assertEqual(len(fetcher.calls), 1)

    def test_mock_web_fetcher_missing_raises_error(self) -> None:
        """MockWebFetcher raises WebFetchError for unregistered URLs when allow_synthetic is False."""
        fetcher = MockWebFetcher(allow_synthetic=False)
        with self.assertRaises(WebFetchError):
            fetcher.fetch("https://example.com/unknown")

    def test_successful_html_fetch(self) -> None:
        """HTTPXWebFetcher successfully fetches and returns HTML content."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"Content-Type": "text/html; charset=utf-8"},
                text="<!DOCTYPE html><html><head><title>Test</title></head><body>Hello World</body></html>",
            )

        fetcher = HTTPXWebFetcher(
            transport=httpx.MockTransport(handler),
            resolve_dns=False,
        )
        res = fetcher.fetch("https://example.com/test")

        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.content_type)
        self.assertIn("Hello World", res.text)
        self.assertEqual(res.requested_url, "https://example.com/test")
        self.assertEqual(res.final_url, "https://example.com/test")

    def test_xhtml_content_type_accepted(self) -> None:
        """application/xhtml+xml is accepted as a valid HTML content type."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"Content-Type": "application/xhtml+xml"},
                text="<html xmlns='http://www.w3.org/1999/xhtml'><body>XHTML</body></html>",
            )

        fetcher = HTTPXWebFetcher(
            transport=httpx.MockTransport(handler),
            resolve_dns=False,
        )
        res = fetcher.fetch("https://example.com/doc.xhtml")
        self.assertEqual(res.status_code, 200)
        self.assertIn("XHTML", res.text)

    def test_unsupported_content_type_rejected(self) -> None:
        """Non-HTML content types (e.g. PDF, binary, JSON) are rejected before loading."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"Content-Type": "application/pdf"},
                content=b"%PDF-1.4 ... binary data",
            )

        fetcher = HTTPXWebFetcher(
            transport=httpx.MockTransport(handler),
            resolve_dns=False,
        )
        with self.assertRaises(UnsupportedContentTypeError) as ctx:
            fetcher.fetch("https://example.com/file.pdf")
        self.assertIn("application/pdf", str(ctx.exception))

    def test_missing_content_type_header_rejected(self) -> None:
        """A response with no Content-Type header is rejected with UnsupportedContentTypeError."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=b"<!DOCTYPE html><html><body>Missing content type header</body></html>",
            )

        fetcher = HTTPXWebFetcher(
            transport=httpx.MockTransport(handler),
            resolve_dns=False,
        )
        with self.assertRaises(UnsupportedContentTypeError) as ctx:
            fetcher.fetch("https://example.com/no-content-type")
        self.assertIn("missing", str(ctx.exception).lower())

    def test_empty_content_type_header_rejected(self) -> None:
        """A response with an empty Content-Type header is rejected."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"Content-Type": ""},
                content=b"<!DOCTYPE html><html><body>Empty content type header</body></html>",
            )

        fetcher = HTTPXWebFetcher(
            transport=httpx.MockTransport(handler),
            resolve_dns=False,
        )
        with self.assertRaises(UnsupportedContentTypeError) as ctx:
            fetcher.fetch("https://example.com/empty-content-type")
        self.assertIn("empty", str(ctx.exception).lower())

    def test_fetch_path_without_trailing_slash(self) -> None:
        """Direct fetch of /docs preserves path without forcing a trailing slash."""
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/docs")
            return httpx.Response(
                200,
                headers={"Content-Type": "text/html"},
                text="<html><body>Docs without slash</body></html>",
            )

        fetcher = HTTPXWebFetcher(
            transport=httpx.MockTransport(handler),
            resolve_dns=False,
        )
        res = fetcher.fetch("https://example.com/docs")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.final_url, "https://example.com/docs")

    def test_fetch_path_with_trailing_slash(self) -> None:
        """Direct fetch of /docs/ preserves path without stripping trailing slash."""
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/docs/")
            return httpx.Response(
                200,
                headers={"Content-Type": "text/html"},
                text="<html><body>Docs with slash</body></html>",
            )

        fetcher = HTTPXWebFetcher(
            transport=httpx.MockTransport(handler),
            resolve_dns=False,
        )
        res = fetcher.fetch("https://example.com/docs/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.final_url, "https://example.com/docs/")

    def test_redirect_docs_to_docs_slash(self) -> None:
        """Redirect from /docs -> /docs/ works without entering an infinite redirect loop."""
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/docs":
                return httpx.Response(301, headers={"Location": "/docs/"})
            elif request.url.path == "/docs/":
                return httpx.Response(
                    200,
                    headers={"Content-Type": "text/html"},
                    text="<html><body>Redirected docs slash</body></html>",
                )
            raise AssertionError(f"Unexpected request path: {request.url.path}")

        fetcher = HTTPXWebFetcher(
            transport=httpx.MockTransport(handler),
            resolve_dns=False,
        )
        res = fetcher.fetch("https://example.com/docs")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.requested_url, "https://example.com/docs")
        self.assertEqual(res.final_url, "https://example.com/docs/")
        self.assertEqual(res.metadata["redirect_count"], 1)

    def test_redirect_docs_slash_to_docs(self) -> None:
        """Redirect from /docs/ -> /docs works without entering an infinite redirect loop."""
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/docs/":
                return httpx.Response(301, headers={"Location": "/docs"})
            elif request.url.path == "/docs":
                return httpx.Response(
                    200,
                    headers={"Content-Type": "text/html"},
                    text="<html><body>Redirected docs no slash</body></html>",
                )
            raise AssertionError(f"Unexpected request path: {request.url.path}")

        fetcher = HTTPXWebFetcher(
            transport=httpx.MockTransport(handler),
            resolve_dns=False,
        )
        res = fetcher.fetch("https://example.com/docs/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.requested_url, "https://example.com/docs/")
        self.assertEqual(res.final_url, "https://example.com/docs")
        self.assertEqual(res.metadata["redirect_count"], 1)

    def test_dns_rebinding_cannot_cause_internal_request(self) -> None:
        """A hostname whose DNS resolution changes from public to private cannot cause an internal request.

        Tests that the fetch path pins the validated public destination and connects
        exclusively to the validated IP, preventing TOCTOU SSRF exploitation.
        """
        dns_query_count = 0

        def dynamic_dns(host, port, *args, **kwargs):
            nonlocal dns_query_count
            dns_query_count += 1
            if dns_query_count == 1:
                # Initial validation lookup returns safe public IP
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]
            else:
                # Attacker switches DNS to internal private loopback IP
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))]

        mock_socket = MagicMock()
        mock_socket.fileno.return_value = 10
        mock_socket.recv.side_effect = [
            b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: 13\r\n\r\nHello, World!",
            b"",
        ]

        with patch("socket.getaddrinfo", side_effect=dynamic_dns):
            with patch("socket.create_connection", return_value=mock_socket) as mock_connect:
                with patch("select.select", return_value=([], [], [])):
                    fetcher = HTTPXWebFetcher(resolve_dns=True)
                    res = fetcher.fetch("http://rebind.attacker.com/article")

                    self.assertEqual(res.status_code, 200)
                    # Verify socket connection connected to the validated public address
                    destination_ip = mock_connect.call_args[0][0][0]
                    self.assertEqual(destination_ip, "93.184.216.34")
                    # Under NO circumstance could it connect to the private address
                    self.assertNotEqual(destination_ip, "127.0.0.1")

    def test_oversized_response_rejected(self) -> None:
        """Responses exceeding max_body_bytes raise ContentTooLargeError."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"Content-Type": "text/html"},
                content=b"A" * 5000,
            )

        fetcher = HTTPXWebFetcher(
            transport=httpx.MockTransport(handler),
            max_body_bytes=1000,
            resolve_dns=False,
        )
        with self.assertRaises(ContentTooLargeError) as ctx:
            fetcher.fetch("https://example.com/huge-page")
        self.assertIn("1000 bytes", str(ctx.exception))

    def test_http_status_errors(self) -> None:
        """HTTP 404 and 500 error responses raise WebFetchError."""
        for code in [404, 500, 503]:
            with self.subTest(status_code=code):
                def handler(request: httpx.Request, status_code=code) -> httpx.Response:
                    return httpx.Response(status_code, text="Error page")

                fetcher = HTTPXWebFetcher(
                    transport=httpx.MockTransport(handler),
                    resolve_dns=False,
                )
                with self.assertRaises(WebFetchError) as ctx:
                    fetcher.fetch("https://example.com/error")
                self.assertIn(str(code), str(ctx.exception))

    def test_timeout_handling(self) -> None:
        """Timeouts during connection or reading raise WebFetchError."""
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("Read timed out")

        fetcher = HTTPXWebFetcher(
            transport=httpx.MockTransport(handler),
            resolve_dns=False,
        )
        with self.assertRaises(WebFetchError) as ctx:
            fetcher.fetch("https://example.com/timeout")
        self.assertIn("timed out", str(ctx.exception).lower())

    def test_redirect_public_to_public_succeeds(self) -> None:
        """Legitimate redirects between public URLs are followed and tracked."""
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/old-url":
                return httpx.Response(
                    301,
                    headers={"Location": "https://example.com/new-url"},
                )
            return httpx.Response(
                200,
                headers={"Content-Type": "text/html"},
                text="<html><body>Redirected destination</body></html>",
            )

        fetcher = HTTPXWebFetcher(
            transport=httpx.MockTransport(handler),
            resolve_dns=False,
        )
        res = fetcher.fetch("https://example.com/old-url")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.requested_url, "https://example.com/old-url")
        self.assertEqual(res.final_url, "https://example.com/new-url")
        self.assertIn("Redirected destination", res.text)

    def test_redirect_to_localhost_blocked_by_ssrf(self) -> None:
        """A redirect to localhost or 127.0.0.1 is blocked by per-hop SSRF validation."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                302,
                headers={"Location": "http://127.0.0.1:8000/internal-admin"},
            )

        fetcher = HTTPXWebFetcher(
            transport=httpx.MockTransport(handler),
            resolve_dns=False,
        )
        with self.assertRaises(SSRFError) as ctx:
            fetcher.fetch("https://example.com/trap")
        self.assertIn("blocked", str(ctx.exception).lower())

    def test_redirect_to_private_network_blocked(self) -> None:
        """A redirect to RFC1918 private IP is blocked by per-hop SSRF validation."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                302,
                headers={"Location": "http://192.168.1.1/setup"},
            )

        fetcher = HTTPXWebFetcher(
            transport=httpx.MockTransport(handler),
            resolve_dns=False,
        )
        with self.assertRaises(SSRFError) as ctx:
            fetcher.fetch("https://example.com/trap2")
        self.assertIn("blocked", str(ctx.exception).lower())

    def test_redirect_limit_exceeded(self) -> None:
        """Redirect loops exceeding max_redirects raise WebFetchError."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                302,
                headers={"Location": "https://example.com/loop"},
            )

        fetcher = HTTPXWebFetcher(
            transport=httpx.MockTransport(handler),
            max_redirects=3,
            resolve_dns=False,
        )
        with self.assertRaises(WebFetchError) as ctx:
            fetcher.fetch("https://example.com/loop")
        self.assertIn("redirect limit", str(ctx.exception).lower())
