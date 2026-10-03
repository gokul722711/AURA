"""Unit tests for URL validation, canonicalization, and SSRF protection."""

import socket
from unittest.mock import MagicMock, patch

from django.test import TestCase

from rag.exceptions import SSRFError, URLSecurityError
from rag.web.security import (
    SSRFSafeSyncBackend,
    ValidatedURL,
    canonicalize_url,
    is_ip_blocked,
    resolve_and_validate_destination,
    validate_url_security,
)


class URLSecurityTests(TestCase):
    """Deterministic security tests for URL validation and SSRF defenses."""

    def test_canonicalize_url(self) -> None:
        """URL canonicalization normalizes schemes, hosts, ports, and removes fragments."""
        # Strips fragment
        self.assertEqual(
            canonicalize_url("https://example.com/article#section-1"),
            "https://example.com/article",
        )
        # Normalizes scheme and host case
        self.assertEqual(
            canonicalize_url("HTTPS://EXAMPLE.COM/Article"),
            "https://example.com/Article",
        )
        # Strips standard ports
        self.assertEqual(
            canonicalize_url("https://example.com:443/docs"),
            "https://example.com/docs",
        )
        self.assertEqual(
            canonicalize_url("http://example.com:80/"),
            "http://example.com/",
        )
        # Retains custom port
        self.assertEqual(
            canonicalize_url("https://example.com:8443/docs"),
            "https://example.com:8443/docs",
        )
        # Root path empty normalized to /
        self.assertEqual(
            canonicalize_url("https://example.com"),
            "https://example.com/",
        )

    def test_validate_url_preserves_path_and_trailing_slash(self) -> None:
        """validate_url_security preserves resource identity including trailing slashes.

        Separates URL security validation from URL canonicalization:
        - canonicalize_url: strips trailing slashes for duplicate detection.
        - validate_url_security: preserves trailing slashes to prevent redirect loops.
        """
        # Trailing slash preserved
        res_slash = validate_url_security("https://example.com/docs/", resolve_dns=False)
        self.assertEqual(res_slash, "https://example.com/docs/")
        self.assertTrue(res_slash.endswith("/docs/"))

        # No trailing slash preserved
        res_no_slash = validate_url_security("https://example.com/docs", resolve_dns=False)
        self.assertEqual(res_no_slash, "https://example.com/docs")
        self.assertTrue(res_no_slash.endswith("/docs"))

        # Subdirectory trailing slash preserved
        res_sub = validate_url_security("https://example.com/guide/v1/", resolve_dns=False)
        self.assertEqual(res_sub, "https://example.com/guide/v1/")

        # In contrast, canonicalize_url strips trailing slashes for duplicate detection
        self.assertEqual(canonicalize_url("https://example.com/docs/"), "https://example.com/docs")
        self.assertEqual(canonicalize_url("https://example.com/docs"), "https://example.com/docs")

    def test_is_ip_blocked_loopback_and_unspecified(self) -> None:
        """Loopback and unspecified IPv4/IPv6 addresses are blocked."""
        self.assertTrue(is_ip_blocked("127.0.0.1"))
        self.assertTrue(is_ip_blocked("127.0.0.2"))
        self.assertTrue(is_ip_blocked("127.255.255.255"))
        self.assertTrue(is_ip_blocked("0.0.0.0"))
        self.assertTrue(is_ip_blocked("::1"))
        self.assertTrue(is_ip_blocked("::"))

    def test_is_ip_blocked_rfc1918_private(self) -> None:
        """RFC1918 private IPv4 ranges are blocked."""
        self.assertTrue(is_ip_blocked("10.0.0.1"))
        self.assertTrue(is_ip_blocked("10.254.0.1"))
        self.assertTrue(is_ip_blocked("172.16.0.1"))
        self.assertTrue(is_ip_blocked("172.31.255.255"))
        self.assertTrue(is_ip_blocked("192.168.0.1"))
        self.assertTrue(is_ip_blocked("192.168.1.100"))

    def test_is_ip_blocked_cloud_metadata_link_local(self) -> None:
        """Link-local addresses (including AWS/GCP/Azure 169.254.169.254) are blocked."""
        self.assertTrue(is_ip_blocked("169.254.169.254"))
        self.assertTrue(is_ip_blocked("169.254.0.1"))
        self.assertTrue(is_ip_blocked("fe80::1"))

    def test_is_ip_blocked_multicast_and_reserved(self) -> None:
        """Multicast and reserved IPv4 addresses are blocked."""
        self.assertTrue(is_ip_blocked("224.0.0.1"))
        self.assertTrue(is_ip_blocked("240.0.0.1"))
        self.assertTrue(is_ip_blocked("255.255.255.255"))

    def test_is_ip_blocked_ipv6_unique_local(self) -> None:
        """IPv6 unique local addresses (fc00::/7) are blocked."""
        self.assertTrue(is_ip_blocked("fc00::1"))
        self.assertTrue(is_ip_blocked("fd12:3456:789a::1"))

    def test_is_ip_blocked_public_ips_allowed(self) -> None:
        """Legitimate public IP addresses are not blocked."""
        self.assertFalse(is_ip_blocked("93.184.216.34"))  # example.com
        self.assertFalse(is_ip_blocked("8.8.8.8"))
        self.assertFalse(is_ip_blocked("1.1.1.1"))
        self.assertFalse(is_ip_blocked("2606:4700:4700::1111"))

    def test_validate_allowed_schemes(self) -> None:
        """Only http and https schemes are permitted."""
        with patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))]):
            canon_http = validate_url_security("http://example.com/article")
            self.assertEqual(canon_http, "http://example.com/article")

            canon_https = validate_url_security("https://example.com/article")
            self.assertEqual(canon_https, "https://example.com/article")

    def test_validate_rejects_unsupported_schemes(self) -> None:
        """Dangerous or unsupported schemes are rejected."""
        unsupported = [
            "file:///etc/passwd",
            "ftp://ftp.example.com/file.txt",
            "javascript:alert(1)",
            "data:text/html,<h1>PWNED</h1>",
            "gopher://gopher.example.com/",
            "ws://example.com/socket",
            "ssh://git@github.com",
        ]
        for url in unsupported:
            with self.subTest(url=url):
                with self.assertRaises(URLSecurityError):
                    validate_url_security(url)

    def test_validate_rejects_embedded_credentials(self) -> None:
        """URLs with embedded credentials must be rejected without credential leakage."""
        with self.assertRaises(URLSecurityError) as ctx:
            validate_url_security("https://admin:secret123@example.com/dashboard")
        self.assertNotIn("secret123", str(ctx.exception))
        self.assertIn("credentials", str(ctx.exception).lower())

    def test_validate_rejects_internal_hostnames(self) -> None:
        """Internal hostnames and patterns are rejected by name."""
        internal_hosts = [
            "http://localhost/admin",
            "http://localhost:8000/api",
            "http://server.local/page",
            "http://database.internal/data",
            "http://gateway.lan/setup",
            "http://kubernetes.default.svc.cluster.local/pods",
        ]
        for url in internal_hosts:
            with self.subTest(url=url):
                with self.assertRaises(SSRFError):
                    validate_url_security(url, resolve_dns=False)

    def test_validate_rejects_direct_private_ips(self) -> None:
        """Direct IP hostnames pointing to private ranges are rejected."""
        private_urls = [
            "http://127.0.0.1/status",
            "http://127.0.0.5:8080/info",
            "http://10.0.1.5/keys",
            "http://192.168.1.1/router",
            "http://172.16.5.10/admin",
            "http://169.254.169.254/latest/meta-data/",
            "http://[::1]/secret",
        ]
        for url in private_urls:
            with self.subTest(url=url):
                with self.assertRaises(SSRFError):
                    validate_url_security(url, resolve_dns=False)

    def test_validate_dns_resolution_blocks_ssrf(self) -> None:
        """DNS resolution resolving to a private/loopback IP is blocked."""
        with patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]):
            with self.assertRaises(SSRFError):
                validate_url_security("http://evil-rebind.example.com/leak", resolve_dns=True)

    def test_validate_dns_resolution_failure(self) -> None:
        """Unresolvable hostnames raise SSRFError cleanly."""
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("Name or service not known")):
            with self.assertRaises(SSRFError) as ctx:
                validate_url_security("https://nonexistent-domain-xyz1234567.org/page", resolve_dns=True)
            self.assertIn("resolve", str(ctx.exception).lower())

    def test_validate_malformed_urls(self) -> None:
        """Malformed URLs raise URLSecurityError."""
        malformed = [
            "",
            "   ",
            "http://",
            "https://",
            "://missing-scheme.com",
            "http://[invalid-ipv6",
        ]
        for url in malformed:
            with self.subTest(url=url):
                with self.assertRaises(URLSecurityError):
                    validate_url_security(url)

    def test_validate_pins_verified_ip(self) -> None:
        """validate_url_security populates pinned_hosts and returns ValidatedURL with resolved_ip."""
        pinned_map: dict[str, str] = {}
        with patch(
            "socket.getaddrinfo",
            return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))],
        ):
            res = validate_url_security(
                "https://example.com/docs/",
                resolve_dns=True,
                pinned_hosts=pinned_map,
            )
            self.assertIsInstance(res, ValidatedURL)
            self.assertEqual(res.resolved_ip, "93.184.216.34")
            self.assertEqual(pinned_map.get("example.com"), "93.184.216.34")

    def test_ssrf_safe_backend_blocks_private_resolution(self) -> None:
        """SSRFSafeSyncBackend blocks connections when hostname resolves to private IP."""
        backend = SSRFSafeSyncBackend(resolve_dns=True)
        with patch(
            "socket.getaddrinfo",
            return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))],
        ):
            with self.assertRaises(SSRFError):
                backend.connect_tcp("rebind.example.com", 80)

    def test_ssrf_safe_backend_connects_to_pinned_destination(self) -> None:
        """SSRFSafeSyncBackend connects directly to the pinned IP without subsequent DNS queries."""
        backend = SSRFSafeSyncBackend(
            pinned_hosts={"pinned.example.com": "93.184.216.34"},
            resolve_dns=True,
        )
        mock_sock = MagicMock()
        with patch("socket.create_connection", return_value=mock_sock) as mock_connect:
            with patch("socket.getaddrinfo") as mock_dns:
                backend.connect_tcp("pinned.example.com", 80)
                mock_dns.assert_not_called()
                mock_connect.assert_called_once_with(
                    ("93.184.216.34", 80),
                    None,
                    source_address=None,
                )
