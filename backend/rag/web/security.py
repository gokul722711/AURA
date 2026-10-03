"""URL validation and SSRF protection for AURA web page ingestion (M12).

Provides strict validation against internal/private destinations, DNS rebinding,
embedded credentials, unsupported schemes, and malformed URLs.
Ensures validated destinations are pinned to the HTTP connection to eliminate
DNS TOCTOU vulnerabilities while preserving HTTPS certificate validation,
HTTP Host semantics, and redirect security.
"""

import ipaddress
import logging
import socket
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpcore
from httpcore import SyncBackend
from httpcore._backends.sync import SyncStream
from httpcore._exceptions import ConnectError, ConnectTimeout, map_exceptions
import httpx

from rag.exceptions import SSRFError, URLSecurityError

logger = logging.getLogger(__name__)

ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})

FORBIDDEN_HOSTNAME_SUFFIXES: tuple[str, ...] = (
    ".local",
    ".internal",
    ".lan",
    ".corp",
    ".home",
    ".test",
    ".arpa",
)

BLOCKED_HOSTNAMES: frozenset[str] = frozenset({
    "localhost",
    "localhost.localdomain",
    "broadcasthost",
    "ip6-localhost",
    "ip6-loopback",
})


class ValidatedURL(str):
    """String subclass representing a security-validated URL.

    Carries the resolved and verified IP address to guarantee that the destination
    used for the actual HTTP connection is identical to the one validated by the
    SSRF security layer.
    """

    resolved_ip: str | None = None


def is_ip_blocked(ip: str | ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Check if an IP address belongs to loopback, private, link-local, or reserved ranges."""
    if isinstance(ip, str):
        try:
            ip = ipaddress.ip_address(ip.strip("[]"))
        except ValueError:
            return True

    # Loopback (127.0.0.0/8, ::1)
    if ip.is_loopback:
        return True
    # RFC1918 (10/8, 172.16/12, 192.168/16) and RFC4193 (fc00::/7)
    if ip.is_private:
        return True
    # Link-local (169.254.0.0/16, fe80::/10)
    if ip.is_link_local:
        return True
    # Unspecified (0.0.0.0, ::)
    if ip.is_unspecified:
        return True
    # Multicast (224.0.0.0/4, ff00::/8)
    if ip.is_multicast:
        return True
    # Reserved (240.0.0.0/4, etc.)
    if ip.is_reserved:
        return True
    # IPv4-mapped IPv6 addresses (e.g. ::ffff:127.0.0.1)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        return is_ip_blocked(ip.ipv4_mapped)
    return False


def canonicalize_url(url: str) -> str:
    """Normalize a URL for consistent storage, deduplication, and database retrieval.

    - Trims whitespace
    - Lowercases scheme and hostname
    - Strips default ports (80 for http, 443 for https)
    - Strips fragment identifiers (#anchor)
    - Strips trailing slash on non-root paths for duplicate detection

    Note: This is used for deduplication comparisons in the database.
    It must NOT be used to rewrite URLs dispatched to the HTTP server,
    which would break trailing-slash redirects.
    """
    if not url or not isinstance(url, str):
        raise URLSecurityError("URL must be a non-empty string.")

    cleaned = url.strip()
    if not cleaned:
        raise URLSecurityError("URL cannot be empty or whitespace-only.")

    try:
        parts = urlsplit(cleaned)
    except Exception as exc:
        raise URLSecurityError(f"Malformed URL: {exc}") from exc

    scheme = parts.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise URLSecurityError(
            f"Unsupported URL scheme '{scheme or 'none'}'. Only http and https are permitted."
        )

    # Check for credentials
    if parts.username or parts.password or "@" in parts.netloc:
        raise URLSecurityError("URLs with embedded credentials are not permitted.")

    hostname = parts.hostname
    if not hostname:
        raise URLSecurityError("URL must include a valid hostname.")

    hostname = hostname.lower()

    # Determine port and netloc (ensure IPv6 literals are bracketed)
    port = parts.port
    netloc = f"[{hostname}]" if ":" in hostname and not hostname.startswith("[") else hostname
    if port is not None:
        if (scheme == "http" and port != 80) or (scheme == "https" and port != 443):
            netloc = f"{netloc}:{port}"

    # Path normalization for deduplication
    path = parts.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    # Query string retained as-is, fragment removed
    return urlunsplit((scheme, netloc, path, parts.query, ""))


def resolve_and_validate_destination(hostname: str, port: int) -> tuple[str, list[str]]:
    """Resolve a hostname via DNS and validate that all returned IP addresses are safe.

    Args:
        hostname: Domain name to resolve.
        port: TCP port number.

    Returns:
        A tuple of (first_safe_ip, list_of_all_resolved_ips).

    Raises:
        SSRFError: If resolution fails, returns no addresses, or ANY resolved IP is blocked.
    """
    try:
        addr_info = socket.getaddrinfo(
            hostname,
            port,
            proto=socket.IPPROTO_TCP,
        )
    except socket.gaierror as exc:
        raise SSRFError(f"Failed to resolve hostname '{hostname}'.") from exc

    if not addr_info:
        raise SSRFError(f"Could not resolve any network addresses for '{hostname}'.")

    resolved_ips: list[str] = []
    for entry in addr_info:
        sockaddr = entry[4]
        ip_str = sockaddr[0]
        try:
            resolved_ip = ipaddress.ip_address(ip_str)
            if is_ip_blocked(resolved_ip):
                raise SSRFError(
                    f"Hostname '{hostname}' resolves to blocked network address."
                )
            resolved_ips.append(ip_str)
        except ValueError:
            continue

    if not resolved_ips:
        raise SSRFError(f"No valid IP addresses resolved for hostname '{hostname}'.")

    return resolved_ips[0], resolved_ips


def validate_url_security(
    url: str,
    resolve_dns: bool = True,
    pinned_hosts: dict[str, str] | None = None,
) -> ValidatedURL:
    """Validate a URL against SSRF and security constraints without mutating resource paths.

    Security validation preserves path identity (including trailing slashes) so that
    legitimate trailing-slash redirects work without entering infinite loops.
    When resolve_dns is True, resolves and validates IP addresses, attaching the verified
    IP to the returned ValidatedURL and updating pinned_hosts if provided.

    Args:
        url: The candidate URL string.
        resolve_dns: Whether to perform DNS resolution to inspect resolved IP addresses.
        pinned_hosts: Optional dictionary to populate with {hostname: resolved_safe_ip}.

    Returns:
        ValidatedURL (a str subclass) containing the safe URL and .resolved_ip attribute.

    Raises:
        URLSecurityError: If URL format, scheme, or hostname is invalid.
        SSRFError: If destination resolves to a private, loopback, or reserved network.
    """
    if not url or not isinstance(url, str):
        raise URLSecurityError("URL must be a non-empty string.")

    cleaned = url.strip()
    if not cleaned:
        raise URLSecurityError("URL cannot be empty or whitespace-only.")

    try:
        parts = urlsplit(cleaned)
    except Exception as exc:
        raise URLSecurityError(f"Malformed URL: {exc}") from exc

    scheme = parts.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise URLSecurityError(
            f"Unsupported URL scheme '{scheme or 'none'}'. Only http and https are permitted."
        )

    # Check for credentials
    if parts.username or parts.password or "@" in parts.netloc:
        raise URLSecurityError("URLs with embedded credentials are not permitted.")

    hostname = parts.hostname
    if not hostname:
        raise URLSecurityError("URL must include a valid hostname.")

    hostname_lower = hostname.lower()

    # Reject obvious internal hostnames
    if hostname_lower in BLOCKED_HOSTNAMES:
        raise SSRFError(f"Access to blocked hostname '{hostname}' is not permitted.")

    for suffix in FORBIDDEN_HOSTNAME_SUFFIXES:
        if hostname_lower.endswith(suffix):
            raise SSRFError(f"Access to internal domain suffix '{suffix}' is not permitted.")

    port = parts.port
    default_port = 443 if scheme == "https" else 80
    effective_port = port or default_port

    # Reconstruct netloc with lowercased host and explicit non-default port
    netloc = f"[{hostname_lower}]" if ":" in hostname_lower and not hostname_lower.startswith("[") else hostname_lower
    if port is not None and port != default_port:
        netloc = f"{netloc}:{port}"

    # Path normalization:
    # Empty path is normalized to '/' so HTTP request line has a valid path.
    # Crucially, resource paths (e.g. /docs, /docs/, /api/v1/) preserve trailing slashes.
    path = parts.path or "/"

    # Fragment identifiers are removed (not sent in HTTP requests)
    safe_url_str = urlunsplit((scheme, netloc, path, parts.query, ""))
    validated = ValidatedURL(safe_url_str)

    # Check if hostname is an IP literal
    raw_ip_str = hostname_lower.strip("[]")
    try:
        ip = ipaddress.ip_address(raw_ip_str)
        if is_ip_blocked(ip):
            raise SSRFError(f"Access to blocked IP address '{ip}' is not permitted.")
        validated.resolved_ip = raw_ip_str
        if pinned_hosts is not None:
            pinned_hosts[hostname_lower] = raw_ip_str
        return validated
    except ValueError:
        # Hostname is a domain name, proceed to DNS resolution if requested
        pass

    if resolve_dns:
        safe_ip, _ = resolve_and_validate_destination(hostname_lower, effective_port)
        validated.resolved_ip = safe_ip
        if pinned_hosts is not None:
            pinned_hosts[hostname_lower] = safe_ip

    return validated


class SSRFSafeSyncBackend(SyncBackend):
    """Network backend for httpcore that enforces SSRF checks and pins validated destinations.

    Eliminates DNS TOCTOU rebinding attacks by ensuring the exact IP validated
    by the SSRF security layer is the IP used to establish the TCP connection,
    while preserving TLS SNI, certificate validation, and HTTP Host header semantics.
    """

    def __init__(
        self,
        pinned_hosts: dict[str, str] | None = None,
        resolve_dns: bool = True,
    ) -> None:
        super().__init__()
        self.pinned_hosts: dict[str, str] = dict(pinned_hosts) if pinned_hosts else {}
        self.resolve_dns = resolve_dns

    def pin_host(self, hostname: str, ip: str) -> None:
        """Explicitly pin a hostname to a verified safe IP address."""
        self.pinned_hosts[hostname.lower()] = ip

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.NetworkStream:
        host_lower = host.lower()
        if host_lower in self.pinned_hosts:
            safe_ip = self.pinned_hosts[host_lower]
        else:
            raw_ip_str = host.strip("[]")
            try:
                ip_obj = ipaddress.ip_address(raw_ip_str)
                if is_ip_blocked(ip_obj):
                    raise SSRFError(f"Access to blocked IP address '{host}' is not permitted.")
                safe_ip = raw_ip_str
            except ValueError:
                if not self.resolve_dns:
                    safe_ip = host
                else:
                    safe_ip, _ = resolve_and_validate_destination(host_lower, port)
                    self.pinned_hosts[host_lower] = safe_ip

        if socket_options is None:
            socket_options = []
        address = (safe_ip, port)
        source_address = None if local_address is None else (local_address, 0)
        exc_map = {
            socket.timeout: ConnectTimeout,
            OSError: ConnectError,
        }

        with map_exceptions(exc_map):
            sock = socket.create_connection(
                address,
                timeout,
                source_address=source_address,
            )
            for option in socket_options:
                sock.setsockopt(*option)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return SyncStream(sock)


class SSRFSafeTransport(httpx.HTTPTransport):
    """HTTPX HTTPTransport subclass configured with SSRFSafeSyncBackend.

    Ensures that connections are pinned to SSRF-validated IPs while preserving
    TLS certificate verification and standard HTTP semantics.
    """

    def __init__(
        self,
        pinned_hosts: dict[str, str] | None = None,
        resolve_dns: bool = True,
        **kwargs: Any,
    ) -> None:
        self.backend = SSRFSafeSyncBackend(pinned_hosts=pinned_hosts, resolve_dns=resolve_dns)
        super().__init__(**kwargs)
        # Recreate pool with SSRFSafeSyncBackend
        self._pool = httpcore.ConnectionPool(
            ssl_context=self._pool._ssl_context,
            max_connections=self._pool._max_connections,
            max_keepalive_connections=self._pool._max_keepalive_connections,
            keepalive_expiry=self._pool._keepalive_expiry,
            http1=self._pool._http1,
            http2=self._pool._http2,
            retries=self._pool._retries,
            network_backend=self.backend,
        )

    def pin_host(self, host: str, ip: str) -> None:
        self.backend.pin_host(host, ip)
