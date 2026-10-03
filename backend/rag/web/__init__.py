"""Web fetching and URL security subsystem for AURA (M12)."""

from rag.web.base import WebFetcher, WebFetchResult
from rag.web.fetcher import HTTPXWebFetcher
from rag.web.mock import MockWebFetcher
from rag.web.security import (
    SSRFSafeSyncBackend,
    SSRFSafeTransport,
    ValidatedURL,
    canonicalize_url,
    is_ip_blocked,
    resolve_and_validate_destination,
    validate_url_security,
)

__all__ = [
    "HTTPXWebFetcher",
    "MockWebFetcher",
    "SSRFSafeSyncBackend",
    "SSRFSafeTransport",
    "ValidatedURL",
    "WebFetcher",
    "WebFetchResult",
    "canonicalize_url",
    "is_ip_blocked",
    "resolve_and_validate_destination",
    "validate_url_security",
]
