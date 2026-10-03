"""Exceptions for the AURA Model Gateway."""


class GatewayError(Exception):
    """Base exception for all Model Gateway errors."""

    pass


class ProviderUnavailableError(GatewayError):
    """Raised when the configured provider cannot be reached, initialized, or found."""

    pass


class UnsupportedCapabilityError(GatewayError):
    """Raised when a requested capability is not supported by the provider."""

    pass


class InvalidRequestError(GatewayError):
    """Raised when a request is invalid or malformed."""

    pass


class ProviderConfigurationError(GatewayError):
    """Raised when provider configuration is invalid or missing required settings."""

    pass


class GenerationError(GatewayError):
    """Raised when model generation fails during execution."""

    pass


class TransientModelProviderError(GenerationError):
    """Raised when model generation fails due to a transient provider or network condition.

    Examples:
    - HTTP 429 (rate limit exceeded / quota burst)
    - HTTP 500, 502, 503, 504 (temporary server overload, bad gateway, service unavailable)
    - Transient socket/connection reset or read timeout
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


# Sets of status codes and names
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
NON_TRANSIENT_STATUS_CODES = {400, 401, 403, 404, 405, 422}

TRANSIENT_CLASS_NAMES = {
    "RateLimitError",
    "InternalServerError",
    "APITimeoutError",
    "APIConnectionError",
    "ServiceUnavailableError",
    "GatewayTimeoutError",
    "BadGatewayError",
    "ConnectError",
    "ConnectTimeout",
    "ReadTimeout",
    "NetworkError",
    "ConnectionResetError",
    "ConnectionRefusedError",
    "RemoteDisconnected",
}

NON_TRANSIENT_CLASS_NAMES = {
    "AuthenticationError",
    "PermissionDeniedError",
    "BadRequestError",
    "NotFoundError",
    "UnprocessableEntityError",
    "ProviderConfigurationError",
    "InvalidRequestError",
    "UnsupportedCapabilityError",
    "ExecutionCancelledError",
}


def _is_transient_text(msg: str) -> bool:
    """Analyze error message string for transient vs non-transient indicators."""
    lower = msg.lower()

    # Definite non-transient markers take precedence
    non_transient_markers = [
        "401", "unauthorized", "invalid api key", "api key not configured", "api_key",
        "403", "forbidden", "permission denied", "permission_denied",
        "400", "bad request", "invalid request", "invalid_request",
        "404", "not found",
        "422", "unprocessable",
        "syntaxerror", "nameerror", "typeerror", "keyerror", "attributeerror", "indexerror",
        "cancelled", "cancellation",
    ]
    if any(m in lower for m in non_transient_markers):
        return False

    # Transient markers
    transient_markers = [
        "503", "service unavailable", "service_unavailable", "temporarily overloaded", "temporarily unavailable",
        "502", "bad gateway", "bad_gateway",
        "504", "gateway timeout", "gateway_timeout",
        "500", "internal server error", "internal_server_error",
        "429", "rate limit", "rate_limit", "too many requests",
        "connection reset", "connection refused", "remote disconnected", "broken pipe",
        "read timeout", "connect timeout", "timed out",
    ]
    return any(m in lower for m in transient_markers)


def is_transient_provider_error(exc_or_msg: Any) -> bool:
    """Determine whether an error represents a transient model-provider failure eligible for retry.

    Returns True for:
    - HTTP 429, 500, 502, 503, 504
    - TransientModelProviderError
    - Connection resets, socket timeouts, temporary network transport drops

    Returns False for:
    - Authentication / permission errors (401, 403, invalid API key)
    - Malformed requests / validation errors (400, 422)
    - Resource not found (404)
    - Application/retrieval/programming errors
    - User cancellation
    """
    if exc_or_msg is None:
        return False

    # 1. Direct instance check
    if isinstance(exc_or_msg, TransientModelProviderError):
        return True

    # 2. Check non-transient class instances
    if isinstance(exc_or_msg, (ProviderConfigurationError, InvalidRequestError, UnsupportedCapabilityError)):
        return False

    # 3. String input handling
    if isinstance(exc_or_msg, str):
        return _is_transient_text(exc_or_msg)

    # 4. Check status_code attributes on exc or exc.response or exc.__cause__
    for obj in (exc_or_msg, getattr(exc_or_msg, "__cause__", None), getattr(exc_or_msg, "response", None)):
        if obj is None:
            continue
        sc = getattr(obj, "status_code", None)
        if isinstance(sc, int):
            if sc in TRANSIENT_STATUS_CODES:
                return True
            if sc in NON_TRANSIENT_STATUS_CODES:
                return False

    # 5. Check class names of exc and exc.__cause__
    for obj in (exc_or_msg, getattr(exc_or_msg, "__cause__", None)):
        if obj is None:
            continue
        cls_name = obj.__class__.__name__
        if cls_name in NON_TRANSIENT_CLASS_NAMES:
            return False
        if cls_name in TRANSIENT_CLASS_NAMES:
            return True

    # 6. Check standard socket / network exceptions
    import socket
    if isinstance(exc_or_msg, (ConnectionResetError, ConnectionRefusedError, BrokenPipeError, TimeoutError, socket.timeout)):
        return True

    # 7. Fallback to message text inspection
    return _is_transient_text(str(exc_or_msg))
