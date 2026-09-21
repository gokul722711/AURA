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
