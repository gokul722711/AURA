"""Model Gateway implementation for routing AI operations."""

import re
from collections.abc import Iterator

from gateway.base import LLMProvider
from gateway.config import GatewayConfig
from gateway.exceptions import (
    GatewayError,
    GenerationError,
    InvalidRequestError,
    UnsupportedCapabilityError,
)
from gateway.registry import create_provider
from gateway.types import (
    CAPABILITY_GENERATION,
    CAPABILITY_STREAMING,
    CAPABILITY_STRUCTURED_OUTPUT,
    GenerationRequest,
    GenerationResponse,
    ProviderMetadata,
    StreamChunk,
    StructuredOutputRequest,
    StructuredOutputResponse,
)

# Pattern to mask sensitive strings in error messages if any appear
_SECRET_PATTERN = re.compile(
    r"(?i)(key|secret|token|password|auth|bearer)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{8,})['\"]?"
)


def _sanitize_error_message(message: str) -> str:
    """Ensure error messages do not leak secrets or tokens."""
    return _SECRET_PATTERN.sub(r"\1=***", str(message))


class ModelGateway:
    """Internal gateway that routes model operations to the configured provider."""

    def __init__(
        self,
        config: GatewayConfig | None = None,
        provider: LLMProvider | None = None,
    ) -> None:
        if provider is not None:
            self._provider = provider
            self._config = config or GatewayConfig(
                provider=provider.metadata().provider,
                model=provider.metadata().model,
            )
        else:
            self._config = config or GatewayConfig.from_settings()
            self._provider = create_provider(self._config)

    @property
    def provider(self) -> LLMProvider:
        """Return the active provider instance."""
        return self._provider

    @property
    def config(self) -> GatewayConfig:
        """Return the active gateway configuration."""
        return self._config

    def _check_capability(self, capability: str) -> None:
        """Verify that the underlying provider supports the requested capability."""
        meta = self._provider.metadata()
        if not meta.supports(capability):
            raise UnsupportedCapabilityError(
                f"Provider '{meta.provider}' does not support capability '{capability}'."
            )

    def metadata(self) -> ProviderMetadata:
        """Return provider and model metadata."""
        return self._provider.metadata()

    def get_metadata(self) -> ProviderMetadata:
        """Alias for metadata()."""
        return self.metadata()

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Route a text generation request through the configured provider."""
        if not isinstance(request, GenerationRequest):
            raise InvalidRequestError("request must be an instance of GenerationRequest.")

        self._check_capability(CAPABILITY_GENERATION)

        try:
            return self._provider.generate(request)
        except GatewayError:
            raise
        except Exception as exc:
            sanitized = _sanitize_error_message(str(exc))
            raise GenerationError(f"Model generation failed: {sanitized}") from exc

    def stream(self, request: GenerationRequest) -> Iterator[StreamChunk]:
        """Route a streaming request through the configured provider."""
        if not isinstance(request, GenerationRequest):
            raise InvalidRequestError("request must be an instance of GenerationRequest.")

        self._check_capability(CAPABILITY_STREAMING)

        try:
            iterator = self._provider.stream(request)
            for chunk in iterator:
                yield chunk
        except GatewayError:
            raise
        except Exception as exc:
            sanitized = _sanitize_error_message(str(exc))
            raise GenerationError(f"Model streaming failed: {sanitized}") from exc

    def structured_output(
        self, request: StructuredOutputRequest
    ) -> StructuredOutputResponse:
        """Route a structured output request through the configured provider."""
        if not isinstance(request, StructuredOutputRequest):
            raise InvalidRequestError("request must be an instance of StructuredOutputRequest.")

        self._check_capability(CAPABILITY_STRUCTURED_OUTPUT)

        try:
            return self._provider.structured_output(request)
        except GatewayError:
            raise
        except Exception as exc:
            sanitized = _sanitize_error_message(str(exc))
            raise GenerationError(f"Structured output generation failed: {sanitized}") from exc


def get_gateway() -> ModelGateway:
    """Return a ModelGateway configured from application settings."""
    return ModelGateway()
