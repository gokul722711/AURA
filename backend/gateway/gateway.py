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

    @classmethod
    def from_profile(cls, profile: Any) -> "ModelGateway":
        """Instantiate a ModelGateway configured from a ModelProfile."""
        extra = dict(profile.extra_config or {})
        if profile.temperature is not None:
            extra["temperature"] = profile.temperature
        if profile.max_tokens is not None:
            extra["max_tokens"] = profile.max_tokens

        config = GatewayConfig(
            provider=profile.provider,
            model=profile.model,
            endpoint=profile.endpoint,
            timeout=profile.timeout,
            api_key=profile.api_key,
            extra_config=extra,
        )
        return cls(config=config)


def get_gateway(
    profile: Any | None = None,
    profile_id: str | None = None,
    allow_fallback: bool = True,
) -> ModelGateway:
    """Return a ModelGateway configured from an explicit profile, profile ID, or active profile."""
    if profile is not None:
        return ModelGateway.from_profile(profile)

    if profile_id is not None:
        from gateway.exceptions import ProviderConfigurationError
        from gateway.models import ModelProfile

        try:
            target_profile = ModelProfile.objects.get(id=profile_id)
            return ModelGateway.from_profile(target_profile)
        except ModelProfile.DoesNotExist:
            raise ProviderConfigurationError(f"ModelProfile with id '{profile_id}' does not exist.")

    # Try resolving active profile from database if Django apps are ready
    try:
        from django.apps import apps

        if apps.ready:
            from gateway.models import get_active_model_profile

            active_profile = get_active_model_profile()
            if active_profile is not None:
                return ModelGateway.from_profile(active_profile)
    except Exception:
        pass

    if allow_fallback:
        return ModelGateway()

    from gateway.exceptions import NoModelConfiguredError

    raise NoModelConfiguredError(
        "No research model configured. Configure a model profile before starting research."
    )
