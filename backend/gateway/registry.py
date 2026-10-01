"""Provider registry for AURA Model Gateway."""


from typing import Any

from gateway.base import LLMProvider
from gateway.config import GatewayConfig
from gateway.exceptions import ProviderConfigurationError, ProviderUnavailableError
from gateway.providers.mock import MockLLMProvider
from gateway.providers.nvidia import NvidiaLLMProvider

_REGISTRY: dict[str, type[LLMProvider]] = {
    "mock": MockLLMProvider,
    "nvidia": NvidiaLLMProvider,
}


def register_provider(name: str, provider_cls: type[LLMProvider]) -> None:
    """Register a new LLMProvider implementation under a provider name."""
    if not isinstance(name, str):
        raise ProviderConfigurationError("Provider name must be a non-empty string.")
    normalized_name = name.strip().lower()
    if not normalized_name:
        raise ProviderConfigurationError("Provider name must be a non-empty string.")

    try:
        if not issubclass(provider_cls, LLMProvider):
            raise ProviderConfigurationError(
                f"Provider class {provider_cls.__name__} must inherit from LLMProvider."
            )
    except TypeError as exc:
        raise ProviderConfigurationError(
            f"Provider class {provider_cls} must be a class inheriting from LLMProvider."
        ) from exc

    _REGISTRY[normalized_name] = provider_cls


def get_provider_class(name: str) -> type[LLMProvider]:
    """Retrieve the provider class for the given provider name."""
    key = name.lower().strip()
    if key not in _REGISTRY:
        raise ProviderUnavailableError(
            f"Provider '{name}' is not registered or available. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[key]


def create_provider(config: GatewayConfig) -> LLMProvider:
    """Instantiate the configured provider from a GatewayConfig."""
    provider_cls = get_provider_class(config.provider)
    try:
        if config.provider.lower().strip() == "nvidia":
            kwargs: dict[str, Any] = {
                "model": config.model,
                "timeout": config.timeout,
            }
            if config.endpoint:
                kwargs["endpoint"] = config.endpoint
            if config.api_key:
                kwargs["api_key"] = config.api_key
            return provider_cls(**kwargs)

        # Standard instantiation passing model
        return provider_cls(model=config.model)
    except Exception as exc:
        raise ProviderConfigurationError(
            f"Failed to initialize provider '{config.provider}': {exc}"
        ) from exc
