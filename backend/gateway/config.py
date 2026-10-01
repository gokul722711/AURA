"""Configuration for AURA Model Gateway."""

import os
from dataclasses import dataclass, field
from typing import Any

from gateway.exceptions import ProviderConfigurationError


@dataclass
class GatewayConfig:
    """Configuration settings for the Model Gateway and provider."""

    provider: str = "mock"
    model: str = "mock-model"
    endpoint: str = ""
    timeout: float = 30.0
    api_key: str = ""
    extra_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider or not isinstance(self.provider, str):
            raise ProviderConfigurationError("Provider name must be a non-empty string.")
        if not self.model or not isinstance(self.model, str):
            raise ProviderConfigurationError("Model name must be a non-empty string.")
        try:
            self.timeout = float(self.timeout)
        except (ValueError, TypeError) as exc:
            raise ProviderConfigurationError(f"Invalid timeout value: {self.timeout}") from exc
        if self.timeout <= 0:
            raise ProviderConfigurationError("Timeout must be greater than 0.")

    @classmethod
    def from_env(cls) -> "GatewayConfig":
        """Load configuration strictly from environment variables."""
        provider = os.environ.get("AI_PROVIDER", "mock").strip() or "mock"
        model = os.environ.get("AI_MODEL", "mock-model").strip() or "mock-model"
        endpoint = os.environ.get("AI_ENDPOINT", "").strip()
        api_key = os.environ.get("AI_API_KEY", "").strip()
        timeout_raw = os.environ.get("AI_TIMEOUT", "30.0").strip() or "30.0"
        try:
            timeout = float(timeout_raw)
        except ValueError as exc:
            raise ProviderConfigurationError(f"Invalid AI_TIMEOUT environment variable: {timeout_raw}") from exc

        return cls(
            provider=provider,
            model=model,
            endpoint=endpoint,
            timeout=timeout,
            api_key=api_key,
        )

    @classmethod
    def from_settings(cls) -> "GatewayConfig":
        """Load configuration from Django settings if available, otherwise fall back to environment."""
        try:
            from django.conf import settings
            from django.core.exceptions import ImproperlyConfigured

            if not settings.configured:
                return cls.from_env()

            if not hasattr(settings, "AI_GATEWAY"):
                return cls.from_env()

            conf = settings.AI_GATEWAY
        except (ImportError, ImproperlyConfigured):
            # Django or its settings are genuinely unavailable
            return cls.from_env()

        if not isinstance(conf, dict):
            raise ProviderConfigurationError("Django setting 'AI_GATEWAY' must be a dictionary.")

        provider = conf.get("PROVIDER", os.environ.get("AI_PROVIDER", "mock"))
        model = conf.get("MODEL", os.environ.get("AI_MODEL", "mock-model"))
        endpoint = conf.get("ENDPOINT", os.environ.get("AI_ENDPOINT", ""))
        timeout = conf.get("TIMEOUT", os.environ.get("AI_TIMEOUT", 30.0))
        api_key = conf.get("API_KEY", os.environ.get("AI_API_KEY", ""))
        extra = {k: v for k, v in conf.items() if k not in ("PROVIDER", "MODEL", "ENDPOINT", "TIMEOUT", "API_KEY")}

        return cls(
            provider=provider,
            model=model,
            endpoint=str(endpoint) if endpoint is not None else "",
            timeout=timeout,
            api_key=str(api_key) if api_key is not None else "",
            extra_config=extra,
        )
