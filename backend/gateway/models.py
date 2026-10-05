"""Django models for User-Configurable Model Profiles (M15)."""

import uuid
from typing import Any

from django.core.exceptions import ValidationError
from django.db import models


class ModelProfile(models.Model):
    """User-configurable LLM profile representing an inference provider, model, and parameters."""

    PROVIDER_NVIDIA = "nvidia"
    PROVIDER_OLLAMA = "ollama"
    PROVIDER_OPENAI_COMPATIBLE = "openai_compatible"
    PROVIDER_MOCK = "mock"

    PROVIDER_CHOICES = [
        (PROVIDER_NVIDIA, "NVIDIA NIM"),
        (PROVIDER_OLLAMA, "Ollama"),
        (PROVIDER_OPENAI_COMPATIBLE, "OpenAI-compatible"),
        (PROVIDER_MOCK, "Mock (Offline Testing)"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(
        max_length=128,
        unique=True,
        help_text="Unique user-friendly name for this profile.",
    )
    provider = models.CharField(
        max_length=64,
        choices=PROVIDER_CHOICES,
        db_index=True,
        help_text="Inference provider type.",
    )
    endpoint = models.CharField(
        max_length=512,
        blank=True,
        default="",
        help_text="API endpoint base URL (e.g. https://integrate.api.nvidia.com/v1 or http://localhost:11434).",
    )
    model = models.CharField(
        max_length=256,
        help_text="Model identifier (e.g. nvidia/nemotron-3-ultra-550b-a55b, llama3.1:8b, Qwen).",
    )
    api_key = models.CharField(
        max_length=512,
        blank=True,
        default="",
        help_text="API key or token for provider authentication. Never exposed via API reads.",
    )
    timeout = models.FloatField(
        default=30.0,
        help_text="Request timeout in seconds.",
    )
    temperature = models.FloatField(
        null=True,
        blank=True,
        help_text="Sampling temperature between 0.0 and 2.0.",
    )
    max_tokens = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Maximum tokens to generate.",
    )
    capabilities = models.JSONField(
        default=list,
        blank=True,
        help_text="List of supported capabilities (generation, streaming, structured_output).",
    )
    extra_config = models.JSONField(
        default=dict,
        blank=True,
        help_text="Additional provider-specific parameters.",
    )
    is_active = models.BooleanField(
        default=False,
        db_index=True,
        help_text="Whether this profile is currently active as the research model.",
    )
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-is_active", "name"]
        indexes = [
            models.Index(fields=["is_active", "-created_at"]),
            models.Index(fields=["provider", "name"]),
        ]

    def __str__(self) -> str:
        active_str = " [active]" if self.is_active else ""
        return f"ModelProfile({self.name}, provider={self.provider}, model={self.model}{active_str})"

    def __repr__(self) -> str:
        return f"<ModelProfile: {self.name} ({self.provider}/{self.model}) active={self.is_active}>"

    def clean(self) -> None:
        """Validate profile fields."""
        if not self.name or not self.name.strip():
            raise ValidationError({"name": "Name cannot be empty or whitespace-only."})
        if not self.model or not self.model.strip():
            raise ValidationError({"model": "Model cannot be empty or whitespace-only."})
        if self.timeout is not None and self.timeout <= 0:
            raise ValidationError({"timeout": "Timeout must be greater than 0."})
        if self.temperature is not None and not (0.0 <= self.temperature <= 2.0):
            raise ValidationError({"temperature": "Temperature must be between 0.0 and 2.0."})

    def save(self, *args: Any, **kwargs: Any) -> None:
        """Ensure only one profile is active at a time and apply clean validation."""
        self.clean()
        if self.is_active:
            ModelProfile.objects.exclude(pk=self.pk).filter(is_active=True).update(is_active=False)
        elif not ModelProfile.objects.filter(is_active=True).exclude(pk=self.pk).exists():
            # If this is the first and only profile being created, make it active by default
            if not ModelProfile.objects.exclude(pk=self.pk).exists():
                self.is_active = True

        super().save(*args, **kwargs)

    @property
    def has_api_key(self) -> bool:
        """Check whether an API key is configured."""
        return bool(self.api_key and self.api_key.strip())

    @property
    def api_key_masked(self) -> str:
        """Return masked representation of API key without exposing secret."""
        if not self.has_api_key:
            return ""
        key = self.api_key.strip()
        if len(key) <= 6:
            return "***"
        return f"...{key[-4:]}"

    def to_dict(self) -> dict[str, Any]:
        """Return safe dictionary representation for API serialization. Never returns raw api_key."""
        return {
            "id": str(self.id),
            "name": self.name,
            "provider": self.provider,
            "endpoint": self.endpoint,
            "model": self.model,
            "has_api_key": self.has_api_key,
            "api_key_masked": self.api_key_masked,
            "timeout": self.timeout,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "capabilities": self.capabilities,
            "extra_config": self.extra_config,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


def get_active_model_profile() -> ModelProfile | None:
    """Return the currently active ModelProfile, or None if none configured."""
    return ModelProfile.objects.filter(is_active=True).first()
