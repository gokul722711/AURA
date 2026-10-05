"""REST API views for User-Configurable Model Profiles (M15)."""

import logging
import time
import uuid
from typing import Any

from django.core.exceptions import ValidationError
from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from agent.security import sanitize_text
from gateway.config import GatewayConfig
from gateway.models import ModelProfile, get_active_model_profile
from gateway.registry import create_provider

logger = logging.getLogger(__name__)

VALID_PROVIDERS = {
    ModelProfile.PROVIDER_NVIDIA,
    ModelProfile.PROVIDER_OLLAMA,
    ModelProfile.PROVIDER_OPENAI_COMPATIBLE,
    ModelProfile.PROVIDER_MOCK,
}


def _parse_uuid(val: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(val))
    except (ValueError, AttributeError):
        return None


def _perform_connection_check(
    provider_name: str,
    model: str,
    endpoint: str,
    api_key: str,
    timeout: float = 15.0,
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Perform a lightweight, bounded connection and usability check for a model profile."""
    t_start = time.monotonic()
    config = GatewayConfig(
        provider=provider_name,
        model=model,
        endpoint=endpoint,
        timeout=timeout,
        api_key=api_key,
        extra_config=extra_config or {},
    )

    try:
        provider = create_provider(config)

        if provider_name == ModelProfile.PROVIDER_MOCK:
            latency_ms = round((time.monotonic() - t_start) * 1000.0, 2)
            return {
                "success": True,
                "message": "Mock provider is ready.",
                "latency_ms": latency_ms,
            }

        if provider_name == ModelProfile.PROVIDER_OLLAMA:
            # Query Ollama native endpoint
            import httpx

            client = getattr(provider, "_client", None)
            if client is None or not hasattr(client, "post"):
                client = httpx.Client(
                    base_url=provider.endpoint,
                    timeout=httpx.Timeout(timeout),
                )
            # Try lightweight 1-token ping
            ping_payload = {
                "model": model,
                "messages": [{"role": "user", "content": "ping"}],
                "stream": False,
                "options": {"num_predict": 1},
            }
            resp = client.post("/api/chat", json=ping_payload)
            latency_ms = round((time.monotonic() - t_start) * 1000.0, 2)
            if resp.status_code == 200:
                return {
                    "success": True,
                    "message": f"Successfully connected to Ollama model '{model}'.",
                    "latency_ms": latency_ms,
                }
            # Fallback to checking /api/version if chat endpoint gave 404
            ver_resp = client.get("/api/version")
            if ver_resp.status_code == 200:
                return {
                    "success": True,
                    "message": f"Ollama endpoint reachable (model verification returned HTTP {resp.status_code}).",
                    "latency_ms": latency_ms,
                }
            return {
                "success": False,
                "error": sanitize_text(f"Ollama returned HTTP {resp.status_code}: {resp.text}"),
                "latency_ms": latency_ms,
            }

        # For NVIDIA and generic OpenAI-compatible
        client = getattr(provider, "_client", None)
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                base_url=endpoint or "http://localhost:8000/v1",
                api_key=api_key or "EMPTY",
                timeout=timeout,
            )

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            timeout=timeout,
        )
        latency_ms = round((time.monotonic() - t_start) * 1000.0, 2)
        return {
            "success": True,
            "message": f"Successfully verified '{model}' via {provider_name}.",
            "latency_ms": latency_ms,
        }

    except Exception as exc:
        latency_ms = round((time.monotonic() - t_start) * 1000.0, 2)
        safe_msg = sanitize_text(str(exc))
        return {
            "success": False,
            "error": safe_msg,
            "latency_ms": latency_ms,
        }


class ModelProfileListCreateView(APIView):
    """List all ModelProfiles or create a new profile."""

    authentication_classes = []
    permission_classes = []

    def get(self, request: Request) -> Response:
        """Return all model profiles ordered by active first."""
        profiles = ModelProfile.objects.all().order_by("-is_active", "name")
        return Response([p.to_dict() for p in profiles], status=status.HTTP_200_OK)

    def post(self, request: Request) -> Response:
        """Create a new ModelProfile."""
        data = request.data
        if not isinstance(data, dict):
            return Response(
                {"error": "Request body must be a JSON object."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        name = data.get("name")
        if not name or not isinstance(name, str) or not name.strip():
            return Response(
                {"error": "Field 'name' is required and must be a non-empty string."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        name = name.strip()

        if ModelProfile.objects.filter(name__iexact=name).exists():
            return Response(
                {"error": f"A model profile named '{name}' already exists."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        provider = data.get("provider")
        if not provider or not isinstance(provider, str) or provider.strip().lower() not in VALID_PROVIDERS:
            return Response(
                {
                    "error": f"Field 'provider' must be one of: {', '.join(sorted(VALID_PROVIDERS))}.",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        provider = provider.strip().lower()

        model_name = data.get("model")
        if not model_name or not isinstance(model_name, str) or not model_name.strip():
            return Response(
                {"error": "Field 'model' is required and must be a non-empty string."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        model_name = model_name.strip()

        endpoint = str(data.get("endpoint", "")).strip()
        api_key = str(data.get("api_key", "")).strip()

        timeout = 30.0
        if "timeout" in data and data["timeout"] is not None:
            try:
                timeout = float(data["timeout"])
                if timeout <= 0:
                    raise ValueError
            except (ValueError, TypeError):
                return Response(
                    {"error": "Field 'timeout' must be a positive number."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        temperature = None
        if "temperature" in data and data["temperature"] is not None:
            try:
                temperature = float(data["temperature"])
                if not (0.0 <= temperature <= 2.0):
                    raise ValueError
            except (ValueError, TypeError):
                return Response(
                    {"error": "Field 'temperature' must be a float between 0.0 and 2.0."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        max_tokens = None
        if "max_tokens" in data and data["max_tokens"] is not None:
            try:
                max_tokens = int(data["max_tokens"])
                if max_tokens <= 0:
                    raise ValueError
            except (ValueError, TypeError):
                return Response(
                    {"error": "Field 'max_tokens' must be a positive integer."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        is_active = bool(data.get("is_active", False))
        capabilities = data.get("capabilities", [])
        if not isinstance(capabilities, list):
            capabilities = []
        extra_config = data.get("extra_config", {})
        if not isinstance(extra_config, dict):
            extra_config = {}

        try:
            profile = ModelProfile(
                name=name,
                provider=provider,
                endpoint=endpoint,
                model=model_name,
                api_key=api_key,
                timeout=timeout,
                temperature=temperature,
                max_tokens=max_tokens,
                capabilities=capabilities,
                extra_config=extra_config,
                is_active=is_active,
            )
            profile.save()
        except ValidationError as val_err:
            return Response(
                {"error": str(val_err)},
                status=status.HTTP_400_BAD_REQUEST,
            )

        return Response(profile.to_dict(), status=status.HTTP_201_CREATED)


class ModelProfileDetailView(APIView):
    """Retrieve, update, or delete a ModelProfile."""

    authentication_classes = []
    permission_classes = []

    def get_object(self, profile_id: str) -> ModelProfile | None:
        uid = _parse_uuid(profile_id)
        if uid is None:
            return None
        return ModelProfile.objects.filter(id=uid).first()

    def get(self, request: Request, profile_id: str) -> Response:
        profile = self.get_object(profile_id)
        if profile is None:
            return Response(
                {"error": f"ModelProfile '{profile_id}' not found."},
                status=status.HTTP_404_NOT_FOUND,
            )
        return Response(profile.to_dict(), status=status.HTTP_200_OK)

    def patch(self, request: Request, profile_id: str) -> Response:
        profile = self.get_object(profile_id)
        if profile is None:
            return Response(
                {"error": f"ModelProfile '{profile_id}' not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        data = request.data
        if not isinstance(data, dict):
            return Response(
                {"error": "Request body must be a JSON object."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if "name" in data:
            name = str(data["name"]).strip()
            if not name:
                return Response(
                    {"error": "Name cannot be empty."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            if ModelProfile.objects.exclude(id=profile.id).filter(name__iexact=name).exists():
                return Response(
                    {"error": f"A model profile named '{name}' already exists."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            profile.name = name

        if "provider" in data:
            provider = str(data["provider"]).strip().lower()
            if provider not in VALID_PROVIDERS:
                return Response(
                    {"error": f"Field 'provider' must be one of: {', '.join(sorted(VALID_PROVIDERS))}."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            profile.provider = provider

        if "model" in data:
            model = str(data["model"]).strip()
            if not model:
                return Response(
                    {"error": "Model identifier cannot be empty."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            profile.model = model

        if "endpoint" in data:
            profile.endpoint = str(data["endpoint"]).strip()

        # Update API key only if explicitly provided non-empty string, or explicitly cleared
        if "clear_api_key" in data and data["clear_api_key"]:
            profile.api_key = ""
        elif "api_key" in data and data["api_key"] is not None:
            key_val = str(data["api_key"]).strip()
            if key_val:
                profile.api_key = key_val

        if "timeout" in data and data["timeout"] is not None:
            try:
                t = float(data["timeout"])
                if t <= 0:
                    raise ValueError
                profile.timeout = t
            except (ValueError, TypeError):
                return Response(
                    {"error": "Timeout must be a positive number."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        if "temperature" in data:
            if data["temperature"] is None:
                profile.temperature = None
            else:
                try:
                    temp = float(data["temperature"])
                    if not (0.0 <= temp <= 2.0):
                        raise ValueError
                    profile.temperature = temp
                except (ValueError, TypeError):
                    return Response(
                        {"error": "Temperature must be between 0.0 and 2.0."},
                        status=status.HTTP_400_BAD_REQUEST,
                    )

        if "max_tokens" in data:
            if data["max_tokens"] is None:
                profile.max_tokens = None
            else:
                try:
                    mt = int(data["max_tokens"])
                    if mt <= 0:
                        raise ValueError
                    profile.max_tokens = mt
                except (ValueError, TypeError):
                    return Response(
                        {"error": "max_tokens must be a positive integer."},
                        status=status.HTTP_400_BAD_REQUEST,
                    )

        if "capabilities" in data and isinstance(data["capabilities"], list):
            profile.capabilities = data["capabilities"]

        if "extra_config" in data and isinstance(data["extra_config"], dict):
            profile.extra_config = data["extra_config"]

        if "is_active" in data:
            profile.is_active = bool(data["is_active"])

        try:
            profile.save()
        except ValidationError as val_err:
            return Response({"error": str(val_err)}, status=status.HTTP_400_BAD_REQUEST)

        return Response(profile.to_dict(), status=status.HTTP_200_OK)

    def delete(self, request: Request, profile_id: str) -> Response:
        profile = self.get_object(profile_id)
        if profile is None:
            return Response(
                {"error": f"ModelProfile '{profile_id}' not found."},
                status=status.HTTP_404_NOT_FOUND,
            )
        profile.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class ModelProfileActivateView(APIView):
    """Set a specific ModelProfile as the active research profile."""

    authentication_classes = []
    permission_classes = []

    def post(self, request: Request, profile_id: str) -> Response:
        uid = _parse_uuid(profile_id)
        if uid is None:
            return Response(
                {"error": f"Invalid profile ID '{profile_id}'."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        profile = ModelProfile.objects.filter(id=uid).first()
        if profile is None:
            return Response(
                {"error": f"ModelProfile '{profile_id}' not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        profile.is_active = True
        profile.save()
        return Response(
            {"status": "activated", "profile": profile.to_dict()},
            status=status.HTTP_200_OK,
        )


class ModelProfileTestConnectionView(APIView):
    """Test connection for a saved or draft ModelProfile."""

    authentication_classes = []
    permission_classes = []

    def post(self, request: Request, profile_id: str | None = None) -> Response:
        data = request.data or {}

        if profile_id:
            uid = _parse_uuid(profile_id)
            if uid is None:
                return Response(
                    {"error": f"Invalid profile ID '{profile_id}'."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            profile = ModelProfile.objects.filter(id=uid).first()
            if profile is None:
                return Response(
                    {"error": f"ModelProfile '{profile_id}' not found."},
                    status=status.HTTP_404_NOT_FOUND,
                )
            provider_name = profile.provider
            model_name = profile.model
            endpoint = profile.endpoint
            api_key = profile.api_key
            timeout = profile.timeout
            extra_config = profile.extra_config
        else:
            provider_name = str(data.get("provider", "")).strip().lower()
            if provider_name not in VALID_PROVIDERS:
                return Response(
                    {"error": f"Provider must be one of: {', '.join(sorted(VALID_PROVIDERS))}."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            model_name = str(data.get("model", "")).strip()
            if not model_name:
                return Response(
                    {"error": "Field 'model' is required for connection testing."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            endpoint = str(data.get("endpoint", "")).strip()
            api_key = str(data.get("api_key", "")).strip()
            timeout = float(data.get("timeout", 15.0))
            extra_config = data.get("extra_config", {})

        result = _perform_connection_check(
            provider_name=provider_name,
            model=model_name,
            endpoint=endpoint,
            api_key=api_key,
            timeout=timeout,
            extra_config=extra_config,
        )
        http_status = status.HTTP_200_OK if result.get("success") else status.HTTP_400_BAD_REQUEST
        return Response(result, status=http_status)
