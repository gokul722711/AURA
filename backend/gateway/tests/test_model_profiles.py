"""Automated tests for User-Configurable Model Profiles (M15)."""

import json
import uuid
from unittest.mock import MagicMock, patch

from django.core.exceptions import ValidationError
from django.test import TestCase
from rest_framework.test import APIClient

from gateway.exceptions import NoModelConfiguredError
from gateway.gateway import ModelGateway, get_gateway
from gateway.models import ModelProfile, get_active_model_profile


class ModelProfileModelTests(TestCase):
    """Unit tests for the ModelProfile model."""

    def setUp(self) -> None:
        ModelProfile.objects.all().delete()

    def test_create_model_profile(self) -> None:
        """Creating a profile sets defaults and enforces active profile rules."""
        profile = ModelProfile.objects.create(
            name="My Test Model",
            provider=ModelProfile.PROVIDER_MOCK,
            model="mock-test-1",
        )
        self.assertIsNotNone(profile.id)
        self.assertEqual(profile.name, "My Test Model")
        self.assertEqual(profile.provider, "mock")
        self.assertEqual(profile.model, "mock-test-1")
        # First profile becomes active automatically
        self.assertTrue(profile.is_active)
        self.assertFalse(profile.has_api_key)
        self.assertEqual(profile.api_key_masked, "")

    def test_single_active_profile_enforcement(self) -> None:
        """Activating a new profile automatically deactivates previously active profiles."""
        p1 = ModelProfile.objects.create(
            name="Model 1",
            provider=ModelProfile.PROVIDER_MOCK,
            model="m1",
            is_active=True,
        )
        self.assertTrue(p1.is_active)

        p2 = ModelProfile.objects.create(
            name="Model 2",
            provider=ModelProfile.PROVIDER_MOCK,
            model="m2",
            is_active=True,
        )
        p1.refresh_from_db()
        self.assertFalse(p1.is_active)
        self.assertTrue(p2.is_active)

        self.assertEqual(get_active_model_profile().id, p2.id)

    def test_api_key_masking(self) -> None:
        """API keys are masked and never exposed via to_dict() or string representations."""
        secret_key = "nvapi-secret-1234567890abcdef"
        profile = ModelProfile.objects.create(
            name="NVIDIA Secret",
            provider=ModelProfile.PROVIDER_NVIDIA,
            model="nemotron",
            api_key=secret_key,
        )
        self.assertTrue(profile.has_api_key)
        self.assertEqual(profile.api_key_masked, "...cdef")

        # to_dict() must not contain the raw secret key
        serialized = profile.to_dict()
        self.assertTrue(serialized["has_api_key"])
        self.assertEqual(serialized["api_key_masked"], "...cdef")
        self.assertNotIn(secret_key, str(serialized))
        self.assertNotIn("api_key", serialized)

        # __str__ and __repr__ must not contain the secret key
        self.assertNotIn(secret_key, str(profile))
        self.assertNotIn(secret_key, repr(profile))

    def test_validation_constraints(self) -> None:
        """Invalid temperature, timeout, or empty names raise validation errors."""
        with self.assertRaises(ValidationError):
            p = ModelProfile(name="", provider="mock", model="m")
            p.clean()

        with self.assertRaises(ValidationError):
            p = ModelProfile(name="Test", provider="mock", model="")
            p.clean()

        with self.assertRaises(ValidationError):
            p = ModelProfile(name="Test", provider="mock", model="m", timeout=-5.0)
            p.clean()

        with self.assertRaises(ValidationError):
            p = ModelProfile(name="Test", provider="mock", model="m", temperature=3.5)
            p.clean()


class ModelProfileAPITests(TestCase):
    """API integration tests for ModelProfile CRUD and activation."""

    def setUp(self) -> None:
        ModelProfile.objects.all().delete()
        self.client = APIClient()

    def test_list_model_profiles_empty(self) -> None:
        """GET /api/models/ returns empty list when no profiles exist."""
        resp = self.client.get("/api/models/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), [])

    def test_create_model_profile_api(self) -> None:
        """POST /api/models/ creates a profile and masks credentials in response."""
        payload = {
            "name": "Local Ollama Llama",
            "provider": "ollama",
            "endpoint": "http://localhost:11434",
            "model": "llama3.1:8b",
            "api_key": "some-secret-token",
            "timeout": 45.0,
            "temperature": 0.5,
            "is_active": True,
        }
        resp = self.client.post("/api/models/", data=payload, format="json")
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data["name"], "Local Ollama Llama")
        self.assertEqual(data["provider"], "ollama")
        self.assertEqual(data["model"], "llama3.1:8b")
        self.assertTrue(data["has_api_key"])
        self.assertEqual(data["api_key_masked"], "...oken")
        self.assertNotIn("some-secret-token", str(data))

        # Check database
        profile = ModelProfile.objects.get(name="Local Ollama Llama")
        self.assertEqual(profile.api_key, "some-secret-token")
        self.assertTrue(profile.is_active)

    def test_reject_duplicate_name(self) -> None:
        """POST /api/models/ rejects duplicate profile names."""
        ModelProfile.objects.create(name="Duplicate Name", provider="mock", model="m1")
        resp = self.client.post(
            "/api/models/",
            data={"name": "Duplicate Name", "provider": "mock", "model": "m2"},
            format="json",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("already exists", resp.json()["error"])

    def test_retrieve_model_profile_detail(self) -> None:
        """GET /api/models/<id>/ returns profile detail with masked key."""
        p = ModelProfile.objects.create(
            name="Detail Model",
            provider="openai_compatible",
            model="Qwen",
            api_key="sk-secret-token-12345",
        )
        resp = self.client.get(f"/api/models/{p.id}/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["name"], "Detail Model")
        self.assertEqual(data["provider"], "openai_compatible")
        self.assertEqual(data["model"], "Qwen")
        self.assertTrue(data["has_api_key"])
        self.assertNotIn("sk-secret-token-12345", str(data))

    def test_patch_model_profile_preserves_unmodified_key(self) -> None:
        """PATCH /api/models/<id>/ preserves stored API key when key is omitted in payload."""
        p = ModelProfile.objects.create(
            name="Preserve Key",
            provider="nvidia",
            model="nemotron",
            api_key="original-secret-key-1234",
            temperature=0.7,
        )
        resp = self.client.patch(
            f"/api/models/{p.id}/",
            data={"temperature": 0.2},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["temperature"], 0.2)

        p.refresh_from_db()
        self.assertEqual(p.api_key, "original-secret-key-1234")

    def test_delete_model_profile(self) -> None:
        """DELETE /api/models/<id>/ removes the profile."""
        p = ModelProfile.objects.create(name="To Delete", provider="mock", model="m")
        resp = self.client.delete(f"/api/models/{p.id}/")
        self.assertEqual(resp.status_code, 204)
        self.assertFalse(ModelProfile.objects.filter(id=p.id).exists())

    def test_activate_model_profile(self) -> None:
        """POST /api/models/<id>/activate/ marks the profile as active."""
        p1 = ModelProfile.objects.create(name="Model 1", provider="mock", model="m1", is_active=True)
        p2 = ModelProfile.objects.create(name="Model 2", provider="mock", model="m2", is_active=False)

        resp = self.client.post(f"/api/models/{p2.id}/activate/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "activated")

        p1.refresh_from_db()
        p2.refresh_from_db()
        self.assertFalse(p1.is_active)
        self.assertTrue(p2.is_active)

    def test_connection_test_mock_provider(self) -> None:
        """POST /api/models/test/ successfully tests a mock configuration."""
        resp = self.client.post(
            "/api/models/test/",
            data={"provider": "mock", "model": "mock-model"},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["success"])
        self.assertIn("Mock provider is ready", resp.json()["message"])

    def test_connection_test_failure_sanitizes_secrets(self) -> None:
        """Connection failures do not leak the configured API key in error responses."""
        secret_key = "nvapi-leaked-secret-999999"
        with patch("gateway.views._perform_connection_check") as mock_check:
            mock_check.return_value = {
                "success": False,
                "error": "Authentication failed for token=***",
                "latency_ms": 12.5,
            }
            resp = self.client.post(
                "/api/models/test/",
                data={
                    "provider": "nvidia",
                    "model": "nvidia/nemotron",
                    "api_key": secret_key,
                },
                format="json",
            )
            self.assertEqual(resp.status_code, 400)
            self.assertFalse(resp.json()["success"])
            self.assertNotIn(secret_key, str(resp.json()))

    def test_deleted_profile_cannot_be_used(self) -> None:
        """Deleted profiles cannot be resolved by get_gateway."""
        p = ModelProfile.objects.create(name="Short Lived", provider="mock", model="m")
        profile_id = str(p.id)
        p.delete()

        from gateway.exceptions import ProviderConfigurationError

        with self.assertRaises(ProviderConfigurationError):
            get_gateway(profile_id=profile_id)
