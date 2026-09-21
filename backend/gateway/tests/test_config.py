"""Tests for Model Gateway configuration."""

import os
from unittest.mock import patch

from django.test import TestCase, override_settings

from gateway.config import GatewayConfig
from gateway.exceptions import ProviderConfigurationError


class ConfigTests(TestCase):
    """Unit tests for GatewayConfig."""

    def test_default_config(self):
        config = GatewayConfig()
        self.assertEqual(config.provider, "mock")
        self.assertEqual(config.model, "mock-model")
        self.assertEqual(config.endpoint, "")
        self.assertEqual(config.timeout, 30.0)
        self.assertEqual(config.extra_config, {})

    def test_custom_valid_config(self):
        config = GatewayConfig(
            provider="custom",
            model="custom-v1",
            endpoint="http://localhost:8080",
            timeout=15.0,
            extra_config={"api_version": "v1"},
        )
        self.assertEqual(config.provider, "custom")
        self.assertEqual(config.model, "custom-v1")
        self.assertEqual(config.endpoint, "http://localhost:8080")
        self.assertEqual(config.timeout, 15.0)
        self.assertEqual(config.extra_config, {"api_version": "v1"})

    def test_invalid_provider_raises(self):
        with self.assertRaises(ProviderConfigurationError):
            GatewayConfig(provider="")
        with self.assertRaises(ProviderConfigurationError):
            GatewayConfig(provider=None)  # type: ignore

    def test_invalid_model_raises(self):
        with self.assertRaises(ProviderConfigurationError):
            GatewayConfig(model="")
        with self.assertRaises(ProviderConfigurationError):
            GatewayConfig(model=None)  # type: ignore

    def test_invalid_timeout_raises(self):
        with self.assertRaises(ProviderConfigurationError):
            GatewayConfig(timeout=0)
        with self.assertRaises(ProviderConfigurationError):
            GatewayConfig(timeout=-5.0)
        with self.assertRaises(ProviderConfigurationError):
            GatewayConfig(timeout="not-a-number")  # type: ignore

    def test_from_env(self):
        env_vars = {
            "AI_PROVIDER": "env-provider",
            "AI_MODEL": "env-model",
            "AI_ENDPOINT": "http://env-endpoint:8000",
            "AI_TIMEOUT": "45.0",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            config = GatewayConfig.from_env()
            self.assertEqual(config.provider, "env-provider")
            self.assertEqual(config.model, "env-model")
            self.assertEqual(config.endpoint, "http://env-endpoint:8000")
            self.assertEqual(config.timeout, 45.0)

    def test_from_env_invalid_timeout_raises(self):
        with patch.dict(os.environ, {"AI_TIMEOUT": "invalid"}):
            with self.assertRaises(ProviderConfigurationError):
                GatewayConfig.from_env()

    @override_settings(
        AI_GATEWAY={
            "PROVIDER": "settings-mock",
            "MODEL": "settings-model",
            "ENDPOINT": "http://settings-host:11434",
            "TIMEOUT": 60.0,
            "CUSTOM_KEY": "custom_value",
        }
    )
    def test_from_settings(self):
        config = GatewayConfig.from_settings()
        self.assertEqual(config.provider, "settings-mock")
        self.assertEqual(config.model, "settings-model")
        self.assertEqual(config.endpoint, "http://settings-host:11434")
        self.assertEqual(config.timeout, 60.0)
        self.assertEqual(config.extra_config.get("CUSTOM_KEY"), "custom_value")

    @override_settings(AI_GATEWAY="not-a-dict")
    def test_from_settings_non_dict_raises(self):
        with self.assertRaises(ProviderConfigurationError) as ctx:
            GatewayConfig.from_settings()
        self.assertIn("must be a dictionary", str(ctx.exception))

    @override_settings(AI_GATEWAY=None)
    def test_from_settings_none_raises(self):
        with self.assertRaises(ProviderConfigurationError) as ctx:
            GatewayConfig.from_settings()
        self.assertIn("must be a dictionary", str(ctx.exception))

    @override_settings(AI_GATEWAY={"PROVIDER": ""})
    def test_from_settings_invalid_provider_raises(self):
        with self.assertRaises(ProviderConfigurationError):
            GatewayConfig.from_settings()

    @override_settings(AI_GATEWAY={"MODEL": ""})
    def test_from_settings_invalid_model_raises(self):
        with self.assertRaises(ProviderConfigurationError):
            GatewayConfig.from_settings()

    @override_settings(AI_GATEWAY={"TIMEOUT": "invalid-num"})
    def test_from_settings_invalid_timeout_string_raises(self):
        with self.assertRaises(ProviderConfigurationError):
            GatewayConfig.from_settings()

    @override_settings(AI_GATEWAY={"TIMEOUT": -5.0})
    def test_from_settings_negative_timeout_raises(self):
        with self.assertRaises(ProviderConfigurationError):
            GatewayConfig.from_settings()

    def test_from_settings_fallback_when_settings_not_configured(self):
        from django.conf import settings
        from unittest.mock import PropertyMock

        with patch.dict(os.environ, {"AI_PROVIDER": "unconfigured-env-mock"}):
            with patch.object(type(settings), "configured", new_callable=PropertyMock, return_value=False):
                config = GatewayConfig.from_settings()
                self.assertEqual(config.provider, "unconfigured-env-mock")

    def test_from_settings_fallback_when_improperly_configured(self):
        from django.conf import settings
        from django.core.exceptions import ImproperlyConfigured
        from unittest.mock import PropertyMock

        with patch.dict(os.environ, {"AI_PROVIDER": "improperly-configured-env-mock"}):
            with patch.object(
                type(settings),
                "configured",
                new_callable=PropertyMock,
                side_effect=ImproperlyConfigured("Settings not configured"),
            ):
                config = GatewayConfig.from_settings()
                self.assertEqual(config.provider, "improperly-configured-env-mock")

    def test_from_settings_fallback_when_ai_gateway_missing(self):
        from django.conf import settings

        orig = getattr(settings, "AI_GATEWAY", None)
        self.addCleanup(lambda: setattr(settings, "AI_GATEWAY", orig) if orig is not None else None)
        del settings.AI_GATEWAY

        with patch.dict(os.environ, {"AI_PROVIDER": "missing-setting-mock"}):
            config = GatewayConfig.from_settings()
            self.assertEqual(config.provider, "missing-setting-mock")
