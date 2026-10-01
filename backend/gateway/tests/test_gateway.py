"""Tests for Model Gateway."""

from unittest.mock import patch

from django.test import TestCase, override_settings

from gateway.config import GatewayConfig
from gateway.exceptions import (
    GenerationError,
    InvalidRequestError,
    ProviderConfigurationError,
    ProviderUnavailableError,
    UnsupportedCapabilityError,
)
from gateway.gateway import ModelGateway, _sanitize_error_message, get_gateway
from gateway.providers.mock import MockLLMProvider
from gateway.registry import get_provider_class, register_provider
from gateway.types import (
    CAPABILITY_GENERATION,
    CAPABILITY_STREAMING,
    CAPABILITY_STRUCTURED_OUTPUT,
    GenerationRequest,
    Message,
    StructuredOutputRequest,
)


@override_settings(
    AI_GATEWAY={
        "PROVIDER": "mock",
        "MODEL": "mock-model",
        "ENDPOINT": "",
        "TIMEOUT": 30.0,
    }
)
class ModelGatewayTests(TestCase):
    """Unit tests for ModelGateway."""

    def test_default_initialization_uses_mock(self):
        gateway = ModelGateway()
        self.assertIsInstance(gateway.provider, MockLLMProvider)
        self.assertEqual(gateway.config.provider, "mock")
        meta = gateway.metadata()
        self.assertEqual(meta.provider, "mock")
        self.assertEqual(meta.model, "mock-model")

    def test_initialization_with_custom_provider(self):
        custom_provider = MockLLMProvider(model="custom-mock-42")
        gateway = ModelGateway(provider=custom_provider)
        self.assertEqual(gateway.provider, custom_provider)
        self.assertEqual(gateway.metadata().model, "custom-mock-42")

    def test_initialization_with_custom_config(self):
        config = GatewayConfig(provider="mock", model="configured-model")
        gateway = ModelGateway(config=config)
        self.assertEqual(gateway.metadata().model, "configured-model")

    def test_get_gateway_factory(self):
        gateway = get_gateway()
        self.assertIsInstance(gateway, ModelGateway)

    def test_metadata_and_get_metadata(self):
        gateway = ModelGateway()
        meta1 = gateway.metadata()
        meta2 = gateway.get_metadata()
        self.assertEqual(meta1, meta2)
        self.assertEqual(meta1.provider, "mock")

    def test_generation_delegation(self):
        gateway = ModelGateway()
        req = GenerationRequest(
            messages=[Message(role="user", content="Test prompt")]
        )
        resp = gateway.generate(req)
        self.assertEqual(resp.text, "Mock response to: Test prompt")
        self.assertEqual(resp.provider, "mock")
        self.assertEqual(resp.model, "mock-model")

    def test_generation_invalid_request_type_raises(self):
        gateway = ModelGateway()
        with self.assertRaises(InvalidRequestError):
            gateway.generate("not a GenerationRequest")  # type: ignore

    def test_streaming_delegation(self):
        gateway = ModelGateway()
        req = GenerationRequest(
            messages=[Message(role="user", content="Stream this message")]
        )
        chunks = list(gateway.stream(req))
        self.assertGreater(len(chunks), 0)
        reconstructed = "".join(chunk.text for chunk in chunks)
        self.assertEqual(reconstructed, "Mock response to: Stream this message")
        self.assertEqual(chunks[-1].finish_reason, "stop")

    def test_streaming_invalid_request_type_raises(self):
        gateway = ModelGateway()
        with self.assertRaises(InvalidRequestError):
            list(gateway.stream("not a GenerationRequest"))  # type: ignore

    def test_structured_output_delegation(self):
        gateway = ModelGateway()
        req = StructuredOutputRequest(
            messages=[Message(role="user", content="Structured prompt")],
            schema={"properties": {"field": {"type": "string"}}},
        )
        resp = gateway.structured_output(req)
        self.assertEqual(resp.provider, "mock")
        self.assertEqual(resp.data, {"field": "mock_field_value"})
        self.assertIn("mock_field_value", resp.raw_text)

    def test_structured_output_invalid_request_type_raises(self):
        gateway = ModelGateway()
        with self.assertRaises(InvalidRequestError):
            gateway.structured_output("not a StructuredOutputRequest")  # type: ignore

    def test_unsupported_generation_capability_raises(self):
        provider = MockLLMProvider(unsupported_capabilities=[CAPABILITY_GENERATION])
        gateway = ModelGateway(provider=provider)
        req = GenerationRequest(
            messages=[Message(role="user", content="Hi")]
        )
        with self.assertRaises(UnsupportedCapabilityError):
            gateway.generate(req)

    def test_unsupported_streaming_capability_raises(self):
        provider = MockLLMProvider(unsupported_capabilities=[CAPABILITY_STREAMING])
        gateway = ModelGateway(provider=provider)
        req = GenerationRequest(
            messages=[Message(role="user", content="Hi")]
        )
        with self.assertRaises(UnsupportedCapabilityError):
            list(gateway.stream(req))

    def test_unsupported_structured_output_capability_raises(self):
        provider = MockLLMProvider(unsupported_capabilities=[CAPABILITY_STRUCTURED_OUTPUT])
        gateway = ModelGateway(provider=provider)
        req = StructuredOutputRequest(
            messages=[Message(role="user", content="Hi")]
        )
        with self.assertRaises(UnsupportedCapabilityError):
            gateway.structured_output(req)

    def test_invalid_provider_configuration_raises(self):
        config = GatewayConfig(provider="nonexistent_provider_xyz")
        with self.assertRaises(ProviderUnavailableError):
            ModelGateway(config=config)

    def test_generation_error_normalization(self):
        failing_provider = MockLLMProvider(
            simulated_failure=ConnectionResetError("Socket reset by peer")
        )
        gateway = ModelGateway(provider=failing_provider)
        req = GenerationRequest(
            messages=[Message(role="user", content="Hi")]
        )
        with self.assertRaises(GenerationError) as ctx:
            gateway.generate(req)
        self.assertIn("Socket reset by peer", str(ctx.exception))

    def test_streaming_error_normalization(self):
        failing_provider = MockLLMProvider(
            simulated_failure=ConnectionResetError("Stream disconnect")
        )
        gateway = ModelGateway(provider=failing_provider)
        req = GenerationRequest(
            messages=[Message(role="user", content="Hi")]
        )
        with self.assertRaises(GenerationError) as ctx:
            list(gateway.stream(req))
        self.assertIn("Stream disconnect", str(ctx.exception))

    def test_structured_output_error_normalization(self):
        failing_provider = MockLLMProvider(
            simulated_failure=ValueError("Parsing failed")
        )
        gateway = ModelGateway(provider=failing_provider)
        req = StructuredOutputRequest(
            messages=[Message(role="user", content="Hi")]
        )
        with self.assertRaises(GenerationError) as ctx:
            gateway.structured_output(req)
        self.assertIn("Parsing failed", str(ctx.exception))

    def test_error_message_sanitization_masks_secrets(self):
        raw_err = "Auth failure: api_key='sk-1234567890abcdef' or token=mysecrettoken12345"
        sanitized = _sanitize_error_message(raw_err)
        self.assertNotIn("sk-1234567890abcdef", sanitized)
        self.assertNotIn("mysecrettoken12345", sanitized)

    def test_register_provider_validation(self):
        class NotAProvider:
            pass

        with self.assertRaises(ProviderConfigurationError):
            register_provider("invalid", NotAProvider)  # type: ignore

        with self.assertRaises(ProviderConfigurationError):
            register_provider("invalid", "not-a-class")  # type: ignore

    def test_register_provider_empty_name_raises(self):
        with self.assertRaises(ProviderConfigurationError):
            register_provider("", MockLLMProvider)

        with self.assertRaises(ProviderConfigurationError):
            register_provider("   ", MockLLMProvider)

    def test_register_provider_non_string_name_raises(self):
        with self.assertRaises(ProviderConfigurationError):
            register_provider(None, MockLLMProvider)  # type: ignore

        with self.assertRaises(ProviderConfigurationError):
            register_provider(123, MockLLMProvider)  # type: ignore

    def test_register_provider_normalizes_name(self):
        class DummyProvider(MockLLMProvider):
            pass

        register_provider("  Custom-Dummy-Name  ", DummyProvider)
        self.assertEqual(get_provider_class("custom-dummy-name"), DummyProvider)
        self.assertEqual(get_provider_class("CUSTOM-DUMMY-NAME"), DummyProvider)
        self.assertEqual(get_provider_class("  custom-dummy-name  "), DummyProvider)
