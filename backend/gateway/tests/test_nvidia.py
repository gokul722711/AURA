"""Deterministic offline unit tests for NvidiaLLMProvider."""

from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase

from gateway.config import GatewayConfig
from gateway.exceptions import (
    GenerationError,
    ProviderConfigurationError,
)
from gateway.providers.nvidia import (
    DEFAULT_NVIDIA_ENDPOINT,
    DEFAULT_NVIDIA_MODEL,
    NvidiaLLMProvider,
)
from gateway.registry import create_provider, get_provider_class
from gateway.types import (
    CAPABILITY_GENERATION,
    CAPABILITY_STREAMING,
    CAPABILITY_STRUCTURED_OUTPUT,
    GenerationRequest,
    Message,
    StructuredOutputRequest,
)


class NvidiaLLMProviderTests(SimpleTestCase):
    """Offline unit tests for NvidiaLLMProvider using mocked OpenAI client."""

    def setUp(self) -> None:
        self.mock_client = MagicMock()
        self.provider = NvidiaLLMProvider(
            model=DEFAULT_NVIDIA_MODEL,
            endpoint=DEFAULT_NVIDIA_ENDPOINT,
            client=self.mock_client,
        )

    def test_default_initialization(self) -> None:
        self.assertEqual(self.provider.model_name, DEFAULT_NVIDIA_MODEL)
        self.assertEqual(self.provider.endpoint, DEFAULT_NVIDIA_ENDPOINT)
        self.assertEqual(self.provider.timeout, 30.0)

    def test_custom_initialization(self) -> None:
        custom_provider = NvidiaLLMProvider(
            model="custom-nemotron",
            endpoint="https://custom.endpoint.com/v1",
            timeout=45.0,
            client=self.mock_client,
        )
        self.assertEqual(custom_provider.model_name, "custom-nemotron")
        self.assertEqual(custom_provider.endpoint, "https://custom.endpoint.com/v1")
        self.assertEqual(custom_provider.timeout, 45.0)

    def test_missing_api_key_raises_configuration_error(self) -> None:
        with patch.dict("os.environ", {"AI_API_KEY": ""}, clear=True):
            with self.assertRaises(ProviderConfigurationError) as ctx:
                NvidiaLLMProvider(api_key="")
            self.assertIn("NVIDIA API key not configured", str(ctx.exception))

    def test_metadata(self) -> None:
        meta = self.provider.metadata()
        self.assertEqual(meta.provider, "nvidia")
        self.assertEqual(meta.model, DEFAULT_NVIDIA_MODEL)
        self.assertTrue(meta.supports(CAPABILITY_GENERATION))
        self.assertTrue(meta.supports(CAPABILITY_STREAMING))
        self.assertTrue(meta.supports(CAPABILITY_STRUCTURED_OUTPUT))

    def test_generate_success(self) -> None:
        choice_mock = MagicMock()
        choice_mock.message.content = "NVIDIA generated text"
        choice_mock.finish_reason = "stop"

        completion_mock = MagicMock()
        completion_mock.choices = [choice_mock]
        completion_mock.usage.prompt_tokens = 10
        completion_mock.usage.completion_tokens = 20
        completion_mock.usage.total_tokens = 30

        self.mock_client.chat.completions.create.return_value = completion_mock

        req = GenerationRequest(
            messages=[Message(role="user", content="Explain quantum computing")],
            temperature=0.2,
            max_tokens=150,
        )
        resp = self.provider.generate(req)

        self.assertEqual(resp.text, "NVIDIA generated text")
        self.assertEqual(resp.provider, "nvidia")
        self.assertEqual(resp.model, DEFAULT_NVIDIA_MODEL)
        self.assertEqual(resp.finish_reason, "stop")
        self.assertEqual(resp.usage.prompt_tokens, 10)
        self.assertEqual(resp.usage.completion_tokens, 20)
        self.assertEqual(resp.usage.total_tokens, 30)

        self.mock_client.chat.completions.create.assert_called_once_with(
            model=DEFAULT_NVIDIA_MODEL,
            messages=[{"role": "user", "content": "Explain quantum computing"}],
            temperature=0.2,
            max_tokens=150,
        )

    def test_generate_failure_wraps_in_generation_error(self) -> None:
        self.mock_client.chat.completions.create.side_effect = RuntimeError("Network timeout")

        req = GenerationRequest(messages=[Message(role="user", content="Hello")])
        with self.assertRaises(GenerationError) as ctx:
            self.provider.generate(req)
        self.assertIn("NVIDIA model generation failed", str(ctx.exception))

    def test_streaming_success(self) -> None:
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock(delta=MagicMock(content="Hello "), finish_reason=None)]
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock(delta=MagicMock(content="world!"), finish_reason="stop")]

        self.mock_client.chat.completions.create.return_value = iter([chunk1, chunk2])

        req = GenerationRequest(messages=[Message(role="user", content="Stream hi")])
        chunks = list(self.provider.stream(req))

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].text, "Hello ")
        self.assertIsNone(chunks[0].finish_reason)
        self.assertEqual(chunks[1].text, "world!")
        self.assertEqual(chunks[1].finish_reason, "stop")

    def test_streaming_failure_wraps_in_generation_error(self) -> None:
        self.mock_client.chat.completions.create.side_effect = RuntimeError("Stream connection dropped")

        req = GenerationRequest(messages=[Message(role="user", content="Stream hi")])
        with self.assertRaises(GenerationError) as ctx:
            list(self.provider.stream(req))
        self.assertIn("NVIDIA streaming failed", str(ctx.exception))

    def test_structured_output_success_json_parsing(self) -> None:
        choice_mock = MagicMock()
        choice_mock.message.content = '```json\n{"decision": "continue", "query": "latest research"}\n```'
        choice_mock.finish_reason = "stop"

        completion_mock = MagicMock()
        completion_mock.choices = [choice_mock]
        completion_mock.usage.prompt_tokens = 15
        completion_mock.usage.completion_tokens = 15
        completion_mock.usage.total_tokens = 30

        self.mock_client.chat.completions.create.return_value = completion_mock

        req = StructuredOutputRequest(
            messages=[Message(role="user", content="Next step?")],
            schema={"properties": {"decision": {"type": "string"}}},
        )
        resp = self.provider.structured_output(req)

        self.assertEqual(resp.provider, "nvidia")
        self.assertEqual(resp.data, {"decision": "continue", "query": "latest research"})
        call_kwargs = self.mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs.get("response_format"), {"type": "json_object"})
        self.assertEqual(
            call_kwargs.get("extra_body"),
            {"chat_template_kwargs": {"enable_thinking": False}},
        )

    def test_structured_output_sends_chat_template_kwargs_disable_thinking(self) -> None:
        """Verify structured_output passes chat_template_kwargs with enable_thinking=False."""
        choice_mock = MagicMock()
        choice_mock.message.content = '{"status": "ok"}'
        choice_mock.finish_reason = "stop"
        completion_mock = MagicMock()
        completion_mock.choices = [choice_mock]
        completion_mock.usage = None
        self.mock_client.chat.completions.create.return_value = completion_mock

        req = StructuredOutputRequest(
            messages=[Message(role="user", content="Ping")],
            schema={"properties": {"status": {"type": "string"}}},
            max_tokens=256,
            temperature=0.0,
        )
        self.provider.structured_output(req)

        call_kwargs = self.mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(
            call_kwargs.get("extra_body"),
            {"chat_template_kwargs": {"enable_thinking": False}},
        )
        self.assertEqual(call_kwargs.get("response_format"), {"type": "json_object"})
        self.assertEqual(call_kwargs.get("max_tokens"), 256)
        self.assertEqual(call_kwargs.get("temperature"), 0.0)

    def test_structured_output_invalid_json_raises_generation_error(self) -> None:
        choice_mock = MagicMock()
        choice_mock.message.content = "Not a json response"
        choice_mock.finish_reason = "stop"

        completion_mock = MagicMock()
        completion_mock.choices = [choice_mock]

        self.mock_client.chat.completions.create.return_value = completion_mock

        req = StructuredOutputRequest(messages=[Message(role="user", content="Next step?")])
        with self.assertRaises(GenerationError) as ctx:
            self.provider.structured_output(req)
        self.assertIn("Failed to parse structured JSON output", str(ctx.exception))

    def test_registry_lookup_and_create_provider(self) -> None:
        cls = get_provider_class("nvidia")
        self.assertIs(cls, NvidiaLLMProvider)

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_openai_cls.return_value = MagicMock()
            config = GatewayConfig(
                provider="nvidia",
                model="nvidia/nemotron-3-ultra-550b-a55b",
                endpoint="https://integrate.api.nvidia.com/v1",
                api_key="test-api-key-123",
                timeout=25.0,
            )
            instance = create_provider(config)
            self.assertIsInstance(instance, NvidiaLLMProvider)
            self.assertEqual(instance.model_name, "nvidia/nemotron-3-ultra-550b-a55b")
            self.assertEqual(instance.endpoint, "https://integrate.api.nvidia.com/v1")
            self.assertEqual(instance.timeout, 25.0)
            mock_openai_cls.assert_called_once_with(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key="test-api-key-123",
                timeout=25.0,
            )
