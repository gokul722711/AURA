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

    def test_provider_timeout_capped_by_configured_ai_timeout(self) -> None:
        """Verify request timeout metadata is capped by configured provider timeout."""
        choice_mock = MagicMock()
        choice_mock.message.content = "Response"
        choice_mock.finish_reason = "stop"
        self.mock_client.chat.completions.create.return_value = MagicMock(
            choices=[choice_mock], usage=None
        )

        # Provider configured with timeout=30.0, request asks for 120.0
        req = GenerationRequest(
            messages=[Message(role="user", content="Hello")],
            metadata={"timeout": 120.0},
        )
        self.provider.generate(req)

        call_kwargs = self.mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs.get("timeout"), 30.0)

    def test_provider_timeout_smaller_when_remaining_budget_smaller(self) -> None:
        """Verify request timeout becomes smaller when remaining agent budget is smaller."""
        choice_mock = MagicMock()
        choice_mock.message.content = "Response"
        choice_mock.finish_reason = "stop"
        self.mock_client.chat.completions.create.return_value = MagicMock(
            choices=[choice_mock], usage=None
        )

        # Provider configured with timeout=30.0, remaining budget is 12.5s
        req = GenerationRequest(
            messages=[Message(role="user", content="Hello")],
            metadata={"timeout": 12.5},
        )
        self.provider.generate(req)

        call_kwargs = self.mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs.get("timeout"), 12.5)

    def test_provider_timeout_preserves_configured_timeout_when_no_metadata(self) -> None:
        """Verify no timeout override is passed to create when metadata has no timeout."""
        choice_mock = MagicMock()
        choice_mock.message.content = "Response"
        choice_mock.finish_reason = "stop"
        self.mock_client.chat.completions.create.return_value = MagicMock(
            choices=[choice_mock], usage=None
        )

        req = GenerationRequest(
            messages=[Message(role="user", content="Hello")],
            metadata={},
        )
        self.provider.generate(req)

        call_kwargs = self.mock_client.chat.completions.create.call_args.kwargs
        self.assertNotIn("timeout", call_kwargs)

    def test_provider_retries_bounded_by_remaining_budget(self) -> None:
        """Verify client max_retries is constrained by remaining budget during execution and restored."""
        self.mock_client.max_retries = 2

        recorded_retries = []

        def fake_create(**kwargs):
            recorded_retries.append(self.mock_client.max_retries)
            choice = MagicMock()
            choice.message.content = "Ok"
            choice.finish_reason = "stop"
            return MagicMock(choices=[choice], usage=None)

        self.mock_client.chat.completions.create.side_effect = fake_create

        # Budget is 15.0s, provider timeout is 30.0s -> effective timeout is 15.0s.
        # int(15 // 15) = 1 attempt. Allowed retries = 0.
        req = GenerationRequest(
            messages=[Message(role="user", content="Hi")],
            metadata={"timeout": 15.0},
        )
        self.provider.generate(req)

        self.assertEqual(recorded_retries, [0])
        # Restored after call
        self.assertEqual(self.mock_client.max_retries, 2)

    def test_structured_output_uses_timeout_budget(self) -> None:
        """Verify structured output calls propagate the timeout budget."""
        choice_mock = MagicMock()
        choice_mock.message.content = '{"decision": "continue", "query": "quantum"}'
        choice_mock.finish_reason = "stop"
        self.mock_client.chat.completions.create.return_value = MagicMock(
            choices=[choice_mock], usage=None
        )

        req = StructuredOutputRequest(
            messages=[Message(role="user", content="Plan next step")],
            schema={"properties": {"decision": {"type": "string"}}},
            metadata={"timeout": 8.0},
        )
        resp = self.provider.structured_output(req)

        self.assertEqual(resp.data, {"decision": "continue", "query": "quantum"})
        call_kwargs = self.mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs.get("timeout"), 8.0)

    def test_existing_timeout_error_behavior_remains_intact(self) -> None:
        """Verify timeout or connection errors are wrapped in GenerationError."""
        self.mock_client.chat.completions.create.side_effect = TimeoutError("Request timed out")

        req = GenerationRequest(
            messages=[Message(role="user", content="Hello")],
            metadata={"timeout": 5.0},
        )
        with self.assertRaises(GenerationError) as ctx:
            self.provider.generate(req)
        self.assertIn("NVIDIA model generation failed", str(ctx.exception))

    def test_provider_retries_accounts_for_backoff_delay(self) -> None:
        """Verify retries account for backoff sleep delay so retries never exceed budget."""
        self.mock_client.max_retries = 2
        recorded_retries = []

        def fake_create(**kwargs):
            recorded_retries.append(self.mock_client.max_retries)
            choice = MagicMock()
            choice.message.content = "Ok"
            choice.finish_reason = "stop"
            return MagicMock(choices=[choice], usage=None)

        self.mock_client.chat.completions.create.side_effect = fake_create

        # Provider timeout is 30.0s.
        # Budget = 60.0s: Attempt 1 takes 30s. Remaining = 30s.
        # A retry takes 30s timeout + 1s backoff = 31s > 30s remaining.
        # Therefore allowed retries must be 0 to prevent exceeding the 60s budget.
        req_60 = GenerationRequest(
            messages=[Message(role="user", content="Hi")],
            metadata={"timeout": 60.0},
        )
        self.provider.generate(req_60)
        self.assertEqual(recorded_retries[-1], 0)

        # Budget = 65.0s: Attempt 1 takes 30s. Remaining = 35s >= 31s.
        # 1 retry fits safely within the remaining budget.
        req_65 = GenerationRequest(
            messages=[Message(role="user", content="Hi")],
            metadata={"timeout": 65.0},
        )
        self.provider.generate(req_65)
        self.assertEqual(recorded_retries[-1], 1)

        # Budget = 95.0s: Attempt 1 (30s) + 2 retries (2 * 31s = 62s) = 92s <= 95s.
        # 2 retries fit safely within the remaining budget.
        req_95 = GenerationRequest(
            messages=[Message(role="user", content="Hi")],
            metadata={"timeout": 95.0},
        )
        self.provider.generate(req_95)
        self.assertEqual(recorded_retries[-1], 2)

    def test_provider_handles_nan_and_inf_timeout_gracefully(self) -> None:
        """Verify NaN or Inf timeout metadata does not crash and preserves default behavior."""
        choice_mock = MagicMock()
        choice_mock.message.content = "Response"
        choice_mock.finish_reason = "stop"
        self.mock_client.chat.completions.create.return_value = MagicMock(
            choices=[choice_mock], usage=None
        )

        for invalid_val in [float("nan"), float("inf"), float("-inf")]:
            req = GenerationRequest(
                messages=[Message(role="user", content="Hi")],
                metadata={"timeout": invalid_val},
            )
            self.provider.generate(req)
            call_kwargs = self.mock_client.chat.completions.create.call_args.kwargs
            self.assertNotIn("timeout", call_kwargs)

    def test_provider_handles_zero_and_negative_timeout_gracefully(self) -> None:
        """Verify non-positive timeout metadata sets minimal positive timeout and 0 retries."""
        choice_mock = MagicMock()
        choice_mock.message.content = "Response"
        choice_mock.finish_reason = "stop"
        self.mock_client.chat.completions.create.return_value = MagicMock(
            choices=[choice_mock], usage=None
        )
        self.mock_client.max_retries = 2
        recorded_retries = []

        def fake_create(**kwargs):
            recorded_retries.append(self.mock_client.max_retries)
            choice = MagicMock()
            choice.message.content = "Ok"
            choice.finish_reason = "stop"
            return MagicMock(choices=[choice], usage=None)

        self.mock_client.chat.completions.create.side_effect = fake_create

        for non_positive_val in [0.0, -10.0]:
            req = GenerationRequest(
                messages=[Message(role="user", content="Hi")],
                metadata={"timeout": non_positive_val},
            )
            self.provider.generate(req)
            call_kwargs = self.mock_client.chat.completions.create.call_args.kwargs
            self.assertEqual(call_kwargs.get("timeout"), 0.001)
            self.assertEqual(recorded_retries[-1], 0)
