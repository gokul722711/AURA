"""Tests for MockLLMProvider."""

import json

from django.test import TestCase

from gateway.providers.mock import MockLLMProvider
from gateway.types import (
    CAPABILITY_GENERATION,
    CAPABILITY_STREAMING,
    CAPABILITY_STRUCTURED_OUTPUT,
    GenerationRequest,
    Message,
    StructuredOutputRequest,
)


class MockLLMProviderTests(TestCase):
    """Unit tests for MockLLMProvider functionality."""

    def setUp(self):
        self.provider = MockLLMProvider(model="test-mock-model")

    def test_metadata(self):
        meta = self.provider.metadata()
        self.assertEqual(meta.provider, "mock")
        self.assertEqual(meta.model, "test-mock-model")
        self.assertTrue(meta.supports(CAPABILITY_GENERATION))
        self.assertTrue(meta.supports(CAPABILITY_STREAMING))
        self.assertTrue(meta.supports(CAPABILITY_STRUCTURED_OUTPUT))

    def test_metadata_with_unsupported_capabilities(self):
        custom_provider = MockLLMProvider(
            model="limited-mock",
            unsupported_capabilities=[CAPABILITY_STREAMING],
        )
        meta = custom_provider.metadata()
        self.assertTrue(meta.supports(CAPABILITY_GENERATION))
        self.assertFalse(meta.supports(CAPABILITY_STREAMING))
        self.assertTrue(meta.supports(CAPABILITY_STRUCTURED_OUTPUT))

    def test_generate_deterministic_response(self):
        req = GenerationRequest(
            messages=[Message(role="user", content="Explain quantum computing")]
        )
        resp = self.provider.generate(req)
        self.assertEqual(resp.provider, "mock")
        self.assertEqual(resp.model, "test-mock-model")
        self.assertEqual(resp.text, "Mock response to: Explain quantum computing")
        self.assertEqual(resp.finish_reason, "stop")
        self.assertGreater(resp.usage.prompt_tokens, 0)
        self.assertGreater(resp.usage.completion_tokens, 0)
        self.assertEqual(
            resp.usage.total_tokens,
            resp.usage.prompt_tokens + resp.usage.completion_tokens,
        )

    def test_generate_model_override(self):
        req = GenerationRequest(
            messages=[Message(role="user", content="Hi")],
            model="override-model",
        )
        resp = self.provider.generate(req)
        self.assertEqual(resp.model, "override-model")

    def test_generate_canned_response(self):
        canned_provider = MockLLMProvider(
            canned_responses={"Special prompt": "Special canned answer"}
        )
        req = GenerationRequest(
            messages=[Message(role="user", content="Special prompt")]
        )
        resp = canned_provider.generate(req)
        self.assertEqual(resp.text, "Special canned answer")

    def test_generate_default_response(self):
        default_provider = MockLLMProvider(default_response="Static default")
        req = GenerationRequest(
            messages=[Message(role="user", content="Any prompt")]
        )
        resp = default_provider.generate(req)
        self.assertEqual(resp.text, "Static default")

    def test_streaming_chunks(self):
        req = GenerationRequest(
            messages=[Message(role="user", content="Count to three")]
        )
        chunks = list(self.provider.stream(req))

        self.assertGreater(len(chunks), 0)
        # Check chunk indexes are sequential
        for idx, chunk in enumerate(chunks):
            self.assertEqual(chunk.index, idx)
            if idx == len(chunks) - 1:
                self.assertEqual(chunk.finish_reason, "stop")
            else:
                self.assertIsNone(chunk.finish_reason)

        # Concatenating chunk texts should match expected full text
        reconstructed = "".join(chunk.text for chunk in chunks)
        self.assertEqual(reconstructed, "Mock response to: Count to three")

    def test_structured_output_without_schema(self):
        req = StructuredOutputRequest(
            messages=[Message(role="user", content="Analyze this")]
        )
        resp = self.provider.structured_output(req)
        self.assertEqual(resp.provider, "mock")
        self.assertEqual(resp.model, "test-mock-model")
        self.assertEqual(resp.finish_reason, "stop")
        self.assertIsInstance(resp.data, dict)
        self.assertEqual(resp.data.get("status"), "ok")
        # raw_text should be valid JSON matching data
        parsed = json.loads(resp.raw_text)
        self.assertEqual(parsed, resp.data)

    def test_structured_output_with_schema(self):
        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "score": {"type": "integer"},
                "verified": {"type": "boolean"},
                "tags": {"type": "array"},
            },
        }
        req = StructuredOutputRequest(
            messages=[Message(role="user", content="Evaluate")],
            schema=schema,
            model="schema-model",
        )
        resp = self.provider.structured_output(req)
        self.assertEqual(resp.model, "schema-model")
        self.assertIsInstance(resp.data, dict)
        self.assertEqual(resp.data.get("summary"), "mock_summary_value")
        self.assertEqual(resp.data.get("score"), 42)
        self.assertEqual(resp.data.get("verified"), True)
        self.assertEqual(resp.data.get("tags"), ["mock_item_1", "mock_item_2"])

    def test_simulated_failure(self):
        failing_provider = MockLLMProvider(
            simulated_failure=RuntimeError("Simulated connection drop")
        )
        req = GenerationRequest(
            messages=[Message(role="user", content="Hi")]
        )
        with self.assertRaises(RuntimeError):
            failing_provider.generate(req)

        with self.assertRaises(RuntimeError):
            list(failing_provider.stream(req))

        struct_req = StructuredOutputRequest(
            messages=[Message(role="user", content="Hi")]
        )
        with self.assertRaises(RuntimeError):
            failing_provider.structured_output(struct_req)
