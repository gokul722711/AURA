"""Tests for Model Gateway data types and contracts."""

from django.test import TestCase

from gateway.exceptions import InvalidRequestError
from gateway.types import (
    CAPABILITY_GENERATION,
    CAPABILITY_STREAMING,
    CAPABILITY_STRUCTURED_OUTPUT,
    GenerationRequest,
    GenerationResponse,
    Message,
    ProviderMetadata,
    StreamChunk,
    StructuredOutputRequest,
    StructuredOutputResponse,
    UsageInfo,
)


class TypesTests(TestCase):
    """Unit tests for gateway message, request, response, and metadata types."""

    def test_message_creation_and_to_dict(self):
        msg = Message(role="user", content="Hello AURA")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "Hello AURA")
        self.assertEqual(msg.to_dict(), {"role": "user", "content": "Hello AURA"})

    def test_generation_request_with_message_objects(self):
        req = GenerationRequest(
            messages=[Message(role="user", content="Hi")],
            model="custom-model",
            temperature=0.7,
            max_tokens=100,
        )
        self.assertEqual(len(req.messages), 1)
        self.assertEqual(req.messages[0].role, "user")
        self.assertEqual(req.model, "custom-model")
        self.assertEqual(req.temperature, 0.7)
        self.assertEqual(req.max_tokens, 100)

    def test_generation_request_with_dict_messages(self):
        req = GenerationRequest(
            messages=[
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": "What is 2+2?"},
            ]
        )
        self.assertEqual(len(req.messages), 2)
        self.assertEqual(req.messages[0].role, "system")
        self.assertEqual(req.messages[0].content, "Be concise")
        self.assertEqual(req.messages[1].role, "user")
        self.assertEqual(req.messages[1].content, "What is 2+2?")

    def test_generation_request_empty_messages_raises(self):
        with self.assertRaises(InvalidRequestError):
            GenerationRequest(messages=[])

    def test_generation_request_invalid_messages_container_raises(self):
        with self.assertRaises(InvalidRequestError):
            GenerationRequest(messages="not a list")  # type: ignore

    def test_generation_request_invalid_message_element_raises(self):
        with self.assertRaises(InvalidRequestError):
            GenerationRequest(messages=[123])  # type: ignore

    def test_generation_request_missing_role_raises(self):
        with self.assertRaises(InvalidRequestError):
            GenerationRequest(messages=[{"content": "missing role"}])

    def test_generation_request_missing_content_raises(self):
        with self.assertRaises(InvalidRequestError):
            GenerationRequest(messages=[{"role": "user"}])

    def test_generation_request_invalid_temperature_raises(self):
        with self.assertRaises(InvalidRequestError):
            GenerationRequest(
                messages=[Message(role="user", content="Hi")],
                temperature=2.5,
            )
        with self.assertRaises(InvalidRequestError):
            GenerationRequest(
                messages=[Message(role="user", content="Hi")],
                temperature=-0.1,
            )

    def test_generation_request_invalid_max_tokens_raises(self):
        with self.assertRaises(InvalidRequestError):
            GenerationRequest(
                messages=[Message(role="user", content="Hi")],
                max_tokens=0,
            )
        with self.assertRaises(InvalidRequestError):
            GenerationRequest(
                messages=[Message(role="user", content="Hi")],
                max_tokens=-10,
            )

    def test_structured_output_request_validation(self):
        req = StructuredOutputRequest(
            messages=[Message(role="user", content="Extract data")],
            schema={"type": "object", "properties": {"name": {"type": "string"}}},
        )
        self.assertEqual(len(req.messages), 1)
        self.assertIsNotNone(req.schema)

        with self.assertRaises(InvalidRequestError):
            StructuredOutputRequest(messages=[], schema={})

        with self.assertRaises(InvalidRequestError):
            StructuredOutputRequest(
                messages=[Message(role="user", content="Hi")],
                temperature=3.0,
            )

    def test_provider_metadata_supports(self):
        meta = ProviderMetadata(
            provider="mock",
            model="mock-model",
            capabilities=(CAPABILITY_GENERATION, CAPABILITY_STREAMING),
        )
        self.assertTrue(meta.supports(CAPABILITY_GENERATION))
        self.assertTrue(meta.supports(CAPABILITY_STREAMING))
        self.assertFalse(meta.supports(CAPABILITY_STRUCTURED_OUTPUT))
        self.assertFalse(meta.supports("unknown_capability"))

    def test_response_and_chunk_dataclasses(self):
        usage = UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        resp = GenerationResponse(
            text="hello",
            provider="mock",
            model="m1",
            usage=usage,
            finish_reason="stop",
        )
        self.assertEqual(resp.text, "hello")
        self.assertEqual(resp.usage.total_tokens, 15)

        chunk = StreamChunk(text="hello", index=0, finish_reason=None)
        self.assertEqual(chunk.text, "hello")
        self.assertEqual(chunk.index, 0)
        self.assertIsNone(chunk.finish_reason)

        struct_resp = StructuredOutputResponse(
            data={"result": 42},
            raw_text='{"result": 42}',
            provider="mock",
            model="m1",
            usage=usage,
        )
        self.assertEqual(struct_resp.data, {"result": 42})
        self.assertEqual(struct_resp.raw_text, '{"result": 42}')
