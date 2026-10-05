"""Unit tests for M15 Providers: OpenAICompatibleLLMProvider and OllamaLLMProvider."""

import json
from unittest.mock import MagicMock
from django.test import TestCase

from gateway.exceptions import GenerationError, TransientModelProviderError
from gateway.providers.ollama import OllamaLLMProvider, _normalize_ollama_endpoint
from gateway.providers.openai_compatible import OpenAICompatibleLLMProvider
from gateway.types import GenerationRequest, Message, StructuredOutputRequest


class OpenAICompatibleProviderTests(TestCase):
    """Tests for OpenAICompatibleLLMProvider using injected mock clients."""

    def test_metadata(self) -> None:
        provider = OpenAICompatibleLLMProvider(
            model="Qwen/Qwen2.5-7B",
            endpoint="http://localhost:8000/v1",
            client=MagicMock(),
        )
        meta = provider.metadata()
        self.assertEqual(meta.provider, "openai_compatible")
        self.assertEqual(meta.model, "Qwen/Qwen2.5-7B")
        self.assertTrue(meta.supports("generation"))
        self.assertTrue(meta.supports("streaming"))
        self.assertTrue(meta.supports("structured_output"))

    def test_generate(self) -> None:
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "OpenAI compatible response."
        mock_choice.finish_reason = "stop"
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.usage.prompt_tokens = 12
        mock_completion.usage.completion_tokens = 6
        mock_completion.usage.total_tokens = 18
        mock_client.chat.completions.create.return_value = mock_completion

        provider = OpenAICompatibleLLMProvider(
            model="test-model",
            endpoint="http://localhost:8000/v1",
            client=mock_client,
        )
        req = GenerationRequest(
            messages=[Message(role="user", content="Hello")],
            temperature=0.7,
            max_tokens=100,
        )
        resp = provider.generate(req)
        self.assertEqual(resp.text, "OpenAI compatible response.")
        self.assertEqual(resp.provider, "openai_compatible")
        self.assertEqual(resp.model, "test-model")
        self.assertEqual(resp.usage.total_tokens, 18)
        mock_client.chat.completions.create.assert_called_once_with(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=100,
        )

    def test_stream(self) -> None:
        mock_client = MagicMock()
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock(delta=MagicMock(content="Hello"), finish_reason=None)]
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock(delta=MagicMock(content=" world!"), finish_reason="stop")]
        mock_client.chat.completions.create.return_value = [chunk1, chunk2]

        provider = OpenAICompatibleLLMProvider(
            model="stream-model",
            endpoint="http://localhost:8000/v1",
            client=mock_client,
        )
        req = GenerationRequest(messages=[Message(role="user", content="Hi")])
        chunks = list(provider.stream(req))
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].text, "Hello")
        self.assertEqual(chunks[1].text, " world!")
        self.assertEqual(chunks[1].finish_reason, "stop")

    def test_structured_output(self) -> None:
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = '{"decision": "finish", "answer": "42"}'
        mock_choice.finish_reason = "stop"
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 10
        mock_completion.usage.total_tokens = 20
        mock_client.chat.completions.create.return_value = mock_completion

        provider = OpenAICompatibleLLMProvider(
            model="struct-model",
            endpoint="http://localhost:8000/v1",
            client=mock_client,
        )
        req = StructuredOutputRequest(
            messages=[Message(role="user", content="What is the answer?")],
            schema={"type": "object", "properties": {"answer": {"type": "string"}}},
        )
        resp = provider.structured_output(req)
        self.assertEqual(resp.data, {"decision": "finish", "answer": "42"})
        self.assertEqual(resp.provider, "openai_compatible")


class OllamaProviderTests(TestCase):
    """Tests for OllamaLLMProvider using injected mock HTTP clients."""

    def test_endpoint_normalization(self) -> None:
        self.assertEqual(_normalize_ollama_endpoint("http://localhost:11434/v1"), "http://localhost:11434")
        self.assertEqual(_normalize_ollama_endpoint("http://localhost:11434/v1/"), "http://localhost:11434")
        self.assertEqual(_normalize_ollama_endpoint("http://localhost:11434/"), "http://localhost:11434")
        self.assertEqual(_normalize_ollama_endpoint("http://my-ollama:11434"), "http://my-ollama:11434")

    def test_metadata(self) -> None:
        provider = OllamaLLMProvider(
            model="llama3.1:8b",
            endpoint="http://localhost:11434",
            client=MagicMock(),
        )
        meta = provider.metadata()
        self.assertEqual(meta.provider, "ollama")
        self.assertEqual(meta.model, "llama3.1:8b")
        self.assertTrue(meta.supports("generation"))

    def test_generate(self) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "llama3.1:8b",
            "message": {"role": "assistant", "content": "Ollama generated answer."},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 20,
            "eval_count": 40,
        }
        mock_client.post.return_value = mock_response

        provider = OllamaLLMProvider(
            model="llama3.1:8b",
            endpoint="http://localhost:11434",
            client=mock_client,
        )
        req = GenerationRequest(
            messages=[Message(role="user", content="Explain RAG.")],
            temperature=0.8,
            max_tokens=256,
        )
        resp = provider.generate(req)
        self.assertEqual(resp.text, "Ollama generated answer.")
        self.assertEqual(resp.provider, "ollama")
        self.assertEqual(resp.model, "llama3.1:8b")
        self.assertEqual(resp.usage.total_tokens, 60)

        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        self.assertEqual(call_args[0][0], "/api/chat")
        payload = call_args[1]["json"]
        self.assertEqual(payload["model"], "llama3.1:8b")
        self.assertEqual(payload["options"]["temperature"], 0.8)
        self.assertEqual(payload["options"]["num_predict"], 256)

    def test_structured_output(self) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "llama3.1:8b",
            "message": {"role": "assistant", "content": '{"summary": "Success"}'},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 15,
            "eval_count": 10,
        }
        mock_client.post.return_value = mock_response

        provider = OllamaLLMProvider(
            model="llama3.1:8b",
            endpoint="http://localhost:11434",
            client=mock_client,
        )
        req = StructuredOutputRequest(
            messages=[Message(role="user", content="Summarize")],
            schema={"type": "object", "properties": {"summary": {"type": "string"}}},
        )
        resp = provider.structured_output(req)
        self.assertEqual(resp.data, {"summary": "Success"})
        self.assertEqual(resp.provider, "ollama")

    def test_error_handling(self) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "Service Unavailable"
        mock_client.post.return_value = mock_response

        provider = OllamaLLMProvider(
            model="llama3.1:8b",
            endpoint="http://localhost:11434",
            client=mock_client,
        )
        req = GenerationRequest(messages=[Message(role="user", content="Hi")])
        with self.assertRaises(TransientModelProviderError):
            provider.generate(req)
