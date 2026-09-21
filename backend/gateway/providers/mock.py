"""Deterministic Mock LLM Provider for testing and local development."""

import json
from collections.abc import Iterator
from typing import Any

from gateway.base import LLMProvider
from gateway.types import (
    ALL_CAPABILITIES,
    GenerationRequest,
    GenerationResponse,
    ProviderMetadata,
    StreamChunk,
    StructuredOutputRequest,
    StructuredOutputResponse,
    UsageInfo,
)


class MockLLMProvider(LLMProvider):
    """Deterministic, local mock provider requiring no external API or API key."""

    def __init__(
        self,
        model: str = "mock-model",
        default_response: str | None = None,
        canned_responses: dict[str, str] | None = None,
        simulated_failure: Exception | None = None,
        unsupported_capabilities: tuple[str, ...] | list[str] | set[str] = (),
    ) -> None:
        self.model_name = model
        self.default_response = default_response
        self.canned_responses = canned_responses or {}
        self.simulated_failure = simulated_failure
        self.unsupported_capabilities = set(unsupported_capabilities)

    def metadata(self) -> ProviderMetadata:
        capabilities = tuple(
            cap for cap in ALL_CAPABILITIES if cap not in self.unsupported_capabilities
        )
        return ProviderMetadata(
            provider="mock",
            model=self.model_name,
            capabilities=capabilities,
        )

    def _determine_text(self, request: GenerationRequest) -> str:
        if not request.messages:
            return "Mock response"
        last_msg = request.messages[-1].content
        if last_msg in self.canned_responses:
            return self.canned_responses[last_msg]
        if self.default_response is not None:
            return self.default_response
        return f"Mock response to: {last_msg}"

    def _calculate_usage(self, request_messages: list[Any], output_text: str) -> UsageInfo:
        prompt_tokens = sum(max(1, len(m.content.split())) for m in request_messages)
        completion_tokens = max(1, len(output_text.split()))
        return UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if self.simulated_failure is not None:
            raise self.simulated_failure

        text = self._determine_text(request)
        model = request.model or self.model_name
        usage = self._calculate_usage(request.messages, text)

        return GenerationResponse(
            text=text,
            provider="mock",
            model=model,
            usage=usage,
            finish_reason="stop",
            metadata={"mock": True, "request_metadata": request.metadata},
        )

    def stream(self, request: GenerationRequest) -> Iterator[StreamChunk]:
        if self.simulated_failure is not None:
            raise self.simulated_failure

        text = self._determine_text(request)
        words = text.split(" ")
        if not words:
            yield StreamChunk(text="", index=0, finish_reason="stop")
            return

        for idx, word in enumerate(words):
            is_last = idx == len(words) - 1
            chunk_text = word if is_last else word + " "
            finish_reason = "stop" if is_last else None
            yield StreamChunk(
                text=chunk_text,
                index=idx,
                finish_reason=finish_reason,
                metadata={"mock": True},
            )

    def structured_output(
        self, request: StructuredOutputRequest
    ) -> StructuredOutputResponse:
        if self.simulated_failure is not None:
            raise self.simulated_failure

        last_msg = request.messages[-1].content if request.messages else ""
        if request.schema and isinstance(request.schema, dict):
            properties = request.schema.get("properties", {})
            if properties:
                data: dict[str, Any] = {}
                for key, prop_schema in properties.items():
                    prop_type = prop_schema.get("type", "string")
                    if prop_type == "string":
                        data[key] = f"mock_{key}_value"
                    elif prop_type in ("integer", "number"):
                        data[key] = 42
                    elif prop_type == "boolean":
                        data[key] = True
                    elif prop_type == "array":
                        data[key] = ["mock_item_1", "mock_item_2"]
                    else:
                        data[key] = {}
            else:
                data = {"status": "ok", "result": f"Structured mock output for: {last_msg}"}
        else:
            data = {"status": "ok", "result": f"Structured mock output for: {last_msg}"}

        raw_text = json.dumps(data)
        model = request.model or self.model_name
        usage = self._calculate_usage(request.messages, raw_text)

        return StructuredOutputResponse(
            data=data,
            raw_text=raw_text,
            provider="mock",
            model=model,
            usage=usage,
            finish_reason="stop",
            metadata={"mock": True},
        )
