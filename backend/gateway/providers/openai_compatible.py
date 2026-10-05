"""Generic OpenAI-compatible LLM Provider for local or hosted inference servers (vLLM, etc.)."""

import json
import math
import os
import re
from collections.abc import Iterator
from typing import Any

from gateway.base import LLMProvider
from gateway.exceptions import (
    GenerationError,
    ProviderConfigurationError,
    ProviderUnavailableError,
    TransientModelProviderError,
    is_transient_provider_error,
)
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

DEFAULT_OPENAI_COMPATIBLE_ENDPOINT = "http://localhost:8000/v1"
DEFAULT_OPENAI_COMPATIBLE_MODEL = "default"


class OpenAICompatibleLLMProvider(LLMProvider):
    """Generic provider targeting any OpenAI-compatible chat completions endpoint (vLLM, LocalAI, etc.)."""

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_COMPATIBLE_MODEL,
        endpoint: str = DEFAULT_OPENAI_COMPATIBLE_ENDPOINT,
        api_key: str | None = None,
        timeout: float = 30.0,
        client: Any = None,
        extra_config: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = (model or DEFAULT_OPENAI_COMPATIBLE_MODEL).strip()
        self.endpoint = (endpoint or DEFAULT_OPENAI_COMPATIBLE_ENDPOINT).strip()
        self.timeout = float(timeout) if timeout else 30.0
        self.extra_config = extra_config or {}

        resolved_key = (api_key or os.environ.get("OPENAI_API_KEY", "")).strip()
        # Most local servers (vLLM, LM Studio) do not check API keys, but the official
        # OpenAI python SDK requires a non-empty string.
        client_key = resolved_key if resolved_key else "EMPTY"

        if client is not None:
            self._client = client
        else:
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    base_url=self.endpoint,
                    api_key=client_key,
                    timeout=self.timeout,
                )
            except Exception as exc:
                raise ProviderUnavailableError(
                    f"Failed to initialize OpenAI-compatible client for '{self.endpoint}': {exc}"
                ) from exc

    def metadata(self) -> ProviderMetadata:
        """Return provider and model metadata and supported capabilities."""
        return ProviderMetadata(
            provider="openai_compatible",
            model=self.model_name,
            capabilities=ALL_CAPABILITIES,
        )

    def _resolve_timeout_and_retries(
        self, metadata: dict[str, Any] | None
    ) -> tuple[float | None, int | None]:
        """Resolve effective per-request timeout and max_retries bounded by runtime budget."""
        if not metadata:
            return None, None

        timeout_val = metadata.get("timeout")
        if timeout_val is None:
            timeout_val = metadata.get("timeout_seconds")
        if timeout_val is None:
            timeout_val = metadata.get("remaining_seconds")

        if timeout_val is None:
            return None, None

        try:
            budget = float(timeout_val)
        except (ValueError, TypeError):
            return None, None

        if math.isnan(budget) or math.isinf(budget):
            return None, None

        if budget <= 0.0:
            return 0.001, 0

        effective_timeout = min(self.timeout, max(0.001, budget))

        client_max_retries = getattr(self._client, "max_retries", 2)
        if not isinstance(client_max_retries, int) or client_max_retries < 0:
            client_max_retries = 2

        retry_delay_buffer = 1.0
        time_per_retry = effective_timeout + retry_delay_buffer
        available_retry_time = budget - effective_timeout

        if available_retry_time <= 0.0:
            allowed_retries = 0
        else:
            allowed_retries = max(0, int(available_retry_time // time_per_retry))

        effective_retries = min(client_max_retries, allowed_retries)
        return effective_timeout, effective_retries

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text response using OpenAI-compatible chat completions API."""
        model = request.model or self.model_name
        messages = [m.to_dict() for m in request.messages]
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens

        effective_timeout, effective_retries = self._resolve_timeout_and_retries(request.metadata)
        if effective_timeout is not None:
            kwargs["timeout"] = effective_timeout

        original_retries = getattr(self._client, "max_retries", None)
        try:
            if effective_retries is not None and hasattr(self._client, "max_retries"):
                self._client.max_retries = effective_retries
            completion = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            if is_transient_provider_error(exc):
                raise TransientModelProviderError(
                    f"OpenAI-compatible generation transient failure: {exc}",
                    status_code=getattr(exc, "status_code", None),
                ) from exc
            raise GenerationError(f"OpenAI-compatible model generation failed: {exc}") from exc
        finally:
            if original_retries is not None and effective_retries is not None and hasattr(self._client, "max_retries"):
                self._client.max_retries = original_retries

        choice = completion.choices[0]
        text = choice.message.content or ""
        finish_reason = getattr(choice, "finish_reason", "stop") or "stop"

        usage_info = UsageInfo()
        if hasattr(completion, "usage") and completion.usage:
            usage_info = UsageInfo(
                prompt_tokens=getattr(completion.usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(completion.usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(completion.usage, "total_tokens", 0) or 0,
            )

        return GenerationResponse(
            text=text,
            provider="openai_compatible",
            model=model,
            usage=usage_info,
            finish_reason=finish_reason,
            metadata={"request_metadata": request.metadata},
        )

    def stream(self, request: GenerationRequest) -> Iterator[StreamChunk]:
        """Generate a stream of chunks incrementally."""
        model = request.model or self.model_name
        messages = [m.to_dict() for m in request.messages]
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens

        effective_timeout, effective_retries = self._resolve_timeout_and_retries(request.metadata)
        if effective_timeout is not None:
            kwargs["timeout"] = effective_timeout

        original_retries = getattr(self._client, "max_retries", None)
        try:
            if effective_retries is not None and hasattr(self._client, "max_retries"):
                self._client.max_retries = effective_retries
            stream_resp = self._client.chat.completions.create(**kwargs)
            for idx, chunk in enumerate(stream_resp):
                if not getattr(chunk, "choices", None):
                    continue
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", "") or ""
                finish_reason = getattr(chunk.choices[0], "finish_reason", None)
                yield StreamChunk(
                    text=content,
                    index=idx,
                    finish_reason=finish_reason,
                )
        except Exception as exc:
            if is_transient_provider_error(exc):
                raise TransientModelProviderError(
                    f"OpenAI-compatible streaming transient failure: {exc}",
                    status_code=getattr(exc, "status_code", None),
                ) from exc
            raise GenerationError(f"OpenAI-compatible streaming failed: {exc}") from exc
        finally:
            if original_retries is not None and effective_retries is not None and hasattr(self._client, "max_retries"):
                self._client.max_retries = original_retries

    def structured_output(
        self, request: StructuredOutputRequest
    ) -> StructuredOutputResponse:
        """Generate structured output adhering to a schema or specification."""
        model = request.model or self.model_name
        messages = [m.to_dict() for m in request.messages]

        if request.schema:
            schema_instr = (
                f"\nYou must format your entire output as a valid JSON object matching this schema:\n"
                f"{json.dumps(request.schema)}\n"
                f"Output ONLY valid JSON. Do not enclose in markdown code blocks."
            )
            if messages:
                messages[-1]["content"] = messages[-1]["content"] + schema_instr
            else:
                messages.append({"role": "user", "content": schema_instr})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens

        effective_timeout, effective_retries = self._resolve_timeout_and_retries(request.metadata)
        if effective_timeout is not None:
            kwargs["timeout"] = effective_timeout

        original_retries = getattr(self._client, "max_retries", None)
        try:
            if effective_retries is not None and hasattr(self._client, "max_retries"):
                self._client.max_retries = effective_retries
            completion = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            if is_transient_provider_error(exc):
                raise TransientModelProviderError(
                    f"OpenAI-compatible structured output transient failure: {exc}",
                    status_code=getattr(exc, "status_code", None),
                ) from exc
            raise GenerationError(f"OpenAI-compatible structured output failed: {exc}") from exc
        finally:
            if original_retries is not None and effective_retries is not None and hasattr(self._client, "max_retries"):
                self._client.max_retries = original_retries

        choice = completion.choices[0]
        raw_text = choice.message.content or ""
        finish_reason = getattr(choice, "finish_reason", "stop") or "stop"

        clean_text = raw_text.strip()
        if clean_text.startswith("```"):
            lines = clean_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            clean_text = "\n".join(lines).strip()

        try:
            data = json.loads(clean_text)
        except Exception as exc:
            raise GenerationError(
                f"Failed to parse structured JSON output: {exc}. Raw text: {raw_text}"
            ) from exc

        usage_info = UsageInfo()
        if hasattr(completion, "usage") and completion.usage:
            usage_info = UsageInfo(
                prompt_tokens=getattr(completion.usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(completion.usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(completion.usage, "total_tokens", 0) or 0,
            )

        return StructuredOutputResponse(
            data=data,
            raw_text=raw_text,
            provider="openai_compatible",
            model=model,
            usage=usage_info,
            finish_reason=finish_reason,
            metadata={"request_metadata": request.metadata},
        )
