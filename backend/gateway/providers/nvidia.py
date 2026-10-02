"""NVIDIA LLM Provider using OpenAI-compatible client."""

import json
import math
import os
from collections.abc import Iterator
from typing import Any

from gateway.base import LLMProvider
from gateway.exceptions import (
    GenerationError,
    ProviderConfigurationError,
    ProviderUnavailableError,
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

DEFAULT_NVIDIA_ENDPOINT = "https://integrate.api.nvidia.com/v1"
DEFAULT_NVIDIA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"


class NvidiaLLMProvider(LLMProvider):
    """NVIDIA NIM / API LLM provider using OpenAI-compatible interface."""

    def __init__(
        self,
        model: str = DEFAULT_NVIDIA_MODEL,
        endpoint: str = DEFAULT_NVIDIA_ENDPOINT,
        api_key: str | None = None,
        timeout: float = 30.0,
        client: Any = None,
    ) -> None:
        self.model_name = model or DEFAULT_NVIDIA_MODEL
        self.endpoint = endpoint or DEFAULT_NVIDIA_ENDPOINT
        self.timeout = float(timeout) if timeout else 30.0

        resolved_key = (api_key or os.environ.get("AI_API_KEY", "")).strip()

        if client is not None:
            self._client = client
        else:
            if not resolved_key:
                raise ProviderConfigurationError(
                    "NVIDIA API key not configured. Set AI_API_KEY environment variable."
                )
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    base_url=self.endpoint,
                    api_key=resolved_key,
                    timeout=self.timeout,
                )
            except Exception as exc:
                raise ProviderUnavailableError(
                    f"Failed to initialize OpenAI client for NVIDIA provider: {exc}"
                ) from exc

    def metadata(self) -> ProviderMetadata:
        """Return provider and model metadata and supported capabilities."""
        return ProviderMetadata(
            provider="nvidia",
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

        # If budget is already exhausted or non-positive, floor to minimal positive timeout
        # to trigger immediate timeout without retries.
        if budget <= 0.0:
            return 0.001, 0

        # Effective timeout is capped by configured provider timeout (self.timeout) and budget
        effective_timeout = min(self.timeout, max(0.001, budget))

        client_max_retries = getattr(self._client, "max_retries", 2)
        if not isinstance(client_max_retries, int) or client_max_retries < 0:
            client_max_retries = 2

        # Provider retries introduce both an additional request attempt (costing up to effective_timeout)
        # and exponential backoff delay (at least ~0.5s - 1.0s per retry).
        # To strictly ensure retries cannot cause the model call to exceed the remaining runtime budget:
        # (1 + retries) * effective_timeout + retries * retry_delay_buffer <= budget
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
        """Generate text response using NVIDIA OpenAI-compatible API."""
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
            raise GenerationError(f"NVIDIA model generation failed: {exc}") from exc
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
            provider="nvidia",
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
            raise GenerationError(f"NVIDIA streaming failed: {exc}") from exc
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
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": False,
                }
            },
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
            raise GenerationError(f"NVIDIA structured output generation failed: {exc}") from exc
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
            provider="nvidia",
            model=model,
            usage=usage_info,
            finish_reason=finish_reason,
            metadata={"request_metadata": request.metadata},
        )
