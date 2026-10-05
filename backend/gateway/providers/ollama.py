"""Ollama LLM Provider using Ollama's native HTTP API."""

import json
import logging
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

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"


def _normalize_ollama_endpoint(endpoint: str) -> str:
    """Normalize endpoint by removing trailing slashes and /v1 suffix if present."""
    ep = (endpoint or DEFAULT_OLLAMA_ENDPOINT).strip().rstrip("/")
    if ep.endswith("/v1"):
        ep = ep[:-3].rstrip("/")
    return ep or DEFAULT_OLLAMA_ENDPOINT


class OllamaLLMProvider(LLMProvider):
    """Local inference provider targeting Ollama native HTTP API."""

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
        api_key: str | None = None,
        timeout: float = 30.0,
        client: Any = None,
        extra_config: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = (model or DEFAULT_OLLAMA_MODEL).strip()
        self.endpoint = _normalize_ollama_endpoint(endpoint)
        self.timeout = float(timeout) if timeout else 30.0
        self.extra_config = extra_config or {}

        if client is not None:
            self._client = client
        else:
            try:
                import httpx

                self._client = httpx.Client(
                    base_url=self.endpoint,
                    timeout=httpx.Timeout(self.timeout),
                )
            except Exception as exc:
                raise ProviderUnavailableError(
                    f"Failed to initialize HTTP client for Ollama at '{self.endpoint}': {exc}"
                ) from exc

    def metadata(self) -> ProviderMetadata:
        """Return provider and model metadata and supported capabilities."""
        return ProviderMetadata(
            provider="ollama",
            model=self.model_name,
            capabilities=ALL_CAPABILITIES,
        )

    def _build_options(
        self, temperature: float | None = None, max_tokens: int | None = None
    ) -> dict[str, Any]:
        """Construct Ollama options dictionary."""
        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = float(temperature)
        if max_tokens is not None:
            options["num_predict"] = int(max_tokens)
        return options

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text response using Ollama /api/chat endpoint."""
        model = request.model or self.model_name
        messages = [m.to_dict() for m in request.messages]

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        options = self._build_options(request.temperature, request.max_tokens)
        if options:
            payload["options"] = options

        timeout_override = None
        if request.metadata and ("timeout" in request.metadata or "timeout_seconds" in request.metadata):
            t_val = request.metadata.get("timeout", request.metadata.get("timeout_seconds"))
            try:
                budget = float(t_val)
                if budget > 0:
                    timeout_override = min(self.timeout, budget)
            except (ValueError, TypeError):
                pass

        try:
            kwargs: dict[str, Any] = {"json": payload}
            if timeout_override is not None:
                kwargs["timeout"] = timeout_override
            resp = self._client.post("/api/chat", **kwargs)
        except Exception as exc:
            if is_transient_provider_error(exc):
                raise TransientModelProviderError(
                    f"Ollama connection error (transient failure): {exc}"
                ) from exc
            raise GenerationError(f"Ollama connection failed: {exc}") from exc

        if resp.status_code != 200:
            err_msg = f"Ollama HTTP {resp.status_code}: {resp.text}"
            if resp.status_code in (502, 503, 504):
                raise TransientModelProviderError(err_msg, status_code=resp.status_code)
            raise GenerationError(err_msg)

        try:
            data = resp.json()
        except Exception as exc:
            raise GenerationError(f"Failed to parse Ollama JSON response: {exc}") from exc

        msg_obj = data.get("message", {})
        text = msg_obj.get("content", "") or ""
        finish_reason = data.get("done_reason", "stop") or "stop"

        p_tokens = data.get("prompt_eval_count", 0) or 0
        c_tokens = data.get("eval_count", 0) or 0
        usage = UsageInfo(
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=p_tokens + c_tokens,
        )

        return GenerationResponse(
            text=text,
            provider="ollama",
            model=model,
            usage=usage,
            finish_reason=finish_reason,
            metadata={"request_metadata": request.metadata},
        )

    def stream(self, request: GenerationRequest) -> Iterator[StreamChunk]:
        """Generate a stream of chunks incrementally from Ollama."""
        model = request.model or self.model_name
        messages = [m.to_dict() for m in request.messages]

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        options = self._build_options(request.temperature, request.max_tokens)
        if options:
            payload["options"] = options

        try:
            if hasattr(self._client, "stream"):
                with self._client.stream("POST", "/api/chat", json=payload) as response:
                    if response.status_code != 200:
                        raise GenerationError(f"Ollama streaming HTTP {response.status_code}: {response.text}")
                    idx = 0
                    for line in response.iter_lines():
                        if not line or not line.strip():
                            continue
                        chunk_data = json.loads(line)
                        content = chunk_data.get("message", {}).get("content", "") or ""
                        finish_reason = chunk_data.get("done_reason") if chunk_data.get("done") else None
                        yield StreamChunk(
                            text=content,
                            index=idx,
                            finish_reason=finish_reason,
                        )
                        idx += 1
            else:
                # Handle test mock client that may not have stream() context manager
                resp = self._client.post("/api/chat", json=payload)
                lines = resp.text.strip().splitlines()
                for idx, line in enumerate(lines):
                    if not line.strip():
                        continue
                    chunk_data = json.loads(line)
                    content = chunk_data.get("message", {}).get("content", "") or ""
                    finish_reason = chunk_data.get("done_reason") if chunk_data.get("done") else None
                    yield StreamChunk(
                        text=content,
                        index=idx,
                        finish_reason=finish_reason,
                    )
        except Exception as exc:
            if is_transient_provider_error(exc):
                raise TransientModelProviderError(
                    f"Ollama streaming transient failure: {exc}"
                ) from exc
            raise GenerationError(f"Ollama streaming failed: {exc}") from exc

    def structured_output(
        self, request: StructuredOutputRequest
    ) -> StructuredOutputResponse:
        """Generate structured output adhering to a schema or specification using format='json'."""
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

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": "json",
        }
        options = self._build_options(request.temperature, request.max_tokens)
        if options:
            payload["options"] = options

        try:
            resp = self._client.post("/api/chat", json=payload)
        except Exception as exc:
            if is_transient_provider_error(exc):
                raise TransientModelProviderError(
                    f"Ollama structured output transient failure: {exc}"
                ) from exc
            raise GenerationError(f"Ollama structured output failed: {exc}") from exc

        if resp.status_code != 200:
            err_msg = f"Ollama HTTP {resp.status_code}: {resp.text}"
            if resp.status_code in (502, 503, 504):
                raise TransientModelProviderError(err_msg, status_code=resp.status_code)
            raise GenerationError(err_msg)

        data_raw = resp.json()
        raw_text = data_raw.get("message", {}).get("content", "") or ""
        finish_reason = data_raw.get("done_reason", "stop") or "stop"

        clean_text = raw_text.strip()
        if clean_text.startswith("```"):
            lines = clean_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            clean_text = "\n".join(lines).strip()

        try:
            parsed = json.loads(clean_text)
        except Exception as exc:
            raise GenerationError(
                f"Failed to parse Ollama structured JSON output: {exc}. Raw text: {raw_text}"
            ) from exc

        p_tokens = data_raw.get("prompt_eval_count", 0) or 0
        c_tokens = data_raw.get("eval_count", 0) or 0
        usage = UsageInfo(
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=p_tokens + c_tokens,
        )

        return StructuredOutputResponse(
            data=parsed,
            raw_text=raw_text,
            provider="ollama",
            model=model,
            usage=usage,
            finish_reason=finish_reason,
            metadata={"request_metadata": request.metadata},
        )
