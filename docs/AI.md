# AURA — AI Architecture & Model Gateway

## Overview

AURA (Autonomous Research & Engineering Agent) is an **open-model-first, LLM-agnostic agentic AI platform**.

The AI subsystem is designed to ensure that application code, agent workflows, RAG pipelines, and evaluation logic never depend directly on a specific model vendor or inference provider.

---

## 1. Architecture

The Model Gateway provides an abstraction boundary between application logic and inference providers:

```text
Application / Agent Layer
          │
          ▼
    ModelGateway (backend/gateway/gateway.py)
          │  Routes, validates requests, checks capabilities, normalizes errors
          ▼
     LLMProvider (backend/gateway/base.py — Abstract Base Class)
          │
          ├── MockLLMProvider (backend/gateway/providers/mock.py)
          └── [Future providers: Local/vLLM/Ollama/OpenAI]
```

### Key Principles

1. **Vendor Independence**: Application code communicates exclusively through `ModelGateway`. Provider-specific details are hidden behind `LLMProvider`.
2. **Open-Model-First**: Open and self-hosted models are primary targets. Initial development operates entirely offline without external model APIs.
3. **Deterministic Testing**: A built-in `MockLLMProvider` enables testing generation, streaming, structured output, and error conditions with zero API keys and zero cost.
4. **Decoupled Configuration**: Providers, models, endpoints, and timeouts are externalized via environment variables and Django settings.
5. **No Secret Leakage**: Gateway error normalization ensures credentials or sensitive headers are never propagated in error messages or logs.

---

## 2. Core Components

### `LLMProvider` Interface

Defined in `backend/gateway/base.py`, `LLMProvider` is an abstract base class establishing four essential capabilities:

```python
class LLMProvider(ABC):
    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse: ...

    @abstractmethod
    def stream(self, request: GenerationRequest) -> Iterator[StreamChunk]: ...

    @abstractmethod
    def structured_output(self, request: StructuredOutputRequest) -> StructuredOutputResponse: ...

    @abstractmethod
    def metadata(self) -> ProviderMetadata: ...
```

### `ModelGateway`

Defined in `backend/gateway/gateway.py`, `ModelGateway` is the single entry point for application code:

* Receives and validates model requests (`InvalidRequestError` on bad inputs).
* Selects and instantiates the configured provider via the provider registry.
* Verifies capability support before dispatching (`UnsupportedCapabilityError`).
* Delegates generation, streaming, and structured-output requests.
* Exposes provider/model metadata (`metadata()` and `get_metadata()`).
* Normalizes provider-facing errors into `GenerationError` while masking secrets.

### `MockLLMProvider`

Defined in `backend/gateway/providers/mock.py`:

* Requires no external network access or API key.
* Produces deterministic responses derived predictably from input messages.
* Streams word-by-word chunks with sequential indexes and terminal `finish_reason="stop"`.
* Emits schema-conforming mock structured data and valid JSON `raw_text`.
* Calculates simulated token usage (`prompt_tokens`, `completion_tokens`, `total_tokens`).
* Allows simulating failures and restricted capabilities for test scenarios.

### Provider Registry

Defined in `backend/gateway/registry.py`:

* Decouples provider implementations from the core gateway.
* Allows runtime registration via `register_provider(name, provider_cls)`.
* Factory function `create_provider(config)` instantiates providers dynamically.

---

## 3. Data Contracts

Defined in `backend/gateway/types.py`:

| Contract | Purpose | Key Fields |
|---|---|---|
| `Message` | Provider-agnostic message | `role`, `content` |
| `UsageInfo` | Token observability | `prompt_tokens`, `completion_tokens`, `total_tokens` |
| `GenerationRequest` | Normal text generation input | `messages`, `model`, `temperature`, `max_tokens`, `metadata` |
| `GenerationResponse` | Text generation output | `text`, `provider`, `model`, `usage`, `finish_reason`, `metadata` |
| `StreamChunk` | Incremental stream chunk | `text`, `index`, `finish_reason`, `metadata` |
| `StructuredOutputRequest` | Structured generation input | `messages`, `schema`, `model`, `temperature`, `max_tokens`, `metadata` |
| `StructuredOutputResponse` | Structured generation output | `data`, `raw_text`, `provider`, `model`, `usage`, `finish_reason` |
| `ProviderMetadata` | Provider identification & capabilities | `provider`, `model`, `capabilities`, `supports(capability)` |

---

## 4. Configuration

Configured via `GatewayConfig` in `backend/gateway/config.py`:

```ini
# Model Gateway (.env)
AI_PROVIDER=mock
AI_MODEL=mock-model
AI_ENDPOINT=
AI_TIMEOUT=30
```

Mapped in `backend/config/settings.py`:

```python
AI_GATEWAY = {
    "PROVIDER": os.environ.get("AI_PROVIDER", "mock"),
    "MODEL": os.environ.get("AI_MODEL", "mock-model"),
    "ENDPOINT": os.environ.get("AI_ENDPOINT", ""),
    "TIMEOUT": float(os.environ.get("AI_TIMEOUT", "30.0")),
}
```

---

## 5. Error Hierarchy

All exceptions inherit from `GatewayError` (`backend/gateway/exceptions.py`):

```text
GatewayError
├── ProviderUnavailableError
├── UnsupportedCapabilityError
├── InvalidRequestError
├── ProviderConfigurationError
└── GenerationError
```

---

## 6. Provider Progression Roadmap

```text
M1: Mock Provider (Local deterministic mock)
  ↓
M7: Local / Open Models (vLLM, Ollama, OpenAI-compatible servers)
  ↓
Later: Optional Hosted Providers
```
