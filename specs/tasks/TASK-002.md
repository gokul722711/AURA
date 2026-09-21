# TASK-002 — M1 Model Gateway

## Objective

Implement AURA's internal **Model Gateway** and provider abstraction.

The gateway must allow the rest of AURA to interact with language models without knowing which model, inference server, or provider is being used.

M1 must work entirely without an external LLM API.

---

## Required Reading

Before implementation, read:

```text
AGENTS.md
docs/STATUS.md
docs/AI.md
specs/tasks/TASK-001.md
```

Inspect the existing backend implementation before making changes.

Do not assume architecture that is not supported by these documents or the existing code.

---

## Scope

Implement:

1. `LLMProvider` abstraction
2. Model Gateway
3. `MockLLMProvider`
4. Provider/model configuration
5. Generation capability
6. Streaming capability
7. Structured-output capability
8. Provider metadata
9. Error handling
10. Unit tests

---

## Architecture

The intended dependency direction is:

```text
Application / Agent
        │
        ▼
  Model Gateway
        │
        ▼
   LLMProvider
        │
        ▼
 MockLLMProvider
```

The application must not directly instantiate or call a provider implementation.

Provider-specific logic belongs behind the provider abstraction.

---

## 1. LLMProvider

Create a clear Python interface or protocol representing an LLM provider.

The interface should support the following capabilities:

```text
generate()
stream()
structured_output()
metadata()
```

Keep the interface minimal.

Do not introduce unnecessary framework-specific abstractions.

The exact Python representation may be an abstract base class or `typing.Protocol`, provided it remains clear, testable, and extensible.

---

## 2. Model Gateway

Create a gateway responsible for routing model operations through the configured provider.

Conceptually:

```text
ModelGateway
      │
      ▼
LLMProvider
```

The gateway should:

* Receive model requests.
* Select/use the configured provider.
* Delegate generation operations.
* Delegate streaming operations.
* Delegate structured output.
* Expose provider/model metadata.
* Normalize provider-facing errors where appropriate.

Application code should interact with the gateway rather than directly with `LLMProvider` implementations.

---

## 3. MockLLMProvider

Implement a deterministic mock provider.

It must:

* Require no external API.
* Require no API key.
* Produce deterministic output suitable for tests.
* Support the provider interface.
* Support generation.
* Support streaming.
* Support structured output.
* Expose basic metadata.

The mock provider should make it possible to develop and test the gateway without model costs.

Do not simulate complex model intelligence.

---

## 4. Generation

Define a provider-independent request/response representation for normal generation.

The design should be capable of representing concepts such as:

```text
messages
model
temperature
max_tokens
metadata
```

Do not hard-code a vendor-specific request format.

The response should provide enough information for future observability, including where appropriate:

```text
text
provider
model
usage
finish_reason
```

Keep the first implementation simple.

---

## 5. Streaming

The gateway must expose a provider-independent streaming abstraction.

The abstraction should allow callers to consume generated output incrementally.

The implementation may use a Python iterator/generator or another simple Python-native mechanism.

Do not couple the interface to:

* Server-Sent Events
* WebSockets
* A specific frontend implementation
* A specific provider SDK

Transport-level streaming will be handled separately at the API layer when required.

---

## 6. Structured Output

Provide a provider-independent mechanism for requesting structured model output.

The first implementation only needs to support deterministic mock behavior.

Do not introduce a large schema framework unless required by the implementation.

The abstraction should leave room for future providers to implement native structured-output mechanisms.

---

## 7. Provider Metadata

The gateway/provider layer should expose metadata such as:

```text
provider
model
capabilities
```

Potential capabilities include:

```text
generation
streaming
structured_output
```

Do not hard-code specific future providers into the core gateway.

---

## 8. Configuration

Provider configuration must come from application configuration/environment rather than hard-coded values.

Support concepts such as:

```text
provider
model
endpoint
timeout
```

Do not require credentials for the mock provider.

Do not add real provider credentials to the repository.

Do not create an OpenAI-specific configuration as part of M1.

---

## 9. Error Handling

Define clear provider/gateway errors for situations such as:

* Provider unavailable
* Unsupported capability
* Invalid request
* Provider configuration error
* Generation failure

Errors should not expose secrets.

Do not build a complex retry system in M1.

---

## 10. Testing

Add deterministic unit tests covering at minimum:

### Provider

* Mock generation
* Mock streaming
* Mock structured output
* Mock metadata

### Gateway

* Provider routing
* Generation delegation
* Streaming delegation
* Structured-output delegation
* Metadata access
* Invalid configuration
* Unsupported capability/error behavior

Tests must not require:

* Internet access
* API keys
* External model servers
* Paid services

The complete M1 test suite must run locally.

---

## 11. API Boundary

Do not expose the model gateway through a Django endpoint yet.

M1 is an internal backend capability.

Do not implement:

* Chat endpoint
* Generation endpoint
* Streaming HTTP endpoint
* Authentication
* User-facing AI UI

These belong to later tasks.

---

## 12. Dependencies

Prefer Python standard-library functionality where practical.

Do not add a model SDK merely to implement the mock provider.

Do not add:

* OpenAI SDK
* NVIDIA SDK
* Ollama SDK
* vLLM SDK
* LangChain
* LangGraph

for M1 unless an existing project dependency genuinely requires it.

The gateway should remain provider-independent.

---

## 13. Security

Never commit:

* API keys
* Provider credentials
* Secrets
* Local `.env` files

Provider endpoints and credentials must be configurable through environment/application settings.

Treat all model output as untrusted data.

---

## 14. Backward Compatibility

M0 functionality must continue to work.

The following must remain functional:

```text
Django startup
Django system checks
Django tests
/api/health/
PostgreSQL configuration
Next.js frontend
```

Do not modify unrelated M0 functionality.

---

## 15. Definition of Done

M1 is complete when:

* `LLMProvider` abstraction exists.
* `ModelGateway` exists.
* `MockLLMProvider` exists.
* Generation works through the gateway.
* Streaming works through the gateway.
* Structured output works through the gateway.
* Provider/model metadata is available.
* Configuration is externalized.
* Appropriate errors exist.
* Deterministic tests cover the gateway and provider.
* Tests pass.
* M0 functionality still passes.
* No external LLM API is required.
* No model vendor SDK is required.
* No TypeScript is introduced.
* No secrets are committed.
* Implementation follows `AGENTS.md` and `docs/AI.md`.

---

## Out of Scope

Do not implement:

* OpenAI integration
* NVIDIA NIM integration
* Ollama integration
* vLLM integration
* Real model inference
* Embeddings
* RAG
* Advanced RAG
* Agents
* LangGraph
* Autonomous research
* Tool execution
* Evaluation framework
* AI frontend
* Authentication
* Deployment

These will be handled by later tasks.

---

## Required Final Report

After implementation, report:

1. Files created/modified.
2. Architecture implemented.
3. Tests added.
4. Commands executed.
5. Test results.
6. M0 regression results.
7. Any limitations or design decisions.
8. Any follow-up work required.

Do not claim completion without running the relevant tests.

Do not commit automatically.
