# AURA — Development Status

## Current Phase

**M1 — Model Gateway: COMPLETE**

**Next: M2 — Basic RAG**

---

## Completed

### M0 — Foundation

Repository and application foundation established.

Implemented:

* Git repository and project structure
* Django backend
* Django REST Framework
* PostgreSQL configuration
* Environment-based configuration
* Backend health endpoint
* Backend automated tests
* Next.js frontend
* React frontend using JavaScript/JSX
* Frontend development/build configuration
* Initial project documentation

Verified:

* Django migrations pass
* Django tests pass
* Django system checks pass
* Backend health endpoint returns successfully
* Next.js production build passes
* Next.js development server starts successfully
* Frontend contains no TypeScript source files
* `git diff --check` passes

### M1 — Model Gateway

Internal model gateway and provider abstraction implemented without requiring external LLM APIs.

Implemented:

* `LLMProvider` abstract base interface (`generate`, `stream`, `structured_output`, `metadata`)
* `ModelGateway` routing, validation, error normalization, and metadata delegation
* `MockLLMProvider` deterministic offline mock provider
* Provider-independent data contracts (`Message`, `GenerationRequest`, `GenerationResponse`, `StreamChunk`, `StructuredOutputRequest`, `StructuredOutputResponse`, `ProviderMetadata`, `UsageInfo`)
* Externalized configuration (`GatewayConfig`, Django `settings.AI_GATEWAY`, `.env.example`)
* Decoupled provider registry (`register_provider`, `create_provider`)
* Normalized error hierarchy preventing credential exposure (`GatewayError`, `ProviderUnavailableError`, `UnsupportedCapabilityError`, `InvalidRequestError`, `ProviderConfigurationError`, `GenerationError`)
* AI architecture documentation (`docs/AI.md`)
* Deterministic unit tests covering types, configuration, mock provider, and gateway

Verified:

* 63 gateway unit tests pass
* Full backend test suite passes (66/66 tests)
* Django system checks pass
* Backend health check endpoint returns 200 OK
* Next.js production build passes
* `git diff --check` passes
* Zero external model SDKs or paid APIs required

---

## Current Repository State

```text
Backend
  Python
  Django
  Django REST Framework
  PostgreSQL
  Model Gateway (M1)

Frontend
  Next.js
  React
  JavaScript / JSX

Infrastructure
  PostgreSQL
  Redis planned
  pgvector planned

AI
  Model Gateway implemented
  Mock provider implemented
  RAG planned
  Agent runtime planned
```

No external AI provider API or key is required.

---

## M2 — Basic RAG

**Status: NEXT**

### Objective

Introduce document ingestion, chunking, embeddings, pgvector storage, and basic grounded retrieval.

---

## Development Roadmap

```text
M0  Foundation              COMPLETE
M1  Model Gateway           COMPLETE
M2  Basic RAG               NEXT
M3  Advanced RAG
M4  Agent System
M5  Autonomous Research
M6  Evaluation
M7  Local/Open Model Expansion
M8  Deployment
```

Milestones are incremental. Each milestone should produce a working, tested increment.

---

## Development Workflow

For each milestone:

```text
Specification
     ↓
Implementation
     ↓
Tests
     ↓
Review
     ↓
Fix
     ↓
Commit
     ↓
Next milestone
```

The repository should remain buildable and testable throughout development.

---

## Architecture Principle

AURA is **open-model-first and LLM-agnostic**.

No core component may depend directly on a specific model vendor or inference provider.

Model-specific and provider-specific logic belongs behind the internal model/provider abstraction.

The intended progression is:

```text
Provider Interface
       ↓
Mock Provider
       ↓
Local/Open Model
       ↓
Additional Providers
```

AURA must not require a paid external LLM API for core development.
