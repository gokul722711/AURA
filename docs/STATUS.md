# AURA — Development Status

## Current Phase

**M5 — Autonomous Research: COMPLETE**

**Next: M6 — Evaluation**

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

### M2 — Basic RAG

Minimal, production-oriented basic retrieval-augmented generation pipeline implemented on top of the Model Gateway foundation.

Implemented:

* Document and DocumentChunk data models with pgvector `VectorField(dimensions=384)`
* Initial migration with `VectorExtension()` operation
* Provider-agnostic `EmbeddingProvider` interface (`embed_query`, `embed_texts`, `dimensions`)
* Deterministic `MockEmbeddingProvider` (hash-based unit-normalized 384-dimensional vectors)
* Embedding provider registry (`register_embedding_provider`, `create_embedding_provider`)
* Deterministic character-based chunking with configurable overlap (`chunk_size=512`, `chunk_overlap=50`)
* Ingestion pipeline (`ingest_document`) with atomic chunk persistence and status transitions
* Database-side exact cosine distance retrieval using `pgvector.django.CosineDistance`
* Similarity score calculation (`1.0 - cosine_distance`) and threshold/top-k filtering
* Structured context assembly formatting retrieved sources and chunk text for LLM prompts
* RAG pipeline orchestration (`RAGPipeline`) routing generation strictly through `ModelGateway.generate()`
* Configuration via Django settings (`AI_EMBEDDINGS`, `AI_RAG`) and `.env.example`
* Comprehensive test suite covering models, chunking, embeddings, ingestion, retrieval, context, and pipeline

Verified:

* PostgreSQL 18.4 with pgvector 0.8.0 server extension
* Python `pgvector` package installed in backend environment
* Django migration `rag.0001_initial` applied successfully
* 164/164 backend automated tests pass (including 98 RAG tests)
* Django system checks pass (`python manage.py check`)
* Model migration check passes with no pending changes
* Frontend Next.js production build passes with zero TypeScript
* `git diff --check` passes with zero whitespace issues
* Zero vendor LLM/embedding SDK dependencies introduced

### M3 — Advanced RAG

Retrieval pipeline correctness, configurability, and measurability improvements. No agent system, no hybrid search, no reranking, no external model calls.

Implemented:

* **Retrieval correctness**: Similarity threshold filtering applied database-side BEFORE top-K slicing, ensuring K qualifying results are returned when K or more exist. Deterministic tie-breaking using `(distance ASC, pk ASC)` ordering.
* **Retrieval result contract**: Enriched `RetrievalResult` exposing `chunk_id`, `document_id`, `content`, `score`, `rank`, `chunk_index`, `document_title`, `document_source`, `start_offset`, `end_offset`, `chunk_metadata`, and backward-compatible `chunk` field.
* **Query processing**: Deterministic, provider-independent `process_query()` with whitespace trimming, repeated whitespace normalization, empty query validation. Original query preserved separately from normalized retrieval query. No LLM calls.
* **Context optimization**: Configurable context budget (`CONTEXT_MAX_CHARS`) via `ContextConfig`. Context assembler preserves retrieval ranking, includes highest-ranked results first, stops when budget is exhausted, retains source/chunk attribution, and handles overlapping-chunk redundancy via character offset comparison. No learned reranker.
* **Explicit no-context behavior**: `RAGResponse.has_context` flag distinguishes context-grounded from ungrounded responses. `metadata["context_grounded"]` makes the distinction explicit. No-context results do not appear document-grounded.
* **RAG evaluation framework**: Under `backend/rag/evaluation/` with deterministic, LLM-independent retrieval metrics (Recall@K, Precision@K, Hit Rate@K, MRR), evaluation dataset representation (`EvaluationExample`, `EvaluationDataset`), evaluator comparing retrieved identifiers against expected relevant identifiers, and a small repository-local test dataset.
* **Configuration**: `AI_RAG` extended with `SIMILARITY_THRESHOLD` and `CONTEXT_MAX_CHARS`. All M3 configuration centralized in Django settings.
* **Comprehensive test suite**: 85 new tests covering retrieval correctness, query processing, context optimization, evaluation metrics, and pipeline integration.

Verified:

* 249/249 backend automated tests pass (164 M0–M2 + 85 M3)
* Django system checks pass (`python manage.py check`)
* Model migration check passes with no pending changes (`makemigrations --check --dry-run`)
* Frontend Next.js production build passes
* `git diff --check` passes with zero whitespace issues
* Zero vendor LLM/embedding SDK dependencies introduced
* No new database migrations required

---

## Current Repository State

```text
Backend
  Python
  Django
  Django REST Framework
  PostgreSQL
  Model Gateway (M1)
  Basic RAG (M2)
  Advanced RAG (M3)
  Agent System (M4)

Frontend
  Next.js
  React
  JavaScript / JSX

Infrastructure
  PostgreSQL (18.4)
  pgvector (0.8.0)
  Redis planned

AI
  Model Gateway implemented
  Mock LLM provider implemented
  EmbeddingProvider abstraction implemented
  Mock embedding provider implemented
  Basic RAG pipeline implemented
  Advanced RAG pipeline implemented
  Query processing implemented
  Context optimization implemented
  RAG evaluation framework implemented
  Agent runtime implemented
```

No external AI provider API or key is required.

---

## M4 — Agent System

**Status: COMPLETE**

### Objective

Introduce agent runtime with explicit, stateful, observable, bounded execution.

Implemented:

* **State Model** (`backend/agent/state.py`): In-memory, serializable `AgentState` tracking `run_id`, `objective`, `status` (`pending`, `running`, `waiting`, `completed`, `failed`, `cancelled`), `iteration`, `step_history`, `tool_results`, `final_output`, `errors`, and `metadata`. Strict transition validation.
* **Planning** (`backend/agent/planning/`): Provider-independent `Planner` abstraction, structured `Plan` and `AgentStep` with `ActionType` (`MODEL`, `TOOL`, `FINISH`). Primary deterministic `MockPlanner` with preconfigured and rule-based step generation for offline testing.
* **Tool System & Registry** (`backend/agent/tools/`): Abstract `Tool` contract with JSON schemas and `ToolResult`. Centralized `ToolRegistry` with duplicate name prevention and schema introspection.
* **Authoritative Policy Layer** (`backend/agent/tools/policy.py`): Strict `ToolPolicy` / `DefaultToolPolicy` evaluating `ALLOWED`, `DENIED`, `UNKNOWN`. Tool existence checks, explicit allowlists, denylists, and input validation. No arbitrary shell, filesystem, or network execution.
* **Safe Builtin Tools** (`backend/agent/tools/builtin/`): Safe `CalculatorTool` using strict AST allowlisting (no `eval`, names, calls, or imports; exponent DoS protection) and `MockEchoTool` for pipeline validation.
* **Execution & Limits** (`backend/agent/execution/`): `StepExecutor` dispatching actions to `ModelGateway`, `ToolRegistry`, or `FINISH`. Strict `ExecutionLimits` and `LimitTracker` enforcing bounds on `max_iterations`, `max_tool_calls`, and `max_time_seconds`.
* **Execution Events & Trace** (`backend/agent/events/`): Structured, immutable `ExecutionEvent` dataclasses and `ExecutionTrace` container providing full auditability without external observability platforms.
* **Agent Runtime & Cancellation** (`backend/agent/runtime.py`): Top-level `AgentRuntime` orchestrating the state machine lifecycle. Synchronous cancellation support checked at safe boundaries (pre-planning, pre-step, post-step).
* **RAG Tool Integration** (`backend/agent/tools/builtin/rag.py`): `RAGSearchTool` wrapping existing M3 `retrieve_chunks` without direct coupling between `AgentRuntime` and RAG internals.
* **Comprehensive Test Suite**: 75 tests covering state, transitions, serialization, tools, calculator safety, registry, policy, limits, events, trace, executor, runtime lifecycle, limits enforcement, cancellation, and RAG search.

---

## M5 — Autonomous Research

**Status: COMPLETE**

### Objective

Deliver autonomous, iterative, LLM-driven research capability grounded in the indexed knowledge base using NVIDIA Nemotron behind the Model Gateway abstraction.

Implements:

* **Real Provider Integration** (`backend/gateway/providers/nvidia.py`): OpenAI-compatible `NvidiaLLMProvider` targeting NVIDIA NIM (`https://integrate.api.nvidia.com/v1`, model: `nvidia/nemotron-3-ultra-550b-a55b`). Provider is completely isolated behind `LLMProvider` and `ModelGateway`. Reads API key securely from `AI_API_KEY` without logging, printing, or hardcoding secrets. Client dependency injection supports deterministic offline testing.
* **Autonomous Research Planner** (`backend/agent/planning/research.py`): `ResearchPlanner` implementing the iterative LLM-driven research loop:
  `Objective → LLM decision (continue with query / finish) → RAGSearchTool → Accumulated Evidence → LLM decision → ... → Grounded Final Synthesis`.
  Enforces constrained structured JSON decisions (`{"decision": "continue", "query": "..."}` or `{"decision": "finish"}`).
* **Evidence Accumulation**: Accumulated uniquely across queries in `state.tool_results` preserving `chunk_id`, `document_title`, `document_source`, `score`, and `content`.
* **Grounded Final Synthesis**: Strict grounding rules ensuring the final response references supporting evidence chunks. If insufficient evidence exists in the knowledge base, the agent explicitly states this without hallucinating.
* **Bounded Execution**: Full reuse of M4 `AgentRuntime`, `ExecutionLimits`, `LimitTracker`, `ToolPolicy`, `ExecutionTrace`, and cancellation boundaries.
* **Factory Helper** (`backend/agent/research.py`): `create_research_runtime()` wiring `ResearchPlanner`, `RAGSearchTool`, and `AgentRuntime`.
* **Deterministic Tests**: 22 new tests (11 provider tests in `backend/gateway/tests/test_nvidia.py` and 11 research tests in `backend/agent/tests/test_research.py`) covering all single-query, multi-query, insufficient evidence, limits, cancellation, and error cases offline without requiring network access or external API keys.

Verified:

* 360/360 backend tests pass (100% offline, deterministic)
* System check (`python manage.py check`) passes with zero issues
* Model migrations check (`makemigrations --check --dry-run`) passes
* Frontend Next.js production build passes with zero TypeScript
* `git diff --check` passes
* Zero secrets in code, docs, tests, or .env.example

---

## Development Roadmap

```text
M0  Foundation              COMPLETE
M1  Model Gateway           COMPLETE
M2  Basic RAG               COMPLETE
M3  Advanced RAG            COMPLETE
M4  Agent System            COMPLETE
M5  Autonomous Research     COMPLETE
M6  Evaluation              NEXT
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
