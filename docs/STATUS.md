# AURA — Development Status

## Current Phase

**M3 — Advanced RAG: COMPLETE**

**Next: M4 — Agent System**

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
  Agent runtime planned
```

No external AI provider API or key is required.

---

## M4 — Agent System

**Status: NEXT**

### Objective

Introduce agent runtime with explicit, stateful, observable, bounded execution.

---

## Development Roadmap

```text
M0  Foundation              COMPLETE
M1  Model Gateway           COMPLETE
M2  Basic RAG               COMPLETE
M3  Advanced RAG            COMPLETE
M4  Agent System            NEXT
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
