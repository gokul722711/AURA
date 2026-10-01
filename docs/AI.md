# AURA — AI Architecture

## 1. Purpose

AURA is an **open-model-first, LLM-agnostic agentic AI platform**.

The AI architecture must allow models and inference providers to change without requiring changes to the core application, agent runtime, RAG pipeline, API layer, or frontend.

No core component may directly depend on a specific model vendor.

---

## 2. Model Gateway

All LLM interaction must pass through the internal Model Gateway.

```text
Agent / RAG
     │
     ▼
Model Gateway
     │
     ▼
LLMProvider
     │
 ┌───┼──────────────┐
 ▼   ▼              ▼
Mock Local/Open   Hosted
     Models       Providers
```

The agent must not directly call an inference SDK or model endpoint.

---

## 3. LLM Provider Interface

The internal provider abstraction should expose capabilities rather than vendor-specific behavior.

Core capabilities:

```text
generate()
stream()
structured_output()
metadata()
```

The exact Python interface should be determined during M1 implementation.

Provider implementations may internally use:

* Local inference servers
* OpenAI-compatible APIs
* NVIDIA NIM
* Ollama
* vLLM
* Other inference systems
* Hosted providers

The rest of AURA must not need to know which implementation is being used.

---

## 4. Initial Provider Strategy

M1 must not require an external model API.

The first implementation should use:

```text
LLMProvider
     │
     ▼
MockLLMProvider
```

The mock provider exists for:

* Deterministic tests
* Gateway development
* Agent development without model costs
* CI
* Interface validation

After the gateway is stable, connect it to a real open/local model.

Potential targets include:

```text
NVIDIA NIM
Ollama
vLLM
Other OpenAI-compatible inference servers
```

These are integration targets, not core dependencies.

---

## 5. Model Configuration

Model configuration must be externalized.

Application code must not contain hard-coded model names, endpoints, credentials, or provider-specific configuration.

Configuration should eventually support concepts such as:

```text
provider
model
endpoint
temperature
max_tokens
timeout
```

Secrets must come from environment variables or an appropriate secret-management system.

---

## 6. Embeddings

Embedding generation follows an internal provider abstraction:

```text
EmbeddingProvider
```

Application and RAG code depend on this interface rather than directly on vendor embedding APIs or local model implementations.

### Interface

Defined in `backend/rag/embeddings/base.py`:

```python
class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_query(self, query: str) -> list[float]: ...

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def dimensions(self) -> int: ...
```

### Provider Registry

Defined in `backend/rag/embeddings/registry.py`:

* Decoupled provider registration via `register_embedding_provider(name, provider_cls)`
* Dynamic instantiation via `create_embedding_provider(name, dimensions=None)`

### `MockEmbeddingProvider`

Defined in `backend/rag/embeddings/mock.py`:

* Generates deterministic, hash-based (SHA-256) float vectors.
* Vectors are unit-normalized for exact cosine distance calculation.
* Default dimensionality is 384 (configured via `AI_EMBEDDINGS["DIMENSIONS"]`).
* Enables offline testing and CI without external API dependencies or heavy local model downloads.

---

## 7. RAG Architecture

The RAG pipeline (M2 + M3) implements the end-to-end flow:

```text
Document
   ↓
Ingestion
   ↓
Deterministic Chunking
   ↓
EmbeddingProvider
   ↓
PostgreSQL / pgvector
   ↓
Query Processing (M3)
   ↓
Cosine Distance Retrieval (M3: threshold before top-K)
   ↓
Context Assembly (M3: budget + redundancy handling)
   ↓
ModelGateway
   ↓
RAG Response (M3: explicit no-context indicator)
```

### 1. Documents & Data Model

Defined in `backend/rag/models.py`:

* `Document`: Stores raw text content, title, source origin, metadata, and ingestion lifecycle status (`pending`, `processing`, `ready`, `error`).
* `DocumentChunk`: Stores chunk text, character offsets (`start_offset`, `end_offset`), sequential index (`chunk_index`), foreign key to `Document`, and vector embedding.
* Vector storage uses `pgvector.django.VectorField(dimensions=384)` with PostgreSQL extension enabled via `pgvector.django.VectorExtension()`.

### 2. Ingestion & Deterministic Chunking

* **Chunking** (`backend/rag/chunking.py`):
  - Deterministic character-based chunking with configurable `chunk_size` (default 512) and `chunk_overlap` (default 50).
  - Produces sequential, non-empty `ChunkResult` structures with exact character offsets.
* **Ingestion** (`backend/rag/ingestion.py`):
  - Validates document inputs and sets initial status to `processing`.
  - Splits text into chunks and generates embeddings batch-wise via `EmbeddingProvider.embed_texts()`.
  - Atomically bulk-persists `DocumentChunk` records inside a `transaction.atomic()` block.
  - Transitions document status to `ready` upon success, or `error` with rollback on failure, preventing partial persistence.

### 3. Query Processing (M3)

Defined in `backend/rag/query_processing.py`:

* Deterministic, provider-independent query normalization.
* Trims surrounding whitespace, normalizes repeated internal whitespace to single spaces.
* Validates that the result is non-empty.
* Returns `ProcessedQuery` with both `original` (unmodified) and `normalized` (for retrieval) forms.
* Does NOT call any LLM or external model.

### 4. Vector Retrieval

Defined in `backend/rag/retrieval.py`:

* Encodes queries via `EmbeddingProvider.embed_query()`.
* Executes database-side exact cosine distance calculation using `pgvector.django.CosineDistance("embedding", query_embedding)`.
* Filters only chunks from documents with `status="ready"`.
* Computes normalized similarity scores: `similarity = 1.0 - distance`.
* **M3 correctness**: Similarity threshold filtering is applied database-side BEFORE the top-K slice, ensuring K qualifying results are returned when K or more exist.
* **M3 determinism**: Deterministic tie-breaking using `(distance ASC, pk ASC)` ordering.
* **M3 result contract**: Enriched `RetrievalResult` exposes `chunk_id`, `document_id`, `content`, `score`, `rank`, `chunk_index`, `document_title`, `document_source`, `start_offset`, `end_offset`, `chunk_metadata`, and backward-compatible `chunk` field.
* Configurable via `AI_RAG["TOP_K"]` (default 5) and `AI_RAG["SIMILARITY_THRESHOLD"]` (default 0.0).
* No approximate vector indexes (HNSW, IVFFlat) are used to ensure exact retrieval.

### 5. Context Assembly

Defined in `backend/rag/context.py`:

* Formats retrieved chunks into a structured prompt block with clear boundary markers.
* Preserves chunk metadata (document title/source, chunk index, similarity score).
* **M3 budget**: Configurable `CONTEXT_MAX_CHARS` budget. Includes highest-ranked results first, stops when budget is exhausted. `None` means no limit (M2 behavior).
* **M3 redundancy**: Deterministic overlapping-chunk redundancy handling. Skips chunks whose character span is fully contained within an already-included chunk from the same document.
* Handles empty retrieval gracefully by prompting the model to indicate absence of relevant context.

### 6. Grounded Generation via Model Gateway

Defined in `backend/rag/pipeline.py`:

* `RAGPipeline.query(question, config)` orchestrates the full process-retrieve-assemble-generate workflow.
* Generation **must** go through `ModelGateway.generate()`. RAG components never directly call an LLM provider or vendor SDK.
* **M3 query processing**: Applies `process_query()` to normalize the query before retrieval.
* **M3 no-context behavior**: `RAGResponse.has_context` distinguishes context-grounded from ungrounded responses. `metadata["context_grounded"]` makes this explicit. No-context responses do not appear document-grounded.
* Returns a structured `RAGResponse` containing:
  - Generated answer text
  - Source citations (`RetrievalResult` records)
  - Original query and `ProcessedQuery` (M3)
  - `has_context` flag (M3)
  - Token usage info (`prompt_tokens`, `completion_tokens`, `total_tokens`)
  - Execution metadata (provider, model, finish reason, retrieval count, context_grounded)

### 7. RAG Evaluation Framework (M3)

Defined in `backend/rag/evaluation/`:

* **Metrics** (`metrics.py`): Deterministic, LLM-independent retrieval quality metrics — Recall@K, Precision@K, Hit Rate@K, MRR. Compare retrieved identifiers against expected relevant identifiers.
* **Dataset** (`dataset.py`): `EvaluationExample` (query + relevant IDs) and `EvaluationDataset` (named collection of examples). Frozen, immutable representations.
* **Evaluator** (`evaluator.py`): Runs evaluation examples through the retrieval pipeline, computes per-example and aggregate metrics. Does NOT use an LLM.
* **Test dataset** (`test_dataset.py`): Small, repository-local deterministic dataset for testing framework mechanics. Does NOT claim meaningful semantic retrieval quality with MockEmbeddingProvider.

### 8. RAG Configuration

Centralized in Django `settings.AI_RAG`:

```text
CHUNK_SIZE          Default 512
CHUNK_OVERLAP       Default 50
TOP_K               Default 5
SIMILARITY_THRESHOLD Default 0.0  (M3)
CONTEXT_MAX_CHARS   Default None  (M3, no limit)
```

All configuration is environment-variable driven via `AI_RAG_*` prefixed variables.


---

## 8. Agent Boundary & Runtime (M4)

The agent runtime (`backend/agent/`) coordinates controlled, stateful, observable, and bounded execution:

```text
Objective
   │
   ▼
AgentRuntime
   │
   ├──► AgentState (serializable: status, plan, step_history, tool_results, errors)
   │
   ├──► Planner (MockPlanner / provider-independent abstraction)
   │       └── Plan [AgentStep: MODEL | TOOL | FINISH]
   │
   ├──► StepExecutor
   │       ├── MODEL ──► ModelGateway (generate)
   │       ├── TOOL  ──► ToolPolicy ──► ToolRegistry ──► Tool
   │       └── FINISH ─► Conclude run with final_output
   │
   ├──► LimitTracker (max_iterations, max_tool_calls, max_time_seconds)
   │
   ├──► Cancellation (synchronous checks at safe boundaries)
   │
   └──► ExecutionTrace (structured immutable events: started, step, tool, model, completed, failed)
```

Key architectural properties:

* **Stateful & Serializable**: `AgentState` tracks lifecycle transitions (`pending` $\rightarrow$ `running` $\rightarrow$ `completed` / `failed` / `cancelled`) without database persistence in M4.
* **Provider-Agnostic Planning**: `Planner` converts `(objective, state)` into a declarative `Plan`. M4 provides `MockPlanner` for deterministic offline testing.
* **Authoritative Tool Policy**: All tool calls pass through `ToolPolicy`. No arbitrary shell, filesystem, or network execution is permitted.
* **Safe Builtin Tools**: `CalculatorTool` uses strict AST allowlisting (no `eval()`, names, or calls; exponent DoS protection). `MockEchoTool` enables deterministic reflection.
* **External RAG Tool**: `RAGSearchTool` bridges to existing M3 retrieval abstractions without coupling runtime internals to vector tables.
* **Bounded Execution**: Hard constraints enforced by `ExecutionLimits` terminate safely with structured failure events upon limit breach.
* **Full Observability**: Emits typed `ExecutionEvent` objects collected in `ExecutionTrace` for complete auditability.

---

## 9. Structured Output

When an agent requires structured model output, the provider abstraction should expose a provider-independent structured-output capability.

Application logic should define the expected structure.

Provider-specific mechanisms for producing that structure belong inside the provider implementation.

---

## 10. Streaming

Streaming should be exposed through the Model Gateway rather than directly through provider SDKs.

The frontend should receive streamed results through the Django API boundary.

The frontend must never communicate directly with an LLM inference server.

---

## 11. Observability

AI operations should expose enough metadata to understand and debug execution.

Important fields include:

```text
run_id
task_id
step
provider
model
latency
tokens
status
error
timestamp
```

Never log API keys, credentials, or other secrets.

---

## 12. Testing

Provider-independent tests should use `MockLLMProvider`.

Tests must not depend on exact wording from a real model.

Prefer testing:

* Gateway behavior
* Provider contracts
* Structured-output handling
* Streaming behavior
* Error handling
* Retries
* Timeouts
* Agent state transitions
* RAG retrieval behavior
* Evaluation metrics

Real-model integration tests should be separate from deterministic unit tests.

---

## 13. Provider Independence

AURA must remain portable across inference systems.

Changing:

```text
Model
Provider
Inference server
Embedding model
```

should not require rewriting:

```text
Agent logic
RAG logic
Django APIs
Frontend
Evaluation system
```

Vendor-specific code belongs at the provider boundary.

---

## 14. Security

External model output is untrusted data.

Model output must not automatically become:

* System instructions
* Tool permissions
* Shell commands
* Filesystem operations
* Arbitrary code
* Trusted facts

Tool execution requires explicit contracts and appropriate validation.

Prompt injection defenses must be considered wherever AURA processes external or retrieved content.

---

## 15. Development Sequence

AI development should proceed incrementally:

```text
M1
Model Gateway
    ↓
Mock Provider
    ↓
Provider Contract Tests

M2
Basic RAG
    ↓
Embedding Provider
    ↓
Retrieval
    ↓
Grounded Generation

M3
Advanced RAG
    ↓
Query Processing
    ↓
Retrieval Correctness
    ↓
Context Optimization
    ↓
Evaluation Framework


M4
Agent Runtime

M5
Autonomous Research

M6
Evaluation

M7
Local/Open Model Expansion

M8
Deployment
```

Local/open inference can be integrated earlier when useful for testing or development. M7 represents expansion and hardening of that capability, not its first introduction.

---

## 16. Core Rule

> **AURA is open-model-first and LLM-agnostic.**

No core AI component should assume OpenAI, NVIDIA, Ollama, vLLM, or any other specific provider.

Providers are implementations behind the AURA model abstraction.
