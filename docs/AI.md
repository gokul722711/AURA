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

The Basic RAG pipeline (M2) implements the end-to-end flow:

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
Cosine Distance Retrieval
   ↓
Context Assembly
   ↓
ModelGateway
   ↓
RAG Response
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

### 3. Vector Retrieval

Defined in `backend/rag/retrieval.py`:

* Encodes queries via `EmbeddingProvider.embed_query()`.
* Executes database-side exact cosine distance calculation using `pgvector.django.CosineDistance("embedding", query_embedding)`.
* Filters only chunks from documents with `status="ready"`.
* Computes normalized similarity scores: `similarity = 1.0 - distance`.
* Applies `top_k` limiting (default 5) and optional `similarity_threshold` filtering.
* No approximate vector indexes (HNSW, IVFFlat) are used in M2 to ensure exact retrieval.

### 4. Context Assembly

Defined in `backend/rag/context.py`:

* Formats retrieved chunks into a structured prompt block with clear boundary markers.
* Preserves chunk metadata (document title/source, chunk index, similarity score).
* Handles empty retrieval gracefully by prompting the model to indicate absence of relevant context.

### 5. Grounded Generation via Model Gateway

Defined in `backend/rag/pipeline.py`:

* `RAGPipeline.query(question, config)` orchestrates the full retrieve-assemble-generate workflow.
* Generation **must** go through `ModelGateway.generate()`. RAG components never directly call an LLM provider or vendor SDK.
* Returns a structured `RAGResponse` containing:
  - Generated answer text
  - Source citations (`RetrievalResult` records)
  - Original query
  - Token usage info (`prompt_tokens`, `completion_tokens`, `total_tokens`)
  - Execution metadata (provider, model, finish reason, retrieval count)

---

## 8. Agent Boundary

The future agent runtime may contain:

```text
Planner
Researcher
Retriever
Tool Executor
Synthesizer
Critic
Evaluator
```

These components communicate with models through the Model Gateway.

They must not contain provider-specific code.

Agent execution should be:

* Explicit
* Stateful
* Observable
* Bounded
* Testable

Execution limits should exist for:

```text
steps
retries
tool calls
tokens
time
```

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
