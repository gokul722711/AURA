# AURA

**Autonomous Research & Engineering Agent**

AURA is an **open-model-first, LLM-agnostic agentic AI platform** designed to handle complex research and engineering objectives through planning, retrieval, tool use, synthesis, evaluation, and iterative improvement.

## Vision

Given a complex objective, AURA should be able to:

1. Understand the objective.
2. Break it into actionable tasks.
3. Retrieve relevant information.
4. Use appropriate tools.
5. Reason over evidence.
6. Produce a grounded result.
7. Critically evaluate the result.
8. Improve or retry when necessary.
9. Provide sources and execution traces.
10. Measure the quality of the result.

## Technology

### Backend

* Python
* Django
* Django REST Framework

### Frontend

* React
* Next.js
* JavaScript

**TypeScript is not used in AURA.**

### Data & Infrastructure

* PostgreSQL
* pgvector
* Redis

### AI

* LLM provider abstraction
* Embedding provider abstraction
* Retrieval-Augmented Generation (RAG)
* Agent workflows
* LangGraph
* Selective LangChain integrations
* Open/local model inference

AURA does not depend on a specific LLM vendor or inference provider.

Potential model and inference targets include:

* NVIDIA NIM
* Ollama
* vLLM
* Other OpenAI-compatible inference servers
* Other local/self-hosted models
* Hosted model providers

These providers are accessed through internal abstractions so that changing the model or inference system does not require changes to the core application architecture.

## Architecture

```text
Next.js / React
      │
      ▼
Django REST API
      │
      ├── Projects
      ├── Documents
      ├── Tasks
      ├── Agent Runs
      └── Evaluations
             │
             ▼
       Agent Runtime
             │
      ┌──────┼──────┐
      ▼      ▼      ▼
     RAG    Tools   LLM Gateway
      │              │
      ▼              ▼
PostgreSQL       LLM Provider
+ pgvector       Abstraction
      │
      ▼
    Redis
```

The LLM Gateway isolates the application from specific models and inference providers.

For example:

```text
Agent Runtime
      │
      ▼
  LLM Gateway
      │
      ▼
  LLMProvider
      │
 ┌────┼──────────┐
 ▼    ▼          ▼
Mock  Local    Hosted
      Models   Providers
```

This architecture is intentionally high-level. Detailed architecture belongs in `docs/ARCHITECTURE.md`.

## Development

AURA is being developed incrementally.

```text
M0  Foundation
M1  Model Gateway
M2  Basic RAG
M3  Advanced RAG
M4  Agent System
M5  Autonomous Research
M6  Evaluation
M7  Local/Open Model Expansion
M8  Deployment
```

Each milestone should result in a working and testable increment.

The initial AI development path is:

```text
Provider Interface
       ↓
Mock Provider
       ↓
Local/Open Model
       ↓
Additional Providers
```

The early development process must not require a paid external LLM API.

## Repository Structure

```text
AURA/

├── AGENTS.md
├── README.md
├── docs/
├── specs/
├── backend/
└── frontend/
```

The repository documentation and implementation together form the project source of truth.

## Current Status

**Phase:** M4 — Agent System (Complete)

Milestones M0 through M4 are fully implemented and verified:

* **M0 Foundation**: Django backend, DRF, Next.js frontend, PostgreSQL/pgvector.
* **M1 Model Gateway**: Provider-agnostic LLM Gateway with MockLLMProvider.
* **M2 Basic RAG**: Ingestion, chunking, embeddings, pgvector cosine retrieval, grounded generation.
* **M3 Advanced RAG**: Database-side threshold filtering, context budget, redundancy handling, query processing, retrieval evaluation framework.
* **M4 Agent System**: Stateful agent runtime, deterministic MockPlanner, tool registry, authoritative security policy, safe calculator and echo tools, execution limits, structured execution events/trace, and RAG tool integration.

See `docs/STATUS.md` for detailed milestone implementation reports.

## Development Principle

> **AURA is not an OpenAI application. AURA is an open-model-first, LLM-agnostic agentic AI platform.**

No core component should depend directly on a specific model vendor or inference provider.

Model and provider-specific logic belongs behind the provider abstraction.

The architecture should allow AURA to move between models and inference systems without requiring changes to the core agent, RAG, API, frontend, or evaluation architecture.
