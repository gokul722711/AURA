# TASK-001 — M0 Foundation

## Objective

Initialize the AURA application foundation as a clean monorepo containing:

* Django backend
* Django REST Framework
* Next.js frontend
* TypeScript
* PostgreSQL configuration
* Environment-based configuration
* Basic health-check endpoint

The result must run locally and provide a clean foundation for later AI functionality.

---

## Read First

Before implementation, read:

```text
AGENTS.md
README.md
docs/STATUS.md
```

Inspect the repository before creating files.

---

## Target Structure

Create a structure similar to:

```text
AURA/
├── AGENTS.md
├── README.md
├── .gitignore
├── .env.example
├── docs/
├── specs/
├── backend/
│   ├── manage.py
│   ├── config/
│   └── ...
└── frontend/
    ├── package.json
    ├── next.config.*
    └── ...
```

The exact framework-generated files may differ.

---

## Backend Requirements

Create a Django project under:

```text
backend/
```

Install/configure:

* Django
* Django REST Framework
* PostgreSQL driver

Create a basic API health endpoint.

Example:

```text
GET /api/health/
```

It should return a small JSON response indicating that the backend is running.

Do not implement authentication yet.

Do not implement AI functionality yet.

---

## Frontend Requirements

Create a Next.js application under:

```text
frontend/
```

Use:

* TypeScript
* React
* Next.js

Create a minimal landing page identifying the application as AURA.

The frontend should run independently during development.

---

## Database Configuration

Configure Django for PostgreSQL using environment variables.

Do not hardcode:

* database host
* database name
* username
* password
* port

Provide:

```text
.env.example
```

with safe placeholder values.

Do not commit a real `.env` file.

A local development PostgreSQL instance may be used for now.

Do not add unnecessary database abstractions.

---

## Environment Configuration

Environment-specific values must be configurable without modifying source code.

At minimum support:

```text
DJANGO_SECRET_KEY
DJANGO_DEBUG
DATABASE_NAME
DATABASE_USER
DATABASE_PASSWORD
DATABASE_HOST
DATABASE_PORT
```

Keep configuration simple at this stage.

---

## Dependency Management

Use conventional dependency files appropriate for each ecosystem.

Backend dependencies should be reproducible.

Frontend dependencies should be managed through the generated Next.js package configuration.

Do not add AI/LLM dependencies yet.

Do not add LangChain or LangGraph yet.

---

## Testing

Add a basic backend test confirming:

```text
GET /api/health/
```

returns a successful response.

Verify the Django application can start.

Verify the Next.js application can start.

---

## Constraints

Do not implement:

* OpenAI integration
* LLMProvider
* EmbeddingProvider
* RAG
* pgvector
* Redis
* LangGraph
* LangChain
* Agents
* Tools
* Evaluation
* Authentication
* Background workers
* Production deployment

Those belong to later tasks.

Do not perform unrelated refactoring.

Do not introduce large dependencies without justification.

---

## Acceptance Criteria

The task is complete when:

* [ ] Django backend exists under `backend/`.
* [ ] Django REST Framework is configured.
* [ ] PostgreSQL configuration uses environment variables.
* [ ] `/api/health/` returns a successful JSON response.
* [ ] Backend health endpoint has a test.
* [ ] Next.js + TypeScript frontend exists under `frontend/`.
* [ ] Frontend displays a basic AURA page.
* [ ] `.env.example` exists.
* [ ] Real secrets are not committed.
* [ ] Backend dependencies are reproducible.
* [ ] Frontend dependencies are reproducible.
* [ ] Backend starts successfully.
* [ ] Frontend starts successfully.
* [ ] Existing repository documentation remains intact.

---

## Expected Report

After implementation, report:

1. Files created/changed.
2. Dependencies added.
3. Commands used to run the backend.
4. Commands used to run the frontend.
5. Tests executed.
6. Test results.
7. Any limitations or unresolved issues.

Do not claim completion if the acceptance criteria were not verified.
