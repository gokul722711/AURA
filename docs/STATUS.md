# AURA — Project Status

## Current Phase

**M0 — Foundation**

## Repository State

The repository contains the application foundation: a Django backend with
Django REST Framework, a Next.js + TypeScript frontend, and PostgreSQL
configuration via environment variables.

Current structure:

```text
AURA/
├── AGENTS.md
├── README.md
├── .gitignore
├── .env.example
├── docs/
│   └── STATUS.md
├── specs/
│   └── tasks/
│       └── TASK-001.md
├── backend/
│   ├── manage.py
│   ├── requirements.txt
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   ├── urls.py
│   │   ├── wsgi.py
│   │   └── asgi.py
│   └── health/
│       ├── __init__.py
│       ├── views.py
│       ├── urls.py
│       └── tests.py
└── frontend/
    ├── package.json
    ├── package-lock.json
    ├── next.config.ts
    ├── tsconfig.json
    ├── eslint.config.mjs
    └── src/
        └── app/
            ├── globals.css
            ├── layout.tsx
            └── page.tsx
```

## Implemented

* Git repository initialized
* GitHub repository created
* Initial agent instructions created
* Initial project README created
* Documentation structure created
* Django backend under `backend/`
* Django REST Framework configured
* PostgreSQL configuration via environment variables
* Health-check endpoint: `GET /api/health/`
* Health-check endpoint tests
* Next.js + TypeScript frontend under `frontend/`
* AURA landing page
* `.env.example` with safe placeholder values
* `.gitignore` for Python, Node.js, environment files

## Not Yet Implemented

* PostgreSQL database and user creation (manual setup required)
* pgvector
* Redis
* LLM provider abstraction
* OpenAI integration
* Embeddings
* RAG
* Agent runtime
* LangGraph
* Tools
* Evaluation system
* Authentication
* Production deployment

## Immediate Next Task

Set up the PostgreSQL database and user, then run migrations and tests.

After that, proceed to M1 — LLM Gateway.

## Development Rule

AURA is built incrementally.

Each task should produce a working and testable change before the next capability is introduced.
