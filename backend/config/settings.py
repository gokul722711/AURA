"""
Django settings for AURA project.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/topics/settings/
"""

import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from project root (.env) if present
try:
    from dotenv import load_dotenv

    env_path = BASE_DIR.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

DEBUG = os.environ.get("DJANGO_DEBUG", "True").lower() in ("true", "1", "yes")

_SECRET_KEY_FALLBACK = "django-insecure-dev-only-key-do-not-use-in-production"

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "")
if not SECRET_KEY:
    if DEBUG:
        SECRET_KEY = _SECRET_KEY_FALLBACK
    else:
        from django.core.exceptions import ImproperlyConfigured

        raise ImproperlyConfigured(
            "DJANGO_SECRET_KEY environment variable is required when DEBUG is False."
        )

ALLOWED_HOSTS = ["localhost", "127.0.0.1", "0.0.0.0"]

# ---------------------------------------------------------------------------
# Application definition
# ---------------------------------------------------------------------------

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # Third-party
    "rest_framework",
    # Local
    "health",
    "gateway",
    "rag",
    "agent",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

# ---------------------------------------------------------------------------
# Database
# https://docs.djangoproject.com/en/5.2/ref/settings/#databases
# ---------------------------------------------------------------------------

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ.get("DATABASE_NAME", "aura"),
        "USER": os.environ.get("DATABASE_USER", "aura"),
        "PASSWORD": os.environ.get("DATABASE_PASSWORD", "aura"),
        "HOST": os.environ.get("DATABASE_HOST", "localhost"),
        "PORT": os.environ.get("DATABASE_PORT", "5432"),
    }
}

# ---------------------------------------------------------------------------
# Password validation
# https://docs.djangoproject.com/en/5.2/ref/settings/#auth-password-validators
# ---------------------------------------------------------------------------

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

# ---------------------------------------------------------------------------
# Internationalization
# https://docs.djangoproject.com/en/5.2/topics/i18n/
# ---------------------------------------------------------------------------

LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_TZ = True

# ---------------------------------------------------------------------------
# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.2/howto/static-files/
# ---------------------------------------------------------------------------

STATIC_URL = "static/"

# ---------------------------------------------------------------------------
# Default primary key field type
# https://docs.djangoproject.com/en/5.2/ref/settings/#default-auto-field
# ---------------------------------------------------------------------------

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ---------------------------------------------------------------------------
# Django REST Framework
# ---------------------------------------------------------------------------

REST_FRAMEWORK = {
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.AllowAny",
    ],
}

# ---------------------------------------------------------------------------
# Model Gateway
# ---------------------------------------------------------------------------

AI_GATEWAY = {
    "PROVIDER": os.environ.get("AI_PROVIDER", "mock"),
    "MODEL": os.environ.get("AI_MODEL", "mock-model"),
    "ENDPOINT": os.environ.get("AI_ENDPOINT", ""),
    "TIMEOUT": float(os.environ.get("AI_TIMEOUT", "30.0")),
    "API_KEY": os.environ.get("AI_API_KEY", ""),
    "MAX_RETRIES": int(os.environ.get("AI_GATEWAY_MAX_RETRIES", "2")),
    "RETRY_BACKOFF_BASE": float(os.environ.get("AI_GATEWAY_RETRY_BACKOFF_BASE", "5.0")),
    "RETRY_BACKOFF_FACTOR": float(os.environ.get("AI_GATEWAY_RETRY_BACKOFF_FACTOR", "3.0")),
}

# ---------------------------------------------------------------------------
# Embedding Configuration
# ---------------------------------------------------------------------------
# Embedding dimensions are fixed per schema. Changing dimensions requires
# a database migration and schema change.

AI_EMBEDDINGS = {
    "PROVIDER": os.environ.get("AI_EMBEDDING_PROVIDER", "mock"),
    "MODEL": os.environ.get("AI_EMBEDDING_MODEL", "mock-embedding"),
    "DIMENSIONS": int(os.environ.get("AI_EMBEDDING_DIMENSIONS", "384")),
}

# ---------------------------------------------------------------------------
# RAG Configuration
# ---------------------------------------------------------------------------

AI_RAG = {
    "CHUNK_SIZE": int(os.environ.get("AI_RAG_CHUNK_SIZE", "512")),
    "CHUNK_OVERLAP": int(os.environ.get("AI_RAG_CHUNK_OVERLAP", "50")),
    "TOP_K": int(os.environ.get("AI_RAG_TOP_K", "5")),
    "SIMILARITY_THRESHOLD": float(os.environ.get("AI_RAG_SIMILARITY_THRESHOLD", "0.0")),
    "CONTEXT_MAX_CHARS": (
        int(os.environ.get("AI_RAG_CONTEXT_MAX_CHARS"))
        if os.environ.get("AI_RAG_CONTEXT_MAX_CHARS")
        else None
    ),
    "RERANKER_MODEL": os.environ.get("AI_RAG_RERANKER_MODEL", "ms-marco-TinyBERT-L-2-v2"),
    "RERANKER_CACHE_DIR": os.environ.get("AI_RAG_RERANKER_CACHE_DIR", "/tmp"),
    "RERANKER_MIN_SCORE": float(os.environ.get("AI_RAG_RERANKER_MIN_SCORE", "0.0001")),
}

# ---------------------------------------------------------------------------
# Agent Runtime Configuration
# ---------------------------------------------------------------------------

AI_AGENT = {
    "MAX_ITERATIONS": int(os.environ.get("AI_AGENT_MAX_ITERATIONS", "10")),
    "MAX_TOOL_CALLS": int(os.environ.get("AI_AGENT_MAX_TOOL_CALLS", "15")),
    "MAX_TIME_SECONDS": float(os.environ.get("AI_AGENT_MAX_TIME_SECONDS", "60.0")),
}

# ---------------------------------------------------------------------------
# Celery Configuration
# ---------------------------------------------------------------------------

CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_TIMEZONE = TIME_ZONE
CELERY_TASK_ALWAYS_EAGER = os.environ.get("CELERY_TASK_ALWAYS_EAGER", "False").lower() in ("true", "1", "yes")
CELERY_TASK_EAGER_PROPAGATES = True

# ---------------------------------------------------------------------------
# Web Search Configuration (M14)
# ---------------------------------------------------------------------------

AI_WEB_SEARCH = {
    "PROVIDER": os.environ.get("AI_WEB_SEARCH_PROVIDER", "searxng").strip().lower(),
    "SEARXNG_URL": os.environ.get("AI_SEARXNG_URL", "http://localhost:8080").strip(),
    "TIMEOUT": float(os.environ.get("AI_WEB_SEARCH_TIMEOUT", "10.0")),
    "SEARCH_TOP_K": int(os.environ.get("AI_WEB_SEARCH_TOP_K", "10")),
    "MAX_FETCH_PAGES": int(os.environ.get("AI_WEB_SEARCH_MAX_FETCH_PAGES", "5")),
    "MAX_EXTRACTED_CHARS": int(os.environ.get("AI_WEB_SEARCH_MAX_EXTRACTED_CHARS", "100000")),
}
