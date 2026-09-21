"""Django app configuration for AURA RAG."""

from django.apps import AppConfig


class RagConfig(AppConfig):
    """Configuration for the RAG application."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "rag"
    verbose_name = "AURA RAG"
