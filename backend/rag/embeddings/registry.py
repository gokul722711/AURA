"""Embedding provider registry for AURA RAG.

Simple in-memory registry that maps provider names to EmbeddingProvider classes.
Mirrors the M1 LLM provider registry pattern.
"""

from rag.embeddings.base import EmbeddingProvider
from rag.embeddings.mock import MockEmbeddingProvider
from rag.exceptions import EmbeddingError

_REGISTRY: dict[str, type[EmbeddingProvider]] = {
    "mock": MockEmbeddingProvider,
}


def register_embedding_provider(
    name: str, provider_cls: type[EmbeddingProvider]
) -> None:
    """Register a new EmbeddingProvider implementation under a provider name."""
    if not isinstance(name, str):
        raise EmbeddingError("Embedding provider name must be a non-empty string.")
    normalized_name = name.strip().lower()
    if not normalized_name:
        raise EmbeddingError("Embedding provider name must be a non-empty string.")

    try:
        if not issubclass(provider_cls, EmbeddingProvider):
            raise EmbeddingError(
                f"Provider class {provider_cls.__name__} must inherit from EmbeddingProvider."
            )
    except TypeError as exc:
        raise EmbeddingError(
            f"Provider class {provider_cls} must be a class inheriting from EmbeddingProvider."
        ) from exc

    _REGISTRY[normalized_name] = provider_cls


def get_embedding_provider_class(name: str) -> type[EmbeddingProvider]:
    """Retrieve the embedding provider class for the given name."""
    key = name.lower().strip()
    if key not in _REGISTRY:
        raise EmbeddingError(
            f"Embedding provider '{name}' is not registered. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[key]


def create_embedding_provider(
    provider_name: str, dimensions: int
) -> EmbeddingProvider:
    """Instantiate an embedding provider by name with the given dimensions."""
    provider_cls = get_embedding_provider_class(provider_name)
    try:
        return provider_cls(dimensions=dimensions)
    except Exception as exc:
        raise EmbeddingError(
            f"Failed to initialize embedding provider '{provider_name}': {exc}"
        ) from exc
