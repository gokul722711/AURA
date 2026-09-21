"""Abstract base interface for embedding providers."""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    All embedding providers must implement this interface.
    The provider layer hides inference-provider details from RAG code.
    """

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Used for embedding document chunks during ingestion.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.
            Each vector has length equal to self.dimensions().
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a single query string.

        Used for embedding search queries during retrieval.
        Some providers use different models for queries vs documents;
        this method allows that distinction.

        Args:
            query: The query text to embed.

        Returns:
            Embedding vector with length equal to self.dimensions().
        """
        pass

    @abstractmethod
    def dimensions(self) -> int:
        """Return the dimensionality of the embedding vectors produced.

        Returns:
            Integer dimension count (e.g. 384, 768, 1536).
        """
        pass
