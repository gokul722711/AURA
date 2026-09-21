"""Vector retrieval for AURA RAG.

Performs database-side cosine distance search using pgvector,
returning the most similar document chunks for a query.

Internally, pgvector CosineDistance computes cosine distance (0 = identical, 2 = opposite).
The public RetrievalResult exposes a similarity score: similarity = 1 - distance.
This means:
  - similarity = 1.0: identical vectors
  - similarity = 0.0: orthogonal vectors
  - similarity < 0.0: opposite vectors

Higher similarity = more relevant.

No approximate vector indexes (HNSW, IVFFlat) are used in M2.
Retrieval uses exact database-side cosine distance for correctness.
"""

from dataclasses import dataclass

from django.conf import settings
from pgvector.django import CosineDistance

from rag.embeddings.base import EmbeddingProvider
from rag.exceptions import RetrievalError
from rag.models import DocumentChunk


@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration for chunk retrieval.

    Attributes:
        top_k: Maximum number of chunks to return.
        similarity_threshold: Minimum similarity score (1 - cosine_distance).
            Chunks below this threshold are excluded. Default 0.0 (no filtering).
    """

    top_k: int = 5
    similarity_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise RetrievalError("top_k must be greater than 0.")


@dataclass(frozen=True)
class RetrievalResult:
    """A single retrieval result.

    Attributes:
        chunk: The matched DocumentChunk instance.
        score: Similarity score (1 - cosine_distance). Higher = more similar.
        rank: 1-based rank in the result set.
    """

    chunk: DocumentChunk
    score: float
    rank: int


def _get_retrieval_config() -> RetrievalConfig:
    """Build RetrievalConfig from Django settings."""
    rag_settings = getattr(settings, "AI_RAG", {})
    return RetrievalConfig(
        top_k=rag_settings.get("TOP_K", 5),
    )


def retrieve_chunks(
    query: str,
    embedding_provider: EmbeddingProvider,
    config: RetrievalConfig | None = None,
) -> list[RetrievalResult]:
    """Retrieve the most similar document chunks for a query.

    Embeds the query, then performs exact cosine distance search in PostgreSQL
    using pgvector. Results are ordered by similarity (highest first).

    Args:
        query: The search query text.
        embedding_provider: Provider to embed the query.
        config: Retrieval configuration. Uses settings defaults if None.

    Returns:
        List of RetrievalResult ordered by descending similarity.

    Raises:
        RetrievalError: If retrieval fails.
    """
    if not query or not query.strip():
        raise RetrievalError("Query must be a non-empty string.")

    if config is None:
        config = _get_retrieval_config()

    try:
        query_embedding = embedding_provider.embed_query(query)
    except Exception as exc:
        raise RetrievalError(f"Failed to embed query: {exc}") from exc

    # Only search chunks from documents with 'ready' status
    queryset = DocumentChunk.objects.filter(
        document__status="ready"
    ).annotate(
        distance=CosineDistance("embedding", query_embedding)
    ).order_by("distance")[: config.top_k]

    results = []
    for rank, chunk in enumerate(queryset, start=1):
        # Convert cosine distance to similarity: similarity = 1 - distance
        similarity = 1.0 - chunk.distance
        if similarity >= config.similarity_threshold:
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=similarity,
                    rank=rank,
                )
            )

    return results
