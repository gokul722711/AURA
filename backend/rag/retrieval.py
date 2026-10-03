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

M3 correctness fix:
  Similarity threshold filtering is applied database-side BEFORE the top-K
  slice. This ensures that exactly K qualifying results are returned when
  K or more exist, even if some candidates would have been filtered out.

  Deterministic tie-breaking uses (distance ASC, chunk pk ASC) ordering.

No approximate vector indexes (HNSW, IVFFlat) are used.
Retrieval uses exact database-side cosine distance for correctness.
"""

from dataclasses import dataclass

from django.conf import settings
from django.contrib.postgres.search import SearchQuery, SearchRank, SearchVector
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
        document_ids: Optional list of document UUIDs to restrict retrieval to.
        max_chunks_per_document: Optional maximum number of chunks to retrieve from
            any single document (prevents a single document from dominating context).
    """

    top_k: int = 5
    similarity_threshold: float = 0.0
    document_ids: list[str] | None = None
    max_chunks_per_document: int | None = None

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise RetrievalError("top_k must be greater than 0.")
        if self.max_chunks_per_document is not None and self.max_chunks_per_document <= 0:
            raise RetrievalError("max_chunks_per_document must be greater than 0.")


@dataclass(frozen=True)
class RetrievalResult:
    """A single retrieval result.

    Attributes:
        chunk_id: UUID of the matched chunk.
        document_id: UUID of the source document.
        content: Text content of the chunk.
        score: Similarity score (1 - cosine_distance). Higher = more similar.
        rank: 1-based rank in the result set.
        chunk_index: Position of this chunk within its source document (0-based).
        document_title: Title of the source document.
        document_source: Source origin of the document.
        start_offset: Character start offset of this chunk in the source document.
        end_offset: Character end offset of this chunk in the source document.
        chunk_metadata: Metadata dictionary from the chunk record.
        chunk: The matched DocumentChunk instance (backward compatibility with M2).
    """

    chunk_id: str
    document_id: str
    content: str
    score: float
    rank: int
    chunk_index: int
    document_title: str
    document_source: str
    start_offset: int
    end_offset: int
    chunk_metadata: dict
    chunk: DocumentChunk


def _get_retrieval_config() -> RetrievalConfig:
    """Build RetrievalConfig from Django settings."""
    rag_settings = getattr(settings, "AI_RAG", {})
    return RetrievalConfig(
        top_k=rag_settings.get("TOP_K", 5),
        similarity_threshold=rag_settings.get("SIMILARITY_THRESHOLD", 0.0),
        max_chunks_per_document=rag_settings.get("MAX_CHUNKS_PER_DOC"),
    )


def retrieve_chunks(
    query: str,
    embedding_provider: EmbeddingProvider,
    config: RetrievalConfig | None = None,
) -> list[RetrievalResult]:
    """Retrieve the most similar document chunks for a query.

    Embeds the query, then performs exact cosine distance search in PostgreSQL
    using pgvector. Results are ordered by similarity (highest first) with
    deterministic tie-breaking by chunk primary key.

    M3/M11 enhancements:
      - Similarity threshold filtering applied database-side before top-K slice.
      - Document ID filtering (document_ids) restricts search to target documents.
      - Multi-document fairness (max_chunks_per_document) prevents one document
        from dominating context.
      - Deterministic tie-breaking uses (distance ASC, chunk pk ASC) ordering.

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

    # Build queryset: annotate with distance, filter ready documents
    queryset = DocumentChunk.objects.filter(
        document__status="ready"
    ).select_related("document").annotate(
        distance=CosineDistance("embedding", query_embedding)
    )

    # Document ID filtering (M11)
    if config.document_ids is not None:
        clean_doc_ids = [str(did).strip() for did in config.document_ids if str(did).strip()]
        queryset = queryset.filter(document_id__in=clean_doc_ids)

    # M3: Apply threshold filtering BEFORE top-K slice.
    # similarity = 1 - distance, so threshold T means distance <= 1 - T.
    if config.similarity_threshold > 0.0:
        max_distance = 1.0 - config.similarity_threshold
        queryset = queryset.filter(distance__lte=max_distance)

    # Deterministic ordering: by distance ASC (best first), then pk ASC for tie-breaking
    queryset = queryset.order_by("distance", "pk")

    results: list[RetrievalResult] = []

    # Multi-document fairness cap (M11)
    if config.max_chunks_per_document is not None:
        doc_chunk_counts: dict[str, int] = {}
        candidate_pool_limit = max(config.top_k * 5, 100)
        for chunk in queryset[:candidate_pool_limit]:
            doc_id_str = str(chunk.document_id)
            if doc_chunk_counts.get(doc_id_str, 0) >= config.max_chunks_per_document:
                continue
            doc_chunk_counts[doc_id_str] = doc_chunk_counts.get(doc_id_str, 0) + 1
            similarity = 1.0 - chunk.distance
            results.append(
                RetrievalResult(
                    chunk_id=str(chunk.pk),
                    document_id=doc_id_str,
                    content=chunk.content,
                    score=similarity,
                    rank=len(results) + 1,
                    chunk_index=chunk.chunk_index,
                    document_title=chunk.document.title,
                    document_source=chunk.document.source,
                    start_offset=chunk.start_offset,
                    end_offset=chunk.end_offset,
                    chunk_metadata=chunk.metadata or {},
                    chunk=chunk,
                )
            )
            if len(results) >= config.top_k:
                break
    else:
        for rank, chunk in enumerate(queryset[: config.top_k], start=1):
            similarity = 1.0 - chunk.distance
            results.append(
                RetrievalResult(
                    chunk_id=str(chunk.pk),
                    document_id=str(chunk.document_id),
                    content=chunk.content,
                    score=similarity,
                    rank=rank,
                    chunk_index=chunk.chunk_index,
                    document_title=chunk.document.title,
                    document_source=chunk.document.source,
                    start_offset=chunk.start_offset,
                    end_offset=chunk.end_offset,
                    chunk_metadata=chunk.metadata or {},
                    chunk=chunk,
                )
            )

    return results


def retrieve_lexical_chunks(
    query: str,
    config: RetrievalConfig | None = None,
) -> list[RetrievalResult]:
    """Retrieve document chunks using PostgreSQL Full-Text Search (M13).

    Uses native PostgreSQL tsvector / tsquery via Django's SearchVector and SearchQuery.
    Results are ordered by SearchRank descending with deterministic tie-breaking by primary key.

    Args:
        query: The search query text.
        config: Retrieval configuration. Uses settings defaults if None.

    Returns:
        List of RetrievalResult ordered by descending lexical rank.

    Raises:
        RetrievalError: If query is empty or retrieval fails.
    """
    if not query or not query.strip():
        raise RetrievalError("Query must be a non-empty string.")

    if config is None:
        config = _get_retrieval_config()

    try:
        search_query = SearchQuery(query.strip(), config="english")
        search_vector = SearchVector("content", config="english")

        queryset = (
            DocumentChunk.objects.filter(document__status="ready")
            .select_related("document")
            .annotate(search=search_vector, rank=SearchRank(search_vector, search_query))
            .filter(search=search_query)
        )

        # Document ID filtering
        if config.document_ids is not None:
            clean_doc_ids = [str(did).strip() for did in config.document_ids if str(did).strip()]
            queryset = queryset.filter(document_id__in=clean_doc_ids)

        # Deterministic ordering: rank DESC (best match first), then pk ASC for tie-breaking
        queryset = queryset.order_by("-rank", "pk")

        results: list[RetrievalResult] = []

        # Multi-document fairness cap
        if config.max_chunks_per_document is not None:
            doc_chunk_counts: dict[str, int] = {}
            candidate_pool_limit = max(config.top_k * 5, 100)
            for chunk in queryset[:candidate_pool_limit]:
                doc_id_str = str(chunk.document_id)
                if doc_chunk_counts.get(doc_id_str, 0) >= config.max_chunks_per_document:
                    continue
                doc_chunk_counts[doc_id_str] = doc_chunk_counts.get(doc_id_str, 0) + 1
                results.append(
                    RetrievalResult(
                        chunk_id=str(chunk.pk),
                        document_id=doc_id_str,
                        content=chunk.content,
                        score=float(chunk.rank),
                        rank=len(results) + 1,
                        chunk_index=chunk.chunk_index,
                        document_title=chunk.document.title,
                        document_source=chunk.document.source,
                        start_offset=chunk.start_offset,
                        end_offset=chunk.end_offset,
                        chunk_metadata=chunk.metadata or {},
                        chunk=chunk,
                    )
                )
                if len(results) >= config.top_k:
                    break
        else:
            for rank_idx, chunk in enumerate(queryset[: config.top_k], start=1):
                results.append(
                    RetrievalResult(
                        chunk_id=str(chunk.pk),
                        document_id=str(chunk.document_id),
                        content=chunk.content,
                        score=float(chunk.rank),
                        rank=rank_idx,
                        chunk_index=chunk.chunk_index,
                        document_title=chunk.document.title,
                        document_source=chunk.document.source,
                        start_offset=chunk.start_offset,
                        end_offset=chunk.end_offset,
                        chunk_metadata=chunk.metadata or {},
                        chunk=chunk,
                    )
                )

        return results
    except Exception as exc:
        raise RetrievalError(f"Lexical retrieval failed: {exc}") from exc
