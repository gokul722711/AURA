"""Document ingestion for AURA RAG.

Handles the full ingestion pipeline:
document creation → chunking → embedding → storage.
"""

import logging
from typing import Any

from django.conf import settings
from django.db import transaction

from rag.chunking import ChunkingConfig, chunk_text
from rag.embeddings.base import EmbeddingProvider
from rag.exceptions import DocumentError, EmbeddingError
from rag.models import Document, DocumentChunk

logger = logging.getLogger(__name__)


def _get_chunking_config() -> ChunkingConfig:
    """Build ChunkingConfig from Django settings."""
    rag_settings = getattr(settings, "AI_RAG", {})
    return ChunkingConfig(
        chunk_size=rag_settings.get("CHUNK_SIZE", 512),
        chunk_overlap=rag_settings.get("CHUNK_OVERLAP", 50),
    )


def ingest_document(
    title: str,
    content: str,
    embedding_provider: EmbeddingProvider,
    source: str = "",
    metadata: dict[str, Any] | None = None,
    chunking_config: ChunkingConfig | None = None,
) -> Document:
    """Ingest a text document into the RAG system.

    Creates a Document record, chunks the content, generates embeddings
    via the provided EmbeddingProvider, and stores DocumentChunk records
    with their embedding vectors.

    Args:
        title: Document title.
        content: Full text content of the document.
        embedding_provider: Provider to generate embeddings.
        source: Optional origin identifier (file path, URL).
        metadata: Optional metadata dictionary.
        chunking_config: Optional chunking configuration. Uses settings defaults if None.

    Returns:
        The created Document instance with status 'ready' or 'error'.

    Raises:
        DocumentError: If input validation fails.
    """
    if not title or not title.strip():
        raise DocumentError("Document title must be a non-empty string.")
    if not content or not content.strip():
        raise DocumentError("Document content must be a non-empty string.")

    if chunking_config is None:
        chunking_config = _get_chunking_config()

    document = Document.objects.create(
        title=title.strip(),
        content=content,
        source=source,
        metadata=metadata or {},
        status=Document.STATUS_PROCESSING,
    )

    try:
        # Chunk the document
        chunks = chunk_text(content, config=chunking_config)
        if not chunks:
            document.status = Document.STATUS_READY
            document.save(update_fields=["status", "updated_at"])
            logger.info(
                "Document '%s' ingested with no chunks (content may be whitespace-only).",
                document.title,
            )
            return document

        # Generate embeddings for all chunk texts
        chunk_texts = [chunk.content for chunk in chunks]
        try:
            embeddings = embedding_provider.embed_texts(chunk_texts)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to generate embeddings for document '{title}': {exc}"
            ) from exc

        if len(embeddings) != len(chunks):
            raise EmbeddingError(
                f"Embedding count ({len(embeddings)}) does not match "
                f"chunk count ({len(chunks)})."
            )

        # Create chunk records with embeddings atomically
        chunk_records = [
            DocumentChunk(
                document=document,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
                embedding=embedding,
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        with transaction.atomic():
            DocumentChunk.objects.bulk_create(chunk_records)
            document.status = Document.STATUS_READY
            document.save(update_fields=["status", "updated_at"])

        logger.info(
            "Document '%s' ingested successfully: %d chunks created.",
            document.title,
            len(chunk_records),
        )

    except Exception:
        document.status = Document.STATUS_ERROR
        document.error_message = "Ingestion failed. See logs for details."
        document.save(update_fields=["status", "error_message", "updated_at"])
        raise

    return document
