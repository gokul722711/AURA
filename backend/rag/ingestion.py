"""Document ingestion for AURA RAG.

Handles the full ingestion pipeline:
document extraction → normalization → chunking → embedding → storage.
Supports TXT, Markdown, PDF, and DOCX formats while preserving source metadata.
"""

import logging
import os
from urllib.parse import urlsplit

from django.conf import settings
from django.db import models, transaction
from django.utils import timezone

from rag.chunking import ChunkingConfig, chunk_extracted_document, chunk_text
from rag.embeddings.base import EmbeddingProvider
from rag.embeddings.registry import create_embedding_provider
from rag.exceptions import DocumentError, DuplicateURLError, EmbeddingError
from rag.extraction import (
    ExtractedDocument,
    get_extractor,
    normalize_document,
)
from rag.extraction.web import WebPageExtractor
from rag.models import Document, DocumentChunk
from rag.web.base import WebFetcher
from rag.web.fetcher import HTTPXWebFetcher
from rag.web.security import canonicalize_url

logger = logging.getLogger(__name__)


def _get_chunking_config() -> ChunkingConfig:
    """Build ChunkingConfig from Django settings."""
    rag_settings = getattr(settings, "AI_RAG", {})
    return ChunkingConfig(
        chunk_size=rag_settings.get("CHUNK_SIZE", 512),
        chunk_overlap=rag_settings.get("CHUNK_OVERLAP", 50),
    )


def get_default_embedding_provider() -> EmbeddingProvider:
    """Instantiate the default embedding provider from Django settings."""
    rag_settings = getattr(settings, "AI_EMBEDDINGS", {})
    provider_name = rag_settings.get("PROVIDER", "mock")
    dimensions = rag_settings.get("DIMENSIONS", 384)
    return create_embedding_provider(provider_name, dimensions)


def ingest_document(
    title: str,
    content: str,
    embedding_provider: EmbeddingProvider | None = None,
    source: str = "",
    metadata: dict[str, Any] | None = None,
    chunking_config: ChunkingConfig | None = None,
) -> Document:
    """Ingest a text document into the RAG system.

    Creates a Document record, chunks the content, generates embeddings
    via the provided EmbeddingProvider (or default provider if None),
    and stores DocumentChunk records with their embedding vectors and metadata.

    Args:
        title: Document title.
        content: Full text content of the document.
        embedding_provider: Provider to generate embeddings. Defaults to settings-configured provider.
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

    if embedding_provider is None:
        embedding_provider = get_default_embedding_provider()

    if chunking_config is None:
        chunking_config = _get_chunking_config()

    doc_metadata = dict(metadata or {})

    document = Document.objects.create(
        title=title.strip(),
        content=content,
        source=source.strip() if source else "",
        metadata=doc_metadata,
        status=Document.STATUS_PROCESSING,
    )

    try:
        chunk_meta = {"source_type": doc_metadata.get("source_type", "text")}
        chunks = chunk_text(content, config=chunking_config, metadata=chunk_meta)
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

        # Create chunk records with embeddings and chunk metadata atomically
        chunk_records = [
            DocumentChunk(
                document=document,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
                embedding=embedding,
                metadata=chunk.metadata or {},
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


def ingest_extracted_document(
    title: str,
    extracted_doc: ExtractedDocument,
    embedding_provider: EmbeddingProvider | None = None,
    source: str = "",
    metadata: dict[str, Any] | None = None,
    chunking_config: ChunkingConfig | None = None,
) -> Document:
    """Ingest a normalized ExtractedDocument into the RAG system.

    Preserves source-location metadata (e.g. PDF page numbers, DOCX block types)
    on DocumentChunk records.

    Args:
        title: Document title.
        extracted_doc: Normalized ExtractedDocument instance.
        embedding_provider: Provider to generate embeddings.
        source: Optional source origin.
        metadata: Optional additional metadata.
        chunking_config: Optional chunking configuration.

    Returns:
        The created Document instance with status 'ready' or 'error'.

    Raises:
        DocumentError: If validation fails.
    """
    if not title or not title.strip():
        raise DocumentError("Document title must be a non-empty string.")

    if embedding_provider is None:
        embedding_provider = get_default_embedding_provider()

    if chunking_config is None:
        chunking_config = _get_chunking_config()

    doc_metadata = dict(extracted_doc.metadata)
    doc_metadata["source_type"] = extracted_doc.source_type
    if metadata:
        doc_metadata.update(metadata)

    # Chunk the extracted document while preserving block-level metadata
    assembled_text, chunks = chunk_extracted_document(extracted_doc, config=chunking_config)

    if not assembled_text or not assembled_text.strip():
        raise DocumentError("Document content must be a non-empty string.")

    document = Document.objects.create(
        title=title.strip(),
        content=assembled_text,
        source=source.strip() if source else "",
        metadata=doc_metadata,
        status=Document.STATUS_PROCESSING,
    )

    try:
        if not chunks:
            document.status = Document.STATUS_READY
            document.save(update_fields=["status", "updated_at"])
            logger.info(
                "Extracted document '%s' ingested with 0 chunks.",
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

        # Create chunk records with embeddings and chunk metadata atomically
        chunk_records = [
            DocumentChunk(
                document=document,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
                embedding=embedding,
                metadata={**doc_metadata, **(chunk.metadata or {})},
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        with transaction.atomic():
            DocumentChunk.objects.bulk_create(chunk_records)
            document.status = Document.STATUS_READY
            document.save(update_fields=["status", "updated_at"])

        logger.info(
            "Extracted document '%s' ingested successfully: %d chunks created.",
            document.title,
            len(chunk_records),
        )

    except Exception:
        document.status = Document.STATUS_ERROR
        document.error_message = "Ingestion failed. See logs for details."
        document.save(update_fields=["status", "error_message", "updated_at"])
        raise

    return document


def ingest_file(
    file_bytes: bytes,
    filename: str,
    title: str | None = None,
    source: str | None = None,
    metadata: dict[str, Any] | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    chunking_config: ChunkingConfig | None = None,
) -> Document:
    """Ingest a file (TXT, Markdown, PDF, or DOCX) into the RAG system.

    Full pipeline:
    raw file bytes → extractor → normalizer → chunker → embedding → pgvector.

    Args:
        file_bytes: Raw file bytes.
        filename: Original filename (including extension).
        title: Optional custom title; defaults to filename without extension.
        source: Optional source origin; defaults to filename.
        metadata: Optional additional metadata dictionary.
        embedding_provider: Optional EmbeddingProvider.
        chunking_config: Optional ChunkingConfig.

    Returns:
        The created Document instance with status 'ready'.

    Raises:
        DocumentError: If format is unsupported or file content is empty/corrupt.
    """
    if not file_bytes:
        raise DocumentError("Uploaded file is empty.")

    extractor = get_extractor(filename)
    extracted = extractor.extract(file_bytes, filename=filename)
    normalized = normalize_document(extracted)

    clean_title = (
        title.strip()
        if (title and isinstance(title, str) and title.strip())
        else os.path.splitext(filename)[0]
    )
    clean_source = (
        source.strip()
        if (source and isinstance(source, str) and source.strip())
        else filename
    )

    file_metadata: dict[str, Any] = {
        "filename": filename,
        "file_size": len(file_bytes),
        "source_type": normalized.source_type,
    }
    if metadata and isinstance(metadata, dict):
        file_metadata.update(metadata)

    return ingest_extracted_document(
        title=clean_title,
        extracted_doc=normalized,
        embedding_provider=embedding_provider,
        source=clean_source,
        metadata=file_metadata,
        chunking_config=chunking_config,
    )


def ingest_url(
    url: str,
    title: str | None = None,
    fetcher: WebFetcher | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    chunking_config: ChunkingConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> Document:
    """Ingest a public web page from a URL into the RAG system.

    Full pipeline:
    URL validation / SSRF protection
      ↓
    WebFetcher (HTTPX)
      ↓
    WebPageExtractor (Trafilatura)
      ↓
    normalization
      ↓
    chunking
      ↓
    EmbeddingProvider
      ↓
    pgvector storage.

    Args:
        url: Public web page URL.
        title: Optional custom document title; defaults to page title or domain.
        fetcher: WebFetcher instance; defaults to HTTPXWebFetcher.
        embedding_provider: Optional EmbeddingProvider.
        chunking_config: Optional ChunkingConfig.
        metadata: Optional additional metadata.

    Returns:
        The created Document instance with status 'ready'.

    Raises:
        URLSecurityError / SSRFError: If URL fails security/SSRF constraints.
        DuplicateURLError: If the URL has already been ingested.
        WebFetchError: If HTTP retrieval fails.
        ExtractionError: If page has no extractable text.
        DocumentError: If validation or persistence fails.
    """
    if not url or not isinstance(url, str) or not url.strip():
        raise DocumentError("URL must be a non-empty string.")

    canonical_url = canonicalize_url(url)

    # Prevent accidental duplicate ingestion of the same URL
    canonical_no_slash = canonical_url.rstrip("/")
    canonical_with_slash = canonical_no_slash + "/"

    existing = (
        Document.objects.filter(
            models.Q(source=canonical_url)
            | models.Q(source=canonical_no_slash)
            | models.Q(source=canonical_with_slash)
            | models.Q(metadata__url=canonical_url)
            | models.Q(metadata__canonical_url=canonical_url)
        )
        .exclude(status=Document.STATUS_ERROR)
        .first()
    )
    if existing:
        raise DuplicateURLError(
            f"URL '{canonical_url}' has already been indexed as document '{existing.title}'."
        )

    active_fetcher = fetcher or HTTPXWebFetcher()
    fetch_result = active_fetcher.fetch(canonical_url)

    extractor = WebPageExtractor()
    extracted_doc = extractor.extract(fetch_result.body, filename=fetch_result.final_url)
    normalized = normalize_document(extracted_doc)

    # Determine title
    clean_title = (
        title.strip()
        if (title and isinstance(title, str) and title.strip())
        else ""
    )
    if not clean_title:
        clean_title = (
            normalized.metadata.get("title", "").strip()
            if isinstance(normalized.metadata, dict)
            else ""
        )
    if not clean_title:
        parsed_final = urlsplit(fetch_result.final_url)
        clean_title = parsed_final.path.strip("/").split("/")[-1] or parsed_final.netloc

    parsed_final = urlsplit(fetch_result.final_url)
    domain = parsed_final.netloc or urlsplit(canonical_url).netloc

    web_metadata: dict[str, Any] = {
        "source_type": "web_page",
        "url": canonical_url,
        "final_url": fetch_result.final_url,
        "canonical_url": normalized.metadata.get("canonical_url") or canonical_url,
        "domain": domain,
        "status_code": fetch_result.status_code,
        "retrieved_at": timezone.now().isoformat(),
    }
    for k in ("author", "date", "description", "site_name"):
        val = normalized.metadata.get(k)
        if val:
            web_metadata[k] = val

    if metadata and isinstance(metadata, dict):
        web_metadata.update(metadata)

    return ingest_extracted_document(
        title=clean_title,
        extracted_doc=normalized,
        embedding_provider=embedding_provider,
        source=canonical_url,
        metadata=web_metadata,
        chunking_config=chunking_config,
    )
