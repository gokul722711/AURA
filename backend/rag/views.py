"""API views for AURA RAG document management and ingestion."""

import logging
import uuid
from typing import Any

from django.db import models
import os
from rest_framework import status
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from agent.security import sanitize_data, sanitize_text
from rag.embeddings.base import EmbeddingProvider
from rag.exceptions import (
    DocumentError,
    DuplicateURLError,
    URLSecurityError,
    WebFetchError,
)
from rag.ingestion import (
    get_default_embedding_provider,
    ingest_document,
    ingest_file,
    ingest_url,
)
from rag.models import Document

logger = logging.getLogger(__name__)

MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB limit
SUPPORTED_FILE_EXTENSIONS = (".txt", ".md", ".pdf", ".docx")


def _serialize_document(doc: Document, include_content: bool = True) -> dict[str, Any]:
    """Serialize a Document model instance into a JSON-safe dictionary.

    Internal embedding and vector representations are never exposed.
    """
    chunk_count = getattr(doc, "chunk_count", None)
    if chunk_count is None:
        chunk_count = doc.chunks.count()

    doc_meta = doc.metadata if isinstance(doc.metadata, dict) else {}
    data: dict[str, Any] = {
        "id": str(doc.id),
        "title": doc.title,
        "source": doc.source,
        "source_type": doc.source_type,
        "filename": doc.filename,
        "file_size": doc.file_size,
        "url": getattr(doc, "url", ""),
        "canonical_url": getattr(doc, "canonical_url", ""),
        "domain": getattr(doc, "domain", ""),
        "metadata": doc_meta,
        "status": doc.status,
        "chunk_count": chunk_count,
        "error_message": doc.error_message,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
        "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
    }
    if include_content:
        data["content"] = doc.content
    return sanitize_data(data)


class DocumentListCreateView(APIView):
    """List knowledge documents or ingest a new document into the RAG system.

    Endpoints:
        GET  /api/documents/   -> List documents with metadata and chunk counts
        POST /api/documents/   -> Ingest document (multipart file upload OR JSON body)
    """

    authentication_classes = []
    permission_classes = []
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def get_embedding_provider(self) -> EmbeddingProvider:
        """Return the active EmbeddingProvider for ingestion."""
        return get_default_embedding_provider()

    def get(self, request: Request) -> Response:
        """Return a list of all ingested documents without vector/embedding data."""
        docs = (
            Document.objects.annotate(chunk_count=models.Count("chunks"))
            .order_by("-created_at")
        )
        serialized = [_serialize_document(doc, include_content=False) for doc in docs]
        return Response(serialized, status=status.HTTP_200_OK)

    def post(self, request: Request) -> Response:
        """Ingest a new document into the RAG system.

        Supports:
        1. Multipart file upload: file (.txt, .md, .pdf, .docx), optional title, optional source
        2. JSON body: title, content, optional source, optional metadata
        """
        # --- Option 1: Multipart file upload ---
        if "file" in request.FILES:
            file_obj = request.FILES["file"]

            if file_obj.size > MAX_UPLOAD_SIZE_BYTES:
                return Response(
                    {"error": f"File size ({file_obj.size} bytes) exceeds maximum limit of 20MB."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            if file_obj.size == 0:
                return Response(
                    {"error": "Uploaded file is empty."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            filename = file_obj.name or "uploaded_file"
            ext = os.path.splitext(filename)[1].lower()
            if ext not in SUPPORTED_FILE_EXTENSIONS:
                supported_str = ", ".join(SUPPORTED_FILE_EXTENSIONS)
                return Response(
                    {
                        "error": f"Unsupported file format '{ext or 'unknown'}'. Supported formats are: {supported_str}."
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            content_bytes = file_obj.read()
            if not content_bytes:
                return Response(
                    {"error": "Uploaded file is empty."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            title = request.data.get("title")
            if title is not None and not isinstance(title, str):
                return Response(
                    {"error": "Field 'title' must be a string."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            source = request.data.get("source")
            if source is not None and not isinstance(source, str):
                return Response(
                    {"error": "Field 'source' must be a string."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            try:
                embedding_provider = self.get_embedding_provider()
                document = ingest_file(
                    file_bytes=content_bytes,
                    filename=filename,
                    title=title,
                    source=source,
                    embedding_provider=embedding_provider,
                )
            except DocumentError as exc:
                return Response(
                    {"error": sanitize_text(str(exc))},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            except Exception as exc:
                logger.warning("Document file ingestion failed: %s", sanitize_text(str(exc)))
                return Response(
                    {
                        "error": "Document ingestion failed.",
                        "detail": sanitize_text(str(exc)),
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

            result_dict = _serialize_document(document, include_content=True)
            return Response(result_dict, status=status.HTTP_201_CREATED)

        # --- Option 2: Web URL ingestion ---
        if isinstance(request.data, dict) and "url" in request.data:
            url_val = request.data.get("url")
            if not url_val or not isinstance(url_val, str) or not url_val.strip():
                return Response(
                    {"error": "Field 'url' must be a non-empty string."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            custom_title = request.data.get("title")
            if custom_title is not None and not isinstance(custom_title, str):
                return Response(
                    {"error": "Field 'title' must be a string."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            try:
                embedding_provider = self.get_embedding_provider()
                document = ingest_url(
                    url=url_val.strip(),
                    title=custom_title.strip() if custom_title else None,
                    embedding_provider=embedding_provider,
                )
            except DuplicateURLError as exc:
                return Response(
                    {"error": sanitize_text(str(exc))},
                    status=status.HTTP_409_CONFLICT,
                )
            except (URLSecurityError, DocumentError) as exc:
                return Response(
                    {"error": sanitize_text(str(exc))},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            except WebFetchError as exc:
                logger.warning("Web page fetch failed for %s: %s", sanitize_text(url_val), exc)
                return Response(
                    {"error": sanitize_text(str(exc))},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            except Exception as exc:
                logger.warning("Web page ingestion failed: %s", sanitize_text(str(exc)))
                return Response(
                    {
                        "error": "Web page ingestion failed.",
                        "detail": sanitize_text(str(exc)),
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

            result_dict = _serialize_document(document, include_content=True)
            return Response(result_dict, status=status.HTTP_201_CREATED)

        # --- Option 3: Existing JSON pasted-text ingestion ---
        data = request.data
        if not isinstance(data, dict):
            return Response(
                {"error": "Invalid request body; JSON object expected."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if "title" not in data:
            return Response(
                {"error": "Missing required field: 'title'."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        title = data["title"]
        if not isinstance(title, str) or not title.strip():
            return Response(
                {"error": "Field 'title' must be a non-empty string."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if "content" not in data:
            return Response(
                {"error": "Missing required field: 'content'."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        content = data["content"]
        if not isinstance(content, str) or not content.strip():
            return Response(
                {"error": "Field 'content' must be a non-empty string."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        source = data.get("source", "")
        if not isinstance(source, str):
            return Response(
                {"error": "Field 'source' must be a string."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            return Response(
                {"error": "Field 'metadata' must be an object."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            embedding_provider = self.get_embedding_provider()
            document = ingest_document(
                title=title.strip(),
                content=content,
                embedding_provider=embedding_provider,
                source=source.strip(),
                metadata=metadata,
            )
        except DocumentError as exc:
            return Response(
                {"error": sanitize_text(str(exc))},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as exc:
            logger.warning("Document ingestion failed: %s", sanitize_text(str(exc)))
            return Response(
                {
                    "error": "Document ingestion failed.",
                    "detail": sanitize_text(str(exc)),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        result_dict = _serialize_document(document, include_content=True)
        return Response(result_dict, status=status.HTTP_201_CREATED)


class DocumentDetailView(APIView):
    """Retrieve metadata and content or delete an individual document.

    Endpoints:
        GET    /api/documents/<id>/  -> Retrieve document details and content
        DELETE /api/documents/<id>/  -> Delete document and all associated chunks
    """

    authentication_classes = []
    permission_classes = []

    def _parse_uuid(self, document_id: str) -> uuid.UUID | None:
        try:
            return uuid.UUID(str(document_id))
        except (ValueError, AttributeError):
            return None

    def get(self, request: Request, document_id: str) -> Response:
        """Retrieve document metadata and content without exposing embeddings."""
        doc_uuid = self._parse_uuid(document_id)
        if doc_uuid is None:
            return Response(
                {"error": "Document not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        try:
            doc = (
                Document.objects.annotate(chunk_count=models.Count("chunks"))
                .get(id=doc_uuid)
            )
        except Document.DoesNotExist:
            return Response(
                {"error": "Document not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        return Response(_serialize_document(doc, include_content=True), status=status.HTTP_200_OK)

    def delete(self, request: Request, document_id: str) -> Response:
        """Delete the document and all associated chunks cleanly via database cascade."""
        doc_uuid = self._parse_uuid(document_id)
        if doc_uuid is None:
            return Response(
                {"error": "Document not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        try:
            doc = Document.objects.get(id=doc_uuid)
        except Document.DoesNotExist:
            return Response(
                {"error": "Document not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        doc.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
