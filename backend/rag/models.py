"""Django models for AURA RAG document storage.

Document and DocumentChunk models store ingested documents and their
chunked/embedded representations for vector retrieval.

The VectorField uses a fixed dimension (384) defined in Django settings
as AI_EMBEDDINGS["DIMENSIONS"]. Changing embedding dimensions requires
a database migration and schema change.
"""

import uuid

from django.conf import settings
from django.db import models
from pgvector.django import VectorField


# Read dimension from settings; default 384 for M2.
_EMBEDDING_DIMENSIONS = getattr(settings, "AI_EMBEDDINGS", {}).get("DIMENSIONS", 384)


class Document(models.Model):
    """A source document ingested into the RAG system."""

    STATUS_PENDING = "pending"
    STATUS_PROCESSING = "processing"
    STATUS_READY = "ready"
    STATUS_ERROR = "error"

    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_PROCESSING, "Processing"),
        (STATUS_READY, "Ready"),
        (STATUS_ERROR, "Error"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=512)
    content = models.TextField(help_text="Full source text of the document.")
    source = models.CharField(
        max_length=1024, blank=True, default="",
        help_text="Origin of the document (file path, URL, etc.)."
    )
    metadata = models.JSONField(default=dict, blank=True)
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING,
        db_index=True,
    )
    error_message = models.TextField(
        blank=True, default="",
        help_text="Error details if status is 'error'."
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"Document({self.title}, status={self.status})"


class DocumentChunk(models.Model):
    """A chunk of a document with its embedding vector."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name="chunks",
    )
    content = models.TextField(help_text="Text content of this chunk.")
    chunk_index = models.IntegerField(
        help_text="Position of this chunk within the source document (0-based)."
    )
    start_offset = models.IntegerField(
        help_text="Character offset where this chunk starts in the source document."
    )
    end_offset = models.IntegerField(
        help_text="Character offset where this chunk ends in the source document."
    )
    embedding = VectorField(
        dimensions=_EMBEDDING_DIMENSIONS,
        help_text="Embedding vector for similarity search.",
    )
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["document", "chunk_index"]
        indexes = [
            models.Index(fields=["document", "chunk_index"]),
        ]

    def __str__(self) -> str:
        return f"Chunk({self.document.title}[{self.chunk_index}])"
