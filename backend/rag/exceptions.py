"""Exceptions for the AURA RAG subsystem."""


class RAGError(Exception):
    """Base exception for all RAG errors."""

    pass


class DocumentError(RAGError):
    """Raised when document input is invalid or a document operation fails."""

    pass


class ExtractionError(DocumentError):
    """Raised when document extraction fails or format contains no extractable text."""

    pass


class UnsupportedFormatError(DocumentError):
    """Raised when an uploaded document has an unsupported file format or extension."""

    pass


class ChunkingError(RAGError):
    """Raised when text chunking fails."""

    pass


class EmbeddingError(RAGError):
    """Raised when embedding generation fails."""

    pass


class RetrievalError(RAGError):
    """Raised when retrieval/search fails."""

    pass


class ContextAssemblyError(RAGError):
    """Raised when context assembly for the LLM prompt fails."""

    pass


class QueryProcessingError(RAGError):
    """Raised when query processing/normalization fails."""

    pass


class EvaluationError(RAGError):
    """Raised when RAG evaluation fails."""

    pass
