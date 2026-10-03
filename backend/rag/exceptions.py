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


class URLSecurityError(DocumentError):
    """Raised when a URL fails format validation or security checks."""

    pass


class SSRFError(URLSecurityError):
    """Raised when a URL attempts to target private, loopback, or internal networks."""

    pass


class DuplicateURLError(DocumentError):
    """Raised when attempting to ingest a URL that has already been indexed."""

    pass


class WebFetchError(DocumentError):
    """Raised when an error occurs during HTTP retrieval of a web page."""

    pass


class ContentTooLargeError(WebFetchError):
    """Raised when the fetched web page response body exceeds configured size limit."""

    pass


class UnsupportedContentTypeError(WebFetchError):
    """Raised when the fetched web page returns an unsupported Content-Type (non-HTML)."""

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
