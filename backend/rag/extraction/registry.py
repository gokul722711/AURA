"""Extractor registry and factory for AURA document extraction."""

import os
from rag.exceptions import UnsupportedFormatError
from rag.extraction.base import DocumentExtractor
from rag.extraction.docx import DocxExtractor
from rag.extraction.markdown import MarkdownExtractor
from rag.extraction.pdf import PDFExtractor
from rag.extraction.txt import TextExtractor
from rag.extraction.web import WebPageExtractor

_EXTRACTORS: dict[str, type[DocumentExtractor]] = {
    ".txt": TextExtractor,
    ".md": MarkdownExtractor,
    ".markdown": MarkdownExtractor,
    ".pdf": PDFExtractor,
    ".docx": DocxExtractor,
    ".html": WebPageExtractor,
    ".htm": WebPageExtractor,
}

_CONTENT_TYPE_MAP: dict[str, type[DocumentExtractor]] = {
    "text/plain": TextExtractor,
    "text/markdown": MarkdownExtractor,
    "text/x-markdown": MarkdownExtractor,
    "application/pdf": PDFExtractor,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocxExtractor,
    "text/html": WebPageExtractor,
    "application/xhtml+xml": WebPageExtractor,
}

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".txt", ".md", ".pdf", ".docx")


def get_extractor(filename: str = "", content_type: str | None = None) -> DocumentExtractor:
    """Retrieve an appropriate DocumentExtractor instance for a filename or content type.

    Args:
        filename: Document filename with extension (e.g. 'report.pdf').
        content_type: Optional MIME content type.

    Returns:
        DocumentExtractor instance.

    Raises:
        UnsupportedFormatError: If the extension or content type is not supported.
    """
    ext = os.path.splitext(filename)[1].lower() if filename else ""

    if ext in _EXTRACTORS:
        return _EXTRACTORS[ext]()

    if content_type:
        clean_ct = content_type.split(";")[0].strip().lower()
        if clean_ct in _CONTENT_TYPE_MAP:
            return _CONTENT_TYPE_MAP[clean_ct]()

    display_ext = ext or "unknown"
    supported_list = ", ".join(SUPPORTED_EXTENSIONS)
    raise UnsupportedFormatError(
        f"Unsupported file format '{display_ext}'. Supported formats are: {supported_list}."
    )
