"""Document extraction and normalization layer for AURA RAG."""

from rag.extraction.base import (
    DocumentExtractor,
    ExtractedBlock,
    ExtractedDocument,
)
from rag.extraction.docx import DocxExtractor
from rag.extraction.markdown import MarkdownExtractor
from rag.extraction.normalization import normalize_document, normalize_text
from rag.extraction.pdf import PDFExtractor
from rag.extraction.registry import SUPPORTED_EXTENSIONS, get_extractor
from rag.extraction.txt import TextExtractor
from rag.extraction.web import WebPageExtractor

__all__ = [
    "DocumentExtractor",
    "ExtractedBlock",
    "ExtractedDocument",
    "TextExtractor",
    "MarkdownExtractor",
    "PDFExtractor",
    "DocxExtractor",
    "WebPageExtractor",
    "get_extractor",
    "SUPPORTED_EXTENSIONS",
    "normalize_text",
    "normalize_document",
]
