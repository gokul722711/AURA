"""Extractor for PDF documents using PyMuPDF."""

import logging

from rag.exceptions import ExtractionError
from rag.extraction.base import DocumentExtractor, ExtractedBlock, ExtractedDocument

logger = logging.getLogger(__name__)


class PDFExtractor(DocumentExtractor):
    """Extracts text page-by-page from PDF files using PyMuPDF.

    Preserves 1-based page metadata for each page block.
    Fails with an ExtractionError if the PDF is corrupt, encrypted,
    or contains no extractable text (e.g. scanned image without OCR).
    """

    @property
    def supported_source_type(self) -> str:
        return "pdf"

    def extract(self, content: bytes, filename: str = "") -> ExtractedDocument:
        if not content:
            raise ExtractionError("PDF file is empty.")

        try:
            import pymupdf
        except ImportError as exc:
            raise ExtractionError(
                "PyMuPDF library is required for PDF extraction but is not installed."
            ) from exc

        try:
            doc = pymupdf.open(stream=content, filetype="pdf")
        except Exception as exc:
            raise ExtractionError(f"Failed to parse PDF document: {exc}") from exc

        try:
            if doc.is_encrypted:
                raise ExtractionError("PDF document is encrypted and cannot be extracted.")

            if len(doc) == 0:
                raise ExtractionError("PDF document contains no pages.")

            blocks: list[ExtractedBlock] = []
            for page_idx in range(len(doc)):
                page_num = page_idx + 1
                try:
                    page = doc[page_idx]
                    page_text = page.get_text()
                except Exception as exc:
                    raise ExtractionError(
                        f"Failed to extract text from PDF page {page_num}: {exc}"
                    ) from exc

                if page_text and page_text.strip():
                    blocks.append(
                        ExtractedBlock(
                            content=page_text.strip(),
                            metadata={
                                "source_type": "pdf",
                                "page": page_num,
                            },
                        )
                    )

            if not blocks:
                raise ExtractionError(
                    "PDF contains no extractable text. Scanned or image-only PDFs "
                    "are not supported without OCR."
                )

            total_text = "\n\n".join(b.content for b in blocks)
            return ExtractedDocument(
                text=total_text,
                blocks=blocks,
                source_type="pdf",
                metadata={
                    "filename": filename,
                    "page_count": len(doc),
                    "source_type": "pdf",
                },
            )
        finally:
            doc.close()
