"""Extractor for Microsoft Word (.docx) documents using python-docx."""

import io
import logging

from rag.exceptions import ExtractionError
from rag.extraction.base import DocumentExtractor, ExtractedBlock, ExtractedDocument

logger = logging.getLogger(__name__)


class DocxExtractor(DocumentExtractor):
    """Extracts text and structural blocks from DOCX files using python-docx.

    Extracts paragraphs, headings, list-like paragraphs, and table contents.
    Assigns block_type metadata ('heading', 'paragraph', 'list', 'table') to each block.
    """

    @property
    def supported_source_type(self) -> str:
        return "docx"

    def _determine_block_type(self, paragraph) -> str:
        """Classify paragraph block type based on style and text prefix."""
        style_name = ""
        if paragraph.style and paragraph.style.name:
            style_name = paragraph.style.name.lower()

        if "heading" in style_name or "title" in style_name:
            return "heading"
        if "list" in style_name:
            return "list"

        stripped = paragraph.text.lstrip()
        if stripped.startswith(("- ", "* ", "• ", "– ", "— ")):
            return "list"
        # Check for numbered list like "1. ", "2. ", "10. "
        parts = stripped.split(" ", 1)
        if len(parts) == 2 and parts[0].rstrip(".").isdigit() and parts[0].endswith("."):
            return "list"

        return "paragraph"

    def extract(self, content: bytes, filename: str = "") -> ExtractedDocument:
        if not content:
            raise ExtractionError("DOCX file is empty.")

        try:
            import docx
            from docx.table import Table
            from docx.text.paragraph import Paragraph
        except ImportError as exc:
            raise ExtractionError(
                "python-docx library is required for DOCX extraction but is not installed."
            ) from exc

        try:
            doc = docx.Document(io.BytesIO(content))
        except Exception as exc:
            raise ExtractionError(f"Failed to parse DOCX document: {exc}") from exc

        blocks: list[ExtractedBlock] = []

        try:
            # Iterate through body elements to preserve natural document ordering
            for child in doc.element.body:
                tag = child.tag.lower()
                if tag.endswith("p"):
                    p = Paragraph(child, doc)
                    text = p.text.strip()
                    if text:
                        b_type = self._determine_block_type(p)
                        blocks.append(
                            ExtractedBlock(
                                content=text,
                                metadata={
                                    "source_type": "docx",
                                    "block_type": b_type,
                                },
                            )
                        )
                elif tag.endswith("tbl"):
                    tbl = Table(child, doc)
                    table_rows: list[str] = []
                    for row in tbl.rows:
                        row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                        if row_cells:
                            table_rows.append(" | ".join(row_cells))
                    if table_rows:
                        blocks.append(
                            ExtractedBlock(
                                content="\n".join(table_rows),
                                metadata={
                                    "source_type": "docx",
                                    "block_type": "table",
                                },
                            )
                        )
        except Exception as exc:
            raise ExtractionError(f"Error processing DOCX content: {exc}") from exc

        if not blocks:
            raise ExtractionError("DOCX contains no extractable text.")

        total_text = "\n\n".join(b.content for b in blocks)
        return ExtractedDocument(
            text=total_text,
            blocks=blocks,
            source_type="docx",
            metadata={
                "filename": filename,
                "source_type": "docx",
            },
        )
