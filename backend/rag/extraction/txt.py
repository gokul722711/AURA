"""Extractor for plain text documents."""

from rag.exceptions import ExtractionError
from rag.extraction.base import DocumentExtractor, ExtractedBlock, ExtractedDocument


class TextExtractor(DocumentExtractor):
    """Extracts plain text (.txt) document content."""

    @property
    def supported_source_type(self) -> str:
        return "txt"

    def extract(self, content: bytes, filename: str = "") -> ExtractedDocument:
        if not content:
            raise ExtractionError("Text file is empty.")

        text = ""
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                text = content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if not text or not text.strip():
            raise ExtractionError("Text file contains no extractable text.")

        cleaned = text.strip()
        blocks = [
            ExtractedBlock(
                content=cleaned,
                metadata={"source_type": "txt"},
            )
        ]
        return ExtractedDocument(
            text=cleaned,
            blocks=blocks,
            source_type="txt",
            metadata={"filename": filename, "source_type": "txt"},
        )
