"""Base abstraction and data models for AURA document extraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExtractedBlock:
    """A discrete block of text extracted from a source document.

    Attributes:
        content: Text content of this block.
        metadata: Block-level metadata (e.g. page number, block_type).
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractedDocument:
    """An in-memory extracted document before chunking.

    Attributes:
        text: Complete raw extracted text across all blocks.
        blocks: List of structured ExtractedBlock items.
        source_type: Document format identifier (e.g. 'txt', 'markdown', 'pdf', 'docx').
        metadata: Document-level metadata (e.g. filename, page_count).
    """

    text: str
    blocks: list[ExtractedBlock] = field(default_factory=list)
    source_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def title(self) -> str:
        """Convenience property returning title from metadata if present."""
        return self.metadata.get("title") or ""


class DocumentExtractor(ABC):
    """Abstract base class for all AURA document format extractors.

    Format-specific libraries (e.g. PyMuPDF, python-docx) must remain
    strictly isolated inside their respective extractor implementations.
    """

    @property
    @abstractmethod
    def supported_source_type(self) -> str:
        """Return the canonical source type identifier (e.g. 'pdf', 'docx')."""
        ...

    @abstractmethod
    def extract(self, content: bytes, filename: str = "") -> ExtractedDocument:
        """Extract text and structural blocks from raw document bytes.

        Args:
            content: Raw document bytes.
            filename: Original filename (used for metadata and source labelling).

        Returns:
            ExtractedDocument containing normalized blocks and metadata.

        Raises:
            ExtractionError: If the document is malformed, corrupted, empty,
                or contains no meaningful extractable text.
        """
        ...
