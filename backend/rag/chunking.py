"""Deterministic text chunking for AURA RAG.

Splits document text into overlapping chunks of a configured size.
Character-based splitting is used for M2 simplicity and determinism.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rag.exceptions import ChunkingError

if TYPE_CHECKING:
    from rag.extraction.base import ExtractedDocument


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for text chunking.

    Attributes:
        chunk_size: Maximum character count per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.
    """

    chunk_size: int = 512
    chunk_overlap: int = 50

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ChunkingError("chunk_size must be greater than 0.")
        if self.chunk_overlap < 0:
            raise ChunkingError("chunk_overlap must be non-negative.")
        if self.chunk_overlap >= self.chunk_size:
            raise ChunkingError("chunk_overlap must be less than chunk_size.")


@dataclass(frozen=True)
class ChunkResult:
    """A single chunk produced by the chunking process.

    Attributes:
        content: The text content of the chunk.
        chunk_index: Position of this chunk in the document (0-based).
        start_offset: Character offset where this chunk starts in the source text.
        end_offset: Character offset where this chunk ends in the source text.
        metadata: Source-location and structural metadata (e.g. page, block_type).
    """

    content: str
    chunk_index: int
    start_offset: int
    end_offset: int
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_text(
    text: str,
    config: ChunkingConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[ChunkResult]:
    """Split text into overlapping chunks.

    Args:
        text: The source text to split.
        config: Chunking configuration. Uses defaults if None.
        metadata: Optional metadata dictionary to associate with each chunk.

    Returns:
        List of ChunkResult objects. Empty list if text is empty or whitespace-only.

    Raises:
        ChunkingError: If chunking fails.
    """
    if config is None:
        config = ChunkingConfig()

    if not text or not text.strip():
        return []

    chunk_meta = dict(metadata) if metadata else {}
    chunks = []
    start = 0
    chunk_index = 0
    text_length = len(text)
    step = config.chunk_size - config.chunk_overlap

    while start < text_length:
        end = min(start + config.chunk_size, text_length)
        chunk_content = text[start:end]

        # Skip chunks that are only whitespace
        if chunk_content.strip():
            chunks.append(
                ChunkResult(
                    content=chunk_content,
                    chunk_index=chunk_index,
                    start_offset=start,
                    end_offset=end,
                    metadata=dict(chunk_meta),
                )
            )
            chunk_index += 1

        if end >= text_length:
            break

        start += step

    return chunks


def chunk_extracted_document(
    doc: "ExtractedDocument",
    config: ChunkingConfig | None = None,
) -> tuple[str, list[ChunkResult]]:
    """Split an ExtractedDocument into overlapping chunks while preserving block metadata.

    Preserves source-location metadata (such as PDF page number or DOCX block_type)
    on each generated chunk, while calculating exact character offsets within the
    overall assembled document text.

    Args:
        doc: ExtractedDocument to chunk.
        config: Optional ChunkingConfig. Uses defaults if None.

    Returns:
        tuple of (assembled_text, list of ChunkResult objects).
    """
    if config is None:
        config = ChunkingConfig()

    if not doc.blocks:
        assembled = doc.text.strip()
        chunks = chunk_text(assembled, config=config, metadata=doc.metadata)
        return assembled, chunks

    assembled_blocks: list[str] = []
    chunks: list[ChunkResult] = []
    current_doc_offset = 0
    global_chunk_index = 0
    separator = "\n\n"

    for idx, block in enumerate(doc.blocks):
        content = block.content.strip()
        if not content:
            continue

        if idx > 0 and assembled_blocks:
            current_doc_offset += len(separator)

        block_chunks = chunk_text(content, config=config, metadata=block.metadata)
        for bc in block_chunks:
            chunks.append(
                ChunkResult(
                    content=bc.content,
                    chunk_index=global_chunk_index,
                    start_offset=current_doc_offset + bc.start_offset,
                    end_offset=current_doc_offset + bc.end_offset,
                    metadata=dict(bc.metadata),
                )
            )
            global_chunk_index += 1

        assembled_blocks.append(content)
        current_doc_offset += len(content)

    full_text = separator.join(assembled_blocks)
    return full_text, chunks
