"""Deterministic text chunking for AURA RAG.

Splits document text into overlapping chunks of a configured size.
Character-based splitting is used for M2 simplicity and determinism.
"""

from dataclasses import dataclass

from rag.exceptions import ChunkingError


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
    """

    content: str
    chunk_index: int
    start_offset: int
    end_offset: int


def chunk_text(text: str, config: ChunkingConfig | None = None) -> list[ChunkResult]:
    """Split text into overlapping chunks.

    Args:
        text: The source text to split.
        config: Chunking configuration. Uses defaults if None.

    Returns:
        List of ChunkResult objects. Empty list if text is empty or whitespace-only.

    Raises:
        ChunkingError: If chunking fails.
    """
    if config is None:
        config = ChunkingConfig()

    if not text or not text.strip():
        return []

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
                )
            )
            chunk_index += 1

        if end >= text_length:
            break

        start += step

    return chunks
