"""Document normalization for AURA extraction pipeline.

Applies faithful text cleaning between extraction and chunking:
- strips trailing whitespace per line
- normalizes carriage returns
- removes trivial extraction artifacts (null bytes, form feeds)
- collapses excessive blank lines (3+ consecutive newlines -> 2)
- drops empty blocks while strictly preserving source text fidelity
"""

import re
from rag.exceptions import ExtractionError
from rag.extraction.base import ExtractedBlock, ExtractedDocument


def normalize_text(text: str) -> str:
    """Normalize raw text content faithfully without rewriting or summarizing.

    Args:
        text: Raw source text.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # Remove null bytes and form feeds (common PyMuPDF/extraction artifacts)
    cleaned = text.replace("\x00", "").replace("\x0c", "\n")

    # Normalize carriage returns
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    # Strip trailing whitespace on each line
    lines = [line.rstrip() for line in cleaned.split("\n")]
    cleaned = "\n".join(lines)

    # Collapse 3 or more consecutive newlines into 2 (standard paragraph break)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def normalize_document(extracted: ExtractedDocument) -> ExtractedDocument:
    """Normalize extracted document blocks and overall text.

    Preserves source-location metadata (pages, block types, etc.) on blocks.

    Args:
        extracted: ExtractedDocument from an extractor.

    Returns:
        Normalized ExtractedDocument.

    Raises:
        ExtractionError: If no meaningful content remains after normalization.
    """
    normalized_blocks: list[ExtractedBlock] = []

    for block in extracted.blocks:
        norm_content = normalize_text(block.content)
        if norm_content:
            normalized_blocks.append(
                ExtractedBlock(
                    content=norm_content,
                    metadata=dict(block.metadata),
                )
            )

    if not normalized_blocks:
        raise ExtractionError(
            "Document contains no meaningful text after normalization."
        )

    full_text = "\n\n".join(b.content for b in normalized_blocks)

    return ExtractedDocument(
        text=full_text,
        blocks=normalized_blocks,
        source_type=extracted.source_type,
        metadata=dict(extracted.metadata),
    )
