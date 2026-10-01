"""Context assembly for AURA RAG.

Formats retrieved document chunks into a structured context string
suitable for inclusion in an LLM prompt.

M3 enhancements:
- Configurable context budget (CONTEXT_MAX_CHARS) limits total context size.
- Retrieval ranking is preserved; highest-ranked results are included first.
- Deterministic redundancy handling detects overlapping chunks based on
  document identity and character offsets, skipping fully redundant chunks.
- Source/chunk attribution is maintained in the formatted output.
"""

from dataclasses import dataclass

from django.conf import settings

from rag.exceptions import ContextAssemblyError
from rag.retrieval import RetrievalResult


@dataclass(frozen=True)
class ContextConfig:
    """Configuration for context assembly.

    Attributes:
        max_chars: Maximum total characters of chunk content to include.
            None means no limit. Loaded from AI_RAG["CONTEXT_MAX_CHARS"].
    """

    max_chars: int | None = None


def _get_context_config() -> ContextConfig:
    """Build ContextConfig from Django settings."""
    rag_settings = getattr(settings, "AI_RAG", {})
    max_chars = rag_settings.get("CONTEXT_MAX_CHARS")
    return ContextConfig(max_chars=max_chars)


def _is_redundant(
    candidate: RetrievalResult,
    accepted: list[RetrievalResult],
) -> bool:
    """Check if a candidate chunk is fully redundant with an already-accepted chunk.

    A chunk is considered redundant if it comes from the same document AND
    its character span is fully contained within an already-accepted chunk's span.

    This uses the deterministic character offsets from chunking rather than
    content string comparison, which is cheaper and reliable for overlapping chunks.
    """
    for existing in accepted:
        if candidate.document_id != existing.document_id:
            continue
        # Candidate is fully contained within existing
        if (
            candidate.start_offset >= existing.start_offset
            and candidate.end_offset <= existing.end_offset
        ):
            return True
    return False


def assemble_context(
    retrieval_results: list[RetrievalResult],
    query: str,
    config: ContextConfig | None = None,
) -> str:
    """Assemble retrieved chunks into a structured context for the LLM.

    Produces a prompt-ready string containing source references, chunk text,
    and clear boundaries between context sections.

    M3 behavior:
    - Preserves retrieval ranking (highest-ranked first).
    - Applies context budget: stops adding chunks when max_chars is exhausted.
    - Skips redundant chunks whose content span is fully contained within
      an already-included chunk from the same document.
    - Maintains source/chunk attribution in the output.

    Args:
        retrieval_results: List of RetrievalResult from retrieval.
        query: The original user query.
        config: Context assembly configuration. Uses settings defaults if None.

    Returns:
        Formatted context string ready for LLM prompt.

    Raises:
        ContextAssemblyError: If assembly fails.
    """
    if not query or not query.strip():
        raise ContextAssemblyError("Query must be a non-empty string.")

    if not retrieval_results:
        return _format_no_context(query)

    if config is None:
        config = _get_context_config()

    sections = []
    accepted: list[RetrievalResult] = []
    chars_used = 0

    for result in retrieval_results:
        # Skip redundant chunks
        if _is_redundant(result, accepted):
            continue

        content = result.content
        content_len = len(content)

        # Check budget
        if config.max_chars is not None:
            if chars_used >= config.max_chars:
                break
            remaining = config.max_chars - chars_used
            if content_len > remaining:
                # Cannot fit this chunk; stop (preserve whole chunks only)
                break

        source_label = result.document_source or result.document_title
        section = (
            f"[Source: {source_label} | "
            f"Chunk {result.chunk_index + 1} | "
            f"Similarity: {result.score:.4f}]\n"
            f"{content}"
        )
        sections.append(section)
        accepted.append(result)
        chars_used += content_len

    if not sections:
        return _format_no_context(query)

    context_block = "\n\n---\n\n".join(sections)

    return (
        "Use the following retrieved context to answer the question. "
        "Base your answer on the provided context. "
        "If the context does not contain sufficient information, say so.\n\n"
        "--- Retrieved Context ---\n\n"
        f"{context_block}\n\n"
        "--- End of Context ---\n\n"
        f"Question: {query}"
    )


def _format_no_context(query: str) -> str:
    """Format a prompt when no relevant context was found."""
    return (
        "No relevant context was found for the following question. "
        "Answer based on your general knowledge and clearly state "
        "that no specific source documents were available.\n\n"
        f"Question: {query}"
    )
