"""Context assembly for AURA RAG.

Formats retrieved document chunks into a structured context string
suitable for inclusion in an LLM prompt.
"""

from rag.exceptions import ContextAssemblyError
from rag.retrieval import RetrievalResult


def assemble_context(
    retrieval_results: list[RetrievalResult],
    query: str,
) -> str:
    """Assemble retrieved chunks into a structured context for the LLM.

    Produces a prompt-ready string containing source references, chunk text,
    and clear boundaries between context sections.

    Args:
        retrieval_results: List of RetrievalResult from retrieval.
        query: The original user query.

    Returns:
        Formatted context string ready for LLM prompt.

    Raises:
        ContextAssemblyError: If assembly fails.
    """
    if not query or not query.strip():
        raise ContextAssemblyError("Query must be a non-empty string.")

    if not retrieval_results:
        return _format_no_context(query)

    sections = []
    for result in retrieval_results:
        chunk = result.chunk
        source_label = chunk.document.source or chunk.document.title
        section = (
            f"[Source: {source_label} | "
            f"Chunk {chunk.chunk_index + 1} | "
            f"Similarity: {result.score:.4f}]\n"
            f"{chunk.content}"
        )
        sections.append(section)

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
