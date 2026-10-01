"""Deterministic, provider-independent query processing for AURA RAG.

Provides local query normalization without calling any external model.
The original user query is always preserved separately from the
normalized retrieval query.

This module does NOT implement:
- LLM-based query rewriting
- Query expansion
- Query decomposition
- Any external model calls
"""

import re
from dataclasses import dataclass

from rag.exceptions import QueryProcessingError


@dataclass(frozen=True)
class ProcessedQuery:
    """Result of query processing.

    Attributes:
        original: The original user query, unmodified.
        normalized: The normalized query for retrieval.
            Trimmed, whitespace-normalized, suitable for embedding.
    """

    original: str
    normalized: str


def process_query(query: str) -> ProcessedQuery:
    """Process a raw user query for retrieval.

    Performs deterministic, local normalization:
    1. Preserves the original query.
    2. Trims surrounding whitespace.
    3. Normalizes repeated internal whitespace to single spaces.
    4. Validates that the result is non-empty.

    No external model or LLM is called.

    Args:
        query: The raw user query string.

    Returns:
        ProcessedQuery with original and normalized forms.

    Raises:
        QueryProcessingError: If the query is empty or whitespace-only.
    """
    if query is None:
        raise QueryProcessingError("Query must not be None.")

    original = query

    # Trim surrounding whitespace
    trimmed = query.strip()

    if not trimmed:
        raise QueryProcessingError("Query must be a non-empty string.")

    # Normalize repeated whitespace to single spaces
    normalized = re.sub(r"\s+", " ", trimmed)

    return ProcessedQuery(original=original, normalized=normalized)
