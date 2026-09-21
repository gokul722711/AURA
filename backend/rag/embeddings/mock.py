"""Deterministic mock embedding provider for testing and local development.

Produces hash-based vectors that are deterministic (same input -> same output)
but do NOT have meaningful semantic properties. This provider exists for:

- Testing pipeline correctness
- Local development without external APIs
- Verifying deterministic vector generation

Do not use this provider for semantic search quality evaluation.
"""

import hashlib
import math

from rag.embeddings.base import EmbeddingProvider


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic mock embedding provider.

    Generates fixed-dimension vectors derived from text hashing.
    Same input text always produces the same output vector.

    These vectors do NOT have meaningful semantic similarity properties.
    They are suitable for testing pipeline mechanics only.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self._dimensions = dimensions

    def dimensions(self) -> int:
        """Return the configured vector dimensionality."""
        return self._dimensions

    def _hash_to_vector(self, text: str) -> list[float]:
        """Convert text to a deterministic unit vector via hashing.

        Uses SHA-256 to generate enough bytes, then normalizes
        the resulting vector to unit length for cosine distance compatibility.
        """
        # Generate enough hash bytes for the required dimensions
        # Each float needs ~4 bytes of hash entropy
        hash_bytes = b""
        counter = 0
        while len(hash_bytes) < self._dimensions * 4:
            data = f"{text}:{counter}".encode("utf-8")
            hash_bytes += hashlib.sha256(data).digest()
            counter += 1

        # Convert bytes to floats in [-1, 1]
        raw = []
        for i in range(self._dimensions):
            # Use 4 bytes per dimension, convert to float in [-1, 1]
            offset = i * 4
            value = int.from_bytes(hash_bytes[offset : offset + 4], "big")
            normalized = (value / (2**32 - 1)) * 2 - 1
            raw.append(normalized)

        # Normalize to unit vector for cosine distance compatibility
        magnitude = math.sqrt(sum(x * x for x in raw))
        if magnitude > 0:
            raw = [x / magnitude for x in raw]

        return raw

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic embeddings for a batch of texts."""
        return [self._hash_to_vector(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        """Generate a deterministic embedding for a query.

        Uses the same hashing as embed_texts — no document/query
        distinction for the mock provider.
        """
        return self._hash_to_vector(query)
