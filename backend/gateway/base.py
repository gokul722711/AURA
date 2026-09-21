"""Abstract base interface for LLM providers."""

from abc import ABC, abstractmethod
from collections.abc import Iterator

from gateway.types import (
    GenerationRequest,
    GenerationResponse,
    ProviderMetadata,
    StreamChunk,
    StructuredOutputRequest,
    StructuredOutputResponse,
)


class LLMProvider(ABC):
    """Abstract base class representing an LLM inference provider."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a complete text response for the given request."""
        pass

    @abstractmethod
    def stream(self, request: GenerationRequest) -> Iterator[StreamChunk]:
        """Generate a stream of chunks incrementally for the given request."""
        pass

    @abstractmethod
    def structured_output(
        self, request: StructuredOutputRequest
    ) -> StructuredOutputResponse:
        """Generate structured output adhering to a schema or specification."""
        pass

    @abstractmethod
    def metadata(self) -> ProviderMetadata:
        """Return provider and model metadata and supported capabilities."""
        pass
