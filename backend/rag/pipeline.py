"""RAG pipeline orchestration for AURA.

Orchestrates the full retrieval-augmented generation flow:
query → retrieve → assemble context → generate via ModelGateway → response.

Generation MUST go through ModelGateway.generate().
RAG code must never directly call an LLM provider.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from gateway.gateway import ModelGateway
from gateway.types import GenerationRequest, Message

from rag.context import assemble_context
from rag.embeddings.base import EmbeddingProvider
from rag.retrieval import RetrievalConfig, RetrievalResult, retrieve_chunks

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RAGConfig:
    """Configuration for the RAG pipeline.

    Attributes:
        retrieval_config: Configuration for chunk retrieval.
        system_prompt: System prompt template for the LLM.
        temperature: LLM generation temperature.
        max_tokens: Maximum tokens for LLM generation.
    """

    retrieval_config: RetrievalConfig = field(default_factory=RetrievalConfig)
    system_prompt: str = (
        "You are a helpful research assistant. "
        "Answer questions based on the provided context. "
        "Cite your sources when possible. "
        "If the context is insufficient, clearly state what is missing."
    )
    temperature: float = 0.1
    max_tokens: int | None = None


@dataclass
class RAGResponse:
    """Response from the RAG pipeline.

    Attributes:
        answer: The generated answer text.
        sources: List of RetrievalResult showing which chunks were used.
        query: The original query.
        usage: Token usage information from the LLM.
        metadata: Additional metadata about the generation.
    """

    answer: str
    sources: list[RetrievalResult]
    query: str
    usage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGPipeline:
    """Orchestrates retrieval-augmented generation.

    Uses an EmbeddingProvider for query/document embedding and
    ModelGateway for text generation. Never calls an LLM provider directly.
    """

    def __init__(
        self,
        gateway: ModelGateway,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._gateway = gateway
        self._embedding_provider = embedding_provider

    @property
    def gateway(self) -> ModelGateway:
        """Return the active ModelGateway."""
        return self._gateway

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        """Return the active EmbeddingProvider."""
        return self._embedding_provider

    def query(
        self,
        question: str,
        config: RAGConfig | None = None,
    ) -> RAGResponse:
        """Execute a full RAG query.

        1. Retrieve relevant chunks via embedding similarity search
        2. Assemble retrieved context into a structured prompt
        3. Generate an answer via ModelGateway.generate()
        4. Return RAGResponse with answer, sources, and usage

        Args:
            question: The user's question.
            config: RAG pipeline configuration. Uses defaults if None.

        Returns:
            RAGResponse with the generated answer and retrieval sources.
        """
        if config is None:
            config = RAGConfig()

        # Step 1: Retrieve relevant chunks
        retrieval_results = retrieve_chunks(
            query=question,
            embedding_provider=self._embedding_provider,
            config=config.retrieval_config,
        )

        logger.info(
            "RAG retrieval: %d chunks found for query '%s'",
            len(retrieval_results),
            question[:80],
        )

        # Step 2: Assemble context
        assembled_context = assemble_context(retrieval_results, question)

        # Step 3: Generate via ModelGateway (never call provider directly)
        messages = [
            Message(role="system", content=config.system_prompt),
            Message(role="user", content=assembled_context),
        ]

        generation_request = GenerationRequest(
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        response = self._gateway.generate(generation_request)

        logger.info(
            "RAG generation complete: %d tokens used.",
            response.usage.total_tokens,
        )

        # Step 4: Build RAGResponse
        return RAGResponse(
            answer=response.text,
            sources=retrieval_results,
            query=question,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            metadata={
                "provider": response.provider,
                "model": response.model,
                "finish_reason": response.finish_reason,
                "retrieval_count": len(retrieval_results),
            },
        )
