"""RAG pipeline orchestration for AURA.

Orchestrates the full retrieval-augmented generation flow:
query → process → retrieve → assemble context → generate via ModelGateway → response.

Generation MUST go through ModelGateway.generate().
RAG code must never directly call an LLM provider.

M3 enhancements:
- Query processing: deterministic local normalization before retrieval.
- Context budget: configurable CONTEXT_MAX_CHARS limits context size.
- Explicit no-context behavior: RAGResponse.has_context distinguishes
  context-grounded from ungrounded responses. Metadata includes
  context_grounded flag.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from gateway.gateway import ModelGateway
from gateway.types import GenerationRequest, Message

from rag.context import ContextConfig, assemble_context
from rag.embeddings.base import EmbeddingProvider
from rag.query_processing import ProcessedQuery, process_query
from rag.retrieval import RetrievalConfig, RetrievalResult, retrieve_chunks

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RAGConfig:
    """Configuration for the RAG pipeline.

    Attributes:
        retrieval_config: Configuration for chunk retrieval.
        context_config: Configuration for context assembly (budget).
        system_prompt: System prompt template for the LLM.
        temperature: LLM generation temperature.
        max_tokens: Maximum tokens for LLM generation.
    """

    retrieval_config: RetrievalConfig = field(default_factory=RetrievalConfig)
    context_config: ContextConfig = field(default_factory=ContextConfig)
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
        processed_query: The ProcessedQuery with original and normalized forms.
        has_context: True if relevant retrieved context was used;
            False if no sufficiently relevant context exists.
        usage: Token usage information from the LLM.
        metadata: Additional metadata about the generation.
    """

    answer: str
    sources: list[RetrievalResult]
    query: str
    processed_query: ProcessedQuery
    has_context: bool
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
        document_ids: list[str] | None = None,
    ) -> RAGResponse:
        """Execute a full RAG query.

        1. Process/normalize the query
        2. Retrieve relevant chunks via embedding similarity search
        3. Assemble retrieved context into a structured prompt
        4. Generate an answer via ModelGateway.generate()
        5. Return RAGResponse with answer, sources, context status, and usage

        Args:
            question: The user's question.
            config: RAG pipeline configuration. Uses defaults if None.
            document_ids: Optional list of document UUIDs to restrict retrieval to.

        Returns:
            RAGResponse with the generated answer, retrieval sources,
            and explicit has_context flag.
        """
        if config is None:
            config = RAGConfig()

        retrieval_cfg = config.retrieval_config
        if document_ids is not None:
            retrieval_cfg = RetrievalConfig(
                top_k=retrieval_cfg.top_k,
                similarity_threshold=retrieval_cfg.similarity_threshold,
                document_ids=document_ids,
                max_chunks_per_document=retrieval_cfg.max_chunks_per_document,
            )

        # Step 1: Process query (deterministic, local normalization)
        processed = process_query(question)

        # Step 2: Retrieve relevant chunks using the normalized query
        retrieval_results = retrieve_chunks(
            query=processed.normalized,
            embedding_provider=self._embedding_provider,
            config=retrieval_cfg,
        )

        has_context = len(retrieval_results) > 0

        logger.info(
            "RAG retrieval: %d chunks found for query '%s' (has_context=%s)",
            len(retrieval_results),
            processed.normalized[:80],
            has_context,
        )

        # Step 3: Assemble context with budget
        assembled_context = assemble_context(
            retrieval_results,
            processed.normalized,
            config=config.context_config,
        )

        # Step 4: Generate via ModelGateway (never call provider directly)
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

        # Step 5: Build RAGResponse with explicit no-context indicator
        return RAGResponse(
            answer=response.text,
            sources=retrieval_results,
            query=question,
            processed_query=processed,
            has_context=has_context,
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
                "context_grounded": has_context,
            },
        )
