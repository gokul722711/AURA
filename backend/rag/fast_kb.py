"""Deterministic Knowledge Base Fast Path Pipeline for AURA (M13).

Executes ordinary Knowledge Base questions through a deterministic retrieval-first pipeline:
  User Question
        ↓
  Query Processing (process_query)
        ↓
  Dense Retrieval (pgvector) + Lexical Retrieval (PostgreSQL FTS)
        ↓
  Reciprocal Rank Fusion (RRF)
        ↓
  FlashRank Reranking (Reranker)
        ↓
  Bounded Context Budget
        ↓
  ONE ModelGateway.generate() Call
        ↓
  Grounded Answer + Citations in canonical ResearchResult

Guarantees:
- Planner calls: 0
- Structured-output calls: 0
- Agent/tool loops: 0
- Generation calls: exactly 1 (when context is found; 0 when no context)
"""

from dataclasses import dataclass, field
import logging
import time
from typing import Any

from django.conf import settings

from agent.results import ResearchEvidence, ResearchResult, _aggregate_sources
from agent.state import AgentStatus
from gateway.gateway import ModelGateway, get_gateway
from gateway.types import GenerationRequest, Message
from rag.context import ContextConfig, select_context_chunks
from rag.embeddings.base import EmbeddingProvider
from rag.embeddings.registry import create_embedding_provider
from rag.fusion import reciprocal_rank_fusion
from rag.query_processing import process_query
from rag.rerankers.base import Reranker
from rag.rerankers.flashrank import FlashRankReranker
from rag.retrieval import RetrievalConfig, RetrievalResult, retrieve_chunks, retrieve_lexical_chunks

logger = logging.getLogger(__name__)


def _get_default_embedding_provider() -> EmbeddingProvider:
    """Instantiate the default embedding provider from Django settings."""
    try:
        conf = getattr(settings, "AI_EMBEDDINGS", {})
        provider = conf.get("PROVIDER", "mock")
        dims = conf.get("DIMENSIONS", 384)
    except Exception:
        provider = "mock"
        dims = 384
    return create_embedding_provider(provider, dims)


@dataclass(frozen=True)
class KBConfig:
    """Configuration for DeterministicKBPipeline.

    Attributes:
        dense_top_k: Number of candidates to retrieve via dense pgvector search.
        lexical_top_k: Number of candidates to retrieve via PostgreSQL FTS search.
        rrf_k: Smoothing constant for Reciprocal Rank Fusion.
        dense_weight: Weight multiplier for dense rankings in RRF.
        lexical_weight: Weight multiplier for lexical rankings in RRF.
        rerank_top_k: Number of candidates to retain after reranking.
        reranker_min_score: Minimum relevance score required to enter final context.
        similarity_threshold: Minimum cosine similarity for dense retrieval.
        max_chunks_per_document: Fairness cap per document.
        context_config: Context budget configuration (max_chars, etc.).
        temperature: LLM generation temperature.
        max_tokens: Maximum tokens for LLM generation.
    """

    dense_top_k: int = 10
    lexical_top_k: int = 10
    rrf_k: int = 60
    dense_weight: float = 1.0
    lexical_weight: float = 1.0
    rerank_top_k: int = 5
    reranker_min_score: float = 0.0001
    similarity_threshold: float = 0.0
    max_chunks_per_document: int | None = None
    context_config: ContextConfig = field(default_factory=ContextConfig)
    temperature: float = 0.1
    max_tokens: int = 1024

    @classmethod
    def from_settings(cls) -> "KBConfig":
        """Build KBConfig from Django settings."""
        rag_conf = getattr(settings, "AI_RAG", {})
        top_k = rag_conf.get("TOP_K", 5)
        sim_thresh = rag_conf.get("SIMILARITY_THRESHOLD", 0.0)
        max_chars = rag_conf.get("CONTEXT_MAX_CHARS")
        max_per_doc = rag_conf.get("MAX_CHUNKS_PER_DOC")
        reranker_min_score = rag_conf.get("RERANKER_MIN_SCORE", 0.0001)

        return cls(
            dense_top_k=max(top_k * 2, 10),
            lexical_top_k=max(top_k * 2, 10),
            rrf_k=60,
            dense_weight=1.0,
            lexical_weight=1.0,
            rerank_top_k=top_k,
            reranker_min_score=reranker_min_score,
            similarity_threshold=sim_thresh,
            max_chunks_per_document=max_per_doc,
            context_config=ContextConfig(
                max_chars=max_chars,
                max_chunks_per_document=max_per_doc,
            ),
            temperature=0.1,
            max_tokens=1024,
        )


class DeterministicKBPipeline:
    """Deterministic, retrieval-first Knowledge Base pipeline for M13."""

    def __init__(
        self,
        gateway: ModelGateway | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        reranker: Reranker | None = None,
        config: KBConfig | None = None,
    ) -> None:
        self.gateway = gateway or get_gateway()
        self.embedding_provider = embedding_provider or _get_default_embedding_provider()
        self.reranker = reranker or FlashRankReranker()
        self.config = config or KBConfig.from_settings()

    def run(
        self,
        objective: str,
        document_ids: list[str] | None = None,
    ) -> ResearchResult:
        """Execute deterministic knowledge base research and produce a ResearchResult.

        Workflow:
        1. Local query normalization.
        2. Dense pgvector retrieval.
        3. Lexical PostgreSQL FTS retrieval.
        4. Reciprocal Rank Fusion (RRF).
        5. FlashRank reranking.
        6. Context selection and bounding.
        7. Exactly ONE ModelGateway.generate() call (or 0 if no context).
        8. Construct canonical ResearchResult with verifiable citations.
        """
        start_time = time.monotonic()
        timings: dict[str, float] = {}

        # 1. Query Processing
        t0 = time.monotonic()
        processed_query = process_query(objective)
        normalized_query = processed_query.normalized
        timings["query_processing_ms"] = round((time.monotonic() - t0) * 1000.0, 2)

        # Build retrieval configurations
        dense_retrieval_cfg = RetrievalConfig(
            top_k=self.config.dense_top_k,
            similarity_threshold=self.config.similarity_threshold,
            document_ids=document_ids,
            max_chunks_per_document=self.config.max_chunks_per_document,
        )
        lexical_retrieval_cfg = RetrievalConfig(
            top_k=self.config.lexical_top_k,
            similarity_threshold=0.0,
            document_ids=document_ids,
            max_chunks_per_document=self.config.max_chunks_per_document,
        )

        # 2. Dense pgvector retrieval
        t_dense_start = time.monotonic()
        try:
            dense_results = retrieve_chunks(
                query=normalized_query,
                embedding_provider=self.embedding_provider,
                config=dense_retrieval_cfg,
            )
        except Exception as exc:
            logger.warning("Dense retrieval encountered error: %s", exc)
            dense_results = []
        timings["dense_retrieval_ms"] = round((time.monotonic() - t_dense_start) * 1000.0, 2)

        # 3. Lexical PostgreSQL FTS retrieval
        t_lex_start = time.monotonic()
        try:
            lexical_results = retrieve_lexical_chunks(
                query=normalized_query,
                config=lexical_retrieval_cfg,
            )
        except Exception as exc:
            logger.warning("Lexical retrieval encountered error: %s", exc)
            lexical_results = []
        timings["lexical_retrieval_ms"] = round((time.monotonic() - t_lex_start) * 1000.0, 2)

        # 4. Reciprocal Rank Fusion (RRF)
        t_rrf_start = time.monotonic()
        fused_candidates = reciprocal_rank_fusion(
            dense_results=dense_results,
            lexical_results=lexical_results,
            k=self.config.rrf_k,
            dense_weight=self.config.dense_weight,
            lexical_weight=self.config.lexical_weight,
            top_k=max(self.config.dense_top_k, self.config.lexical_top_k),
        )
        timings["rrf_ms"] = round((time.monotonic() - t_rrf_start) * 1000.0, 2)

        # 5. FlashRank Reranking & Relevance Filtering
        t_rerank_start = time.monotonic()
        try:
            try:
                reranked_candidates = self.reranker.rerank(
                    query=normalized_query,
                    candidates=fused_candidates,
                    top_k=self.config.rerank_top_k,
                    min_score=self.config.reranker_min_score,
                )
            except TypeError:
                # Custom reranker that does not accept min_score argument
                reranked_candidates = self.reranker.rerank(
                    query=normalized_query,
                    candidates=fused_candidates,
                    top_k=self.config.rerank_top_k,
                )
        except Exception as exc:
            logger.warning("Reranker encountered error, falling back to fused candidates: %s", exc)
            fallback = fused_candidates
            if self.config.reranker_min_score > 0.0:
                fallback = [c for c in fallback if c.score >= self.config.reranker_min_score]
            reranked_candidates = fallback[: self.config.rerank_top_k]

        # Enforce that candidates strictly meet the minimum relevance threshold
        if self.config.reranker_min_score > 0.0:
            reranked_candidates = [
                c for c in reranked_candidates
                if c.score >= self.config.reranker_min_score
            ]
        timings["reranking_ms"] = round((time.monotonic() - t_rerank_start) * 1000.0, 2)

        # 6. Check No-Context Case
        if not reranked_candidates:
            total_duration_ms = round((time.monotonic() - start_time) * 1000.0, 2)
            timings["total_ms"] = total_duration_ms
            return ResearchResult(
                objective=objective,
                final_answer=(
                    f"The available knowledge base did not provide sufficient supporting evidence "
                    f"to answer the research objective: '{objective}'."
                ),
                evidence=[],
                sources=[],
                queries=[normalized_query],
                iteration_count=1,
                has_evidence=False,
                status=AgentStatus.COMPLETED,
                duration_ms=total_duration_ms,
                errors=[],
                metadata={
                    "mode": "knowledge_base",
                    "pipeline": "DeterministicKBPipeline",
                    "context_grounded": False,
                    "timings": timings,
                },
            )

        # 7. Context Selection & Bounding
        accepted_chunks = select_context_chunks(
            reranked_candidates,
            config=self.config.context_config,
        )

        if not accepted_chunks:
            total_duration_ms = round((time.monotonic() - start_time) * 1000.0, 2)
            timings["total_ms"] = total_duration_ms
            return ResearchResult(
                objective=objective,
                final_answer=(
                    f"The available knowledge base did not provide sufficient supporting evidence "
                    f"to answer the research objective: '{objective}'."
                ),
                evidence=[],
                sources=[],
                queries=[normalized_query],
                iteration_count=1,
                has_evidence=False,
                status=AgentStatus.COMPLETED,
                duration_ms=total_duration_ms,
                errors=[],
                metadata={
                    "mode": "knowledge_base",
                    "pipeline": "DeterministicKBPipeline",
                    "context_grounded": False,
                    "timings": timings,
                },
            )

        # 8. Transform accepted chunks into ResearchEvidence
        evidence: list[ResearchEvidence] = []
        for c in accepted_chunks:
            ev = ResearchEvidence(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                document_title=c.document_title,
                document_source=c.document_source,
                content=c.content,
                score=c.score,
                metadata=c.chunk_metadata or {},
            )
            evidence.append(ev)

        sources = _aggregate_sources(evidence)

        # 9. Format grounded evidence context for LLM
        evidence_items = []
        for idx, ev in enumerate(evidence, start=1):
            title = ev.document_title or "Untitled"
            source = ev.document_source or "Unknown"
            cid = ev.chunk_id
            page_str = f" | Page: {ev.page}" if ev.page is not None else ""
            evidence_items.append(
                f"[{idx}] Source: {title} | Document: {source}{page_str} | Chunk: {cid}\n{ev.content.strip()}"
            )
        evidence_block = "\n\n".join(evidence_items)

        system_prompt = (
            "You are AURA's Knowledge Base Assistant. You synthesize comprehensive, "
            "factually grounded answers based strictly on retrieved knowledge base evidence.\n\n"
            "Grounding Rules:\n"
            "1. Rely strictly on the provided evidence. Do NOT extrapolate, hallucinate, "
            "or assume facts not supported by the evidence.\n"
            "2. If the evidence is insufficient to answer parts of the objective, clearly "
            "indicate that the available knowledge base did not provide sufficient supporting evidence.\n"
            "3. Cite supporting sources using references like [Source Title, Chunk: ID] or "
            "[Source Title, Page: X, Chunk: ID] where appropriate. Never invent citations or reference sources "
            "not present in the provided evidence."
        )

        user_content = (
            f"Research Objective:\n{objective}\n\n"
            f"Retrieved Evidence:\n{evidence_block}\n\n"
            f"Provide a grounded, comprehensive answer addressing the research objective "
            f"based strictly on the evidence above."
        )

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_content),
        ]

        # 10. Single ModelGateway Generation Call
        t_gen_start = time.monotonic()
        gen_request = GenerationRequest(
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        response = self.gateway.generate(gen_request)
        timings["generation_ms"] = round((time.monotonic() - t_gen_start) * 1000.0, 2)

        total_duration_ms = round((time.monotonic() - start_time) * 1000.0, 2)
        timings["total_ms"] = total_duration_ms

        final_answer = response.text.strip()

        return ResearchResult(
            objective=objective,
            final_answer=final_answer,
            evidence=evidence,
            sources=sources,
            queries=[normalized_query],
            iteration_count=1,
            has_evidence=True,
            status=AgentStatus.COMPLETED,
            duration_ms=total_duration_ms,
            errors=[],
            metadata={
                "mode": "knowledge_base",
                "pipeline": "DeterministicKBPipeline",
                "context_grounded": True,
                "model": response.model,
                "provider": response.provider,
                "timings": timings,
            },
        )


def create_deterministic_kb_pipeline(
    gateway: ModelGateway | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    reranker: Reranker | None = None,
    config: KBConfig | None = None,
) -> DeterministicKBPipeline:
    """Factory function to build a configured DeterministicKBPipeline."""
    active_reranker = reranker
    if active_reranker is None:
        rag_conf = getattr(settings, "AI_RAG", {})
        model_name = rag_conf.get("RERANKER_MODEL", "ms-marco-TinyBERT-L-2-v2")
        cache_dir = rag_conf.get("RERANKER_CACHE_DIR", "/tmp")
        min_score = rag_conf.get("RERANKER_MIN_SCORE", 0.0001)
        active_reranker = FlashRankReranker(model_name=model_name, cache_dir=cache_dir, min_score=min_score)

    return DeterministicKBPipeline(
        gateway=gateway,
        embedding_provider=embedding_provider,
        reranker=active_reranker,
        config=config,
    )
