"""Autonomous research runtime factory and helpers for AURA M5."""

from agent.execution.limits import ExecutionLimits
from agent.planning.research import ResearchPlanner
from agent.runtime import AgentRuntime
from agent.tools.builtin.rag import RAGSearchTool
from agent.tools.policy import DefaultToolPolicy, ToolPolicy
from agent.tools.registry import ToolRegistry
from gateway.gateway import ModelGateway, get_gateway
from rag.embeddings.base import EmbeddingProvider
from rag.embeddings.registry import create_embedding_provider
from rag.retrieval import RetrievalConfig


def _get_default_embedding_provider() -> EmbeddingProvider:
    try:
        from django.conf import settings

        conf = getattr(settings, "AI_EMBEDDINGS", {})
        provider = conf.get("PROVIDER", "mock")
        dims = conf.get("DIMENSIONS", 384)
    except Exception:
        provider = "mock"
        dims = 384
    return create_embedding_provider(provider, dims)


def create_research_runtime(
    gateway: ModelGateway | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    limits: ExecutionLimits | None = None,
    policy: ToolPolicy | None = None,
    retrieval_config: RetrievalConfig | None = None,
    max_queries: int | None = None,
) -> AgentRuntime:
    """Create an AgentRuntime configured for autonomous research with RAGSearchTool and ResearchPlanner."""
    active_gateway = gateway or get_gateway()
    active_embeddings = embedding_provider or _get_default_embedding_provider()

    registry = ToolRegistry()
    rag_tool = RAGSearchTool(
        embedding_provider=active_embeddings,
        config=retrieval_config,
    )
    registry.register(rag_tool)

    planner = ResearchPlanner(
        gateway=active_gateway,
        max_queries=max_queries,
    )

    active_policy = policy or DefaultToolPolicy(allowed_tools={"rag_search"})

    return AgentRuntime(
        planner=planner,
        gateway=active_gateway,
        registry=registry,
        policy=active_policy,
        limits=limits,
    )
