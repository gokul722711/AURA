"""Autonomous research runtime factory, evidence representations, and structured result models for AURA."""

from typing import Any

from agent.execution.limits import ExecutionLimits
from agent.planning.research import ResearchPlanner
from agent.results import ResearchEvidence, ResearchResult
from agent.runtime import AgentRunResult, AgentRuntime
from agent.tools.builtin.rag import RAGSearchTool
from agent.tools.builtin.web import WebSearchProvider, WebSearchTool
from agent.tools.policy import DefaultToolPolicy, ToolPolicy
from agent.tools.registry import ToolRegistry
from gateway.gateway import ModelGateway, get_gateway
from rag.embeddings.base import EmbeddingProvider
from rag.embeddings.registry import create_embedding_provider
from rag.retrieval import RetrievalConfig

__all__ = [
    "ResearchEvidence",
    "ResearchResult",
    "ResearchRuntime",
    "create_research_runtime",
]


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


class ResearchRuntime(AgentRuntime):
    """Specialized AgentRuntime for autonomous research workflows."""

    def run_research(
        self,
        objective: str,
        metadata: dict[str, Any] | None = None,
    ) -> ResearchResult:
        """Execute autonomous research and return a structured ResearchResult."""
        run_result = self.run(objective, metadata=metadata)
        return ResearchResult.from_run_result(run_result)


def create_research_runtime(
    gateway: ModelGateway | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    web_search_provider: WebSearchProvider | None = None,
    limits: ExecutionLimits | None = None,
    policy: ToolPolicy | None = None,
    retrieval_config: RetrievalConfig | None = None,
    max_queries: int | None = None,
    decision_max_tokens: int = 256,
    synthesis_max_tokens: int = 1024,
    mode: str = "knowledge_base",
) -> ResearchRuntime:
    """Create an AgentRuntime configured for autonomous research with permitted tools for the mode."""
    active_gateway = gateway or get_gateway()
    registry = ToolRegistry()
    allowed_tools: set[str] = set()

    # 1. Register Knowledge Base tool if allowed by mode
    if mode in ("knowledge_base", "web_knowledge_base"):
        active_embeddings = embedding_provider or _get_default_embedding_provider()
        rag_tool = RAGSearchTool(
            embedding_provider=active_embeddings,
            config=retrieval_config,
        )
        registry.register(rag_tool)
        allowed_tools.add("rag_search")

    # 2. Register Web Search tool if allowed by mode
    if mode in ("web", "web_knowledge_base"):
        web_tool = WebSearchTool(provider=web_search_provider)
        registry.register(web_tool)
        allowed_tools.add("web_search")

    # 3. Model Knowledge mode registers no search tools and has empty allowed_tools

    active_policy = policy or DefaultToolPolicy(allowed_tools=allowed_tools)

    planner = ResearchPlanner(
        gateway=active_gateway,
        max_queries=max_queries,
        decision_max_tokens=decision_max_tokens,
        synthesis_max_tokens=synthesis_max_tokens,
        mode=mode,
        available_tools=allowed_tools,
    )

    return ResearchRuntime(
        planner=planner,
        gateway=active_gateway,
        registry=registry,
        policy=active_policy,
        limits=limits,
    )
