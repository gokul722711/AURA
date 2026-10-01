"""RAG retrieval tool bridging agent runtime to existing RAG retrieval."""

from typing import Any

from agent.tools.base import Tool, ToolResult
from rag.embeddings.base import EmbeddingProvider
from rag.retrieval import RetrievalConfig, retrieve_chunks


class RAGSearchTool(Tool):
    """Tool that queries AURA's vector retrieval pipeline for relevant chunks."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.config = config or RetrievalConfig()

    @property
    def name(self) -> str:
        return "rag_search"

    @property
    def description(self) -> str:
        return (
            "Search indexed knowledge base documents for relevant chunks "
            "matching a natural language query."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query to search knowledge base.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of chunks to retrieve (optional).",
                },
            },
            "required": ["query"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        query = kwargs.get("query")
        if not query or not isinstance(query, str):
            return ToolResult(
                tool_name=self.name,
                output=None,
                is_error=True,
                error_message="Parameter 'query' must be a non-empty string.",
            )

        top_k = kwargs.get("top_k")
        if top_k is not None:
            if not isinstance(top_k, int) or top_k <= 0:
                return ToolResult(
                    tool_name=self.name,
                    output=None,
                    is_error=True,
                    error_message="Parameter 'top_k' must be a positive integer.",
                )
            retrieval_config = RetrievalConfig(
                top_k=top_k,
                similarity_threshold=self.config.similarity_threshold,
            )
        else:
            retrieval_config = self.config

        try:
            results = retrieve_chunks(
                query=query,
                embedding_provider=self.embedding_provider,
                config=retrieval_config,
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                output=None,
                is_error=True,
                error_message=f"RAG retrieval failed: {exc}",
                metadata={"query": query},
            )

        output_chunks = []
        for r in results:
            output_chunks.append({
                "chunk_id": r.chunk_id,
                "document_id": getattr(r, "document_id", None),
                "document_title": r.document_title,
                "document_source": r.document_source,
                "score": r.score,
                "content": r.content,
            })

        return ToolResult(
            tool_name=self.name,
            output=output_chunks,
            is_error=False,
            metadata={
                "query": query,
                "retrieval_count": len(output_chunks),
            },
        )
