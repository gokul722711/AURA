"""RAG retrieval tool bridging agent runtime to existing RAG retrieval."""

from typing import Any

from agent.tools.base import Tool, ToolResult
from rag.embeddings.base import EmbeddingProvider
from rag.retrieval import RetrievalConfig, _get_retrieval_config, retrieve_chunks


class RAGSearchTool(Tool):
    """Tool that queries AURA's vector retrieval pipeline for relevant chunks."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.config = config or _get_retrieval_config()

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
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of document UUIDs to restrict retrieval to.",
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
            target_top_k = top_k
        else:
            target_top_k = self.config.top_k

        document_ids = kwargs.get("document_ids")
        if document_ids is not None:
            if not isinstance(document_ids, (list, tuple)):
                return ToolResult(
                    tool_name=self.name,
                    output=None,
                    is_error=True,
                    error_message="Parameter 'document_ids' must be a list of strings.",
                )
            clean_doc_ids = [str(did).strip() for did in document_ids if str(did).strip()]
        else:
            clean_doc_ids = self.config.document_ids

        retrieval_config = RetrievalConfig(
            top_k=target_top_k,
            similarity_threshold=self.config.similarity_threshold,
            document_ids=clean_doc_ids,
            max_chunks_per_document=self.config.max_chunks_per_document,
        )

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
            meta = dict(r.chunk_metadata) if r.chunk_metadata else {}
            chunk_dict = {
                "chunk_id": r.chunk_id,
                "document_id": getattr(r, "document_id", None),
                "document_title": r.document_title,
                "document_source": r.document_source,
                "score": r.score,
                "content": r.content,
                "metadata": meta,
            }
            if meta.get("page") is not None:
                chunk_dict["page"] = meta["page"]

            url = None
            if meta.get("url") is not None:
                url = str(meta["url"])
            elif meta.get("canonical_url") is not None:
                url = str(meta["canonical_url"])
            elif hasattr(r, "chunk") and hasattr(r.chunk, "document") and getattr(r.chunk.document, "url", None):
                url = str(r.chunk.document.url)
            elif r.document_source and str(r.document_source).startswith(("http://", "https://")):
                url = str(r.document_source)

            if url:
                chunk_dict["url"] = url
                if "url" not in meta:
                    meta["url"] = url

            output_chunks.append(chunk_dict)

        return ToolResult(
            tool_name=self.name,
            output=output_chunks,
            is_error=False,
            metadata={
                "query": query,
                "retrieval_count": len(output_chunks),
            },
        )
