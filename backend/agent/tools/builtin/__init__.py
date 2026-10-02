"""Builtin tools for AURA Agent Runtime."""

from agent.tools.builtin.calculator import CalculatorTool
from agent.tools.builtin.echo import MockEchoTool
from agent.tools.builtin.rag import RAGSearchTool
from agent.tools.builtin.web import (
    DuckDuckGoHTMLParser,
    DuckDuckGoWebSearchProvider,
    MockWebSearchProvider,
    WebSearchProvider,
    WebSearchResult,
    WebSearchTool,
    extract_ddg_destination_url,
    validate_web_url,
)

__all__ = [
    "CalculatorTool",
    "DuckDuckGoHTMLParser",
    "DuckDuckGoWebSearchProvider",
    "MockEchoTool",
    "MockWebSearchProvider",
    "RAGSearchTool",
    "WebSearchProvider",
    "WebSearchResult",
    "WebSearchTool",
    "extract_ddg_destination_url",
    "validate_web_url",
]
