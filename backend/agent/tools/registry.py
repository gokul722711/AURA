"""Registry for agent tools."""

import re
from typing import Any

from agent.exceptions import DuplicateToolError, ToolNotFoundError
from agent.tools.base import Tool

_TOOL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


class ToolRegistry:
    """Registry maintaining available tools for the agent runtime."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a new tool instance."""
        if not isinstance(tool, Tool):
            raise TypeError(f"Expected Tool instance, got {type(tool).__name__}.")

        name = tool.name
        if not name or not isinstance(name, str):
            raise ValueError("Tool name must be a non-empty string.")

        if not _TOOL_NAME_PATTERN.match(name):
            raise ValueError(
                f"Tool name '{name}' is invalid. Must match pattern ^[a-zA-Z0-9_-]+$."
            )

        if name in self._tools:
            raise DuplicateToolError(f"Tool with name '{name}' is already registered.")

        self._tools[name] = tool

    def get(self, name: str) -> Tool:
        """Retrieve a registered tool by name."""
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' is not registered.")
        return self._tools[name]

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def list_tools(self) -> list[Tool]:
        """Return a list of all registered tools."""
        return list(self._tools.values())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Return tool definitions with JSON schemas for prompt or function calling."""
        schemas = []
        for tool in self._tools.values():
            schemas.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            })
        return schemas
