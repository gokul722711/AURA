"""Deterministic echo tool for pipeline and testing validation."""

from typing import Any

from agent.tools.base import Tool, ToolResult


class MockEchoTool(Tool):
    """Simple deterministic tool that reflects input messages."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echoes back the provided input message for testing."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Message to echo back.",
                }
            },
            "required": ["message"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        if "message" not in kwargs:
            return ToolResult(
                tool_name=self.name,
                output=None,
                is_error=True,
                error_message="Missing required parameter 'message'.",
            )

        message = kwargs["message"]
        return ToolResult(
            tool_name=self.name,
            output={"echo": str(message)},
            is_error=False,
            metadata={"original_input": message},
        )
