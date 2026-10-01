"""Base contracts for AURA Agent tools."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    """Standardized result returned by tool execution."""

    tool_name: str
    output: Any
    is_error: bool = False
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "output": self.output,
            "is_error": self.is_error,
            "error_message": self.error_message,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolResult":
        return cls(
            tool_name=data["tool_name"],
            output=data.get("output"),
            is_error=data.get("is_error", False),
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
        )


class Tool(ABC):
    """Abstract base class for all agent tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human/LLM readable description of tool function."""
        pass

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """JSON schema defining the expected kwargs."""
        pass

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool deterministically with provided keyword arguments."""
        pass
