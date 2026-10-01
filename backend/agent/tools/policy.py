"""Security and authorization policy layer for tool execution."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from agent.state import AgentState
from agent.tools.registry import ToolRegistry


class PolicyDecision(str, Enum):
    """Possible outcomes of a tool policy check."""

    ALLOWED = "allowed"
    DENIED = "denied"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PolicyCheckResult:
    """Result of evaluating a tool execution against policy."""

    decision: PolicyDecision
    reason: str
    tool_name: str


class ToolPolicy(ABC):
    """Abstract interface for tool authorization policies."""

    @abstractmethod
    def evaluate(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        registry: ToolRegistry,
        state: AgentState,
    ) -> PolicyCheckResult:
        """Evaluate if a tool execution is permitted."""
        pass


class DefaultToolPolicy(ToolPolicy):
    """Default policy implementing explicit allowlists, denylists, and existence checks."""

    def __init__(
        self,
        allowed_tools: set[str] | None = None,
        denied_tools: set[str] | None = None,
    ) -> None:
        self.allowed_tools = set(allowed_tools) if allowed_tools is not None else None
        self.denied_tools = set(denied_tools) if denied_tools is not None else set()

    def evaluate(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        registry: ToolRegistry,
        state: AgentState,
    ) -> PolicyCheckResult:
        # 1. Tool existence check
        if not registry.has_tool(tool_name):
            return PolicyCheckResult(
                decision=PolicyDecision.UNKNOWN,
                reason=f"Tool '{tool_name}' does not exist in registry.",
                tool_name=tool_name,
            )

        # 2. Denylist check
        if tool_name in self.denied_tools:
            return PolicyCheckResult(
                decision=PolicyDecision.DENIED,
                reason=f"Tool '{tool_name}' is explicitly denied by policy.",
                tool_name=tool_name,
            )

        # 3. Allowlist check (if defined)
        if self.allowed_tools is not None and tool_name not in self.allowed_tools:
            return PolicyCheckResult(
                decision=PolicyDecision.DENIED,
                reason=f"Tool '{tool_name}' is not in the allowed tools list.",
                tool_name=tool_name,
            )

        # 4. Input structure check
        if not isinstance(tool_input, dict):
            return PolicyCheckResult(
                decision=PolicyDecision.DENIED,
                reason="Tool input arguments must be provided as a dictionary.",
                tool_name=tool_name,
            )

        tool = registry.get(tool_name)
        schema = tool.input_schema
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in tool_input:
                return PolicyCheckResult(
                    decision=PolicyDecision.DENIED,
                    reason=f"Missing required input parameter '{field}' for tool '{tool_name}'.",
                    tool_name=tool_name,
                )

        return PolicyCheckResult(
            decision=PolicyDecision.ALLOWED,
            reason="Tool execution permitted by policy.",
            tool_name=tool_name,
        )
