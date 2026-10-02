"""Execution limits and budget tracking for AURA Agent Runtime."""

import time
from dataclasses import dataclass
from django.conf import settings

from agent.exceptions import (
    IterationLimitExceededError,
    TimeoutLimitExceededError,
    ToolCallLimitExceededError,
)


@dataclass(frozen=True)
class ExecutionLimits:
    """Configurable boundaries on agent execution."""

    max_iterations: int = 10
    max_tool_calls: int = 15
    max_time_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be greater than 0.")
        if self.max_tool_calls < 0:
            raise ValueError("max_tool_calls must be non-negative.")
        if self.max_time_seconds <= 0.0:
            raise ValueError("max_time_seconds must be greater than 0.0.")

    @classmethod
    def from_settings(cls) -> "ExecutionLimits":
        """Build ExecutionLimits from Django settings.AI_AGENT."""
        agent_settings = getattr(settings, "AI_AGENT", {})
        return cls(
            max_iterations=agent_settings.get("MAX_ITERATIONS", 10),
            max_tool_calls=agent_settings.get("MAX_TOOL_CALLS", 15),
            max_time_seconds=agent_settings.get("MAX_TIME_SECONDS", 60.0),
        )


class LimitTracker:
    """Monitors real-time resource consumption against ExecutionLimits."""

    def __init__(self, limits: ExecutionLimits) -> None:
        self.limits = limits
        self.start_time = time.monotonic()
        self.iterations = 0
        self.tool_calls = 0

    def increment_iteration(self) -> None:
        """Increment and check iteration count."""
        self.iterations += 1
        if self.iterations > self.limits.max_iterations:
            raise IterationLimitExceededError(
                f"Iteration limit of {self.limits.max_iterations} exceeded."
            )

    def increment_tool_call(self) -> None:
        """Increment and check cumulative tool calls."""
        self.tool_calls += 1
        if self.tool_calls > self.limits.max_tool_calls:
            raise ToolCallLimitExceededError(
                f"Tool call limit of {self.limits.max_tool_calls} exceeded."
            )

    def remaining_seconds(self) -> float:
        """Calculate the remaining portion of max_time_seconds using monotonic clock."""
        elapsed = time.monotonic() - self.start_time
        return max(0.0, self.limits.max_time_seconds - elapsed)

    def check_time(self) -> None:
        """Check elapsed execution time against timeout limit."""
        elapsed = time.monotonic() - self.start_time
        if elapsed > self.limits.max_time_seconds:
            raise TimeoutLimitExceededError(
                f"Time limit of {self.limits.max_time_seconds}s exceeded (elapsed: {elapsed:.2f}s)."
            )
