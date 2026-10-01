"""Exceptions for AURA Agent Runtime."""


class AgentError(Exception):
    """Base exception for all agent runtime errors."""
    pass


class AgentRuntimeError(AgentError):
    """Raised when runtime orchestration encounters an unexpected error."""
    pass


class InvalidStateTransitionError(AgentRuntimeError):
    """Raised when an illegal state transition is attempted."""
    pass


class UnhandledActionTypeError(AgentRuntimeError):
    """Raised when a step specifies an unrecognized action type."""
    pass


class ExecutionCancelledError(AgentRuntimeError):
    """Raised when execution is cancelled via runtime cancellation flag."""
    pass


class PlanningError(AgentError):
    """Base exception for planning errors."""
    pass


class InvalidPlanError(PlanningError):
    """Raised when a generated plan is structurally invalid or empty."""
    pass


class StepExecutionError(AgentError):
    """Base exception for step execution failures."""
    pass


class ModelExecutionError(StepExecutionError):
    """Raised when model generation fails during step execution."""
    pass


class LimitExceededError(AgentError):
    """Base exception for execution limit breaches."""
    pass


class IterationLimitExceededError(LimitExceededError):
    """Raised when maximum execution iterations are exceeded."""
    pass


class ToolCallLimitExceededError(LimitExceededError):
    """Raised when maximum cumulative tool calls are exceeded."""
    pass


class TimeoutLimitExceededError(LimitExceededError):
    """Raised when maximum execution time is exceeded."""
    pass


class ToolError(AgentError):
    """Base exception for tool-related errors."""
    pass


class DuplicateToolError(ToolError):
    """Raised when registering a tool with an already-registered name."""
    pass


class ToolNotFoundError(ToolError):
    """Raised when requested tool is not found in the registry."""
    pass


class ToolInputValidationError(ToolError):
    """Raised when tool inputs fail validation."""
    pass


class ToolExecutionError(ToolError):
    """Raised when a tool encounters an error during execution."""
    pass


class ToolPolicyViolationError(ToolError):
    """Raised when a tool execution is denied by the security policy."""
    pass
