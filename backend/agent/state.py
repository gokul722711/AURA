"""State representations and lifecycle management for AURA Agent Runtime."""

import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from agent.exceptions import InvalidStateTransitionError


class AgentStatus(str, Enum):
    """Execution status of an agent run."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class StepExecutionRecord:
    """Record of an executed plan step."""

    step_id: str
    action_type: str
    status: str
    output: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action_type": self.action_type,
            "status": self.status,
            "output": self.output,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StepExecutionRecord":
        return cls(
            step_id=data["step_id"],
            action_type=data["action_type"],
            status=data["status"],
            output=data.get("output"),
            error=data.get("error"),
            duration_ms=data.get("duration_ms", 0.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AgentState:
    """Serializable runtime state of an agent execution."""

    run_id: str
    objective: str
    status: AgentStatus = AgentStatus.PENDING
    iteration: int = 0
    messages: list[dict[str, str]] = field(default_factory=list)
    plan: Any = None
    current_step_index: int = 0
    step_history: list[StepExecutionRecord] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    final_output: str | None = None
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, objective: str, metadata: dict[str, Any] | None = None) -> "AgentState":
        """Factory method to create a new initial agent state."""
        return cls(
            run_id=str(uuid.uuid4()),
            objective=objective,
            status=AgentStatus.PENDING,
            metadata=metadata or {},
        )

    def is_terminal(self) -> bool:
        """Check if current status is terminal."""
        return self.status in (
            AgentStatus.COMPLETED,
            AgentStatus.FAILED,
            AgentStatus.CANCELLED,
        )

    def transition_to(self, new_status: AgentStatus) -> None:
        """Safely transition state to a new status with validation."""
        valid_transitions: dict[AgentStatus, set[AgentStatus]] = {
            AgentStatus.PENDING: {AgentStatus.RUNNING, AgentStatus.CANCELLED},
            AgentStatus.RUNNING: {
                AgentStatus.WAITING,
                AgentStatus.COMPLETED,
                AgentStatus.FAILED,
                AgentStatus.CANCELLED,
            },
            AgentStatus.WAITING: {
                AgentStatus.RUNNING,
                AgentStatus.FAILED,
                AgentStatus.CANCELLED,
            },
            AgentStatus.COMPLETED: set(),
            AgentStatus.FAILED: set(),
            AgentStatus.CANCELLED: set(),
        }

        if new_status not in valid_transitions.get(self.status, set()):
            raise InvalidStateTransitionError(
                f"Cannot transition agent state from '{self.status.value}' to '{new_status.value}'."
            )
        self.status = new_status

    def record_step(self, step_record: StepExecutionRecord) -> None:
        """Append an executed step record to history."""
        self.step_history.append(step_record)

    def record_tool_result(self, result_dict: dict[str, Any]) -> None:
        """Append a tool result dictionary."""
        self.tool_results.append(result_dict)

    def add_error(self, error_message: str) -> None:
        """Record an error string."""
        self.errors.append(error_message)

    def to_dict(self) -> dict[str, Any]:
        """Convert state to a clean JSON-serializable dictionary."""
        plan_dict = None
        if self.plan is not None:
            if hasattr(self.plan, "to_dict"):
                plan_dict = self.plan.to_dict()
            elif isinstance(self.plan, dict):
                plan_dict = self.plan

        return {
            "run_id": self.run_id,
            "objective": self.objective,
            "status": self.status.value,
            "iteration": self.iteration,
            "messages": list(self.messages),
            "plan": plan_dict,
            "current_step_index": self.current_step_index,
            "step_history": [s.to_dict() for s in self.step_history],
            "tool_results": list(self.tool_results),
            "final_output": self.final_output,
            "errors": list(self.errors),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentState":
        """Reconstruct AgentState from a dictionary."""
        return cls(
            run_id=data["run_id"],
            objective=data["objective"],
            status=AgentStatus(data["status"]),
            iteration=data.get("iteration", 0),
            messages=data.get("messages", []),
            plan=data.get("plan"),
            current_step_index=data.get("current_step_index", 0),
            step_history=[
                StepExecutionRecord.from_dict(s)
                for s in data.get("step_history", [])
            ],
            tool_results=data.get("tool_results", []),
            final_output=data.get("final_output"),
            errors=data.get("errors", []),
            metadata=data.get("metadata", {}),
        )
