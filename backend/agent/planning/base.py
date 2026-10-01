"""Base definitions for planning, plans, and steps in AURA Agent Runtime."""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agent.exceptions import InvalidPlanError
from agent.state import AgentState


class ActionType(str, Enum):
    """Supported step action types in M4 runtime."""

    MODEL = "model"
    TOOL = "tool"
    FINISH = "finish"


@dataclass(frozen=True)
class AgentStep:
    """A single declarative step within an agent plan."""

    step_id: str
    action_type: ActionType
    description: str
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.step_id:
            object.__setattr__(self, "step_id", str(uuid.uuid4()))
        if not isinstance(self.action_type, ActionType):
            try:
                object.__setattr__(self, "action_type", ActionType(self.action_type))
            except ValueError:
                raise ValueError(f"Invalid action_type: {self.action_type}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action_type": self.action_type.value,
            "description": self.description,
            "payload": dict(self.payload),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentStep":
        return cls(
            step_id=data["step_id"],
            action_type=ActionType(data["action_type"]),
            description=data.get("description", ""),
            payload=data.get("payload", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class Plan:
    """Ordered sequence of steps to accomplish an objective."""

    plan_id: str
    objective: str
    steps: tuple[AgentStep, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.plan_id:
            object.__setattr__(self, "plan_id", str(uuid.uuid4()))
        if isinstance(self.steps, list):
            object.__setattr__(self, "steps", tuple(self.steps))
        if len(self.steps) == 0:
            raise InvalidPlanError("A Plan must contain at least one step.")

    def __len__(self) -> int:
        return len(self.steps)

    def get_step(self, index: int) -> AgentStep | None:
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "objective": self.objective,
            "steps": [s.to_dict() for s in self.steps],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Plan":
        return cls(
            plan_id=data["plan_id"],
            objective=data["objective"],
            steps=tuple(AgentStep.from_dict(s) for s in data.get("steps", [])),
            metadata=data.get("metadata", {}),
        )


class Planner(ABC):
    """Abstract interface for planning engines."""

    supports_replanning: bool = False

    @abstractmethod
    def plan(self, objective: str, state: AgentState) -> Plan:
        """Produce an actionable Plan for the given objective and state."""
        pass
