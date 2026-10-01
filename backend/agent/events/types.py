"""Structured execution events for AURA Agent Runtime observability."""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExecutionEvent:
    """Base class for all runtime execution events."""

    event_id: str
    run_id: str
    timestamp: float
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "payload": dict(self.payload),
        }


def _make_event(
    event_type: str,
    run_id: str,
    payload: dict[str, Any] | None = None,
) -> ExecutionEvent:
    return ExecutionEvent(
        event_id=str(uuid.uuid4()),
        run_id=run_id,
        timestamp=time.time(),
        event_type=event_type,
        payload=payload or {},
    )


def agent_run_started(run_id: str, objective: str) -> ExecutionEvent:
    return _make_event("AgentRunStarted", run_id, {"objective": objective})


def plan_created(run_id: str, plan_id: str, step_count: int, steps: list[dict[str, Any]]) -> ExecutionEvent:
    return _make_event("PlanCreated", run_id, {"plan_id": plan_id, "step_count": step_count, "steps": steps})


def step_started(run_id: str, step_id: str, action_type: str, description: str) -> ExecutionEvent:
    return _make_event("StepStarted", run_id, {"step_id": step_id, "action_type": action_type, "description": description})


def model_call_requested(run_id: str, step_id: str, prompt: str) -> ExecutionEvent:
    return _make_event("ModelCallRequested", run_id, {"step_id": step_id, "prompt": prompt})


def model_call_completed(
    run_id: str,
    step_id: str,
    text: str,
    usage: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ExecutionEvent:
    return _make_event(
        "ModelCallCompleted",
        run_id,
        {
            "step_id": step_id,
            "text": text,
            "usage": usage or {},
            "metadata": metadata or {},
        },
    )


def tool_call_requested(run_id: str, step_id: str, tool_name: str, tool_input: dict[str, Any]) -> ExecutionEvent:
    return _make_event("ToolCallRequested", run_id, {"step_id": step_id, "tool_name": tool_name, "tool_input": tool_input})


def tool_call_completed(
    run_id: str,
    step_id: str,
    tool_name: str,
    output: Any,
    is_error: bool,
    error_message: str | None = None,
) -> ExecutionEvent:
    return _make_event(
        "ToolCallCompleted",
        run_id,
        {
            "step_id": step_id,
            "tool_name": tool_name,
            "output": output,
            "is_error": is_error,
            "error_message": error_message,
        },
    )


def step_completed(run_id: str, step_id: str, status: str, duration_ms: float = 0.0) -> ExecutionEvent:
    return _make_event("StepCompleted", run_id, {"step_id": step_id, "status": status, "duration_ms": duration_ms})


def agent_run_completed(run_id: str, final_output: str | None, iterations: int) -> ExecutionEvent:
    return _make_event("AgentRunCompleted", run_id, {"final_output": final_output, "iterations": iterations})


def agent_run_failed(run_id: str, error: str, iterations: int) -> ExecutionEvent:
    return _make_event("AgentRunFailed", run_id, {"error": error, "iterations": iterations})


def agent_run_cancelled(run_id: str, reason: str, iterations: int) -> ExecutionEvent:
    return _make_event("AgentRunCancelled", run_id, {"reason": reason, "iterations": iterations})
