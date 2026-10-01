"""Execution trace container collecting and querying runtime events."""

from typing import Any

from agent.events.types import ExecutionEvent


class ExecutionTrace:
    """In-memory ordered audit trail of execution events for an agent run."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._events: list[ExecutionEvent] = []

    def emit(self, event: ExecutionEvent) -> None:
        """Append an event to the trace."""
        self._events.append(event)

    def get_events(self) -> list[ExecutionEvent]:
        """Return a copy of all recorded events in chronological order."""
        return list(self._events)

    def get_events_by_type(self, event_type: str) -> list[ExecutionEvent]:
        """Filter events by event_type string."""
        return [e for e in self._events if e.event_type == event_type]

    def total_duration_seconds(self) -> float:
        """Calculate elapsed seconds from first to last recorded event."""
        if len(self._events) < 2:
            return 0.0
        return max(0.0, self._events[-1].timestamp - self._events[0].timestamp)

    def tool_call_count(self) -> int:
        """Count how many tool calls were requested."""
        return len(self.get_events_by_type("ToolCallRequested"))

    def model_call_count(self) -> int:
        """Count how many model calls were requested."""
        return len(self.get_events_by_type("ModelCallRequested"))

    def to_dict(self) -> dict[str, Any]:
        """Serialize trace and all events to a dictionary."""
        return {
            "run_id": self.run_id,
            "event_count": len(self._events),
            "events": [e.to_dict() for e in self._events],
            "tool_calls": self.tool_call_count(),
            "model_calls": self.model_call_count(),
            "duration_seconds": self.total_duration_seconds(),
        }
