"""Tests for ExecutionEvent types and ExecutionTrace."""

from django.test import SimpleTestCase

from agent.events.trace import ExecutionTrace
from agent.events.types import (
    agent_run_cancelled,
    agent_run_completed,
    agent_run_failed,
    agent_run_started,
    model_call_completed,
    model_call_requested,
    plan_created,
    step_completed,
    step_started,
    tool_call_completed,
    tool_call_requested,
)


class ExecutionEventsTests(SimpleTestCase):
    """Tests for structured events creation and serialization."""

    def test_event_creation(self):
        ev = agent_run_started(run_id="run-1", objective="Perform audit")
        self.assertEqual(ev.run_id, "run-1")
        self.assertEqual(ev.event_type, "AgentRunStarted")
        self.assertEqual(ev.payload["objective"], "Perform audit")
        self.assertGreater(ev.timestamp, 0.0)

        d = ev.to_dict()
        self.assertEqual(d["event_type"], "AgentRunStarted")
        self.assertEqual(d["run_id"], "run-1")

    def test_model_call_events_optional_usage(self):
        # Test model call event when usage is provided
        ev1 = model_call_completed(
            run_id="r1",
            step_id="s1",
            text="Hello world",
            usage={"total_tokens": 25},
        )
        self.assertEqual(ev1.payload["usage"]["total_tokens"], 25)

        # Test model call event when usage is None (not mandatory)
        ev2 = model_call_completed(
            run_id="r1",
            step_id="s2",
            text="No usage text",
            usage=None,
        )
        self.assertEqual(ev2.payload["usage"], {})


class ExecutionTraceTests(SimpleTestCase):
    """Tests for ExecutionTrace audit trail."""

    def setUp(self):
        self.trace = ExecutionTrace(run_id="test-run")

    def test_emit_and_retrieve_events(self):
        self.trace.emit(agent_run_started("test-run", "Start test"))
        self.trace.emit(step_started("test-run", "step-1", "tool", "Run tool"))
        self.trace.emit(tool_call_requested("test-run", "step-1", "calc", {"exp": "1+1"}))
        self.trace.emit(tool_call_completed("test-run", "step-1", "calc", 2, False))
        self.trace.emit(step_completed("test-run", "step-1", "completed", 5.0))
        self.trace.emit(agent_run_completed("test-run", "Done", 1))

        events = self.trace.get_events()
        self.assertEqual(len(events), 6)
        self.assertEqual(self.trace.tool_call_count(), 1)
        self.assertEqual(self.trace.model_call_count(), 0)

        started_events = self.trace.get_events_by_type("StepStarted")
        self.assertEqual(len(started_events), 1)

    def test_trace_to_dict(self):
        self.trace.emit(agent_run_started("test-run", "Trace to dict"))
        self.trace.emit(agent_run_failed("test-run", "Failure reason", 1))

        d = self.trace.to_dict()
        self.assertEqual(d["run_id"], "test-run")
        self.assertEqual(d["event_count"], 2)
        self.assertEqual(len(d["events"]), 2)
