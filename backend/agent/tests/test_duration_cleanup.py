"""Focused regression tests for research duration calculation, persistence, and conversion."""

from datetime import timedelta
from unittest.mock import MagicMock, patch

from django.test import TestCase
from django.utils import timezone

from agent.models import ResearchRun
from agent.research import ResearchResult
from agent.state import AgentState, AgentStatus, StepExecutionRecord
from agent.tasks import execute_research_run


class ResearchDurationCleanupTests(TestCase):
    """Tests verifying accurate duration tracking across runtime, model, tasks, and API."""

    def test_research_result_duration_properties_and_serialization(self) -> None:
        """ResearchResult exposes duration_seconds and serializes both seconds and ms."""
        state = AgentState.create(objective="Duration test")
        state.metadata["duration_ms"] = 34790.0
        state.metadata["duration_seconds"] = 34.79

        result = ResearchResult.from_state(state)
        self.assertEqual(result.duration_ms, 34790.0)
        self.assertEqual(result.duration_seconds, 34.79)

        data = result.to_dict()
        self.assertEqual(data["duration_seconds"], 34.79)
        self.assertEqual(data["duration_ms"], 34790.0)

        # Deserialize from dictionary
        deserialized = ResearchResult.from_dict(data)
        self.assertEqual(deserialized.duration_ms, 34790.0)
        self.assertEqual(deserialized.duration_seconds, 34.79)

    def test_research_run_model_duration_properties(self) -> None:
        """ResearchRun model derives duration_seconds from duration_ms bidirectionally."""
        run = ResearchRun.objects.create(
            objective="Model duration test",
            status=ResearchRun.STATUS_COMPLETED,
            duration_ms=34790.0,
        )
        self.assertEqual(run.duration_seconds, 34.79)

        # Test setter
        run.duration_seconds = 61.44
        self.assertEqual(run.duration_ms, 61440.0)

        # Test summary and detail dictionary serialization
        summary = run.to_summary_dict()
        self.assertEqual(summary["duration_seconds"], 61.44)
        self.assertEqual(summary["duration_ms"], 61440.0)

        detail = run.to_detail_dict()
        self.assertEqual(detail["duration_seconds"], 61.44)
        self.assertEqual(detail["duration_ms"], 61440.0)

    def test_research_run_null_duration_handling(self) -> None:
        """ResearchRun handles None duration cleanly."""
        run = ResearchRun.objects.create(
            objective="Null duration test",
            status=ResearchRun.STATUS_QUEUED,
            duration_ms=None,
        )
        self.assertIsNone(run.duration_seconds)
        summary = run.to_summary_dict()
        self.assertIsNone(summary["duration_seconds"])
        self.assertIsNone(summary["duration_ms"])

        run.duration_seconds = None
        self.assertIsNone(run.duration_ms)

    @patch("agent.tasks.create_research_runtime")
    def test_task_completed_run_records_accurate_duration(
        self, mock_create_runtime: MagicMock
    ) -> None:
        """Celery task execution records accurate duration on completed runs."""
        mock_runtime = MagicMock()
        mock_result = ResearchResult(
            objective="Completed task duration test",
            final_answer="Answer",
            evidence=[],
            sources=[],
            queries=[],
            iteration_count=1,
            has_evidence=False,
            status=AgentStatus.COMPLETED,
            duration_ms=34790.0,
        )
        mock_runtime.run_research.return_value = mock_result
        mock_create_runtime.return_value = mock_runtime

        run = ResearchRun.objects.create(
            objective="Completed task duration test",
            status=ResearchRun.STATUS_QUEUED,
        )

        res = execute_research_run(str(run.id))
        self.assertEqual(res["status"], "completed")

        run.refresh_from_db()
        self.assertEqual(run.status, ResearchRun.STATUS_COMPLETED)
        self.assertEqual(run.duration_ms, 34790.0)
        self.assertEqual(run.duration_seconds, 34.79)
        self.assertIsNotNone(run.started_at)
        self.assertIsNotNone(run.completed_at)

    @patch("agent.tasks.create_research_runtime")
    def test_task_failed_run_records_accurate_duration(
        self, mock_create_runtime: MagicMock
    ) -> None:
        """Celery task execution records non-null duration and completed_at on failed runs."""
        mock_runtime = MagicMock()
        mock_runtime.run_research.side_effect = RuntimeError("Inference connection failed")
        mock_create_runtime.return_value = mock_runtime

        run = ResearchRun.objects.create(
            objective="Failed task duration test",
            status=ResearchRun.STATUS_QUEUED,
        )

        res = execute_research_run(str(run.id))
        self.assertEqual(res["status"], "failed")

        run.refresh_from_db()
        self.assertEqual(run.status, ResearchRun.STATUS_FAILED)
        self.assertIsNotNone(run.duration_ms)
        self.assertGreaterEqual(run.duration_ms, 0.0)
        self.assertIsNotNone(run.duration_seconds)
        self.assertGreaterEqual(run.duration_seconds, 0.0)
        self.assertIsNotNone(run.completed_at)

    def test_cancel_api_sets_accurate_duration(self) -> None:
        """ResearchCancelView populates duration_ms and completed_at on cancellation."""
        past_time = timezone.now() - timedelta(seconds=25)
        run = ResearchRun.objects.create(
            objective="Cancellation duration test",
            status=ResearchRun.STATUS_RUNNING,
            started_at=past_time,
        )

        response = self.client.post(f"/api/research/{run.id}/cancel/")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["status"], "cancelled")
        self.assertIsNotNone(data["duration_seconds"])
        self.assertGreaterEqual(data["duration_seconds"], 24.0)

        run.refresh_from_db()
        self.assertEqual(run.status, ResearchRun.STATUS_CANCELLED)
        self.assertIsNotNone(run.completed_at)
        self.assertGreaterEqual(run.duration_seconds, 24.0)
