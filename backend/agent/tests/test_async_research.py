"""Automated tests for M9 Async Research Run model, Celery task, and REST APIs."""

import json
import uuid
from unittest.mock import MagicMock, patch

from django.test import TestCase, override_settings
from django.utils import timezone
from rest_framework.test import APIClient

from agent.models import ResearchRun
from agent.research import ResearchEvidence, ResearchResult
from agent.state import AgentStatus
from agent.tasks import execute_research_run
from agent.views import ResearchView


class ResearchRunModelTests(TestCase):
    """Tests for the ResearchRun model and serialization."""

    def test_create_research_run_defaults(self) -> None:
        """Creating a run initializes status to queued and sets timestamps."""
        run = ResearchRun.objects.create(objective="What is AURA?")
        self.assertEqual(run.status, ResearchRun.STATUS_QUEUED)
        self.assertEqual(run.objective, "What is AURA?")
        self.assertIsNotNone(run.created_at)
        self.assertIsNone(run.started_at)
        self.assertIsNone(run.completed_at)
        self.assertEqual(run.result, {})
        self.assertEqual(run.error_message, "")

    def test_status_transitions(self) -> None:
        """Run status transitions through lifecycle states."""
        run = ResearchRun.objects.create(objective="Status test")
        self.assertEqual(run.status, ResearchRun.STATUS_QUEUED)

        run.status = ResearchRun.STATUS_RUNNING
        run.started_at = timezone.now()
        run.save()
        self.assertEqual(ResearchRun.objects.get(id=run.id).status, ResearchRun.STATUS_RUNNING)

        run.status = ResearchRun.STATUS_COMPLETED
        run.completed_at = timezone.now()
        run.save()
        self.assertEqual(ResearchRun.objects.get(id=run.id).status, ResearchRun.STATUS_COMPLETED)

    def test_to_summary_dict(self) -> None:
        """to_summary_dict() returns lightweight metadata without evidence content."""
        run = ResearchRun.objects.create(
            objective="Summary test",
            status=ResearchRun.STATUS_COMPLETED,
            duration_ms=450.0,
            result={
                "is_grounded": True,
                "has_evidence": True,
                "citations": ["[Doc, Chunk: 1]"],
                "final_answer": "Very long text that should not appear in history summary",
                "evidence": [{"content": "Heavy evidence content"}],
            },
        )
        summary = run.to_summary_dict()
        self.assertEqual(summary["run_id"], str(run.id))
        self.assertEqual(summary["objective"], "Summary test")
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["duration_ms"], 450.0)
        self.assertTrue(summary["is_grounded"])
        self.assertTrue(summary["has_evidence"])
        self.assertEqual(summary["citation_count"], 1)
        self.assertNotIn("final_answer", summary)
        self.assertNotIn("evidence", summary)

    def test_to_detail_dict_completed(self) -> None:
        """to_detail_dict() exposes full M6 result fields when completed."""
        result_payload = {
            "final_answer": "Synthesized grounded answer.",
            "evidence": [{"chunk_id": "c1", "content": "Text"}],
            "sources": [{"document_title": "Doc 1"}],
            "citations": ["[Doc 1, Chunk: c1]"],
            "queries": ["search query"],
            "iteration_count": 2,
            "is_grounded": True,
            "has_evidence": True,
            "errors": [],
            "metadata": {"test": True},
        }
        run = ResearchRun.objects.create(
            objective="Detail test",
            status=ResearchRun.STATUS_COMPLETED,
            duration_ms=320.0,
            result=result_payload,
        )
        detail = run.to_detail_dict()
        self.assertEqual(detail["run_id"], str(run.id))
        self.assertEqual(detail["status"], "completed")
        self.assertEqual(detail["final_answer"], "Synthesized grounded answer.")
        self.assertEqual(len(detail["evidence"]), 1)
        self.assertEqual(detail["citations"], ["[Doc 1, Chunk: c1]"])
        self.assertEqual(detail["queries"], ["search query"])
        self.assertEqual(detail["iteration_count"], 2)


class ResearchCeleryTaskTests(TestCase):
    """Tests for the execute_research_run Celery task."""

    def test_execute_research_run_success(self) -> None:
        """Task loads run, sets status to running, calls runtime, and saves completed result."""
        run = ResearchRun.objects.create(objective="Task test")

        mock_result = ResearchResult(
            objective="Task test",
            final_answer="The answer grounded in evidence.",
            evidence=[
                ResearchEvidence(
                    chunk_id="c1",
                    document_title="Title",
                    document_source="source.md",
                    content="Content",
                )
            ],
            sources=[{"document_title": "Title"}],
            queries=["query 1"],
            iteration_count=1,
            has_evidence=True,
            status=AgentStatus.COMPLETED,
            duration_ms=250.0,
        )

        mock_runtime = MagicMock()
        mock_runtime.run_research.return_value = mock_result

        with patch("agent.tasks.create_research_runtime", return_value=mock_runtime):
            task_res = execute_research_run(str(run.id))
            self.assertEqual(task_res["status"], "completed")

        run.refresh_from_db()
        self.assertEqual(run.status, ResearchRun.STATUS_COMPLETED)
        self.assertIsNotNone(run.started_at)
        self.assertIsNotNone(run.completed_at)
        self.assertEqual(run.duration_ms, 250.0)
        self.assertEqual(run.result["final_answer"], "The answer grounded in evidence.")
        self.assertTrue(run.result["is_grounded"])

    def test_execute_nonexistent_run_handled_gracefully(self) -> None:
        """Nonexistent run_id does not raise uncaught exception."""
        fake_id = str(uuid.uuid4())
        res = execute_research_run(fake_id)
        self.assertEqual(res["status"], "error")

    def test_execute_already_cancelled_run_does_not_execute(self) -> None:
        """If run was cancelled before task starts, task exits without executing runtime."""
        run = ResearchRun.objects.create(
            objective="Cancelled early",
            status=ResearchRun.STATUS_CANCELLED,
            error_message="Cancelled by user.",
        )
        mock_runtime = MagicMock()
        with patch("agent.tasks.create_research_runtime", return_value=mock_runtime):
            res = execute_research_run(str(run.id))
            self.assertEqual(res["status"], "cancelled")
            mock_runtime.run_research.assert_not_called()

    def test_execute_run_runtime_exception_transitions_to_failed(self) -> None:
        """Runtime exception marks run as failed with sanitized error."""
        run = ResearchRun.objects.create(objective="Failing run")

        mock_runtime = MagicMock()
        mock_runtime.run_research.side_effect = RuntimeError("Inference connection failed")

        with patch("agent.tasks.create_research_runtime", return_value=mock_runtime):
            res = execute_research_run(str(run.id))
            self.assertEqual(res["status"], "failed")

        run.refresh_from_db()
        self.assertEqual(run.status, ResearchRun.STATUS_FAILED)
        self.assertIn("Inference connection failed", run.error_message)

    def test_cancellation_during_runtime_execution_is_not_overwritten_to_completed(self) -> None:
        """If run is cancelled while runtime is executing, worker does not overwrite status to completed."""
        run = ResearchRun.objects.create(objective="Concurrent cancel test")

        mock_result = ResearchResult(
            objective="Concurrent cancel test",
            final_answer="Answer synthesized before cancel was picked up.",
            evidence=[],
            sources=[],
            queries=[],
            iteration_count=1,
            has_evidence=False,
            status=AgentStatus.COMPLETED,
            duration_ms=100.0,
        )

        mock_runtime = MagicMock()

        def side_effect_run_research(obj: str) -> ResearchResult:
            # Simulate user clicking cancel concurrently while runtime was running
            ResearchRun.objects.filter(id=run.id).update(
                status=ResearchRun.STATUS_CANCELLED,
                error_message="Cancelled by user concurrently.",
                completed_at=timezone.now(),
            )
            return mock_result

        mock_runtime.run_research.side_effect = side_effect_run_research

        with patch("agent.tasks.create_research_runtime", return_value=mock_runtime):
            res = execute_research_run(str(run.id))
            self.assertEqual(res["status"], "cancelled")

        run.refresh_from_db()
        self.assertEqual(run.status, ResearchRun.STATUS_CANCELLED)
        self.assertEqual(run.error_message, "Cancelled by user concurrently.")

    def test_cancellation_during_runtime_failure_is_not_overwritten_to_failed(self) -> None:
        """If run is cancelled while runtime is executing and runtime fails, status remains cancelled."""
        run = ResearchRun.objects.create(objective="Cancel before exception test")

        mock_runtime = MagicMock()

        def side_effect_fail(obj: str) -> None:
            ResearchRun.objects.filter(id=run.id).update(
                status=ResearchRun.STATUS_CANCELLED,
                error_message="Cancelled by user concurrently.",
                completed_at=timezone.now(),
            )
            raise RuntimeError("Underlying network error during cancel")

        mock_runtime.run_research.side_effect = side_effect_fail

        with patch("agent.tasks.create_research_runtime", return_value=mock_runtime):
            res = execute_research_run(str(run.id))
            self.assertEqual(res["status"], "cancelled")

        run.refresh_from_db()
        self.assertEqual(run.status, ResearchRun.STATUS_CANCELLED)
        self.assertEqual(run.error_message, "Cancelled by user concurrently.")


class AsyncResearchAPITests(TestCase):
    """Tests for async research endpoints and cancellation."""

    def setUp(self) -> None:
        self.client = APIClient()

    def test_post_research_returns_202_and_queues_task(self) -> None:
        """POST /api/research/ returns 202 quickly with run_id and queued status."""
        with patch.object(ResearchView, "dispatch_task") as mock_dispatch:
            res = self.client.post(
                "/api/research/",
                {"objective": "Explain Model Gateway"},
                format="json",
            )
            self.assertEqual(res.status_code, 202)
            data = res.json()
            self.assertIn("run_id", data)
            self.assertEqual(data["status"], "queued")
            self.assertEqual(data["objective"], "Explain Model Gateway")
            mock_dispatch.assert_called_once_with(data["run_id"])

            run = ResearchRun.objects.get(id=data["run_id"])
            self.assertEqual(run.status, ResearchRun.STATUS_QUEUED)

    def test_post_research_broker_failure_returns_503(self) -> None:
        """When Celery broker is unreachable, returns 503 without obscure traceback."""
        with patch.object(
            ResearchView,
            "dispatch_task",
            side_effect=ConnectionError("Error 111 connecting to localhost:6379"),
        ):
            res = self.client.post(
                "/api/research/",
                {"objective": "Test broker failure"},
                format="json",
            )
            self.assertEqual(res.status_code, 503)
            data = res.json()
            self.assertIn("error", data)
            self.assertEqual(data["status"], "failed")

    def test_get_research_detail_running(self) -> None:
        """GET /api/research/<id>/ returns running status for active runs."""
        run = ResearchRun.objects.create(
            objective="Running run",
            status=ResearchRun.STATUS_RUNNING,
            started_at=timezone.now(),
        )
        res = self.client.get(f"/api/research/{run.id}/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["run_id"], str(run.id))
        self.assertEqual(data["status"], "running")
        self.assertIsNotNone(data["started_at"])

    def test_get_research_detail_not_found(self) -> None:
        """GET /api/research/<id>/ returns 404 for unknown or invalid IDs."""
        res1 = self.client.get(f"/api/research/{uuid.uuid4()}/")
        self.assertEqual(res1.status_code, 404)

        res2 = self.client.get("/api/research/not-a-valid-uuid/")
        self.assertEqual(res2.status_code, 404)

    def test_get_research_history_runs(self) -> None:
        """GET /api/research/runs/ returns history in newest-first order."""
        run1 = ResearchRun.objects.create(objective="Run 1")
        run2 = ResearchRun.objects.create(objective="Run 2")

        res = self.client.get("/api/research/runs/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertGreaterEqual(len(data), 2)
        # Newest run first
        self.assertEqual(data[0]["run_id"], str(run2.id))
        self.assertEqual(data[1]["run_id"], str(run1.id))

    def test_cancel_research_run(self) -> None:
        """POST /api/research/<id>/cancel/ marks queued or running run as cancelled."""
        run = ResearchRun.objects.create(
            objective="To cancel",
            status=ResearchRun.STATUS_QUEUED,
        )
        res = self.client.post(f"/api/research/{run.id}/cancel/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "cancelled")

        run.refresh_from_db()
        self.assertEqual(run.status, ResearchRun.STATUS_CANCELLED)
        self.assertIn("Cancelled", run.error_message)

    def test_cancel_already_completed_run_does_not_mutate_status(self) -> None:
        """POST /api/research/<id>/cancel/ on completed run returns 200 without changing status."""
        run = ResearchRun.objects.create(
            objective="Already completed",
            status=ResearchRun.STATUS_COMPLETED,
            result={"final_answer": "Complete answer."},
        )
        res = self.client.post(f"/api/research/{run.id}/cancel/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "completed")

        run.refresh_from_db()
        self.assertEqual(run.status, ResearchRun.STATUS_COMPLETED)

    def test_cancel_nonexistent_run_returns_404(self) -> None:
        """POST /api/research/<id>/cancel/ returns 404 for unknown run."""
        res = self.client.post(f"/api/research/{uuid.uuid4()}/cancel/")
        self.assertEqual(res.status_code, 404)
