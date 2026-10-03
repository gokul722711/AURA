"""Tests for M13 mode-specific execution dispatch in AURA."""

from unittest.mock import MagicMock, patch
from django.test import TestCase

from agent.models import ResearchRun
from agent.results import ResearchEvidence, ResearchResult
from agent.state import AgentStatus
from agent.tasks import execute_research_run


class M13DispatchTests(TestCase):
    """Tests verifying mode-specific routing between Fast Path and Agent Path."""

    def test_knowledge_base_mode_dispatches_deterministic_kb_pipeline(self) -> None:
        """When mode is knowledge_base, execute_research_run invokes DeterministicKBPipeline."""
        run = ResearchRun.objects.create(
            objective="What is pgvector?",
            mode=ResearchRun.MODE_KNOWLEDGE_BASE,
        )

        mock_result = ResearchResult(
            objective="What is pgvector?",
            final_answer="pgvector is an open-source vector extension [Doc, Chunk: c1].",
            evidence=[
                ResearchEvidence(
                    chunk_id="c1",
                    document_title="Doc",
                    document_source="doc.md",
                    content="pgvector is an open-source vector extension.",
                )
            ],
            sources=[{"document_title": "Doc"}],
            queries=["What is pgvector?"],
            iteration_count=1,
            has_evidence=True,
            status=AgentStatus.COMPLETED,
            duration_ms=150.0,
            metadata={"pipeline": "DeterministicKBPipeline"},
        )

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = mock_result

        with patch("agent.tasks.create_deterministic_kb_pipeline", return_value=mock_pipeline) as mock_create_kb, patch(
            "agent.tasks.create_research_runtime"
        ) as mock_create_agent:
            res = execute_research_run(str(run.id))
            self.assertEqual(res["status"], "completed")
            mock_create_kb.assert_called_once()
            mock_create_agent.assert_not_called()

        run.refresh_from_db()
        self.assertEqual(run.status, ResearchRun.STATUS_COMPLETED)
        self.assertEqual(run.result["metadata"]["pipeline"], "DeterministicKBPipeline")

    def test_web_mode_dispatches_agent_runtime(self) -> None:
        """When mode is web, execute_research_run invokes existing create_research_runtime."""
        run = ResearchRun.objects.create(
            objective="Latest research papers on transformers",
            mode=ResearchRun.MODE_WEB,
        )

        mock_result = ResearchResult(
            objective="Latest research papers",
            final_answer="Recent papers discuss transformers.",
            evidence=[],
            sources=[],
            queries=["transformers"],
            iteration_count=2,
            has_evidence=False,
            status=AgentStatus.COMPLETED,
            duration_ms=500.0,
        )

        mock_runtime = MagicMock()
        mock_runtime.run_research.return_value = mock_result

        with patch("agent.tasks.create_deterministic_kb_pipeline") as mock_create_kb, patch(
            "agent.tasks.create_research_runtime", return_value=mock_runtime
        ) as mock_create_agent:
            res = execute_research_run(str(run.id))
            self.assertEqual(res["status"], "completed")
            mock_create_agent.assert_called_once_with(mode=ResearchRun.MODE_WEB)
            mock_create_kb.assert_not_called()

    def test_web_knowledge_base_mode_dispatches_agent_runtime(self) -> None:
        """When mode is web_knowledge_base, invokes create_research_runtime."""
        run = ResearchRun.objects.create(
            objective="Hybrid research query",
            mode=ResearchRun.MODE_WEB_KNOWLEDGE_BASE,
        )

        mock_result = ResearchResult(
            objective="Hybrid query",
            final_answer="Answer.",
            evidence=[],
            sources=[],
            queries=[],
            iteration_count=1,
            has_evidence=False,
            status=AgentStatus.COMPLETED,
            duration_ms=100.0,
        )

        mock_runtime = MagicMock()
        mock_runtime.run_research.return_value = mock_result

        with patch("agent.tasks.create_deterministic_kb_pipeline") as mock_create_kb, patch(
            "agent.tasks.create_research_runtime", return_value=mock_runtime
        ) as mock_create_agent:
            res = execute_research_run(str(run.id))
            self.assertEqual(res["status"], "completed")
            mock_create_agent.assert_called_once_with(mode=ResearchRun.MODE_WEB_KNOWLEDGE_BASE)
            mock_create_kb.assert_not_called()

    def test_model_knowledge_mode_dispatches_agent_runtime(self) -> None:
        """When mode is model_knowledge, invokes create_research_runtime."""
        run = ResearchRun.objects.create(
            objective="General question",
            mode=ResearchRun.MODE_MODEL_KNOWLEDGE,
        )

        mock_result = ResearchResult(
            objective="General question",
            final_answer="Answer.",
            evidence=[],
            sources=[],
            queries=[],
            iteration_count=1,
            has_evidence=False,
            status=AgentStatus.COMPLETED,
            duration_ms=100.0,
        )

        mock_runtime = MagicMock()
        mock_runtime.run_research.return_value = mock_result

        with patch("agent.tasks.create_deterministic_kb_pipeline") as mock_create_kb, patch(
            "agent.tasks.create_research_runtime", return_value=mock_runtime
        ) as mock_create_agent:
            res = execute_research_run(str(run.id))
            self.assertEqual(res["status"], "completed")
            mock_create_agent.assert_called_once_with(mode=ResearchRun.MODE_MODEL_KNOWLEDGE)
            mock_create_kb.assert_not_called()
