"""Automated tests for Model Profiles integration with Research and Architectural boundaries (M15)."""

import ast
import json
import os
from unittest.mock import MagicMock, patch
from django.test import TestCase
from rest_framework.test import APIClient

from agent.models import ResearchRun
from agent.tasks import execute_research_run
from agent.views import ResearchView
from gateway.models import ModelProfile
from rag.models import Document, DocumentChunk


class ResearchModelProfileIntegrationTests(TestCase):
    """Verify that research pipelines seamlessly use user-configured model profiles."""

    def setUp(self) -> None:
        ModelProfile.objects.all().delete()
        ResearchRun.objects.all().delete()
        self.client = APIClient()

        # Create two distinct profiles
        self.profile_nvidia = ModelProfile.objects.create(
            name="Mocked NVIDIA",
            provider="mock",
            model="nvidia/nemotron-test",
            api_key="nvapi-super-secret-key-9999",
            is_active=True,
        )
        self.profile_ollama = ModelProfile.objects.create(
            name="Mocked Ollama",
            provider="mock",
            model="ollama/llama3.1-test",
            is_active=False,
        )

        # Create KB document
        self.doc = Document.objects.create(
            title="AURA Architecture",
            content="AURA uses ModelGateway for model-agnostic research.",
            source="docs/arch.md",
            status="ready",
        )
        self.chunk = DocumentChunk.objects.create(
            document=self.doc,
            content="AURA uses ModelGateway for model-agnostic research.",
            chunk_index=0,
            start_offset=0,
            end_offset=50,
            embedding=[0.05] * 384,
        )

    def test_research_run_uses_active_profile(self) -> None:
        """Celery task execute_research_run uses active profile when none explicitly specified."""
        run = ResearchRun.objects.create(
            objective="What is AURA?",
            mode=ResearchRun.MODE_KNOWLEDGE_BASE,
        )

        res = execute_research_run(str(run.id))
        self.assertEqual(res["status"], "completed")

        run.refresh_from_db()
        self.assertEqual(run.model_profile_id, self.profile_nvidia.id)
        self.assertEqual(run.model_name, "nvidia/nemotron-test")
        self.assertEqual(run.provider_name, "mock")

        # Metadata in result
        result_meta = run.result.get("metadata", {})
        self.assertEqual(result_meta.get("model"), "nvidia/nemotron-test")
        self.assertEqual(result_meta.get("model_profile_name"), "Mocked NVIDIA")

        # Verify API key is NOT in result or metadata
        self.assertNotIn("nvapi-super-secret-key-9999", str(run.result))

    def test_switching_active_profile_switches_model_used(self) -> None:
        """Switching the active profile causes subsequent runs to use the new profile."""
        # Switch active profile to Ollama
        self.profile_ollama.is_active = True
        self.profile_ollama.save()

        run = ResearchRun.objects.create(
            objective="What is AURA?",
            mode=ResearchRun.MODE_KNOWLEDGE_BASE,
        )

        res = execute_research_run(str(run.id))
        self.assertEqual(res["status"], "completed")

        run.refresh_from_db()
        self.assertEqual(run.model_profile_id, self.profile_ollama.id)
        self.assertEqual(run.model_name, "ollama/llama3.1-test")
        self.assertEqual(run.result["metadata"]["model"], "ollama/llama3.1-test")
        self.assertEqual(run.result["metadata"]["model_profile_name"], "Mocked Ollama")

    def test_post_research_specifying_explicit_profile(self) -> None:
        """POST /api/research/ accepts model_profile_id to override active profile."""
        with patch.object(ResearchView, "dispatch_task"):
            resp = self.client.post(
                "/api/research/",
                data={
                    "objective": "Targeted profile test",
                    "mode": "knowledge_base",
                    "model_profile_id": str(self.profile_ollama.id),
                },
                format="json",
            )
            self.assertEqual(resp.status_code, 202)
            run_id = resp.json()["run_id"]
            run = ResearchRun.objects.get(id=run_id)
            self.assertEqual(run.model_profile_id, self.profile_ollama.id)
            self.assertEqual(run.model_name, "ollama/llama3.1-test")

    def test_post_research_rejects_nonexistent_profile(self) -> None:
        """POST /api/research/ rejects non-existent model_profile_id with 400."""
        import uuid

        fake_id = str(uuid.uuid4())
        resp = self.client.post(
            "/api/research/",
            data={
                "objective": "Fake profile test",
                "model_profile_id": fake_id,
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("not found", resp.json()["error"])

    def test_no_model_configured_behavior(self) -> None:
        """When zero profiles are configured, research requests fail gracefully."""
        ModelProfile.objects.all().delete()

        # 1. API request fails with NO_MODEL_CONFIGURED
        resp = self.client.post(
            "/api/research/",
            data={"objective": "No model test"},
            format="json",
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertEqual(data.get("code"), "NO_MODEL_CONFIGURED")
        self.assertIn("No research model configured", data.get("error", ""))

        # 2. Ingestion & KB operations remain 100% operational
        doc_resp = self.client.post(
            "/api/documents/",
            data={"title": "Doc during no-model", "content": "Sample content without LLM."},
            format="json",
        )
        self.assertEqual(doc_resp.status_code, 201)

        # 3. Direct task execution without profile fails cleanly
        run = ResearchRun.objects.create(
            objective="Orphaned task test",
            mode=ResearchRun.MODE_KNOWLEDGE_BASE,
        )
        task_res = execute_research_run(str(run.id))
        self.assertEqual(task_res["status"], "failed")
        self.assertEqual(task_res["code"], "NO_MODEL_CONFIGURED")


class ArchitecturalBoundaryTests(TestCase):
    """Verify that research and retrieval layers have ZERO vendor-specific provider leakage."""

    def test_no_provider_branching_in_research_layers(self) -> None:
        """Ensure no 'if provider == ...' or vendor names are hardcoded in research pipelines."""
        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        files_to_check = [
            os.path.join(backend_dir, "rag", "retrieval.py"),
            os.path.join(backend_dir, "rag", "fast_kb.py"),
            os.path.join(backend_dir, "rag", "fast_web.py"),
            os.path.join(backend_dir, "agent", "planning", "research.py"),
            os.path.join(backend_dir, "agent", "runtime.py"),
        ]

        forbidden_literals = ["nvidia", "ollama", "vllm", "openai_compatible"]

        for file_path in files_to_check:
            self.assertTrue(os.path.exists(file_path), f"File {file_path} must exist.")
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content, filename=file_path)

            # Inspect all If statements in AST
            for node in ast.walk(tree):
                if isinstance(node, ast.If):
                    # Check if the condition compares a variable named 'provider'
                    test_str = ast.unparse(node.test)
                    self.assertNotIn(
                        "provider ==",
                        test_str,
                        f"Forbidden provider conditional found in {file_path}: {test_str}",
                    )
                    self.assertNotIn(
                        "provider in",
                        test_str,
                        f"Forbidden provider membership test found in {file_path}: {test_str}",
                    )
