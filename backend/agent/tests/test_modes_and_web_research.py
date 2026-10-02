"""Comprehensive automated tests for AURA M10: Research Modes & Web Research."""

import json
from unittest.mock import MagicMock, patch
import uuid

from django.test import TestCase
from rest_framework.test import APIClient

from agent.models import ResearchRun
from agent.planning.base import ActionType
from agent.planning.research import ResearchPlanner
from agent.research import create_research_runtime
from agent.results import ResearchEvidence, ResearchResult
from agent.security import sanitize_data
from agent.state import AgentState, AgentStatus
from agent.tasks import execute_research_run
from agent.tools.builtin.web import (
    DuckDuckGoHTMLParser,
    DuckDuckGoWebSearchProvider,
    MockWebSearchProvider,
    WebSearchResult,
    WebSearchTool,
    extract_ddg_destination_url,
    validate_web_url,
)
from agent.tools.policy import PolicyDecision
from agent.views import ResearchView
from gateway.types import GenerationResponse, ProviderMetadata, StructuredOutputResponse, UsageInfo


class MockGatewayHelper:
    """Helper to mock ModelGateway generation and structured output."""

    @staticmethod
    def create_mock_gateway(
        decision: str = "finish",
        tool: str = "web_search",
        query: str = "test query",
        synthesis: str = "Model synthesized answer.",
    ) -> MagicMock:
        gateway = MagicMock()
        gateway.metadata.return_value = ProviderMetadata(
            provider="mock",
            model="mock-model",
            capabilities=("generation", "streaming", "structured_output"),
        )
        gateway.structured_output.return_value = StructuredOutputResponse(
            data={"decision": decision, "tool": tool, "query": query},
            raw_text=json.dumps({"decision": decision, "tool": tool, "query": query}),
            provider="mock",
            model="mock-model",
            usage=UsageInfo(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            finish_reason="stop",
        )
        gateway.generate.return_value = GenerationResponse(
            text=synthesis,
            provider="mock",
            model="mock-model",
            usage=UsageInfo(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            finish_reason="stop",
        )
        return gateway


class WebSearchToolTests(TestCase):
    """Tests for WebSearchTool and WebSearchProvider abstraction."""

    def test_mock_provider_returns_deterministic_results(self) -> None:
        """MockWebSearchProvider returns structured results for queries."""
        provider = MockWebSearchProvider()
        results = provider.search("python async", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertIn("python async", results[0].title)
        self.assertTrue(results[0].url.startswith("https://"))
        self.assertTrue(bool(results[0].snippet))
        self.assertTrue(bool(results[0].domain))
        self.assertTrue(bool(results[0].timestamp))

    def test_tool_execute_validates_query(self) -> None:
        """WebSearchTool requires non-empty string query."""
        tool = WebSearchTool(provider=MockWebSearchProvider())
        # Missing query
        res1 = tool.execute()
        self.assertTrue(res1.is_error)
        self.assertIn("Parameter 'query' must be a non-empty string", res1.error_message)

        # Whitespace query
        res2 = tool.execute(query="   ")
        self.assertTrue(res2.is_error)
        self.assertIn("cannot be empty or whitespace-only", res2.error_message)

    def test_tool_execute_normalizes_structured_chunks(self) -> None:
        """WebSearchTool normalizes provider output into evidence chunks."""
        custom_item = WebSearchResult(
            title="AURA Agent Docs",
            url="https://aura.ai/docs/agent",
            snippet="Autonomous Research & Engineering Agent documentation.",
            domain="aura.ai",
        )
        provider = MockWebSearchProvider(default_results=[custom_item])
        tool = WebSearchTool(provider=provider)

        result = tool.execute(query="AURA documentation", top_k=5)
        self.assertFalse(result.is_error)
        self.assertEqual(result.metadata["retrieval_count"], 1)
        self.assertEqual(result.metadata["source_type"], "web")

        chunks = result.output
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertTrue(chunk["chunk_id"].startswith("web-"))
        self.assertEqual(chunk["document_title"], "AURA Agent Docs")
        self.assertEqual(chunk["document_source"], "https://aura.ai/docs/agent")
        self.assertEqual(chunk["document_id"], "aura.ai")
        self.assertIn("[AURA Agent Docs, Chunk: web-", chunk["citation"])
        self.assertEqual(chunk["metadata"]["source_type"], "web")

    def test_tool_execute_handles_provider_failure_without_leaks(self) -> None:
        """Provider exception produces clean error result without exposing secrets."""
        provider = MockWebSearchProvider(simulated_failure=RuntimeError("SecretKey=sk-12345 network timeout"))
        tool = WebSearchTool(provider=provider)

        res = tool.execute(query="search something")
        self.assertTrue(res.is_error)
        self.assertIn("Web search failed", res.error_message)
        self.assertNotIn("sk-12345", res.error_message)

    def test_tool_execute_deduplicates_urls(self) -> None:
        """Duplicate URLs in provider results are safely deduplicated."""
        dup1 = WebSearchResult(title="Page 1", url="https://dup.org/page", snippet="Snippet 1")
        dup2 = WebSearchResult(title="Page 1 Alt", url="https://dup.org/page", snippet="Snippet 2")
        provider = MockWebSearchProvider(default_results=[dup1, dup2])
        tool = WebSearchTool(provider=provider)

        res = tool.execute(query="dup test")
        self.assertEqual(len(res.output), 1)


class ResearchModesPolicyAndRuntimeTests(TestCase):
    """Tests for research modes, tool policies, and runtime initialization."""

    def test_knowledge_base_mode_registers_rag_only(self) -> None:
        """Knowledge Base mode registers rag_search and denies web_search."""
        runtime = create_research_runtime(mode="knowledge_base")
        self.assertTrue(runtime.executor.registry.has_tool("rag_search"))
        self.assertFalse(runtime.executor.registry.has_tool("web_search"))

        # Policy evaluation
        dummy_state = AgentState.create("test")
        check_rag = runtime.executor.policy.evaluate("rag_search", {"query": "q"}, runtime.executor.registry, dummy_state)
        self.assertEqual(check_rag.decision, PolicyDecision.ALLOWED)

        check_web = runtime.executor.policy.evaluate("web_search", {"query": "q"}, runtime.executor.registry, dummy_state)
        self.assertEqual(check_web.decision, PolicyDecision.UNKNOWN)

    def test_web_mode_registers_web_only(self) -> None:
        """Web mode registers web_search and denies rag_search."""
        runtime = create_research_runtime(mode="web")
        self.assertTrue(runtime.executor.registry.has_tool("web_search"))
        self.assertFalse(runtime.executor.registry.has_tool("rag_search"))

        dummy_state = AgentState.create("test")
        check_web = runtime.executor.policy.evaluate("web_search", {"query": "q"}, runtime.executor.registry, dummy_state)
        self.assertEqual(check_web.decision, PolicyDecision.ALLOWED)

        check_rag = runtime.executor.policy.evaluate("rag_search", {"query": "q"}, runtime.executor.registry, dummy_state)
        self.assertEqual(check_rag.decision, PolicyDecision.UNKNOWN)

    def test_web_knowledge_base_mode_registers_both_tools(self) -> None:
        """Web + Knowledge Base mode registers and permits both tools."""
        runtime = create_research_runtime(mode="web_knowledge_base")
        self.assertTrue(runtime.executor.registry.has_tool("rag_search"))
        self.assertTrue(runtime.executor.registry.has_tool("web_search"))

        dummy_state = AgentState.create("test")
        check_rag = runtime.executor.policy.evaluate("rag_search", {"query": "q"}, runtime.executor.registry, dummy_state)
        self.assertEqual(check_rag.decision, PolicyDecision.ALLOWED)

        check_web = runtime.executor.policy.evaluate("web_search", {"query": "q"}, runtime.executor.registry, dummy_state)
        self.assertEqual(check_web.decision, PolicyDecision.ALLOWED)

    def test_model_knowledge_mode_has_no_tools(self) -> None:
        """Model Knowledge mode registers no search tools and denies all tool calls."""
        runtime = create_research_runtime(mode="model_knowledge")
        self.assertFalse(runtime.executor.registry.has_tool("rag_search"))
        self.assertFalse(runtime.executor.registry.has_tool("web_search"))

        dummy_state = AgentState.create("test")
        check_rag = runtime.executor.policy.evaluate("rag_search", {"query": "q"}, runtime.executor.registry, dummy_state)
        self.assertEqual(check_rag.decision, PolicyDecision.UNKNOWN)


class ResearchPlannerModesTests(TestCase):
    """Tests for ResearchPlanner behavior across modes."""

    def test_model_knowledge_planner_synthesizes_directly(self) -> None:
        """Model knowledge mode generates a direct answer without search queries."""
        gateway = MockGatewayHelper.create_mock_gateway(synthesis="General model knowledge answer.")
        planner = ResearchPlanner(gateway=gateway, mode="model_knowledge")
        state = AgentState.create("Explain general relativity")

        plan = planner.plan("Explain general relativity", state)
        self.assertEqual(len(plan.steps), 1)
        step = plan.steps[0]
        self.assertEqual(step.action_type, ActionType.FINISH)
        self.assertEqual(step.payload["final_answer"], "General model knowledge answer.")
        self.assertEqual(step.payload["evidence_count"], 0)
        self.assertEqual(step.payload["queries"], [])

        # Verify gateway.generate was called, not structured_output for search
        gateway.generate.assert_called_once()
        gateway.structured_output.assert_not_called()

    def test_web_mode_planner_dispatches_web_search(self) -> None:
        """In web mode, continue decision generates a web_search step."""
        gateway = MockGatewayHelper.create_mock_gateway(decision="continue", query="latest AI developments")
        planner = ResearchPlanner(gateway=gateway, mode="web")
        state = AgentState.create("What are the latest AI developments?")

        plan = planner.plan("What are the latest AI developments?", state)
        self.assertEqual(len(plan.steps), 1)
        step = plan.steps[0]
        self.assertEqual(step.action_type, ActionType.TOOL)
        self.assertEqual(step.payload["tool_name"], "web_search")
        self.assertEqual(step.payload["tool_input"], {"query": "latest AI developments"})

    def test_web_knowledge_base_planner_supports_tool_selection(self) -> None:
        """In web_knowledge_base mode, planner dynamically routes to chosen tool."""
        # 1. Model chooses web_search
        gateway1 = MockGatewayHelper.create_mock_gateway(decision="continue", tool="web_search", query="web query")
        planner1 = ResearchPlanner(gateway=gateway1, mode="web_knowledge_base")
        state1 = AgentState.create("Mixed query")
        plan1 = planner1.plan("Mixed query", state1)
        self.assertEqual(plan1.steps[0].payload["tool_name"], "web_search")

        # 2. Model chooses rag_search
        gateway2 = MockGatewayHelper.create_mock_gateway(decision="continue", tool="rag_search", query="rag query")
        planner2 = ResearchPlanner(gateway=gateway2, mode="web_knowledge_base")
        state2 = AgentState.create("Mixed query")
        plan2 = planner2.plan("Mixed query", state2)
        self.assertEqual(plan2.steps[0].payload["tool_name"], "rag_search")


class EndToEndModesResearchTests(TestCase):
    """End-to-end integration tests for research execution across all 4 modes."""

    def test_model_knowledge_run_research_completes(self) -> None:
        """Model knowledge research execution finishes cleanly with ungrounded result."""
        gateway = MockGatewayHelper.create_mock_gateway(synthesis="General physics knowledge.")
        runtime = create_research_runtime(gateway=gateway, mode="model_knowledge")

        result = runtime.run_research("What is entropy?")
        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertEqual(result.final_answer, "General physics knowledge.")
        self.assertFalse(result.is_grounded)
        self.assertFalse(result.has_evidence)
        self.assertEqual(result.evidence, [])
        self.assertEqual(result.queries, [])

    def test_web_research_accumulates_and_attributes_evidence(self) -> None:
        """Web research accumulates web snippets and correctly reports sources and citations."""
        item = WebSearchResult(
            title="Quantum Computing News",
            url="https://quantum.org/news/breakthrough",
            snippet="Researchers achieved fault-tolerant quantum error correction.",
            domain="quantum.org",
        )
        provider = MockWebSearchProvider(default_results=[item])

        # Step 1: continue with web search
        # Step 2: finish
        call_count = 0

        def structured_output_mock(req):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return StructuredOutputResponse(
                    data={"decision": "continue", "query": "quantum error correction"},
                    raw_text=json.dumps({"decision": "continue", "query": "quantum error correction"}),
                    provider="mock",
                    model="mock-model",
                    usage=UsageInfo(prompt_tokens=5, completion_tokens=5, total_tokens=10),
                    finish_reason="stop",
                )
            return StructuredOutputResponse(
                data={"decision": "finish"},
                raw_text=json.dumps({"decision": "finish"}),
                provider="mock",
                model="mock-model",
                usage=UsageInfo(prompt_tokens=5, completion_tokens=5, total_tokens=10),
                finish_reason="stop",
            )

        gateway = MagicMock()
        gateway.structured_output.side_effect = structured_output_mock
        gateway.metadata.return_value = ProviderMetadata(
            provider="mock",
            model="mock-model",
            capabilities=("generation", "streaming", "structured_output"),
        )
        gateway.generate.return_value = GenerationResponse(
            text="Grounded answer citing [Quantum Computing News, Chunk: web-].",
            provider="mock",
            model="mock-model",
            usage=UsageInfo(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            finish_reason="stop",
        )

        runtime = create_research_runtime(
            gateway=gateway,
            web_search_provider=provider,
            mode="web",
        )

        result = runtime.run_research("What are recent quantum computing breakthroughs?")
        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertTrue(result.is_grounded)
        self.assertTrue(result.has_evidence)
        self.assertEqual(len(result.evidence), 1)
        self.assertEqual(result.evidence[0].document_source, "https://quantum.org/news/breakthrough")
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0]["document_source"], "https://quantum.org/news/breakthrough")
        self.assertIn("quantum error correction", result.queries)


class AsyncAPIModesTests(TestCase):
    """Tests for research modes in Django REST Framework endpoints."""

    def setUp(self) -> None:
        self.client = APIClient()

    def test_post_research_accepts_and_persists_mode(self) -> None:
        """POST /api/research/ accepts valid mode and persists on ResearchRun."""
        with patch.object(ResearchView, "dispatch_task"):
            for mode in ("model_knowledge", "knowledge_base", "web", "web_knowledge_base"):
                res = self.client.post(
                    "/api/research/",
                    {"objective": f"Test {mode}", "mode": mode},
                    format="json",
                )
                self.assertEqual(res.status_code, 202)
                data = res.json()
                self.assertEqual(data["mode"], mode)

                run = ResearchRun.objects.get(id=data["run_id"])
                self.assertEqual(run.mode, mode)

    def test_post_research_defaults_mode_to_knowledge_base(self) -> None:
        """POST /api/research/ without mode defaults to knowledge_base."""
        with patch.object(ResearchView, "dispatch_task"):
            res = self.client.post(
                "/api/research/",
                {"objective": "Backward compatible request"},
                format="json",
            )
            self.assertEqual(res.status_code, 202)
            data = res.json()
            self.assertEqual(data["mode"], "knowledge_base")

            run = ResearchRun.objects.get(id=data["run_id"])
            self.assertEqual(run.mode, "knowledge_base")

    def test_post_research_rejects_invalid_mode(self) -> None:
        """POST /api/research/ rejects unknown mode with 400."""
        res = self.client.post(
            "/api/research/",
            {"objective": "Invalid mode test", "mode": "unsupported_mode"},
            format="json",
        )
        self.assertEqual(res.status_code, 400)
        data = res.json()
        self.assertIn("error", data)
        self.assertIn("unsupported_mode", data["error"])

    def test_get_research_detail_and_history_expose_mode(self) -> None:
        """Detail and history endpoints include mode field."""
        run = ResearchRun.objects.create(
            objective="Inspect mode",
            mode=ResearchRun.MODE_WEB,
            status=ResearchRun.STATUS_COMPLETED,
        )

        # GET detail
        detail_res = self.client.get(f"/api/research/{run.id}/")
        self.assertEqual(detail_res.status_code, 200)
        self.assertEqual(detail_res.json()["mode"], "web")

        # GET history
        history_res = self.client.get("/api/research/runs/")
        self.assertEqual(history_res.status_code, 200)
        history_runs = history_res.json()
        target = next((r for r in history_runs if r["run_id"] == str(run.id)), None)
        self.assertIsNotNone(target)
        self.assertEqual(target["mode"], "web")

    def test_celery_task_passes_mode_to_runtime(self) -> None:
        """Celery task execute_research_run reads mode and passes to create_research_runtime."""
        run = ResearchRun.objects.create(
            objective="Celery mode test",
            mode=ResearchRun.MODE_MODEL_KNOWLEDGE,
        )

        mock_runtime = MagicMock()
        mock_runtime.run_research.return_value = ResearchResult(
            objective=run.objective,
            final_answer="Synthesized",
            evidence=[],
            sources=[],
            queries=[],
            iteration_count=1,
            has_evidence=False,
            status=AgentStatus.COMPLETED,
        )

        with patch("agent.tasks.create_research_runtime", return_value=mock_runtime) as mock_create:
            res = execute_research_run(str(run.id))
            self.assertEqual(res["status"], "completed")
            mock_create.assert_called_once_with(mode="model_knowledge")


class HardenedDuckDuckGoProviderTests(TestCase):
    """Deterministic unit tests for hardened DuckDuckGo HTML parser, URL validator, and provider."""

    def test_valid_http_result_url(self) -> None:
        """Valid HTTP URLs are correctly extracted and preserved."""
        html = """
        <div class="result results_links">
          <h2 class="result__title">
            <a class="result__a" href="http://example.org/http-article">HTTP Architecture Guide</a>
          </h2>
          <a class="result__snippet" href="http://example.org/http-article">
            Overview of communication patterns over plain HTTP.
          </a>
        </div>
        """
        results = DuckDuckGoWebSearchProvider.parse_html_results(html, query="http test")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, "http://example.org/http-article")
        self.assertEqual(results[0].title, "HTTP Architecture Guide")
        self.assertEqual(results[0].snippet, "Overview of communication patterns over plain HTTP.")
        self.assertEqual(results[0].domain, "example.org")

    def test_valid_https_result_url(self) -> None:
        """Valid HTTPS URLs are correctly extracted and preserved."""
        html = """
        <div class="result results_links">
          <h2 class="result__title">
            <a class="result__a" href="https://docs.python.org/3/library/html.parser.html">HTML Parser Docs</a>
          </h2>
          <a class="result__snippet" href="https://docs.python.org/3/library/html.parser.html">
            Standard library module for parsing HTML and XHTML.
          </a>
        </div>
        """
        results = DuckDuckGoWebSearchProvider.parse_html_results(html, query="python parser")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, "https://docs.python.org/3/library/html.parser.html")
        self.assertEqual(results[0].title, "HTML Parser Docs")
        self.assertEqual(results[0].domain, "docs.python.org")

    def test_invalid_and_non_http_urls_ignored(self) -> None:
        """Non-HTTP/HTTPS schemes, malformed URLs, and empty hostnames are safely ignored."""
        html = """
        <div class="results">
          <div class="result">
            <a class="result__title" href="ftp://ftp.archive.org/file.iso">FTP Download</a>
            <div class="result__snippet">FTP download link should be ignored.</div>
          </div>
          <div class="result">
            <a class="result__title" href="javascript:alert(document.cookie)">JS Injection</a>
            <div class="result__snippet">Javascript link should be ignored.</div>
          </div>
          <div class="result">
            <a class="result__title" href="file:///etc/shadow">Local File</a>
            <div class="result__snippet">File URL should be ignored.</div>
          </div>
          <div class="result">
            <a class="result__title" href="mailto:support@aura.local">Mail Link</a>
            <div class="result__snippet">Mailto URL should be ignored.</div>
          </div>
          <div class="result">
            <a class="result__title" href="http:///empty-host">No Host</a>
            <div class="result__snippet">Missing hostname should be ignored.</div>
          </div>
          <div class="result">
            <a class="result__title" href="">Empty URL</a>
            <div class="result__snippet">Empty URL should be ignored.</div>
          </div>
          <div class="result">
            <a class="result__title" href="https://valid-target.com/valid-article">Valid Result</a>
            <div class="result__snippet">This valid HTTPS URL must be preserved.</div>
          </div>
        </div>
        """
        results = DuckDuckGoWebSearchProvider.parse_html_results(html, query="url filter test")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, "https://valid-target.com/valid-article")
        self.assertEqual(results[0].title, "Valid Result")

    def test_html_entities_decoded(self) -> None:
        """HTML entities in titles and snippets are fully unescaped."""
        html = """
        <div class="result">
          <h2 class="result__title">
            <a class="result__a" href="https://example.com/entities">
              Python &amp; Django: &#39;Special&#39; &quot;Symbols&quot; &lt;v1.0&gt;
            </a>
          </h2>
          <a class="result__snippet" href="https://example.com/entities">
            Guide on handling &amp;, &quot;quotes&quot;, &#39;apostrophes&#39;, and &lt;tags&gt;.
          </a>
        </div>
        """
        results = DuckDuckGoWebSearchProvider.parse_html_results(html)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Python & Django: 'Special' \"Symbols\" <v1.0>")
        self.assertEqual(
            results[0].snippet,
            "Guide on handling &, \"quotes\", 'apostrophes', and <tags>.",
        )

    def test_title_and_snippet_extraction(self) -> None:
        """Title and snippet text extraction handles nested formatting elements and whitespace."""
        html = """
        <div class="result results_links">
          <h2 class="result__title">
            <a class="result__a" href="https://example.com/rich">
              AURA <span>Autonomous</span> <b>Research</b> Platform
            </a>
          </h2>
          <div class="result__snippet">
            Line 1 of snippet.<br>Line 2 with <b>bold <i>and italic</i></b> emphasis.
          </div>
        </div>
        """
        results = DuckDuckGoWebSearchProvider.parse_html_results(html)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "AURA Autonomous Research Platform")
        self.assertEqual(
            results[0].snippet,
            "Line 1 of snippet. Line 2 with bold and italic emphasis.",
        )

    def test_ddg_uddg_redirect_resolved(self) -> None:
        """DuckDuckGo uddg= redirect parameter is cleanly unwrapped to destination URL."""
        html = """
        <div class="result">
          <h2 class="result__title">
            <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Ftarget-destination.org%2Fspec%3Fversion%3D2%26format%3Djson&amp;rut=1">
              Redirect Destination Spec
            </a>
          </h2>
          <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Ftarget-destination.org%2Fspec%3Fversion%3D2%26format%3Djson">
            Specification document at destination target.
          </a>
        </div>
        <div class="result">
          <h2 class="result__title">
            <a class="result__a" href="/l/?kh=-1&uddg=https%3A%2F%2Fsecond-target.io%2Foverview">
              Second Destination
            </a>
          </h2>
          <a class="result__snippet" href="/l/?kh=-1&uddg=https%3A%2F%2Fsecond-target.io%2Foverview">
            Overview document.
          </a>
        </div>
        """
        results = DuckDuckGoWebSearchProvider.parse_html_results(html)
        self.assertEqual(len(results), 2)
        self.assertEqual(
            results[0].url,
            "https://target-destination.org/spec?version=2&format=json",
        )
        self.assertEqual(results[0].domain, "target-destination.org")
        self.assertEqual(results[1].url, "https://second-target.io/overview")
        self.assertEqual(results[1].domain, "second-target.io")

    def test_rank_semantics_explicit_without_fake_score(self) -> None:
        """Web evidence chunks provide explicit rank and score: None without fake scores."""
        item1 = WebSearchResult(
            title="Result 1",
            url="https://site1.org/doc",
            snippet="First snippet.",
        )
        item2 = WebSearchResult(
            title="Result 2",
            url="https://site2.org/doc",
            snippet="Second snippet.",
        )
        provider = MockWebSearchProvider(default_results=[item1, item2])
        tool = WebSearchTool(provider=provider)

        result = tool.execute(query="rank test")
        self.assertFalse(result.is_error)
        chunks = result.output
        self.assertEqual(len(chunks), 2)

        # First chunk
        self.assertEqual(chunks[0]["rank"], 1)
        self.assertIsNone(chunks[0]["score"])
        self.assertEqual(chunks[0]["metadata"]["rank"], 1)

        # Second chunk
        self.assertEqual(chunks[1]["rank"], 2)
        self.assertIsNone(chunks[1]["score"])
        self.assertEqual(chunks[1]["metadata"]["rank"], 2)

    def test_validate_web_url_helper(self) -> None:
        """Direct verification of validate_web_url validator utility."""
        # Valid URLs
        self.assertTrue(validate_web_url("https://example.com"))
        self.assertTrue(validate_web_url("http://example.org/path?q=1"))
        self.assertTrue(validate_web_url("https://sub.domain.co.uk:8080/res"))

        # Invalid schemes
        self.assertFalse(validate_web_url("ftp://example.com"))
        self.assertFalse(validate_web_url("javascript:alert(1)"))
        self.assertFalse(validate_web_url("data:text/html,test"))
        self.assertFalse(validate_web_url("file:///etc/hosts"))

        # Missing or malformed hostnames
        self.assertFalse(validate_web_url("http:///empty-host"))
        self.assertFalse(validate_web_url("https://"))
        self.assertFalse(validate_web_url("http://"))
        self.assertFalse(validate_web_url(""))
        self.assertFalse(validate_web_url("   "))
        self.assertFalse(validate_web_url("not a url"))
        self.assertFalse(validate_web_url("https://in valid.com"))

    def test_extract_ddg_destination_url_helper(self) -> None:
        """Direct verification of extract_ddg_destination_url redirect unwrapper."""
        self.assertEqual(
            extract_ddg_destination_url("//duckduckgo.com/l/?uddg=https%3A%2F%2Fpython.org&rut=1"),
            "https://python.org",
        )
        self.assertEqual(
            extract_ddg_destination_url("/l/?kh=-1&uddg=https%3A%2F%2Fdocs.python.org%2F3%2F"),
            "https://docs.python.org/3/",
        )
        self.assertEqual(
            extract_ddg_destination_url("https://example.com/direct"),
            "https://example.com/direct",
        )
        self.assertEqual(extract_ddg_destination_url(""), "")

