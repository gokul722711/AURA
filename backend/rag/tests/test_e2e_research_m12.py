"""End-to-end integration tests for M12 Web-Page Knowledge Ingestion and Research.

Verifies the complete pipeline:
  User URL
    ↓
  SSRF validation & WebFetcher
    ↓
  WebPageExtractor (Trafilatura)
    ↓
  Normalization & Chunking
    ↓
  pgvector embedding & storage
    ↓
  Knowledge Base Research Mode
    ↓
  Evidence with URL provenance
    ↓
  Grounded answer citing [Title, URL, Chunk: ID]
"""

import json
from collections.abc import Iterator
from typing import Any

from django.test import TestCase

from agent.research import create_research_runtime
from agent.results import make_citation
from agent.state import AgentStatus
from gateway.base import LLMProvider
from gateway.gateway import ModelGateway
from gateway.types import (
    ALL_CAPABILITIES,
    GenerationRequest,
    GenerationResponse,
    ProviderMetadata,
    StreamChunk,
    StructuredOutputRequest,
    StructuredOutputResponse,
)
from rag.chunking import ChunkingConfig
from rag.embeddings.mock import MockEmbeddingProvider
from rag.ingestion import ingest_url
from rag.models import Document, DocumentChunk
from rag.retrieval import RetrievalConfig
from rag.web.mock import MockWebFetcher


class ScriptedE2ELLMProvider(LLMProvider):
    """Deterministic LLM Provider returning a scripted sequence of responses."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.call_count = 0
        self.recorded_requests: list[GenerationRequest] = []
        self.recorded_structured_requests: list[StructuredOutputRequest] = []

    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            provider="scripted",
            model="scripted-model",
            capabilities=ALL_CAPABILITIES,
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.recorded_requests.append(request)
        self.call_count += 1
        if self.call_count <= len(self.responses):
            text = self.responses[self.call_count - 1]
        else:
            text = '{"decision": "finish"}'
        return GenerationResponse(
            text=text,
            provider="scripted",
            model="scripted-model",
        )

    def stream(self, request: GenerationRequest) -> Iterator[StreamChunk]:
        resp = self.generate(request)
        yield StreamChunk(text=resp.text, index=0, finish_reason="stop")

    def structured_output(
        self, request: StructuredOutputRequest
    ) -> StructuredOutputResponse:
        self.recorded_structured_requests.append(request)
        gen_req = GenerationRequest(
            messages=request.messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        resp = self.generate(gen_req)
        try:
            data = json.loads(resp.text)
        except Exception:
            data = {"decision": "finish"}
        return StructuredOutputResponse(
            data=data,
            raw_text=resp.text,
            provider="scripted",
            model="scripted-model",
        )


class E2EResearchM12Tests(TestCase):
    """Deterministic end-to-end test proving web page indexing through KB research and citations."""

    def setUp(self) -> None:
        Document.objects.all().delete()
        self.embedding_provider = MockEmbeddingProvider(dimensions=384)
        self.fetcher = MockWebFetcher()

    def test_web_page_e2e_research_and_citation_flow(self) -> None:
        """A public web article is ingested into Knowledge Base and cited in research."""
        target_url = "https://quantum-daily.example/articles/fault-tolerant-computing"
        html_article = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fault-Tolerant Quantum Computing Milestones</title>
            <meta name="author" content="Dr. Sarah Connor"/>
            <meta name="date" content="2026-03-01"/>
            <link rel="canonical" href="https://quantum-daily.example/articles/fault-tolerant-computing"/>
        </head>
        <body>
            <header><nav><a href="/">Home</a></nav></header>
            <main>
                <article>
                    <h1>Fault-Tolerant Quantum Computing Milestones</h1>
                    <p>
                        Researchers have demonstrated fault-tolerant quantum error correction with surface codes,
                        achieving logical qubit error rates well below the physical threshold.
                    </p>
                    <p>
                        This experimental setup utilizes a planar superconducting grid operated in a dilution
                        refrigerator at millikelvin temperatures.
                    </p>
                </article>
            </main>
            <footer><p>&copy; 2026 Quantum Daily</p></footer>
        </body>
        </html>
        """
        self.fetcher.register_html(target_url, html_article)

        # 1. Ingest the web page into Knowledge Base
        doc = ingest_url(
            url=target_url,
            fetcher=self.fetcher,
            embedding_provider=self.embedding_provider,
            chunking_config=ChunkingConfig(chunk_size=500, chunk_overlap=0),
        )

        self.assertEqual(doc.status, "ready")
        self.assertEqual(doc.source_type, "web_page")
        self.assertEqual(doc.url, target_url)
        self.assertEqual(doc.domain, "quantum-daily.example")

        chunks = list(doc.chunks.all().order_by("chunk_index"))
        self.assertGreater(len(chunks), 0)
        target_chunk = chunks[0]
        chunk_id = str(target_chunk.id)

        # Verify chunk retained web provenance
        self.assertEqual(target_chunk.metadata.get("url"), target_url)
        self.assertEqual(target_chunk.metadata.get("domain"), "quantum-daily.example")
        self.assertEqual(target_chunk.metadata.get("source_type"), "web_page")

        # 2. Setup scripted LLM to research, synthesize, and cite the web document
        canonical_citation = make_citation(doc.title, chunk_id, url=target_url)
        self.assertEqual(
            canonical_citation,
            f"[{doc.title}, {target_url}, Chunk: {chunk_id}]",
        )

        step1_search = json.dumps({
            "decision": "continue",
            "query": "fault-tolerant surface codes",
        })
        step2_finish = json.dumps({
            "decision": "finish",
        })
        step3_synthesis = (
            f"Surface codes enable fault-tolerant quantum computing below physical thresholds {canonical_citation}."
        )

        llm_provider = ScriptedE2ELLMProvider([
            step1_search,
            step2_finish,
            step3_synthesis,
        ])
        gateway = ModelGateway(provider=llm_provider)

        # 3. Create Research Runtime in Knowledge Base mode
        runtime = create_research_runtime(
            gateway=gateway,
            embedding_provider=self.embedding_provider,
            retrieval_config=RetrievalConfig(top_k=3),
            mode="knowledge_base",
        )

        # 4. Execute research objective
        result = runtime.run_research("Explain how surface codes enable fault-tolerant quantum computing.")

        # 5. Verify completion, answer, evidence, and citations
        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertIn("Surface codes enable fault-tolerant", result.final_answer)

        # Evidence checks
        self.assertGreater(len(result.evidence), 0)
        ev = result.evidence[0]
        self.assertEqual(ev.url, target_url)
        self.assertEqual(ev.citation, canonical_citation)
        self.assertIn("fault-tolerant", ev.content.lower())

        # Citations list check
        self.assertIn(canonical_citation, result.citations)

        # Sources list check
        self.assertGreater(len(result.sources), 0)
        src = result.sources[0]
        self.assertEqual(src["url"], target_url)
        self.assertEqual(src["document_title"], doc.title)

        # Verified groundedness
        self.assertTrue(result.is_grounded)
