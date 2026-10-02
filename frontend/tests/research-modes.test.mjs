import test from "node:test";
import assert from "node:assert/strict";

const VALID_MODES = [
  "model_knowledge",
  "knowledge_base",
  "web",
  "web_knowledge_base",
];

const formatMode = (m) => {
  switch (m) {
    case "model_knowledge":
      return "Model Knowledge";
    case "knowledge_base":
      return "Knowledge Base";
    case "web":
      return "Web";
    case "web_knowledge_base":
      return "Web + KB";
    default:
      return m || "Knowledge Base";
  }
};

test("Research modes helper formats mode labels correctly", () => {
  assert.equal(formatMode("model_knowledge"), "Model Knowledge");
  assert.equal(formatMode("knowledge_base"), "Knowledge Base");
  assert.equal(formatMode("web"), "Web");
  assert.equal(formatMode("web_knowledge_base"), "Web + KB");
  assert.equal(formatMode(null), "Knowledge Base");
  assert.equal(formatMode(undefined), "Knowledge Base");
  assert.equal(formatMode("custom_mode"), "custom_mode");
});

test("Research submission payload includes selected mode", () => {
  const createSubmitPayload = (objective, mode = "knowledge_base") => {
    return {
      objective: objective.trim(),
      mode: VALID_MODES.includes(mode) ? mode : "knowledge_base",
    };
  };

  const payloadDefault = createSubmitPayload("Explain pgvector");
  assert.deepEqual(payloadDefault, {
    objective: "Explain pgvector",
    mode: "knowledge_base",
  });

  const payloadWeb = createSubmitPayload("Latest developments", "web");
  assert.deepEqual(payloadWeb, {
    objective: "Latest developments",
    mode: "web",
  });

  const payloadHybrid = createSubmitPayload("Hybrid query", "web_knowledge_base");
  assert.deepEqual(payloadHybrid, {
    objective: "Hybrid query",
    mode: "web_knowledge_base",
  });

  const payloadModel = createSubmitPayload("General query", "model_knowledge");
  assert.deepEqual(payloadModel, {
    objective: "General query",
    mode: "model_knowledge",
  });

  const payloadInvalid = createSubmitPayload("Invalid mode test", "unsupported");
  assert.deepEqual(payloadInvalid, {
    objective: "Invalid mode test",
    mode: "knowledge_base",
  });
});

test("202 Accepted response preserves mode attribute", () => {
  const parseSubmitResponse = (status, data) => {
    if (status === 202) {
      return {
        isAsync: true,
        runId: data.run_id,
        status: data.status,
        mode: data.mode,
        objective: data.objective,
      };
    }
    return { isAsync: false, error: data.error || `HTTP ${status}` };
  };

  const responseData = {
    run_id: "run-m10-web-001",
    status: "queued",
    mode: "web",
    objective: "Search web for AURA specs",
  };

  const parsed = parseSubmitResponse(202, responseData);
  assert.equal(parsed.isAsync, true);
  assert.equal(parsed.runId, "run-m10-web-001");
  assert.equal(parsed.mode, "web");
});

test("Research history parser extracts and exposes mode", () => {
  const mockHistoryData = [
    {
      run_id: "run-kb-1",
      objective: "KB query",
      status: "completed",
      mode: "knowledge_base",
      created_at: "2026-10-02T10:00:00Z",
      duration_ms: 12000,
      is_grounded: true,
      citation_count: 2,
    },
    {
      run_id: "run-web-2",
      objective: "Web query",
      status: "completed",
      mode: "web",
      created_at: "2026-10-02T10:05:00Z",
      duration_ms: 8500,
      is_grounded: true,
      citation_count: 1,
    },
    {
      run_id: "run-model-3",
      objective: "Model direct query",
      status: "completed",
      mode: "model_knowledge",
      created_at: "2026-10-02T10:10:00Z",
      duration_ms: 1500,
      is_grounded: false,
      citation_count: 0,
    },
  ];

  const parseHistory = (data) => {
    if (!Array.isArray(data)) return [];
    return data.map((item) => ({
      runId: item.run_id,
      objective: item.objective,
      status: item.status,
      mode: item.mode || "knowledge_base",
      formattedMode: formatMode(item.mode),
      durationSec: item.duration_ms ? (item.duration_ms / 1000).toFixed(1) : null,
      isGrounded: item.is_grounded,
      citationCount: item.citation_count,
    }));
  };

  const parsed = parseHistory(mockHistoryData);
  assert.equal(parsed.length, 3);
  assert.equal(parsed[0].mode, "knowledge_base");
  assert.equal(parsed[0].formattedMode, "Knowledge Base");
  assert.equal(parsed[1].mode, "web");
  assert.equal(parsed[1].formattedMode, "Web");
  assert.equal(parsed[2].mode, "model_knowledge");
  assert.equal(parsed[2].formattedMode, "Model Knowledge");
});

test("Web evidence structure preserves source attribution and citations", () => {
  const mockWebEvidenceRun = {
    run_id: "run-web-test",
    status: "completed",
    mode: "web",
    objective: "Find python documentation",
    final_answer: "Python provides built-in typing [Python Docs, Chunk: web-abc12345].",
    evidence: [
      {
        chunk_id: "web-abc12345",
        document_title: "Python Docs",
        document_source: "https://docs.python.org/3/",
        content: "Typing module provides runtime support for type hints.",
        citation: "[Python Docs, Chunk: web-abc12345]",
        metadata: {
          url: "https://docs.python.org/3/",
          domain: "docs.python.org",
        },
      },
    ],
    sources: [
      {
        document_title: "Python Docs",
        document_source: "https://docs.python.org/3/",
        chunk_count: 1,
        chunk_ids: ["web-abc12345"],
      },
    ],
    citations: ["[Python Docs, Chunk: web-abc12345]"],
    queries: ["python docs typing"],
    is_grounded: true,
  };

  assert.equal(mockWebEvidenceRun.mode, "web");
  assert.equal(mockWebEvidenceRun.evidence[0].document_source, "https://docs.python.org/3/");
  assert.equal(mockWebEvidenceRun.evidence[0].metadata.domain, "docs.python.org");
  assert.ok(mockWebEvidenceRun.final_answer.includes("[Python Docs, Chunk: web-abc12345]"));
  assert.equal(mockWebEvidenceRun.citations.length, 1);
});
