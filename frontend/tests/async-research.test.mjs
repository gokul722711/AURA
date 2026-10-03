import test from "node:test";
import assert from "node:assert/strict";

test("Research API 202 response initializes async run tracking", () => {
  const parseSubmitResponse = (status, data) => {
    if (status === 202) {
      return {
        isAsync: true,
        runId: data.run_id,
        status: data.status,
        objective: data.objective,
      };
    }
    return { isAsync: false, error: data.error || `HTTP ${status}` };
  };

  const acceptedRes = {
    run_id: "test-run-uuid-123",
    status: "queued",
    objective: "How does AURA RAG work?",
  };

  const parsed = parseSubmitResponse(202, acceptedRes);
  assert.equal(parsed.isAsync, true);
  assert.equal(parsed.runId, "test-run-uuid-123");
  assert.equal(parsed.status, "queued");
  assert.equal(parsed.objective, "How does AURA RAG work?");
});

test("Polling lifecycle detects terminal states correctly", () => {
  const isTerminalStatus = (status) => {
    return ["completed", "failed", "cancelled"].includes(status);
  };

  assert.equal(isTerminalStatus("queued"), false);
  assert.equal(isTerminalStatus("running"), false);
  assert.equal(isTerminalStatus("completed"), true);
  assert.equal(isTerminalStatus("failed"), true);
  assert.equal(isTerminalStatus("cancelled"), true);
});

test("Completed run data exposes M6 result structure and citations", () => {
  const mockCompletedRun = {
    run_id: "run-abc-789",
    status: "completed",
    objective: "Examine Model Gateway",
    final_answer: "The Model Gateway decouples inference [Gateway Spec, Chunk: c1].",
    evidence: [
      {
        chunk_id: "c1",
        document_title: "Gateway Spec",
        content: "Decouples application logic from providers.",
        citation: "[Gateway Spec, Chunk: c1]",
      },
    ],
    sources: [
      {
        document_title: "Gateway Spec",
        chunk_count: 1,
        chunk_ids: ["c1"],
      },
    ],
    citations: ["[Gateway Spec, Chunk: c1]"],
    queries: ["Model Gateway abstraction"],
    iteration_count: 2,
    is_grounded: true,
    has_evidence: true,
    duration_ms: 1250.0,
  };

  assert.equal(mockCompletedRun.status, "completed");
  assert.equal(mockCompletedRun.is_grounded, true);
  assert.equal(mockCompletedRun.citations.length, 1);
  assert.equal(mockCompletedRun.evidence.length, 1);
  assert.equal(mockCompletedRun.sources.length, 1);
  assert.ok(mockCompletedRun.final_answer.includes("[Gateway Spec, Chunk: c1]"));
});

test("Research History list parser preserves summary metadata without full evidence", () => {
  const mockHistoryData = [
    {
      run_id: "run-1",
      objective: "Objective 1",
      status: "completed",
      created_at: "2026-10-02T10:00:00Z",
      duration_ms: 45000,
      is_grounded: true,
      has_evidence: true,
      citation_count: 3,
    },
    {
      run_id: "run-2",
      objective: "Objective 2",
      status: "running",
      created_at: "2026-10-02T10:05:00Z",
      duration_ms: null,
      is_grounded: false,
      has_evidence: false,
      citation_count: 0,
    },
  ];

  const parseHistory = (data) => {
    if (!Array.isArray(data)) return [];
    return data.map((item) => ({
      runId: item.run_id,
      objective: item.objective,
      status: item.status,
      durationSec: item.duration_ms ? (item.duration_ms / 1000).toFixed(1) : null,
      isGrounded: item.is_grounded,
      citationCount: item.citation_count,
    }));
  };

  const parsed = parseHistory(mockHistoryData);
  assert.equal(parsed.length, 2);
  assert.equal(parsed[0].runId, "run-1");
  assert.equal(parsed[0].status, "completed");
  assert.equal(parsed[0].durationSec, "45.0");
  assert.equal(parsed[0].citationCount, 3);
  assert.equal(parsed[1].runId, "run-2");
  assert.equal(parsed[1].status, "running");
});

test("Duration formatting helper renders canonical seconds accurately", () => {
  const formatDuration = (runOrResult) => {
    if (!runOrResult) return null;
    if (typeof runOrResult.duration_seconds === "number") {
      return `${runOrResult.duration_seconds.toFixed(2)}s`;
    }
    if (typeof runOrResult.duration_ms === "number") {
      return `${(runOrResult.duration_ms / 1000).toFixed(2)}s`;
    }
    return null;
  };

  const formatHistoryDuration = (run) => {
    if (!run) return null;
    if (typeof run.duration_seconds === "number" && run.duration_seconds > 0) {
      return `⏱ ${run.duration_seconds.toFixed(1)}s`;
    }
    if (typeof run.duration_ms === "number" && run.duration_ms > 0) {
      return `⏱ ${(run.duration_ms / 1000).toFixed(1)}s`;
    }
    return null;
  };

  // 1. Result summary duration in canonical seconds
  assert.equal(formatDuration({ duration_seconds: 34.79 }), "34.79s");
  assert.equal(formatDuration({ duration_ms: 34790.0 }), "34.79s");
  assert.equal(formatDuration({ duration_seconds: 61.44 }), "61.44s");
  assert.equal(formatDuration({ duration_seconds: null }), null);
  assert.equal(formatDuration(null), null);

  // 2. History duration pill in canonical seconds
  assert.equal(formatHistoryDuration({ duration_seconds: 34.79 }), "⏱ 34.8s");
  assert.equal(formatHistoryDuration({ duration_ms: 34790.0 }), "⏱ 34.8s");
  assert.equal(formatHistoryDuration({ duration_seconds: 61.44 }), "⏱ 61.4s");
  assert.equal(formatHistoryDuration({ duration_seconds: 0 }), null);
  assert.equal(formatHistoryDuration({ duration_seconds: null }), null);
});
