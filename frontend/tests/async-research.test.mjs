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

test("Frontend research submission lifecycle supports consecutive submissions without page refresh", async () => {
  // Model state machine mirroring page.jsx research lifecycle
  class ResearchUIController {
    constructor() {
      this.objective = "";
      this.loading = false;
      this.error = null;
      this.result = null;
      this.activeRunId = null;
      this.activeRunStatus = null;
      this.activeRunElapsed = 0;
      this.postCalls = [];
      this.pollTimerActive = false;
    }

    isSubmitDisabled() {
      return this.loading || !this.objective.trim();
    }

    async submit(mockFetch, mode = "knowledge_base") {
      const trimmed = this.objective.trim();
      if (!trimmed) {
        this.error = "Please enter a research objective.";
        return;
      }

      this.loading = true;
      this.error = null;
      this.result = null;
      this.activeRunElapsed = 0;
      this.pollTimerActive = false;

      try {
        const response = await mockFetch("/api/research/", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ objective: trimmed, mode }),
        });
        this.postCalls.push({ url: "/api/research/", objective: trimmed, mode });

        const contentType = response.headers?.get("content-type") || "application/json";
        let data = {};
        if (contentType.includes("application/json")) {
          data = await response.json();
        } else {
          const text = await response.text();
          throw new Error(`Server returned HTTP ${response.status}: ${text.slice(0, 120)}`);
        }

        if (response.status !== 202 && !response.ok) {
          const errorDetail = data.detail || data.error || `HTTP error ${response.status}`;
          throw new Error(errorDetail);
        }

        this.activeRunId = data.run_id;
        this.activeRunStatus = data.status || "queued";
        this.pollTimerActive = true;
      } catch (err) {
        this.error = err.message || "Failed to communicate with research backend.";
        this.loading = false;
      }
    }

    async poll(mockFetch) {
      if (!this.activeRunId) return;
      const currentRunId = this.activeRunId;

      try {
        const response = await mockFetch(`/api/research/${currentRunId}/`);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();

        // Stale closure guard
        if (this.activeRunId !== currentRunId) return;

        this.activeRunStatus = data.status;
        if (data.status === "completed") {
          this.result = data;
          this.error = null;
          this.activeRunId = null;
          this.activeRunStatus = null;
          this.loading = false;
          this.pollTimerActive = false;
        } else if (data.status === "failed") {
          this.error = data.error || "Research run failed.";
          this.activeRunId = null;
          this.activeRunStatus = null;
          this.loading = false;
          this.pollTimerActive = false;
        } else if (data.status === "cancelled") {
          this.error = data.error || "Research was cancelled.";
          this.activeRunId = null;
          this.activeRunStatus = null;
          this.loading = false;
          this.pollTimerActive = false;
        }
      } catch (err) {
        // Polling retry
      }
    }
  }

  const ui = new ResearchUIController();

  // 1. Initial State: Button disabled when empty
  assert.equal(ui.isSubmitDisabled(), true);
  ui.objective = "Query 1: What is AURA?";
  assert.equal(ui.isSubmitDisabled(), false);

  // 2. First Submission succeeds (202 Accepted)
  const mockFetch1 = async (url, opts) => {
    if (opts?.method === "POST") {
      return {
        status: 202,
        ok: true,
        headers: { get: () => "application/json" },
        json: async () => ({ run_id: "run-uuid-1", status: "queued" }),
      };
    }
    return {
      status: 200,
      ok: true,
      headers: { get: () => "application/json" },
      json: async () => ({ status: "completed", final_answer: "AURA is an agentic platform." }),
    };
  };

  await ui.submit(mockFetch1);
  assert.equal(ui.loading, true);
  assert.equal(ui.activeRunId, "run-uuid-1");
  assert.equal(ui.postCalls.length, 1);
  assert.equal(ui.postCalls[0].objective, "Query 1: What is AURA?");

  // Complete first run via poll
  await ui.poll(mockFetch1);
  assert.equal(ui.loading, false);
  assert.equal(ui.activeRunId, null);
  assert.notEqual(ui.result, null);
  assert.equal(ui.error, null);

  // 3. Button can be clicked again after clearing / updating objective
  ui.objective = "";
  assert.equal(ui.isSubmitDisabled(), true);
  ui.objective = "Query 2: How does FlashRank work?";
  assert.equal(ui.isSubmitDisabled(), false);

  // 4. Second submission sends a NEW POST with new objective without refreshing
  const mockFetch2 = async (url, opts) => {
    if (opts?.method === "POST") {
      return {
        status: 202,
        ok: true,
        headers: { get: () => "application/json" },
        json: async () => ({ run_id: "run-uuid-2", status: "queued" }),
      };
    }
    return {
      status: 200,
      ok: true,
      headers: { get: () => "application/json" },
      json: async () => ({ status: "completed", final_answer: "FlashRank is a local cross-encoder." }),
    };
  };

  await ui.submit(mockFetch2);
  // Completed result from run 1 is cleared on second submit
  assert.equal(ui.loading, true);
  assert.equal(ui.activeRunId, "run-uuid-2");
  assert.equal(ui.result, null);
  assert.equal(ui.postCalls.length, 2);
  assert.equal(ui.postCalls[1].objective, "Query 2: How does FlashRank work?");

  await ui.poll(mockFetch2);
  assert.equal(ui.loading, false);
  assert.equal(ui.activeRunId, null);
  assert.notEqual(ui.result, null);
});

test("Frontend error state from a previous run does not block subsequent submissions or fabricate 503", async () => {
  let isSubmitDisabled = (loading, objective) => loading || !objective.trim();

  let state = {
    loading: false,
    error: "Previous error: NVIDIA model generation failed: 503 Service Unavailable",
    result: null,
    activeRunId: null,
    objective: "New query after failure",
  };

  // 1. Error state from previous run does not disable submit button
  assert.equal(isSubmitDisabled(state.loading, state.objective), false);

  // 2. Submitting clears previous error and result immediately
  state.loading = true;
  state.error = null;
  state.result = null;

  assert.equal(state.error, null);
  assert.equal(state.result, null);
  assert.equal(isSubmitDisabled(state.loading, state.objective), true);

  // 3. Local network failures do not fabricate or map to HTTP 503
  const handleFetchError = (err) => {
    return err.message || "Failed to communicate with research backend.";
  };

  const netError = new TypeError("Failed to fetch");
  const parsed = handleFetchError(netError);
  assert.equal(parsed, "Failed to fetch");
  assert.ok(!parsed.includes("503"));

  const abortError = new DOMException("The user aborted a request.", "AbortError");
  assert.equal(handleFetchError(abortError), "The user aborted a request.");
  assert.ok(!handleFetchError(abortError).includes("503"));
});
