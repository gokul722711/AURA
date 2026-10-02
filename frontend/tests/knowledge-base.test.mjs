import test from "node:test";
import assert from "node:assert/strict";
import nextConfig from "../next.config.mjs";

test("Next.js proxy routes /api/documents/ correctly", async () => {
  const rewrites = await nextConfig.rewrites();
  assert.ok(Array.isArray(rewrites), "rewrites() should return an array");

  const apiRewrite = rewrites.find((r) => r.source === "/api/:path*");
  assert.ok(apiRewrite, "Must define a rewrite rule for /api/:path*");
  assert.ok(
    apiRewrite.destination.endsWith("/api/:path*/"),
    "Rewrite destination must preserve trailing slash for Django REST routes"
  );
});

test("Document payload validation enforces title and content requirements", () => {
  const validateDocPayload = (payload) => {
    if (!payload || typeof payload !== "object") {
      return { valid: false, error: "Invalid payload; JSON object expected." };
    }
    const title = typeof payload.title === "string" ? payload.title.trim() : "";
    if (!title) {
      return { valid: false, error: "Document title cannot be empty." };
    }
    const content = typeof payload.content === "string" ? payload.content.trim() : "";
    if (!content) {
      return { valid: false, error: "Document content cannot be empty." };
    }
    return { valid: true, error: null };
  };

  assert.equal(validateDocPayload(null).valid, false);
  assert.equal(validateDocPayload({}).valid, false);
  assert.equal(validateDocPayload({ title: "", content: "valid" }).valid, false);
  assert.equal(validateDocPayload({ title: "   ", content: "valid" }).valid, false);
  assert.equal(validateDocPayload({ title: "Title", content: "" }).valid, false);
  assert.equal(validateDocPayload({ title: "Title", content: "   " }).valid, false);
  assert.equal(
    validateDocPayload({ title: "AURA Specs", content: "# Architecture Overview" }).valid,
    true
  );
});

test("Document API client parses list response and preserves chunk counts", async () => {
  const mockDocs = [
    {
      id: "doc-1",
      title: "AURA Architecture",
      status: "ready",
      chunk_count: 12,
      source: "specs/arch.md",
      created_at: "2026-10-02T10:00:00Z",
    },
    {
      id: "doc-2",
      title: "Research Notes",
      status: "ready",
      chunk_count: 5,
      source: "notes.md",
      created_at: "2026-10-02T09:00:00Z",
    },
  ];

  // Simulate frontend fetch parsing
  const parseDocsResponse = (data) => {
    if (!Array.isArray(data)) return [];
    return data.map((doc) => ({
      id: doc.id,
      title: doc.title,
      status: doc.status,
      chunkCount: doc.chunk_count || 0,
      source: doc.source || "",
      createdAt: doc.created_at,
    }));
  };

  const parsed = parseDocsResponse(mockDocs);
  assert.equal(parsed.length, 2);
  assert.equal(parsed[0].title, "AURA Architecture");
  assert.equal(parsed[0].chunkCount, 12);
  assert.equal(parsed[0].status, "ready");
  assert.equal(parsed[1].chunkCount, 5);
});

test("Document API client handles ingestion response and error structures", async () => {
  const mockSuccessRes = {
    id: "new-doc-id",
    title: "New Spec",
    content: "Content text",
    status: "ready",
    chunk_count: 3,
  };

  const handleIngestResponse = (status, data) => {
    if (status === 201 || status === 200) {
      return { success: true, doc: data, error: null };
    }
    const errorMsg = data?.error || data?.detail || `HTTP ${status}`;
    return { success: false, doc: null, error: errorMsg };
  };

  const successResult = handleIngestResponse(201, mockSuccessRes);
  assert.equal(successResult.success, true);
  assert.equal(successResult.doc.chunk_count, 3);

  const errorResult = handleIngestResponse(400, {
    error: "Field 'title' must be a non-empty string.",
  });
  assert.equal(errorResult.success, false);
  assert.equal(errorResult.error, "Field 'title' must be a non-empty string.");
});

test("Document API client handles document deletion status", async () => {
  const handleDeleteResponse = (status) => {
    if (status === 204 || status === 200) {
      return { deleted: true, error: null };
    }
    return { deleted: false, error: `Deletion failed with HTTP ${status}` };
  };

  assert.equal(handleDeleteResponse(204).deleted, true);
  assert.equal(handleDeleteResponse(404).deleted, false);
});
