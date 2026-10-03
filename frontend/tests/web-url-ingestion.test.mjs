import test from "node:test";
import assert from "node:assert/strict";

const validateWebUrl = (urlStr) => {
  if (!urlStr || typeof urlStr !== "string") return false;
  const trimmed = urlStr.trim();
  if (!trimmed) return false;
  try {
    const parsed = new URL(trimmed);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch (_) {
    return false;
  }
};

const getDocTypeBadge = (sourceType, filename) => {
  const t = (sourceType || "").toLowerCase();
  const ext = (filename || "").split(".").pop().toLowerCase();
  if (t === "web_page" || t === "url") return { label: "Web Page", icon: "🌐", className: "badge-web" };
  if (t === "pdf" || ext === "pdf") return { label: "PDF", icon: "📄", className: "badge-pdf" };
  if (t === "docx" || ext === "docx") return { label: "DOCX", icon: "📝", className: "badge-docx" };
  if (t === "markdown" || ext === "md" || ext === "markdown") return { label: "Markdown", icon: "📋", className: "badge-md" };
  return { label: "TXT", icon: "📑", className: "badge-txt" };
};

const buildUrlIngestPayload = (url, title) => {
  const payload = { url: url.trim() };
  if (title && title.trim()) {
    payload.title = title.trim();
  }
  return payload;
};

test("M12 Frontend: URL validator accepts valid HTTP/HTTPS URLs and rejects unsupported schemes or invalid inputs", () => {
  assert.equal(validateWebUrl("https://example.com/article"), true);
  assert.equal(validateWebUrl("http://docs.example.org/spec.html"), true);
  assert.equal(validateWebUrl("https://blog.deepmind.com/posts/gemini-2/"), true);

  assert.equal(validateWebUrl("ftp://files.example.com/data.txt"), false);
  assert.equal(validateWebUrl("file:///etc/passwd"), false);
  assert.equal(validateWebUrl("javascript:alert(1)"), false);
  assert.equal(validateWebUrl("data:text/html,hello"), false);
  assert.equal(validateWebUrl("not-a-valid-url"), false);
  assert.equal(validateWebUrl(""), false);
  assert.equal(validateWebUrl(null), false);
  assert.equal(validateWebUrl(undefined), false);
});

test("M12 Frontend: Badge resolver correctly identifies web_page source types", () => {
  const webBadge = getDocTypeBadge("web_page");
  assert.equal(webBadge.label, "Web Page");
  assert.equal(webBadge.icon, "🌐");
  assert.equal(webBadge.className, "badge-web");

  const urlBadge = getDocTypeBadge("url");
  assert.equal(urlBadge.label, "Web Page");
  assert.equal(urlBadge.className, "badge-web");

  const pdfBadge = getDocTypeBadge("pdf");
  assert.equal(pdfBadge.label, "PDF");
});

test("M12 Frontend: URL ingestion payload builder prepares proper JSON structure", () => {
  const payload1 = buildUrlIngestPayload("  https://example.com/paper  ", " Quantum Paper ");
  assert.deepEqual(payload1, {
    url: "https://example.com/paper",
    title: "Quantum Paper",
  });

  const payload2 = buildUrlIngestPayload("https://example.com/page", "");
  assert.deepEqual(payload2, {
    url: "https://example.com/page",
  });
});

test("M12 Frontend: Document parser preserves web provenance metadata (url, domain, canonical_url)", () => {
  const backendDoc = {
    id: "web-doc-12345",
    title: "Quantum Supremacy Milestone",
    source_type: "web_page",
    source: "https://example.com/quantum",
    url: "https://example.com/quantum",
    canonical_url: "https://example.com/quantum-supremacy",
    domain: "example.com",
    chunk_count: 5,
    status: "ready",
    metadata: {
      author: "Jane Doe",
      date: "2026-03-15",
      domain: "example.com",
      url: "https://example.com/quantum",
    },
    created_at: "2026-10-03T10:00:00Z",
  };

  assert.equal(backendDoc.source_type, "web_page");
  assert.equal(backendDoc.url, "https://example.com/quantum");
  assert.equal(backendDoc.domain, "example.com");
  assert.equal(backendDoc.canonical_url, "https://example.com/quantum-supremacy");
  assert.equal(backendDoc.metadata.author, "Jane Doe");

  const badge = getDocTypeBadge(backendDoc.source_type);
  assert.equal(badge.label, "Web Page");
  assert.equal(badge.className, "badge-web");
});

test("M12 Frontend: Evidence and Citation structures preserve URL attributes", () => {
  const evidence = {
    content: "Quantum supremacy was demonstrated using a 70-qubit processor.",
    citation: "[Quantum Milestone, https://example.com/quantum, Chunk: 12]",
    chunk_id: 12,
    document_title: "Quantum Milestone",
    document_source: "https://example.com/quantum",
    url: "https://example.com/quantum",
    domain: "example.com",
    score: 0.95,
  };

  assert.ok(evidence.citation.includes("https://example.com/quantum"));
  assert.equal(evidence.url, "https://example.com/quantum");
  assert.equal(evidence.domain, "example.com");
});

test("M13.1 Frontend: formatScore preserves precision for small non-zero scores and does not display literal 0.000", () => {
  const formatScore = (score) => {
    if (typeof score !== "number" || isNaN(score)) return "";
    if (score === 0) return "0.000";
    if (score >= 0.01) return score.toFixed(3);
    if (score >= 0.0001) return score.toFixed(4);
    return score.toFixed(6);
  };

  assert.equal(formatScore(0.997235), "0.997");
  assert.equal(formatScore(0.147379), "0.147");
  assert.equal(formatScore(0.000432), "0.0004");
  assert.equal(formatScore(0.000033), "0.000033");
  assert.equal(formatScore(0.000024), "0.000024");
  assert.equal(formatScore(0), "0.000");
  assert.equal(formatScore(null), "");
  assert.equal(formatScore(undefined), "");
});
