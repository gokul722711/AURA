import test from "node:test";
import assert from "node:assert/strict";

import {
  formatMode,
  getModeDescription,
  formatFileSize,
  getDocTypeBadge,
  formatScore,
  formatDuration,
  formatHistoryDuration,
  formatProviderBadge,
  resolveActiveProfile,
  validateWebUrl,
  validateFileExtension,
  groupHistoryByDate,
  getContextualSuggestions,
  getDynamicActivityText,
} from "../src/utils/helpers.js";

test("Helpers: formatMode formats all research modes canonically", () => {
  assert.equal(formatMode("knowledge_base"), "Knowledge Base");
  assert.equal(formatMode("web"), "Web");
  assert.equal(formatMode("web_knowledge_base"), "Web + KB");
  assert.equal(formatMode("model_knowledge"), "Model Knowledge");
  assert.equal(formatMode(null), "Knowledge Base");
  assert.equal(formatMode("custom"), "custom");
});

test("Helpers: getModeDescription provides truthful mode summaries", () => {
  assert.ok(getModeDescription("knowledge_base").includes("internal documents"));
  assert.ok(getModeDescription("web").includes("SearXNG"));
  assert.ok(getModeDescription("web_knowledge_base").includes("hybrid"));
  assert.ok(getModeDescription("model_knowledge").includes("pretrained LLM"));
});

test("Helpers: formatFileSize handles bytes, KB, MB, and edge cases", () => {
  assert.equal(formatFileSize(0), "0 B");
  assert.equal(formatFileSize(512), "512 B");
  assert.equal(formatFileSize(1024), "1.0 KB");
  assert.equal(formatFileSize(204850), "200.0 KB");
  assert.equal(formatFileSize(5242880), "5.0 MB");
  assert.equal(formatFileSize(null), "");
  assert.equal(formatFileSize(undefined), "");
  assert.equal(formatFileSize(NaN), "");
});

test("Helpers: getDocTypeBadge handles all supported file and web types", () => {
  const pdf = getDocTypeBadge("pdf", "spec.pdf");
  assert.equal(pdf.label, "PDF");
  assert.equal(pdf.className, "badge-pdf");

  const docx = getDocTypeBadge("docx", "doc.docx");
  assert.equal(docx.label, "DOCX");
  assert.equal(docx.className, "badge-docx");

  const md = getDocTypeBadge("markdown", "readme.md");
  assert.equal(md.label, "Markdown");
  assert.equal(md.className, "badge-md");

  const txt = getDocTypeBadge("text", "data.txt");
  assert.equal(txt.label, "TXT");
  assert.equal(txt.className, "badge-txt");

  const web = getDocTypeBadge("web_page", "https://example.com");
  assert.equal(web.label, "Web Page");
  assert.equal(web.className, "badge-web");
});

test("Helpers: formatScore formats similarity and rerank scores with precision", () => {
  assert.equal(formatScore(0), "0.000");
  assert.equal(formatScore(0.87654), "0.877");
  assert.equal(formatScore(0.000432), "0.0004");
  assert.equal(formatScore(0.000033), "0.000033");
  assert.equal(formatScore(null), "");
});

test("Helpers: formatDuration and formatHistoryDuration produce canonical second strings", () => {
  assert.equal(formatDuration({ duration_seconds: 14.2 }), "14.20s");
  assert.equal(formatDuration({ duration_ms: 8500 }), "8.50s");
  assert.equal(formatDuration(null), null);

  assert.equal(formatHistoryDuration({ duration_seconds: 14.2 }), "14.2s");
  assert.equal(formatHistoryDuration({ duration_ms: 8500 }), "8.5s");
  assert.equal(formatHistoryDuration({ duration_seconds: 0 }), null);
});

test("Helpers: formatProviderBadge handles NVIDIA, Ollama, OpenAI-compatible, and custom", () => {
  assert.equal(formatProviderBadge("nvidia"), "NVIDIA NIM");
  assert.equal(formatProviderBadge("ollama"), "Ollama");
  assert.equal(formatProviderBadge("openai_compatible"), "OpenAI Compatible");
  assert.equal(formatProviderBadge("mock"), "Mock Provider");
  assert.equal(formatProviderBadge("vllm"), "vllm");
  assert.equal(formatProviderBadge(null), "Custom");
});

test("Helpers: resolveActiveProfile finds active profile or safe fallback", () => {
  const list = [
    { id: "1", name: "Local", is_active: false },
    { id: "2", name: "Hosted", is_active: true },
  ];
  assert.equal(resolveActiveProfile(list).id, "2");

  const noneActive = [
    { id: "1", name: "Local", is_active: false },
    { id: "2", name: "Hosted", is_active: false },
  ];
  assert.equal(resolveActiveProfile(noneActive).id, "1");
  assert.equal(resolveActiveProfile([]), null);
  assert.equal(resolveActiveProfile(null), null);
});

test("Helpers: validateWebUrl and validateFileExtension check inputs properly", () => {
  assert.equal(validateWebUrl("https://example.com/article"), true);
  assert.equal(validateWebUrl("http://localhost:8000"), true);
  assert.equal(validateWebUrl("ftp://invalid.com"), false);
  assert.equal(validateWebUrl("not-a-url"), false);

  assert.equal(validateFileExtension("paper.pdf"), true);
  assert.equal(validateFileExtension("notes.md"), true);
  assert.equal(validateFileExtension("report.docx"), true);
  assert.equal(validateFileExtension("readme.txt"), true);
  assert.equal(validateFileExtension("script.py"), false);
});

test("Helpers: groupHistoryByDate groups runs into Today, Yesterday, and Earlier correctly", () => {
  const now = new Date();
  const todayIso = new Date(now.getTime() - 1000 * 60 * 30).toISOString(); // 30 min ago
  const yesterdayIso = new Date(now.getTime() - 1000 * 60 * 60 * 26).toISOString(); // 26h ago
  const earlierIso = new Date(now.getTime() - 1000 * 60 * 60 * 72).toISOString(); // 3 days ago

  const runs = [
    { run_id: "r1", objective: "Today run", created_at: todayIso },
    { run_id: "r2", objective: "Yesterday run", created_at: yesterdayIso },
    { run_id: "r3", objective: "Earlier run", created_at: earlierIso },
    { run_id: "r4", objective: "Undated run", created_at: null },
  ];

  const grouped = groupHistoryByDate(runs);
  assert.equal(grouped.Today.length, 1);
  assert.equal(grouped.Today[0].run_id, "r1");
  assert.equal(grouped.Yesterday.length, 1);
  assert.equal(grouped.Yesterday[0].run_id, "r2");
  assert.equal(grouped.Earlier.length, 2);
  assert.equal(grouped.Earlier[0].run_id, "r3");
  assert.equal(grouped.Earlier[1].run_id, "r4");
});

test("Spherical Node Geometry: Fibonacci distribution generates valid points on sphere", () => {
  const nodeCount = 50;
  const radius = 100;
  const phi = Math.PI * (3 - Math.sqrt(5));

  for (let i = 0; i < nodeCount; i++) {
    const y = 1 - (i / (nodeCount - 1)) * 2;
    const radiusAtY = Math.sqrt(1 - y * y);
    const theta = phi * i;

    const x = Math.cos(theta) * radiusAtY;
    const z = Math.sin(theta) * radiusAtY;

    const px = x * radius;
    const py = y * radius;
    const pz = z * radius;

    // Euclidean distance from origin must equal radius within floating point tolerance
    const dist = Math.sqrt(px * px + py * py + pz * pz);
    assert.ok(Math.abs(dist - radius) < 1e-4, `Point ${i} distance ${dist} matches radius ${radius}`);
  }
});

test("Contextual Suggestions: gracefully returns empty array when no context exists", () => {
  assert.deepEqual(getContextualSuggestions([], []), []);
  assert.deepEqual(getContextualSuggestions(null, null), []);
});

test("Contextual Suggestions: extracts topics from research history and uploaded documents", () => {
  const history = [
    { run_id: "r1", objective: "How does RAG work?" },
    { run_id: "r2", objective: "What is quantum computing?" },
  ];
  const docs = [
    { id: 1, title: "AURA Architecture Specifications.pdf" },
  ];

  const suggestions = getContextualSuggestions(history, docs);
  assert.ok(suggestions.length > 0 && suggestions.length <= 3);

  // First suggestion should relate to recent history
  assert.equal(suggestions[0].type, "history");
  assert.ok(suggestions[0].text.includes("RAG"));

  // Next suggestions should incorporate document context
  const docSugg = suggestions.find((s) => s.type === "document");
  assert.ok(docSugg);
  assert.ok(docSugg.text.includes("AURA Architecture"));
});

test("Dynamic Activity Text: returns truthful mode-specific progress text", () => {
  // Model Knowledge mode
  assert.equal(getDynamicActivityText("model_knowledge", 1), "Working from model knowledge");
  assert.equal(getDynamicActivityText("model_knowledge", 8), "Synthesizing research response");

  // Knowledge Base mode with documents
  const docs = [{ title: "AURA Specs.pdf" }];
  assert.ok(getDynamicActivityText("knowledge_base", 1, docs).includes("AURA Specs"));
  assert.equal(getDynamicActivityText("knowledge_base", 4, docs), "Retrieving relevant passages");
  assert.equal(getDynamicActivityText("knowledge_base", 9, docs), "Synthesizing knowledge base findings");

  // Web mode
  assert.equal(getDynamicActivityText("web", 2), "Searching the web");
  assert.equal(getDynamicActivityText("web", 5), "Reviewing retrieved web sources");
  assert.equal(getDynamicActivityText("web", 9), "Synthesizing findings");

  // Web + KB hybrid mode
  assert.equal(getDynamicActivityText("web_knowledge_base", 1, docs), "Searching the web");
  assert.ok(getDynamicActivityText("web_knowledge_base", 4, docs).includes("1 knowledge base documents"));
  assert.equal(getDynamicActivityText("web_knowledge_base", 8, docs), "Cross-checking sources & synthesizing");
});

