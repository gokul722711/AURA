import test from "node:test";
import assert from "node:assert/strict";

const SUPPORTED_EXTENSIONS = [".txt", ".md", ".pdf", ".docx"];

const validateFileExtension = (filename) => {
  if (!filename || typeof filename !== "string") return false;
  const lower = filename.toLowerCase();
  return SUPPORTED_EXTENSIONS.some((ext) => lower.endsWith(ext));
};

const formatFileSize = (bytes) => {
  if (bytes === null || bytes === undefined || isNaN(bytes)) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const getDocTypeBadge = (sourceType) => {
  switch ((sourceType || "").toLowerCase()) {
    case "pdf":
      return { label: "PDF", className: "doc-type-badge badge-pdf" };
    case "docx":
      return { label: "DOCX", className: "doc-type-badge badge-docx" };
    case "markdown":
    case "md":
      return { label: "Markdown", className: "doc-type-badge badge-md" };
    default:
      return { label: "TXT", className: "doc-type-badge badge-txt" };
  }
};

const formatPagesList = (pages) => {
  if (!pages || !Array.isArray(pages) || pages.length === 0) return null;
  return `Pages: ${pages.join(", ")}`;
};

test("M11: File extension validator accepts supported formats and rejects unsupported ones", () => {
  assert.equal(validateFileExtension("research_paper.pdf"), true);
  assert.equal(validateFileExtension("REPORT.PDF"), true);
  assert.equal(validateFileExtension("notes.md"), true);
  assert.equal(validateFileExtension("data.txt"), true);
  assert.equal(validateFileExtension("specification.docx"), true);
  assert.equal(validateFileExtension("DOC.DOCX"), true);

  assert.equal(validateFileExtension("script.py"), false);
  assert.equal(validateFileExtension("malicious.exe"), false);
  assert.equal(validateFileExtension("data.csv"), false);
  assert.equal(validateFileExtension("image.png"), false);
  assert.equal(validateFileExtension("archive.zip"), false);
  assert.equal(validateFileExtension(""), false);
  assert.equal(validateFileExtension(null), false);
});

test("M11: File size formatting helper displays human-readable units", () => {
  assert.equal(formatFileSize(500), "500 B");
  assert.equal(formatFileSize(1024), "1.0 KB");
  assert.equal(formatFileSize(1536), "1.5 KB");
  assert.equal(formatFileSize(1048576), "1.0 MB");
  assert.equal(formatFileSize(5242880), "5.0 MB");
  assert.equal(formatFileSize(null), "");
  assert.equal(formatFileSize(undefined), "");
});

test("M11: Document type badge resolver returns correct labels and styling classes", () => {
  const pdfBadge = getDocTypeBadge("pdf");
  assert.equal(pdfBadge.label, "PDF");
  assert.ok(pdfBadge.className.includes("badge-pdf"));

  const docxBadge = getDocTypeBadge("docx");
  assert.equal(docxBadge.label, "DOCX");
  assert.ok(docxBadge.className.includes("badge-docx"));

  const mdBadge = getDocTypeBadge("markdown");
  assert.equal(mdBadge.label, "Markdown");
  assert.ok(mdBadge.className.includes("badge-md"));

  const txtBadge = getDocTypeBadge("text");
  assert.equal(txtBadge.label, "TXT");
  assert.ok(txtBadge.className.includes("badge-txt"));

  const defaultBadge = getDocTypeBadge(null);
  assert.equal(defaultBadge.label, "TXT");
});

test("M11: Source pages formatter correctly formats page coverage", () => {
  assert.equal(formatPagesList([1, 2, 7]), "Pages: 1, 2, 7");
  assert.equal(formatPagesList([42]), "Pages: 42");
  assert.equal(formatPagesList([]), null);
  assert.equal(formatPagesList(null), null);
  assert.equal(formatPagesList(undefined), null);
});

test("M11: Document API parses extended document fields from backend", () => {
  const apiDoc = {
    id: "doc-pdf-uuid",
    title: "Quantum Cryptography Overview",
    source_type: "pdf",
    filename: "quantum_crypto.pdf",
    file_size: 204850,
    status: "ready",
    chunk_count: 8,
    metadata: {
      page_count: 4,
    },
    created_at: "2026-10-03T10:00:00Z",
  };

  assert.equal(apiDoc.source_type, "pdf");
  assert.equal(apiDoc.filename, "quantum_crypto.pdf");
  assert.equal(apiDoc.file_size, 204850);
  assert.equal(apiDoc.chunk_count, 8);
  assert.equal(apiDoc.metadata.page_count, 4);

  const badge = getDocTypeBadge(apiDoc.source_type);
  assert.equal(badge.label, "PDF");

  const formattedSize = formatFileSize(apiDoc.file_size);
  assert.equal(formattedSize, "200.0 KB");
});
