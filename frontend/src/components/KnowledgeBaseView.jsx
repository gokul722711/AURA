"use client";

import { useRef, useState } from "react";
import { formatFileSize, getDocTypeBadge } from "../utils/helpers";
import {
  RefreshIcon,
  TrashIcon,
  FileIcon,
  GlobeIcon,
  WarningIcon,
  CheckIcon,
  ExternalLinkIcon,
} from "./Icons";

const SAMPLE_MARKDOWN = {
  title: "AURA Architecture & Knowledge System",
  source: "specs/aura-architecture.md",
  content: `# AURA Architecture & Knowledge System

AURA (Autonomous Research & Engineering Agent) is an open-model-first, LLM-agnostic agentic AI platform.

## System Milestones
- **M1 Model Gateway**: Provider-agnostic inference abstraction routing requests to local or hosted models.
- **M2-M3 RAG Engine**: Character chunking, pgvector embedding storage, and similarity threshold retrieval.
- **M4-M6 Autonomous Research**: Multi-step iterative reasoning loop accumulating evidence and verifying citations.
- **M7-M8 Application Layer**: Minimal Django REST APIs and Next.js interface for research queries and knowledge document ingestion.
- **M9 Async Engine**: Persistent research run tracking via Celery and Redis with complete historical trace auditability.

## Ingestion Pipeline
When a document is ingested, it is segmented into chunks, embedded with 384-dimensional vectors, and stored in PostgreSQL with pgvector for instant vector similarity retrieval.`,
};

/**
 * KnowledgeBaseView
 *
 * Workspace for managing knowledge documents, file uploads, web page crawls, and pgvector embeddings.
 */
export default function KnowledgeBaseView({
  documents = [],
  loading = false,
  error = null,
  onRefresh,
  onAddDocument,
  onDeleteDocument,
  onPromptFromDoc,
  getApiUrl,
}) {
  const [showAddForm, setShowAddForm] = useState(false);
  const [ingestMode, setIngestMode] = useState("file"); // 'file' | 'url' | 'paste'
  const [selectedFile, setSelectedFile] = useState(null);
  const [newTitle, setNewTitle] = useState("");
  const [newUrl, setNewUrl] = useState("");
  const [newContent, setNewContent] = useState("");
  const [newSource, setNewSource] = useState("");
  const [ingesting, setIngesting] = useState(false);
  const [addError, setAddError] = useState(null);
  const [ingestSuccess, setIngestSuccess] = useState(null);

  const [expandedDocId, setExpandedDocId] = useState(null);
  const [docDetails, setDocDetails] = useState({});
  const [deleteConfirmId, setDeleteConfirmId] = useState(null);
  const [deletingId, setDeletingId] = useState(null);

  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setAddError(null);
      if (!newTitle.trim()) {
        const baseName = (file.name || "").replace(/\.[^/.]+$/, "");
        setNewTitle(baseName || "");
      }
      if (!newSource.trim()) {
        setNewSource(file.name || "");
      }
    }
  };

  const loadSampleMarkdown = () => {
    setNewTitle(SAMPLE_MARKDOWN.title);
    setNewSource(SAMPLE_MARKDOWN.source);
    setNewContent(SAMPLE_MARKDOWN.content);
    setAddError(null);
  };

  const handleFormSubmit = async (e) => {
    e.preventDefault();
    setAddError(null);
    setIngestSuccess(null);

    if (ingestMode === "file") {
      if (!selectedFile) {
        setAddError("Please select a document file (.txt, .md, .pdf, or .docx).");
        return;
      }
      const ext = selectedFile.name.split(".").pop().toLowerCase();
      if (!["txt", "md", "pdf", "docx"].includes(ext)) {
        setAddError("Unsupported file format. Supported formats are: .txt, .md, .pdf, .docx.");
        return;
      }

      setIngesting(true);
      try {
        const formData = new FormData();
        formData.append("file", selectedFile);
        if (newTitle.trim()) formData.append("title", newTitle.trim());
        if (newSource.trim()) formData.append("source", newSource.trim());

        const res = await onAddDocument(formData, true);
        setIngestSuccess({
          id: res.id,
          title: res.title,
          chunkCount: res.chunk_count,
          sourceType: res.source_type,
        });

        setSelectedFile(null);
        if (fileInputRef.current) fileInputRef.current.value = "";
        setNewTitle("");
        setNewContent("");
        setNewSource("");
        setShowAddForm(false);
      } catch (err) {
        setAddError(err.message || "File ingestion failed.");
      } finally {
        setIngesting(false);
      }
    } else if (ingestMode === "url") {
      const urlTrimmed = newUrl.trim();
      if (!urlTrimmed) {
        setAddError("Please enter a public web page URL.");
        return;
      }
      try {
        const parsed = new URL(urlTrimmed);
        if (!["http:", "https:"].includes(parsed.protocol)) {
          setAddError("Only HTTP and HTTPS URLs are supported.");
          return;
        }
      } catch (_) {
        setAddError("Please enter a valid URL (e.g. https://example.com/article).");
        return;
      }

      setIngesting(true);
      try {
        const payload = { url: urlTrimmed };
        if (newTitle.trim()) payload.title = newTitle.trim();

        const res = await onAddDocument(payload, false);
        setIngestSuccess({
          id: res.id,
          title: res.title,
          chunkCount: res.chunk_count,
          sourceType: res.source_type,
          url: res.url,
        });

        setNewUrl("");
        setNewTitle("");
        setShowAddForm(false);
      } catch (err) {
        setAddError(err.message || "URL ingestion failed.");
      } finally {
        setIngesting(false);
      }
    } else {
      const titleTrimmed = newTitle.trim();
      const contentTrimmed = newContent.trim();

      if (!titleTrimmed) {
        setAddError("Document title cannot be empty.");
        return;
      }
      if (!contentTrimmed) {
        setAddError("Document content cannot be empty.");
        return;
      }

      setIngesting(true);
      try {
        const payload = {
          title: titleTrimmed,
          content: contentTrimmed,
          source: newSource.trim() || undefined,
        };

        const res = await onAddDocument(payload, false);
        setIngestSuccess({
          id: res.id,
          title: res.title,
          chunkCount: res.chunk_count,
          sourceType: res.source_type,
        });

        setNewTitle("");
        setNewContent("");
        setNewSource("");
        setShowAddForm(false);
      } catch (err) {
        setAddError(err.message || "Document ingestion failed.");
      } finally {
        setIngesting(false);
      }
    }
  };

  const handleTogglePreview = async (docId) => {
    if (expandedDocId === docId) {
      setExpandedDocId(null);
      return;
    }
    setExpandedDocId(docId);

    if (!docDetails[docId]) {
      try {
        const res = await fetch(`${getApiUrl("documents")}${docId}/`);
        if (res.ok) {
          const detail = await res.json();
          setDocDetails((prev) => ({ ...prev, [docId]: detail }));
        }
      } catch (err) {
        console.error("Failed to load document content detail:", err);
      }
    }
  };

  const handleDelete = async (docId) => {
    setDeletingId(docId);
    try {
      await onDeleteDocument(docId);
      setDeleteConfirmId(null);
      if (expandedDocId === docId) {
        setExpandedDocId(null);
      }
    } catch (err) {
      console.error("Deletion failed:", err);
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="kb-workspace-container">
      {/* Header & Action Bar */}
      <header className="workspace-header">
        <div>
          <h2 className="workspace-title">Knowledge Base</h2>
          <p className="workspace-subtitle">
            Indexed documents and web pages embedded in pgvector for grounded retrieval.
          </p>
        </div>
        <div className="workspace-header-actions">
          <button
            type="button"
            className="btn-workspace-refresh"
            onClick={onRefresh}
            disabled={loading}
            title="Refresh documents"
          >
            <RefreshIcon size={13} />
            <span>Refresh</span>
          </button>
          <button
            type="button"
            id="btn-add-document"
            className="btn-primary-action"
            onClick={() => {
              setShowAddForm(!showAddForm);
              setAddError(null);
              setIngestSuccess(null);
            }}
          >
            {showAddForm ? "Close Form" : "+ Add Document"}
          </button>
        </div>
      </header>

      {/* Success Notification Banner */}
      {ingestSuccess && (
        <div className="kb-success-banner" role="status" aria-live="polite">
          <span className="success-badge-icon" aria-hidden="true">
            <CheckIcon size={14} />
          </span>
          <div className="success-content">
            <h4 className="success-title">Document Ingested Successfully</h4>
            <p className="success-message">
              &ldquo;{ingestSuccess.title}&rdquo; was processed into {ingestSuccess.chunkCount} vector chunk(s) and is immediately searchable.
            </p>
          </div>
          <button
            type="button"
            className="btn-success-shortcut"
            onClick={() => onPromptFromDoc(ingestSuccess.title)}
          >
            Research this document →
          </button>
        </div>
      )}

      {/* Ingestion Panel Form */}
      {showAddForm && (
        <section className="kb-form-panel" aria-label="Ingest Document Form">
          <div className="kb-form-header">
            <h3 className="kb-form-heading">Ingest Knowledge</h3>
            {ingestMode === "paste" && (
              <button
                type="button"
                className="btn-sample-load"
                onClick={loadSampleMarkdown}
                disabled={ingesting}
              >
                <FileIcon size={13} />
                <span>Load Sample Markdown</span>
              </button>
            )}
          </div>

          {/* Mode Switcher */}
          <div className="kb-mode-pills" role="tablist">
            <button
              type="button"
              role="tab"
              aria-selected={ingestMode === "file"}
              className={`mode-tab-btn ${ingestMode === "file" ? "active" : ""}`}
              onClick={() => {
                setIngestMode("file");
                setAddError(null);
              }}
              disabled={ingesting}
            >
              <FileIcon size={13} />
              <span>File (.txt, .md, .pdf, .docx)</span>
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={ingestMode === "url"}
              className={`mode-tab-btn ${ingestMode === "url" ? "active" : ""}`}
              onClick={() => {
                setIngestMode("url");
                setAddError(null);
              }}
              disabled={ingesting}
            >
              <GlobeIcon size={13} />
              <span>Web Page URL</span>
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={ingestMode === "paste"}
              className={`mode-tab-btn ${ingestMode === "paste" ? "active" : ""}`}
              onClick={() => {
                setIngestMode("paste");
                setAddError(null);
              }}
              disabled={ingesting}
            >
              <FileIcon size={13} />
              <span>Paste Text / MD</span>
            </button>
          </div>

          <form onSubmit={handleFormSubmit} className="kb-form-body">
            {ingestMode === "file" ? (
              <>
                <div className="form-field">
                  <label htmlFor="doc-file-input" className="form-label">
                    Document File <span className="field-required">*</span>
                    <span className="field-hint"> (.txt, .md, .pdf, .docx up to 20MB)</span>
                  </label>
                  <input
                    id="doc-file-input"
                    ref={fileInputRef}
                    type="file"
                    accept=".txt,.md,.pdf,.docx"
                    className="file-input-field"
                    onChange={handleFileChange}
                    disabled={ingesting}
                    required
                  />
                  {selectedFile && (
                    <div className="file-preview-pill">
                      <span className="file-name-with-icon">
                        <FileIcon size={13} />
                        <span>{selectedFile.name}</span>
                      </span>
                      <span className="file-size-tag">({formatFileSize(selectedFile.size)})</span>
                    </div>
                  )}
                </div>

                <div className="form-field">
                  <label htmlFor="doc-title-input" className="form-label">
                    Document Title <span className="field-hint">(optional, defaults to filename)</span>
                  </label>
                  <input
                    id="doc-title-input"
                    type="text"
                    className="text-input-field"
                    placeholder="e.g. Quantum Processor Architecture"
                    value={newTitle}
                    onChange={(e) => setNewTitle(e.target.value)}
                    disabled={ingesting}
                  />
                </div>

                <div className="form-field">
                  <label htmlFor="doc-source-input" className="form-label">
                    Source Identifier <span className="field-hint">(optional origin or path)</span>
                  </label>
                  <input
                    id="doc-source-input"
                    type="text"
                    className="text-input-field"
                    placeholder="e.g. specs/quantum.pdf"
                    value={newSource}
                    onChange={(e) => setNewSource(e.target.value)}
                    disabled={ingesting}
                  />
                </div>
              </>
            ) : ingestMode === "url" ? (
              <>
                <div className="form-field">
                  <label htmlFor="doc-url-input" className="form-label">
                    Web Page URL <span className="field-required">*</span>
                    <span className="field-hint"> (Public HTTP or HTTPS article/doc page)</span>
                  </label>
                  <input
                    id="doc-url-input"
                    type="url"
                    className="text-input-field"
                    placeholder="https://example.com/article"
                    value={newUrl}
                    onChange={(e) => setNewUrl(e.target.value)}
                    disabled={ingesting}
                    required
                  />
                </div>

                <div className="form-field">
                  <label htmlFor="doc-url-title-input" className="form-label">
                    Document Title <span className="field-hint">(optional, defaults to page title)</span>
                  </label>
                  <input
                    id="doc-url-title-input"
                    type="text"
                    className="text-input-field"
                    placeholder="e.g. Quantum Computing Milestone"
                    value={newTitle}
                    onChange={(e) => setNewTitle(e.target.value)}
                    disabled={ingesting}
                  />
                </div>
              </>
            ) : (
              <>
                <div className="form-field">
                  <label htmlFor="doc-paste-title-input" className="form-label">
                    Title <span className="field-required">*</span>
                  </label>
                  <input
                    id="doc-paste-title-input"
                    type="text"
                    className="text-input-field"
                    placeholder="e.g. AURA Architecture Specifications"
                    value={newTitle}
                    onChange={(e) => setNewTitle(e.target.value)}
                    disabled={ingesting}
                    required
                  />
                </div>

                <div className="form-field">
                  <label htmlFor="doc-paste-source-input" className="form-label">
                    Source Identifier <span className="field-hint">(optional file path, URL, or tag)</span>
                  </label>
                  <input
                    id="doc-paste-source-input"
                    type="text"
                    className="text-input-field"
                    placeholder="e.g. docs/architecture.md"
                    value={newSource}
                    onChange={(e) => setNewSource(e.target.value)}
                    disabled={ingesting}
                  />
                </div>

                <div className="form-field">
                  <label htmlFor="doc-content-input" className="form-label">
                    Content <span className="field-required">*</span>
                  </label>
                  <textarea
                    id="doc-content-input"
                    className="textarea-field"
                    placeholder="Paste or write Markdown/plain text content here..."
                    value={newContent}
                    onChange={(e) => setNewContent(e.target.value)}
                    disabled={ingesting}
                    rows={7}
                    required
                  />
                </div>
              </>
            )}

            {addError && (
              <div className="error-notice-banner" role="alert">
                <WarningIcon size={14} />
                <span>{addError}</span>
              </div>
            )}

            <div className="form-actions-row">
              <button
                type="button"
                className="btn-cancel-form"
                onClick={() => {
                  setShowAddForm(false);
                  setSelectedFile(null);
                  setNewUrl("");
                  setAddError(null);
                }}
                disabled={ingesting}
              >
                Cancel
              </button>
              <button
                type="submit"
                id="btn-submit-document"
                className="btn-submit-ingest"
                disabled={
                  ingesting ||
                  (ingestMode === "file" && !selectedFile) ||
                  (ingestMode === "url" && !newUrl.trim()) ||
                  (ingestMode === "paste" && (!newTitle.trim() || !newContent.trim()))
                }
              >
                {ingesting ? (
                  <>
                    <span className="spinner-micro" aria-hidden="true" />
                    <span>Ingesting &amp; Embedding...</span>
                  </>
                ) : (
                  <>
                    <span>
                      {ingestMode === "file"
                        ? "Upload & Ingest File"
                        : ingestMode === "url"
                        ? "Ingest Web Page"
                        : "Ingest Document"}
                    </span>
                    <span aria-hidden="true">→</span>
                  </>
                )}
              </button>
            </div>
          </form>
        </section>
      )}

      {/* Global Document Error */}
      {error && (
        <div className="error-notice-banner" role="alert">
          <WarningIcon size={14} />
          <span>{error}</span>
        </div>
      )}

      {/* Document List */}
      {loading && documents.length === 0 ? (
        <div className="workspace-loading-card" role="status" aria-live="polite">
          <div className="spinner-large" aria-hidden="true" />
          <p className="loading-caption">Loading Knowledge Base...</p>
        </div>
      ) : documents.length === 0 ? (
        <div className="kb-empty-box" id="kb-empty-state">
          <div className="empty-box-icon" aria-hidden="true">
            <FileIcon size={32} />
          </div>
          <h3 className="empty-box-title">No documents yet</h3>
          <p className="empty-box-desc">
            Ingest text, Markdown, PDF, or web pages to enable AURA to ground research against verified knowledge.
          </p>
          <button
            type="button"
            className="btn-primary-action"
            onClick={() => {
              setShowAddForm(true);
              setIngestMode("paste");
              loadSampleMarkdown();
            }}
          >
            + Ingest Sample Document
          </button>
        </div>
      ) : (
        <div className="kb-documents-grid" id="kb-doc-list">
          {documents.map((doc) => {
            const isExpanded = expandedDocId === doc.id;
            const isDeleting = deletingId === doc.id;
            const isConfirming = deleteConfirmId === doc.id;
            const typeBadge = getDocTypeBadge(doc.source_type, doc.filename || doc.source);
            const detailed = docDetails[doc.id] || doc;

            return (
              <article
                key={doc.id}
                className={`kb-doc-item ${isExpanded ? "doc-item-expanded" : ""}`}
                id={`doc-card-${doc.id}`}
              >
                <div className="doc-item-main">
                  <div className="doc-item-info">
                    <h3 className="doc-item-title">{doc.title}</h3>
                    <div className="doc-meta-row">
                      <span
                        className={`doc-status-tag ${
                          doc.status === "ready"
                            ? "status-ready"
                            : doc.status === "error"
                            ? "status-error"
                            : "status-pending"
                        }`}
                      >
                        ● {doc.status === "ready" ? "Ready" : doc.status}
                      </span>
                      <span className={`doc-badge ${typeBadge.className}`}>
                        {typeBadge.iconType === "web" ? <GlobeIcon size={11} /> : <FileIcon size={11} />}
                        <span>{typeBadge.label}</span>
                      </span>
                      <span className="doc-meta-pill">
                        {doc.chunk_count} chunk{doc.chunk_count === 1 ? "" : "s"}
                      </span>
                      {doc.file_size != null && (
                        <span className="doc-meta-pill">
                          {formatFileSize(doc.file_size)}
                        </span>
                      )}
                      {doc.domain && (
                        <span className="doc-meta-pill">
                          <ExternalLinkIcon size={11} />
                          <span>{doc.domain}</span>
                        </span>
                      )}
                      {doc.source && (
                        <span className="doc-meta-pill doc-meta-source" title={doc.source}>
                          {doc.source_type === "web_page" ? (
                            <GlobeIcon size={11} />
                          ) : (
                            <FileIcon size={11} />
                          )}
                          <span>{doc.source}</span>
                        </span>
                      )}
                      {doc.created_at && (
                        <span className="doc-meta-date">
                          {new Date(doc.created_at).toLocaleDateString(undefined, {
                            month: "short",
                            day: "numeric",
                            year: "numeric",
                          })}
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="doc-item-actions">
                    <button
                      type="button"
                      className="btn-doc-view"
                      onClick={() => handleTogglePreview(doc.id)}
                      aria-expanded={isExpanded}
                      title="Toggle preview"
                    >
                      {isExpanded ? "Hide Preview ▲" : "View Details ▼"}
                    </button>

                    {isConfirming ? (
                      <div className="doc-delete-confirm">
                        <span className="confirm-text">Delete?</span>
                        <button
                          type="button"
                          className="btn-confirm-yes"
                          onClick={() => handleDelete(doc.id)}
                          disabled={isDeleting}
                        >
                          {isDeleting ? "..." : "Yes"}
                        </button>
                        <button
                          type="button"
                          className="btn-confirm-no"
                          onClick={() => setDeleteConfirmId(null)}
                          disabled={isDeleting}
                        >
                          No
                        </button>
                      </div>
                    ) : (
                      <button
                        type="button"
                        className="btn-doc-delete"
                        onClick={() => setDeleteConfirmId(doc.id)}
                        disabled={isDeleting}
                        title="Delete document"
                      >
                        <TrashIcon size={13} />
                        <span>Delete</span>
                      </button>
                    )}
                  </div>
                </div>

                {/* Expanded Details & Content Preview */}
                {isExpanded && (
                  <div className="doc-expanded-panel">
                    <div className="doc-specs-grid">
                      <div>
                        <span className="spec-label">Document ID:</span>
                        <code className="spec-code">{doc.id}</code>
                      </div>
                      <div>
                        <span className="spec-label">Format:</span>
                        <span className="spec-val">{typeBadge.label}</span>
                      </div>
                      {doc.url && (
                        <div>
                          <span className="spec-label">URL:</span>
                          <a
                            href={doc.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="spec-link"
                          >
                            {doc.url}
                          </a>
                        </div>
                      )}
                      {doc.domain && (
                        <div>
                          <span className="spec-label">Domain:</span>
                          <span className="spec-val">{doc.domain}</span>
                        </div>
                      )}
                      {doc.metadata?.author && (
                        <div>
                          <span className="spec-label">Author:</span>
                          <span className="spec-val">{doc.metadata.author}</span>
                        </div>
                      )}
                      {doc.metadata?.date && (
                        <div>
                          <span className="spec-label">Date:</span>
                          <span className="spec-val">{doc.metadata.date}</span>
                        </div>
                      )}
                      {doc.filename && (
                        <div>
                          <span className="spec-label">Filename:</span>
                          <span className="spec-val">{doc.filename}</span>
                        </div>
                      )}
                      {doc.metadata?.page_count && (
                        <div>
                          <span className="spec-label">Pages:</span>
                          <span className="spec-val">{doc.metadata.page_count}</span>
                        </div>
                      )}
                      <div>
                        <span className="spec-label">Vector Storage:</span>
                        <span className="spec-val">{doc.chunk_count} chunks in pgvector</span>
                      </div>
                    </div>

                    <div className="doc-content-preview-wrap">
                      <span className="spec-label">Document Content:</span>
                      <pre className="doc-content-pre">
                        {detailed.content || "Loading document content..."}
                      </pre>
                    </div>

                    <div className="doc-panel-actions">
                      <button
                        type="button"
                        className="btn-ask-doc"
                        onClick={() => onPromptFromDoc(doc.title)}
                      >
                        Ask AURA about this document →
                      </button>
                    </div>
                  </div>
                )}
              </article>
            );
          })}
        </div>
      )}
    </div>
  );
}
