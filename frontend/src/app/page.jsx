"use client";

import { useEffect, useState } from "react";

const SAMPLE_OBJECTIVES = [
  "How do the AURA Model Gateway, RAG pipeline, and Agent Runtime work together?",
  "What mechanisms ensure citation integrity and prevent hallucinated sources?",
  "Explain the architectural boundaries and security constraints of the agent runtime.",
];

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

## Ingestion Pipeline
When a document is ingested, it is segmented into chunks, embedded with 384-dimensional vectors, and stored in PostgreSQL with pgvector for instant vector similarity retrieval.`,
};

export default function Home() {
  // Navigation
  const [activeTab, setActiveTab] = useState("research"); // 'research' | 'knowledge'

  // Research State
  const [objective, setObjective] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  // Knowledge Base State
  const [documents, setDocuments] = useState([]);
  const [docsLoading, setDocsLoading] = useState(false);
  const [docsError, setDocsError] = useState(null);

  // Add Document State
  const [showAddForm, setShowAddForm] = useState(false);
  const [newTitle, setNewTitle] = useState("");
  const [newContent, setNewContent] = useState("");
  const [newSource, setNewSource] = useState("");
  const [ingesting, setIngesting] = useState(false);
  const [addError, setAddError] = useState(null);
  const [ingestSuccess, setIngestSuccess] = useState(null);

  // Document Interaction State
  const [expandedDocId, setExpandedDocId] = useState(null);
  const [deletingId, setDeletingId] = useState(null);
  const [deleteConfirmId, setDeleteConfirmId] = useState(null);

  const getApiUrl = (endpoint) => {
    const base = process.env.NEXT_PUBLIC_API_URL
      ? process.env.NEXT_PUBLIC_API_URL.replace(/\/$/, "")
      : "";
    return `${base}/api/${endpoint}/`;
  };

  // Fetch Documents
  const fetchDocuments = async () => {
    setDocsLoading(true);
    setDocsError(null);
    try {
      const response = await fetch(getApiUrl("documents"));
      if (!response.ok) {
        throw new Error(`Failed to load documents (HTTP ${response.status})`);
      }
      const data = await response.json();
      setDocuments(Array.isArray(data) ? data : []);
    } catch (err) {
      setDocsError(err.message || "Failed to load documents from backend.");
    } finally {
      setDocsLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  // Submit Research Objective
  const handleResearchSubmit = async (e) => {
    e.preventDefault();
    const trimmed = objective.trim();
    if (!trimmed) {
      setError("Please enter a research objective.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const apiUrl = getApiUrl("research");

    try {
      const response = await fetch(apiUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ objective: trimmed }),
      });

      const contentType = response.headers.get("content-type") || "";
      let data = {};
      if (contentType.includes("application/json")) {
        data = await response.json();
      } else {
        const text = await response.text();
        throw new Error(
          `Server returned HTTP ${response.status}: ${text.slice(0, 120)}`
        );
      }

      if (!response.ok) {
        const errorDetail =
          data.detail || data.error || `HTTP error ${response.status}`;
        throw new Error(errorDetail);
      }

      setResult(data);
      if (data.status === "failed" && data.errors && data.errors.length > 0) {
        setError(`Research finished with error: ${data.errors.join("; ")}`);
      }
    } catch (err) {
      setError(err.message || "Failed to communicate with research backend.");
    } finally {
      setLoading(false);
    }
  };

  // Ingest New Document
  const handleAddDocument = async (e) => {
    e.preventDefault();
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
    setAddError(null);
    setIngestSuccess(null);

    try {
      const response = await fetch(getApiUrl("documents"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          title: titleTrimmed,
          content: contentTrimmed,
          source: newSource.trim() || undefined,
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || data.detail || `HTTP ${response.status}`);
      }

      setIngestSuccess({
        id: data.id,
        title: data.title,
        chunkCount: data.chunk_count,
      });

      setNewTitle("");
      setNewContent("");
      setNewSource("");
      setShowAddForm(false);
      await fetchDocuments();
    } catch (err) {
      setAddError(err.message || "Document ingestion failed.");
    } finally {
      setIngesting(false);
    }
  };

  // Delete Document
  const handleDeleteDocument = async (docId) => {
    setDeletingId(docId);
    setDocsError(null);
    try {
      const response = await fetch(`${getApiUrl("documents")}${docId}/`, {
        method: "DELETE",
      });

      if (!response.ok && response.status !== 204) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || `HTTP ${response.status}`);
      }

      setDocuments((prev) => prev.filter((d) => d.id !== docId));
      setDeleteConfirmId(null);
      if (expandedDocId === docId) {
        setExpandedDocId(null);
      }
    } catch (err) {
      setDocsError(err.message || "Failed to delete document.");
    } finally {
      setDeletingId(null);
    }
  };

  // Toggle Document Detail Preview
  const handleToggleDocDetail = async (docId) => {
    if (expandedDocId === docId) {
      setExpandedDocId(null);
      return;
    }
    setExpandedDocId(docId);

    const doc = documents.find((d) => d.id === docId);
    if (!doc || !doc.content) {
      try {
        const res = await fetch(`${getApiUrl("documents")}${docId}/`);
        if (res.ok) {
          const detail = await res.json();
          setDocuments((prev) =>
            prev.map((d) => (d.id === docId ? { ...d, ...detail } : d))
          );
        }
      } catch (err) {
        console.error("Failed to load document content detail:", err);
      }
    }
  };

  const loadSampleMarkdown = () => {
    setNewTitle(SAMPLE_MARKDOWN.title);
    setNewSource(SAMPLE_MARKDOWN.source);
    setNewContent(SAMPLE_MARKDOWN.content);
    setAddError(null);
  };

  const handlePromptFromDoc = (title) => {
    setObjective(`Explain the core concepts and architecture described in "${title}".`);
    setActiveTab("research");
  };

  return (
    <div className="app-container">
      <main className="content-wrapper">
        {/* Header */}
        <header className="header">
          <div className="header-top">
            <h1 className="logo">AURA</h1>
            <span className="phase-pill">M8 — Knowledge Base</span>
          </div>
          <p className="subtitle">
            Autonomous Research &amp; Engineering Agent
          </p>
        </header>

        {/* Tab Navigation */}
        <nav className="nav-tabs" aria-label="Main Navigation">
          <button
            type="button"
            id="tab-research"
            className={`nav-tab ${activeTab === "research" ? "active" : ""}`}
            onClick={() => setActiveTab("research")}
          >
            <span className="tab-icon" aria-hidden="true">🔬</span>
            <span>Autonomous Research</span>
          </button>
          <button
            type="button"
            id="tab-knowledge"
            className={`nav-tab ${activeTab === "knowledge" ? "active" : ""}`}
            onClick={() => setActiveTab("knowledge")}
          >
            <span className="tab-icon" aria-hidden="true">📚</span>
            <span>Knowledge Base</span>
            <span className="tab-count-badge" id="kb-count-badge">
              {documents.length}
            </span>
          </button>
        </nav>

        {/* ================================================================= */}
        {/* TAB 1: KNOWLEDGE BASE VIEW */}
        {/* ================================================================= */}
        {activeTab === "knowledge" && (
          <div className="kb-view-container">
            {/* Action Bar */}
            <div className="kb-action-bar">
              <div>
                <h2 className="kb-section-title">Knowledge Documents</h2>
                <p className="kb-section-subtitle">
                  Text &amp; Markdown documents ingested for pgvector RAG retrieval
                </p>
              </div>
              <button
                type="button"
                id="btn-add-document"
                className="btn-add-doc"
                onClick={() => {
                  setShowAddForm(!showAddForm);
                  setAddError(null);
                  setIngestSuccess(null);
                }}
              >
                {showAddForm ? "✕ Close Form" : "+ Add Document"}
              </button>
            </div>

            {/* Ingest Success Banner */}
            {ingestSuccess && (
              <div className="kb-success-banner" role="status" aria-live="polite">
                <span className="success-icon" aria-hidden="true">✓</span>
                <div className="success-content">
                  <p className="success-title">Document Ingested Successfully</p>
                  <p className="success-message">
                    &ldquo;{ingestSuccess.title}&rdquo; was processed into {ingestSuccess.chunkCount} vector chunk(s) and is immediately searchable.
                  </p>
                </div>
                <button
                  type="button"
                  className="btn-research-shortcut"
                  onClick={() => handlePromptFromDoc(ingestSuccess.title)}
                >
                  Research this document →
                </button>
              </div>
            )}

            {/* Add Document Panel Form */}
            {showAddForm && (
              <section className="card kb-form-card" aria-label="Add Document">
                <div className="kb-form-header">
                  <h3 className="kb-form-title">Ingest Knowledge Document</h3>
                  <button
                    type="button"
                    className="btn-secondary-sm"
                    onClick={loadSampleMarkdown}
                    disabled={ingesting}
                  >
                    📄 Load Sample Markdown
                  </button>
                </div>

                <form onSubmit={handleAddDocument} className="form-group">
                  <div>
                    <label htmlFor="doc-title-input" className="label-text">
                      Title <span className="required-star">*</span>
                    </label>
                    <input
                      id="doc-title-input"
                      type="text"
                      className="text-input"
                      placeholder="e.g. AURA Architecture Specifications"
                      value={newTitle}
                      onChange={(e) => setNewTitle(e.target.value)}
                      disabled={ingesting}
                      required
                    />
                  </div>

                  <div>
                    <label htmlFor="doc-source-input" className="label-text">
                      Source Identifier <span className="label-hint">(optional file path, URL, or tag)</span>
                    </label>
                    <input
                      id="doc-source-input"
                      type="text"
                      className="text-input"
                      placeholder="e.g. docs/architecture.md"
                      value={newSource}
                      onChange={(e) => setNewSource(e.target.value)}
                      disabled={ingesting}
                    />
                  </div>

                  <div>
                    <label htmlFor="doc-content-input" className="label-text">
                      Content (Plain Text or Markdown) <span className="required-star">*</span>
                    </label>
                    <textarea
                      id="doc-content-input"
                      className="textarea"
                      placeholder="Paste or write Markdown/text content here..."
                      value={newContent}
                      onChange={(e) => setNewContent(e.target.value)}
                      disabled={ingesting}
                      rows={8}
                      required
                    />
                  </div>

                  {addError && (
                    <div className="error-banner" role="alert">
                      <span className="error-icon" aria-hidden="true">⚠️</span>
                      <p className="error-message">{addError}</p>
                    </div>
                  )}

                  <div className="kb-form-actions">
                    <button
                      type="button"
                      className="btn-cancel"
                      onClick={() => setShowAddForm(false)}
                      disabled={ingesting}
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      id="btn-submit-document"
                      className="btn-ingest"
                      disabled={ingesting || !newTitle.trim() || !newContent.trim()}
                    >
                      {ingesting ? (
                        <>
                          <span className="spinner" aria-hidden="true" />
                          <span>Ingesting &amp; Embedding...</span>
                        </>
                      ) : (
                        <>
                          <span>Ingest Document</span>
                          <span aria-hidden="true">→</span>
                        </>
                      )}
                    </button>
                  </div>
                </form>
              </section>
            )}

            {/* Document List Errors */}
            {docsError && (
              <div className="error-banner" role="alert">
                <span className="error-icon" aria-hidden="true">⚠️</span>
                <p className="error-message">{docsError}</p>
              </div>
            )}

            {/* Documents List */}
            {docsLoading && documents.length === 0 ? (
              <div className="loading-card" role="status" aria-live="polite">
                <div className="loading-spinner-large" aria-hidden="true" />
                <p className="loading-title">Loading Knowledge Base...</p>
              </div>
            ) : documents.length === 0 ? (
              <div className="kb-empty-state" id="kb-empty-state">
                <div className="kb-empty-icon" aria-hidden="true">📖</div>
                <h3 className="kb-empty-title">No documents yet</h3>
                <p className="kb-empty-desc">
                  Add text or Markdown knowledge documents to allow AURA to ground research in your domain evidence.
                </p>
                <button
                  type="button"
                  className="btn-add-doc"
                  onClick={() => {
                    setShowAddForm(true);
                    loadSampleMarkdown();
                  }}
                >
                  + Ingest Sample Document
                </button>
              </div>
            ) : (
              <div className="kb-doc-list" id="kb-doc-list">
                {documents.map((doc) => {
                  const isExpanded = expandedDocId === doc.id;
                  const isDeleting = deletingId === doc.id;
                  const isConfirming = deleteConfirmId === doc.id;

                  return (
                    <article
                      key={doc.id}
                      className={`kb-doc-card ${isExpanded ? "expanded" : ""}`}
                      id={`doc-card-${doc.id}`}
                    >
                      <div className="kb-doc-main">
                        <div className="kb-doc-info">
                          <h3 className="kb-doc-title">{doc.title}</h3>
                          <div className="kb-doc-meta-row">
                            <span
                              className={`status-pill ${
                                doc.status === "ready"
                                  ? "status-ready"
                                  : doc.status === "error"
                                  ? "status-error"
                                  : "status-pending"
                              }`}
                            >
                              ● {doc.status === "ready" ? "Ready" : doc.status}
                            </span>
                            <span className="chunk-count-badge">
                              {doc.chunk_count} chunk{doc.chunk_count === 1 ? "" : "s"}
                            </span>
                            {doc.source && (
                              <span className="doc-source-tag">
                                📁 {doc.source}
                              </span>
                            )}
                            {doc.created_at && (
                              <span className="doc-date-tag">
                                {new Date(doc.created_at).toLocaleDateString(undefined, {
                                  month: "short",
                                  day: "numeric",
                                  year: "numeric",
                                })}
                              </span>
                            )}
                          </div>
                        </div>

                        <div className="kb-doc-actions">
                          <button
                            type="button"
                            className="btn-action-view"
                            onClick={() => handleToggleDocDetail(doc.id)}
                            aria-expanded={isExpanded}
                            title="Toggle document preview"
                          >
                            {isExpanded ? "Hide Preview ▲" : "View Details ▼"}
                          </button>

                          {isConfirming ? (
                            <div className="delete-confirm-group">
                              <span className="delete-confirm-text">Delete?</span>
                              <button
                                type="button"
                                className="btn-confirm-delete"
                                onClick={() => handleDeleteDocument(doc.id)}
                                disabled={isDeleting}
                              >
                                {isDeleting ? "..." : "Yes"}
                              </button>
                              <button
                                type="button"
                                className="btn-cancel-delete"
                                onClick={() => setDeleteConfirmId(null)}
                                disabled={isDeleting}
                              >
                                No
                              </button>
                            </div>
                          ) : (
                            <button
                              type="button"
                              className="btn-action-delete"
                              onClick={() => setDeleteConfirmId(doc.id)}
                              disabled={isDeleting}
                              title="Delete document"
                            >
                              🗑 Delete
                            </button>
                          )}
                        </div>
                      </div>

                      {/* Expandable Preview */}
                      {isExpanded && (
                        <div className="kb-doc-preview-panel">
                          <div className="preview-meta-grid">
                            <div>
                              <span className="preview-label">Document ID:</span>
                              <code className="preview-code">{doc.id}</code>
                            </div>
                            <div>
                              <span className="preview-label">Chunks Stored:</span>
                              <span className="preview-val">{doc.chunk_count} in pgvector</span>
                            </div>
                          </div>

                          <div className="preview-content-box">
                            <span className="preview-label">Document Content:</span>
                            <pre className="preview-text">
                              {doc.content || "Loading document content..."}
                            </pre>
                          </div>

                          <div className="preview-footer-actions">
                            <button
                              type="button"
                              className="btn-research-shortcut-sm"
                              onClick={() => handlePromptFromDoc(doc.title)}
                            >
                              🔬 Ask AURA about this document →
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
        )}

        {/* ================================================================= */}
        {/* TAB 2: AUTONOMOUS RESEARCH VIEW */}
        {/* ================================================================= */}
        {activeTab === "research" && (
          <div className="research-view-container">
            {/* Input Form Card */}
            <section className="card" aria-label="Research input">
              <form onSubmit={handleResearchSubmit} className="form-group">
                <div className="label">
                  <label htmlFor="objective-input">Research Objective</label>
                  <span className="label-hint">
                    Autonomous iterative loop • {documents.length} doc(s) indexed
                  </span>
                </div>

                <textarea
                  id="objective-input"
                  className="textarea"
                  placeholder="Enter a research objective (e.g. How do the AURA Model Gateway, RAG pipeline, and Agent Runtime work together?)..."
                  value={objective}
                  onChange={(e) => setObjective(e.target.value)}
                  disabled={loading}
                  rows={4}
                />

                {/* Quick Suggestions */}
                <div className="suggestions-row">
                  <span className="suggestions-title">Quick prompts:</span>
                  {SAMPLE_OBJECTIVES.map((sample, idx) => (
                    <button
                      key={idx}
                      type="button"
                      className="btn-suggestion"
                      onClick={() => setObjective(sample)}
                      disabled={loading}
                    >
                      {idx === 0
                        ? "Architecture Overview"
                        : idx === 1
                          ? "Citation Integrity"
                          : "Security & Sandbox"}
                    </button>
                  ))}
                  {documents.length > 0 && (
                    <button
                      type="button"
                      className="btn-suggestion btn-suggestion-kb"
                      onClick={() => handlePromptFromDoc(documents[0].title)}
                      disabled={loading}
                    >
                      📖 &ldquo;{documents[0].title.slice(0, 20)}...&rdquo;
                    </button>
                  )}
                </div>

                <div className="actions-row">
                  <button
                    type="submit"
                    id="research-submit-btn"
                    className="btn-research"
                    disabled={loading || !objective.trim()}
                  >
                    {loading ? (
                      <>
                        <span className="spinner" aria-hidden="true" />
                        <span>Researching...</span>
                      </>
                    ) : (
                      <>
                        <span>Research</span>
                        <span aria-hidden="true">→</span>
                      </>
                    )}
                  </button>
                </div>
              </form>
            </section>

            {/* Loading State */}
            {loading && (
              <div className="loading-card" role="status" aria-live="polite">
                <div className="loading-spinner-large" aria-hidden="true" />
                <div className="loading-text-container">
                  <p className="loading-title">Autonomous Research in Progress</p>
                  <p className="loading-subtitle">
                    Iteratively planning searches, retrieving evidence, synthesizing grounded answers, and verifying citations...
                  </p>
                </div>
              </div>
            )}

            {/* Error State */}
            {error && (
              <div className="error-banner" role="alert">
                <span className="error-icon" aria-hidden="true">⚠️</span>
                <div className="error-content">
                  <p className="error-title">Research Notice</p>
                  <p className="error-message">{error}</p>
                </div>
              </div>
            )}

            {/* Results View */}
            {result && (
              <div className="results-container">
                {/* Meta Summary Bar */}
                <div className="meta-summary-bar">
                  <span
                    className={`badge ${
                      result.status === "completed"
                        ? "badge-completed"
                        : "badge-failed"
                    }`}
                  >
                    ● Status: {result.status}
                  </span>
                  <span
                    className={`badge ${
                      result.is_grounded ? "badge-grounded" : "badge-ungrounded"
                    }`}
                  >
                    {result.is_grounded
                      ? "✓ Grounded in Evidence"
                      : "○ Ungrounded / No Context"}
                  </span>
                  <span className="badge badge-info">
                    Iterations: {result.iteration_count}
                  </span>
                  {typeof result.duration_ms === "number" && (
                    <span className="badge badge-info">
                      Duration: {(result.duration_ms / 1000).toFixed(2)}s
                    </span>
                  )}
                  <span className="badge badge-info">
                    Citations: {result.citations?.length || 0}
                  </span>
                </div>

                {/* Final Answer Section */}
                <section className="result-section" aria-labelledby="heading-final-answer">
                  <div className="section-header">
                    <h2 id="heading-final-answer" className="section-title">
                      Final Answer
                    </h2>
                  </div>
                  <div className="final-answer-card">
                    {result.final_answer ? (
                      result.final_answer
                    ) : (
                      <span className="empty-text">
                        No synthesized final answer returned.
                      </span>
                    )}
                  </div>
                </section>

                {/* Queries executed */}
                {result.queries && result.queries.length > 0 && (
                  <section className="result-section" aria-labelledby="heading-queries">
                    <div className="section-header">
                      <h3 id="heading-queries" className="section-title">
                        Executed Search Queries
                      </h3>
                      <span className="section-count">{result.queries.length}</span>
                    </div>
                    <div className="queries-tag-list">
                      {result.queries.map((q, idx) => (
                        <span key={idx} className="query-tag">
                          🔍 {q}
                        </span>
                      ))}
                    </div>
                  </section>
                )}

                {/* Sources Section */}
                <section className="result-section" aria-labelledby="heading-sources">
                  <div className="section-header">
                    <h2 id="heading-sources" className="section-title">
                      Sources
                    </h2>
                    <span className="section-count">
                      {result.sources?.length || 0}
                    </span>
                  </div>
                  {result.sources && result.sources.length > 0 ? (
                    <div className="sources-grid">
                      {result.sources.map((src, idx) => (
                        <div key={idx} className="source-card">
                          <p className="source-title">
                            {src.document_title || "Untitled Document"}
                          </p>
                          <p className="source-origin">
                            {src.document_source || "unknown"}
                          </p>
                          <div className="source-meta">
                            <span className="label-hint">
                              {src.chunk_count || src.chunk_ids?.length || 0} chunk(s)
                            </span>
                            <div className="chunk-pills">
                              {(src.chunk_ids || []).map((cid, cidx) => (
                                <span key={cidx} className="chunk-pill">
                                  #{cid}
                                </span>
                              ))}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="empty-text">
                      No source documents retrieved for this objective.
                    </div>
                  )}
                </section>

                {/* Evidence Section */}
                <section className="result-section" aria-labelledby="heading-evidence">
                  <div className="section-header">
                    <h2 id="heading-evidence" className="section-title">
                      Evidence Chunks
                    </h2>
                    <span className="section-count">
                      {result.evidence?.length || 0}
                    </span>
                  </div>
                  {result.evidence && result.evidence.length > 0 ? (
                    <div className="evidence-list">
                      {result.evidence.map((ev, idx) => (
                        <article key={idx} className="evidence-card">
                          <div className="evidence-header">
                            <span className="citation-tag">
                              {ev.citation || `[Chunk: ${ev.chunk_id}]`}
                            </span>
                            {typeof ev.score === "number" && (
                              <span className="score-badge">
                                Score: {ev.score.toFixed(3)}
                              </span>
                            )}
                          </div>
                          <div className="evidence-content">{ev.content}</div>
                          <div className="evidence-footer">
                            Origin: {ev.document_source || "unknown"}
                            {ev.document_id ? ` (ID: ${ev.document_id})` : ""}
                          </div>
                        </article>
                      ))}
                    </div>
                  ) : (
                    <div className="empty-text">
                      No evidence chunks accumulated.
                    </div>
                  )}
                </section>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
