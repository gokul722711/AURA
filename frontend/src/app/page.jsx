"use client";

import { useState } from "react";

const SAMPLE_OBJECTIVES = [
  "How do the AURA Model Gateway, RAG pipeline, and Agent Runtime work together?",
  "What mechanisms ensure citation integrity and prevent hallucinated sources?",
  "Explain the architectural boundaries and security constraints of the agent runtime.",
];

export default function Home() {
  const [objective, setObjective] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const trimmed = objective.trim();
    if (!trimmed) {
      setError("Please enter a research objective.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const apiUrl = process.env.NEXT_PUBLIC_API_URL
      ? `${process.env.NEXT_PUBLIC_API_URL.replace(/\/$/, "")}/api/research/`
      : "/api/research/";

    try {
      const response = await fetch(apiUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ objective: trimmed }),
      });

      const data = await response.json();

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

  const handleSuggestionClick = (sample) => {
    setObjective(sample);
  };

  return (
    <div className="app-container">
      <main className="content-wrapper">
        {/* Header */}
        <header className="header">
          <div className="header-top">
            <h1 className="logo">AURA</h1>
            <span className="phase-pill">M7 — Research API</span>
          </div>
          <p className="subtitle">
            Autonomous Research &amp; Engineering Agent
          </p>
        </header>

        {/* Input Form Card */}
        <section className="card" aria-label="Research input">
          <form onSubmit={handleSubmit} className="form-group">
            <div className="label">
              <label htmlFor="objective-input">Research Objective</label>
              <span className="label-hint">Autonomous iterative loop</span>
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
                  onClick={() => handleSuggestionClick(sample)}
                  disabled={loading}
                >
                  {idx === 0
                    ? "Architecture Overview"
                    : idx === 1
                    ? "Citation Integrity"
                    : "Security & Sandbox"}
                </button>
              ))}
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
      </main>
    </div>
  );
}
