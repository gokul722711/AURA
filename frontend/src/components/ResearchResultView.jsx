"use client";

import { useState } from "react";
import { formatMode, formatDuration, formatScore } from "../utils/helpers";
import {
  CopyIcon,
  CheckIcon,
  SearchIcon,
  GlobeIcon,
  FileIcon,
} from "./Icons";

/**
 * ResearchResultView
 *
 * Renders completed research as a high-quality editorial research document.
 * Prioritizes answer typography, provenance, structured citations, and collapsible evidence.
 * Strict zero-emoji policy, utilizing restrained semantic SVG icons.
 */
export default function ResearchResultView({
  result,
  objective,
  onNewResearch,
}) {
  const [evidenceExpanded, setEvidenceExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  if (!result) return null;

  const durationStr = formatDuration(result);
  const isGrounded = Boolean(result.is_grounded);
  const isSuccess = result.status === "completed";

  const handleCopyAnswer = async () => {
    if (!result.final_answer) return;
    try {
      await navigator.clipboard.writeText(result.final_answer);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy answer to clipboard:", err);
    }
  };

  return (
    <article className="result-document" aria-label="Research Findings">
      {/* Research Header & Meta */}
      <header className="result-doc-header">
        <div className="result-title-row">
          <h2 className="result-doc-title">
            {objective || result.objective || "Research Findings"}
          </h2>
          <div className="result-header-actions">
            <button
              type="button"
              className="btn-result-action"
              onClick={handleCopyAnswer}
              title="Copy synthesized answer"
            >
              {copied ? (
                <>
                  <CheckIcon size={13} />
                  <span>Copied</span>
                </>
              ) : (
                <>
                  <CopyIcon size={13} />
                  <span>Copy</span>
                </>
              )}
            </button>
            <button
              type="button"
              className="btn-result-action"
              onClick={onNewResearch}
            >
              + New Research
            </button>
          </div>
        </div>

        {/* Editorial Metadata Strip */}
        <div className="result-meta-strip">
          <span
            className={`meta-pill ${
              isSuccess ? "meta-pill-success" : "meta-pill-danger"
            }`}
          >
            ● {result.status}
          </span>
          <span className="meta-pill meta-pill-mode">
            {formatMode(result.mode)}
          </span>
          <span
            className={`meta-pill ${
              isGrounded ? "meta-pill-grounded" : "meta-pill-ungrounded"
            }`}
          >
            {isGrounded ? "Grounded" : "Ungrounded"}
          </span>
          {result.iteration_count != null && (
            <span className="meta-pill">
              Iterations: {result.iteration_count}
            </span>
          )}
          {durationStr && (
            <span className="meta-pill">
              Duration: {durationStr}
            </span>
          )}
          <span className="meta-pill">
            Citations: {result.citations?.length || 0}
          </span>
          {result.model && (
            <span className="meta-pill meta-pill-mono">
              Model: {result.model}
            </span>
          )}
        </div>
      </header>

      {/* Final Synthesized Answer */}
      <section className="result-answer-section" aria-label="Synthesized Answer">
        <div className="result-answer-body">
          {result.final_answer ? (
            <div className="answer-prose">
              {result.final_answer.split("\n\n").map((para, idx) => (
                <p key={idx}>{para}</p>
              ))}
            </div>
          ) : (
            <p className="empty-answer-text">
              No synthesized final answer returned for this run.
            </p>
          )}
        </div>
      </section>

      {/* Executed Search Queries */}
      {result.queries && result.queries.length > 0 && (
        <section className="result-queries-section" aria-label="Search Queries">
          <div className="section-label-row">
            <h3 className="section-subheading">Executed Queries</h3>
            <span className="section-count-badge">{result.queries.length}</span>
          </div>
          <div className="queries-chips-wrap">
            {result.queries.map((q, idx) => (
              <span key={idx} className="query-chip">
                <SearchIcon size={12} className="query-chip-icon-svg" />
                <span className="query-chip-text">{q}</span>
              </span>
            ))}
          </div>
        </section>
      )}

      {/* Sources Section */}
      <section className="result-sources-section" aria-label="Sources">
        <div className="section-label-row">
          <h3 className="section-subheading">Sources</h3>
          <span className="section-count-badge">
            {result.sources?.length || 0}
          </span>
        </div>

        {result.sources && result.sources.length > 0 ? (
          <div className="sources-list-grid">
            {result.sources.map((src, idx) => {
              const isWeb = Boolean(src.url || (src.document_source && src.document_source.startsWith("http")));
              return (
                <div key={idx} className="source-item-card">
                  <div className="source-item-top">
                    <span
                      className={`source-type-indicator ${
                        isWeb ? "type-web" : "type-kb"
                      }`}
                    >
                      {isWeb ? "WEB" : "KB"}
                    </span>
                    <h4 className="source-item-title">
                      {src.document_title || "Untitled Source"}
                    </h4>
                  </div>

                  {src.url ? (
                    <a
                      href={src.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="source-item-link"
                    >
                      <GlobeIcon size={13} />
                      <span>{src.url}</span>
                    </a>
                  ) : src.document_source ? (
                    <span className="source-item-origin">
                      <FileIcon size={13} />
                      <span>{src.document_source}</span>
                    </span>
                  ) : null}

                  <div className="source-item-meta">
                    <span className="source-chunks-tag">
                      {src.chunk_count || src.chunk_ids?.length || 0} chunk(s)
                    </span>
                    {src.pages && src.pages.length > 0 && (
                      <span className="source-pages-tag">
                        Pages: {src.pages.join(", ")}
                      </span>
                    )}
                    {(src.chunk_ids || []).length > 0 && (
                      <div className="source-chunk-pills">
                        {src.chunk_ids.map((cid, cidx) => (
                          <span key={cidx} className="chunk-id-pill">
                            #{cid}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <p className="empty-subtext">No sources retrieved for this objective.</p>
        )}
      </section>

      {/* Collapsible Evidence Section */}
      <section className="result-evidence-section" aria-label="Evidence Chunks">
        <button
          type="button"
          className="evidence-toggle-btn"
          onClick={() => setEvidenceExpanded(!evidenceExpanded)}
          aria-expanded={evidenceExpanded}
        >
          <div className="evidence-toggle-left">
            <span className="evidence-toggle-arrow">
              {evidenceExpanded ? "−" : "+"}
            </span>
            <span className="section-subheading">Supporting Evidence</span>
            <span className="section-count-badge">
              {result.evidence?.length || 0}
            </span>
          </div>
          <span className="evidence-toggle-hint">
            {evidenceExpanded ? "Hide details" : "Show chunks & scores"}
          </span>
        </button>

        {evidenceExpanded && (
          <div className="evidence-expanded-list">
            {result.evidence && result.evidence.length > 0 ? (
              result.evidence.map((ev, idx) => (
                <div key={idx} className="evidence-chunk-item">
                  <div className="evidence-chunk-header">
                    <span className="evidence-citation-pill">
                      {ev.citation || `[Chunk: ${ev.chunk_id}]`}
                    </span>
                    {typeof ev.score === "number" && (
                      <span className="evidence-score-pill">
                        Score: {formatScore(ev.score)}
                      </span>
                    )}
                  </div>

                  <div className="evidence-chunk-content">
                    {ev.content}
                  </div>

                  <div className="evidence-chunk-footer">
                    <span className="chunk-origin">
                      Origin: {ev.document_source || "unknown"}
                    </span>
                    {ev.url && (
                      <a
                        href={ev.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="chunk-url"
                      >
                        <GlobeIcon size={13} />
                        <span>{ev.url}</span>
                      </a>
                    )}
                    {(ev.page != null || ev.metadata?.page != null) && (
                      <span className="chunk-page">
                        Page: {ev.page != null ? ev.page : ev.metadata.page}
                      </span>
                    )}
                    {ev.document_id && (
                      <span className="chunk-doc-id">
                        Doc: {ev.document_id}
                      </span>
                    )}
                  </div>
                </div>
              ))
            ) : (
              <p className="empty-subtext">No evidence chunks recorded.</p>
            )}
          </div>
        )}
      </section>
    </article>
  );
}
