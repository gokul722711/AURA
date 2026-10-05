"use client";

import SphericalNodeField from "./SphericalNodeField";
import { formatMode, getDynamicActivityText } from "../utils/helpers";

/**
 * ResearchActiveView
 *
 * Visualizes in-flight research using the 3D spherical node field.
 * Combines sophisticated motion with truthful status feedback and
 * dynamic, mode-dependent activity indicator.
 */
export default function ResearchActiveView({
  activeRunId,
  activeRunStatus = "running",
  activeRunMode = "knowledge_base",
  elapsedSeconds = 0,
  documents = [],
  onCancel,
}) {
  const dynamicActivity = getDynamicActivityText(activeRunMode, elapsedSeconds, documents);

  return (
    <div className="active-research-view" role="status" aria-live="polite">
      {/* 3D Spherical Node Field - enlarged slightly per design review */}
      <div className="active-sphere-wrap">
        <SphericalNodeField size={215} nodeCount={145} />
      </div>

      {/* Truthful Progress & State */}
      <div className="active-meta-card">
        <div className="active-badges-row">
          <span className="active-status-badge">
            <span className="pulse-indicator" aria-hidden="true" />
            <span>Status: {activeRunStatus || "queued"}</span>
          </span>
          <span className="active-mode-badge">
            Mode: {formatMode(activeRunMode)}
          </span>
          <span className="active-timer-badge">
            {elapsedSeconds}s elapsed
          </span>
        </div>

        <h3 className="active-view-title">Autonomous Research in Progress</h3>

        {/* Dynamic Contextual Activity Text based on Research Mode */}
        <div className="active-activity-container" key={dynamicActivity}>
          <span className="activity-live-dot" aria-hidden="true" />
          <span className="activity-text-line">{dynamicActivity}</span>
        </div>

        <p className="active-view-desc">
          Executing iterative retrieval, cross-source evidence accumulation, and grounded synthesis via background Celery worker.
        </p>

        <div className="active-run-id-row">
          <span className="run-id-label">Run ID:</span>
          <code className="run-id-code">{activeRunId}</code>
        </div>

        <div className="active-actions-row">
          <button
            type="button"
            className="btn-cancel-research"
            onClick={onCancel}
          >
            Cancel Research
          </button>
        </div>
      </div>
    </div>
  );
}
