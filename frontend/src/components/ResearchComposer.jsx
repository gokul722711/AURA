"use client";

import { useEffect, useRef } from "react";
import { formatMode, getModeDescription } from "../utils/helpers";
import { WarningIcon } from "./Icons";

/**
 * ResearchComposer
 *
 * Dual-mode query composer (Hero & Docked).
 * Displays full AURA name ("Autonomous Research & Engineering Agent") beneath the central title.
 * Replaces static prompts with lightweight contextual suggestions.
 */
export default function ResearchComposer({
  mode = "hero", // 'hero' | 'bottom'
  objective,
  setObjective,
  researchMode,
  setResearchMode,
  modelProfiles = [],
  selectedModelProfileId,
  setSelectedModelProfileId,
  activeProfile = null,
  loading = false,
  onSubmit,
  onConfigureModels,
  contextualSuggestions = [],
}) {
  const textareaRef = useRef(null);

  // Auto-resize textarea height to content
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    const newHeight = Math.min(Math.max(el.scrollHeight, mode === "hero" ? 90 : 52), 220);
    el.style.height = `${newHeight}px`;
  }, [objective, mode]);

  const handleKeyDown = (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      if (!loading && objective.trim()) {
        onSubmit(e);
      }
    }
  };

  const isNoModel = modelProfiles.length === 0;

  return (
    <div className={`composer-container composer-${mode}`}>
      {/* Hero Header with AURA full name */}
      {mode === "hero" && (
        <div className="hero-header">
          <div className="hero-branding">
            <h1 className="hero-logo">AURA</h1>
            <p className="hero-descriptor">Autonomous Research &amp; Engineering Agent</p>
          </div>
          <h2 className="hero-title">What would you like to research?</h2>
          <p className="hero-subtitle">
            Grounded investigation across internal knowledge, live web sources, and local or hosted models.
          </p>
        </div>
      )}

      {/* Warning banner if no model configured */}
      {isNoModel && mode === "hero" && (
        <div className="warning-banner" id="no-model-warning-banner" role="alert">
          <div className="warning-banner-text">
            <div className="warning-banner-header">
              <WarningIcon size={16} />
              <span className="warning-banner-title">No inference model configured</span>
            </div>
            <span>Configure a local or hosted model profile before starting research.</span>
          </div>
          <button
            type="button"
            className="btn-configure-banner"
            id="btn-configure-model-banner"
            onClick={onConfigureModels}
          >
            Configure Model →
          </button>
        </div>
      )}

      {/* Main Composer Box */}
      <form onSubmit={onSubmit} className="composer-card">
        <div className="composer-input-wrap">
          <textarea
            ref={textareaRef}
            id="objective-input"
            className="composer-textarea"
            placeholder={
              mode === "hero"
                ? "Ask AURA anything, investigate a topic, or synthesize evidence..."
                : "Ask a follow-up question or start a new inquiry..."
            }
            value={objective ?? ""}
            onChange={(e) => setObjective(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
            rows={mode === "hero" ? 3 : 2}
          />
        </div>

        {/* Composer Controls Bar */}
        <div className="composer-toolbar">
          <div className="composer-toolbar-left">
            {/* Research Mode Selector */}
            <div className="toolbar-pill-group">
              <label htmlFor="mode-select" className="sr-only">
                Research Mode
              </label>
              <select
                id="mode-select"
                className="toolbar-select mode-select-dropdown"
                value={researchMode ?? "knowledge_base"}
                onChange={(e) => setResearchMode(e.target.value)}
                disabled={loading}
                title={getModeDescription(researchMode)}
              >
                <option value="knowledge_base">Knowledge Base (Default)</option>
                <option value="web">Web</option>
                <option value="web_knowledge_base">Web + KB</option>
                <option value="model_knowledge">Model Knowledge</option>
              </select>
            </div>

            {/* Model Profile Selector */}
            <div className="toolbar-pill-group">
              <label htmlFor="model-profile-select" className="sr-only">
                Inference Model Profile
              </label>
              <select
                id="model-profile-select"
                className="toolbar-select mode-select-dropdown"
                value={
                  selectedModelProfileId ||
                  (activeProfile ? activeProfile.id : "")
                }
                onChange={(e) => setSelectedModelProfileId(e.target.value)}
                disabled={loading || isNoModel}
                title={
                  activeProfile
                    ? `Active: ${activeProfile.name} (${activeProfile.model})`
                    : "Select model profile"
                }
              >
                {isNoModel ? (
                  <option value="">No model profile configured</option>
                ) : (
                  modelProfiles.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name} ({p.provider} / {p.model})
                      {p.is_active ? " — Active" : ""}
                    </option>
                  ))
                )}
              </select>
            </div>
          </div>

          <div className="composer-toolbar-right">
            <button
              type="submit"
              id="research-submit-btn"
              className="btn-submit-research"
              disabled={loading || !objective.trim()}
              title="Press ⌘+Enter to submit"
            >
              {loading ? (
                <>
                  <span className="spinner-micro" aria-hidden="true" />
                  <span>Queuing...</span>
                </>
              ) : (
                <>
                  <span>{mode === "hero" ? "Start Research" : "Send"}</span>
                  <span className="submit-arrow" aria-hidden="true">→</span>
                </>
              )}
            </button>
          </div>
        </div>
      </form>

      {/* Contextual Suggestions (only if available context exists in hero mode) */}
      {mode === "hero" && contextualSuggestions.length > 0 && (
        <div className="hero-suggestions">
          <div className="suggestions-list">
            {contextualSuggestions.map((s, idx) => (
              <button
                key={idx}
                type="button"
                className={`btn-suggestion-pill ${
                  s.type === "document" ? "btn-suggestion-doc" : ""
                }`}
                onClick={() => setObjective(s.text)}
                disabled={loading}
                title={s.text}
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
