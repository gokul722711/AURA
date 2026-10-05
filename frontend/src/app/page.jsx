"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import Sidebar from "../components/Sidebar";
import ResearchComposer from "../components/ResearchComposer";
import ResearchActiveView from "../components/ResearchActiveView";
import ResearchResultView from "../components/ResearchResultView";
import KnowledgeBaseView from "../components/KnowledgeBaseView";
import ModelsView from "../components/ModelsView";
import {
  formatMode,
  resolveActiveProfile,
  getContextualSuggestions,
} from "../utils/helpers";
import { SidebarToggleIcon, WarningIcon } from "../components/Icons";

export default function Home() {
  // Navigation & Shell
  const [activeTab, setActiveTab] = useState("research"); // 'research' | 'knowledge' | 'models'
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);

  // Research State
  const [objective, setObjective] = useState("");
  const [researchMode, setResearchMode] = useState("knowledge_base");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [selectedHistoryId, setSelectedHistoryId] = useState(null);

  // Active Run Tracking (M9 Polling)
  const [activeRunId, setActiveRunId] = useState(null);
  const [activeRunStatus, setActiveRunStatus] = useState(null); // 'queued' | 'running'
  const [activeRunMode, setActiveRunMode] = useState(null);
  const [activeRunElapsed, setActiveRunElapsed] = useState(0);

  const pollTimerRef = useRef(null);
  const elapsedTimerRef = useRef(null);

  // Research History State
  const [historyRuns, setHistoryRuns] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  // Knowledge Base State
  const [documents, setDocuments] = useState([]);
  const [docsLoading, setDocsLoading] = useState(false);
  const [docsError, setDocsError] = useState(null);

  // Model Profiles State (M15)
  const [modelProfiles, setModelProfiles] = useState([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelsError, setModelsError] = useState(null);
  const [selectedModelProfileId, setSelectedModelProfileId] = useState("");
  const [testResults, setTestResults] = useState({});

  const activeProfile = resolveActiveProfile(modelProfiles);

  const getApiUrl = useCallback((endpoint) => {
    const base = process.env.NEXT_PUBLIC_API_URL
      ? process.env.NEXT_PUBLIC_API_URL.replace(/\/$/, "")
      : "";
    const cleanEndpoint = endpoint.replace(/^\/|\/$/g, "");
    return `${base}/api/${cleanEndpoint}/`;
  }, []);

  // Fetch Documents
  const fetchDocuments = useCallback(async () => {
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
  }, [getApiUrl]);

  // Fetch Research History
  const fetchHistory = useCallback(async () => {
    setHistoryLoading(true);
    try {
      const response = await fetch(getApiUrl("research/runs"));
      if (response.ok) {
        const data = await response.json();
        setHistoryRuns(Array.isArray(data) ? data : []);
      }
    } catch (err) {
      console.error("Failed to load research history:", err);
    } finally {
      setHistoryLoading(false);
    }
  }, [getApiUrl]);

  // Fetch Model Profiles
  const fetchModelProfiles = useCallback(async () => {
    setModelsLoading(true);
    setModelsError(null);
    try {
      const response = await fetch(getApiUrl("models"));
      if (!response.ok) {
        throw new Error(`Failed to load models (HTTP ${response.status})`);
      }
      const data = await response.json();
      const list = Array.isArray(data) ? data : [];
      setModelProfiles(list);
      const active = list.find((p) => p.is_active);
      if (active && (!selectedModelProfileId || !list.some((p) => p.id === selectedModelProfileId))) {
        setSelectedModelProfileId(active.id);
      }
    } catch (err) {
      setModelsError(err.message || "Failed to load model profiles.");
    } finally {
      setModelsLoading(false);
    }
  }, [getApiUrl, selectedModelProfileId]);

  useEffect(() => {
    fetchDocuments();
    fetchHistory();
    fetchModelProfiles();
  }, [fetchDocuments, fetchHistory, fetchModelProfiles]);

  // Global Keyboard Shortcuts (⌘K for New Research, ⌘B for Sidebar Toggle)
  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        handleNewResearch();
      }
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "b") {
        e.preventDefault();
        setIsSidebarCollapsed((prev) => !prev);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // Elapsed seconds timer during active research run
  useEffect(() => {
    if (activeRunId && (activeRunStatus === "queued" || activeRunStatus === "running")) {
      elapsedTimerRef.current = setInterval(() => {
        setActiveRunElapsed((prev) => prev + 1);
      }, 1000);
    } else {
      if (elapsedTimerRef.current) clearInterval(elapsedTimerRef.current);
    }
    return () => {
      if (elapsedTimerRef.current) clearInterval(elapsedTimerRef.current);
    };
  }, [activeRunId, activeRunStatus]);

  // Polling loop for active research run
  useEffect(() => {
    if (!activeRunId) return;

    let isCurrent = true;

    const pollStatus = async () => {
      try {
        const response = await fetch(`${getApiUrl("research")}${activeRunId}/`);
        if (!isCurrent) return;
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        if (!isCurrent) return;

        setActiveRunStatus(data.status);
        if (data.mode) {
          setActiveRunMode(data.mode);
        }

        if (data.status === "completed") {
          setResult(data);
          setError(null);
          setSelectedHistoryId(activeRunId);
          setActiveRunId(null);
          setActiveRunStatus(null);
          setLoading(false);
          fetchHistory();
        } else if (data.status === "failed") {
          const errMsg = data.error || (data.errors && data.errors.join("; ")) || "Research run failed.";
          setError(errMsg);
          setActiveRunId(null);
          setActiveRunStatus(null);
          setLoading(false);
          fetchHistory();
        } else if (data.status === "cancelled") {
          setError(data.error || "Research was cancelled.");
          setActiveRunId(null);
          setActiveRunStatus(null);
          setLoading(false);
          fetchHistory();
        }
      } catch (err) {
        if (isCurrent) {
          console.warn("Polling retry error:", err);
        }
      }
    };

    pollTimerRef.current = setInterval(pollStatus, 2000);

    return () => {
      isCurrent = false;
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    };
  }, [activeRunId, getApiUrl, fetchHistory]);

  // Start New Research Session
  const handleNewResearch = () => {
    setResult(null);
    setSelectedHistoryId(null);
    setError(null);
    setObjective("");
    setActiveTab("research");
  };

  // Submit Research Objective
  const handleResearchSubmit = async (e) => {
    if (e && e.preventDefault) e.preventDefault();

    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }

    const trimmed = objective.trim();
    if (!trimmed) {
      setError("Please enter a research objective.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    setSelectedHistoryId(null);
    setActiveRunElapsed(0);

    const apiUrl = getApiUrl("research");

    try {
      const requestPayload = { objective: trimmed, mode: researchMode };
      if (selectedModelProfileId) {
        requestPayload.model_profile_id = selectedModelProfileId;
      }

      const response = await fetch(apiUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestPayload),
      });

      const contentType = response.headers.get("content-type") || "";
      let data = {};
      if (contentType.includes("application/json")) {
        data = await response.json();
      } else {
        const text = await response.text();
        throw new Error(`Server returned HTTP ${response.status}: ${text.slice(0, 120)}`);
      }

      if (response.status !== 202 && !response.ok) {
        const errorDetail = data.detail || data.error || `HTTP error ${response.status}`;
        throw new Error(errorDetail);
      }

      // 202 Accepted: Initialize tracking
      setActiveRunId(data.run_id);
      setActiveRunStatus(data.status || "queued");
      setActiveRunMode(data.mode || researchMode);
      setSelectedHistoryId(data.run_id);
      fetchHistory();
    } catch (err) {
      setError(err.message || "Failed to communicate with research backend.");
      setLoading(false);
    }
  };

  // Cancel Running Research
  const handleCancelResearch = async () => {
    if (!activeRunId) return;
    try {
      const response = await fetch(`${getApiUrl("research")}${activeRunId}/cancel/`, {
        method: "POST",
      });
      if (response.ok) {
        setError("Research execution cancelled by user.");
      }
    } catch (err) {
      console.error("Failed to cancel research run:", err);
    } finally {
      setActiveRunId(null);
      setActiveRunStatus(null);
      setLoading(false);
      fetchHistory();
    }
  };

  // Select Historical Run to View
  const handleSelectHistoryRun = async (runId) => {
    try {
      setSelectedHistoryId(runId);
      const response = await fetch(`${getApiUrl("research")}${runId}/`);
      if (response.ok) {
        const data = await response.json();
        setResult(data);
        setObjective(data.objective || "");
        if (data.mode) {
          setResearchMode(data.mode);
        }
        setActiveTab("research");
        setError(null);
        window.scrollTo({ top: 0, behavior: "smooth" });
      }
    } catch (err) {
      console.error("Failed to load historical run details:", err);
    }
  };

  // History Run Deletion Handlers
  const handleDeleteRun = async (runId) => {
    try {
      const resp = await fetch(`${getApiUrl("research")}${runId}/`, {
        method: "DELETE",
      });
      if (resp.ok || resp.status === 204) {
        setHistoryRuns((prev) => prev.filter((r) => r.run_id !== runId && r.id !== runId));
        if (selectedHistoryId === runId) {
          setResult(null);
          setSelectedHistoryId(null);
        }
      }
    } catch (err) {
      console.error("Failed to delete research run:", err);
    }
  };

  const handleDeleteRuns = async (runIds) => {
    if (!Array.isArray(runIds) || runIds.length === 0) return;
    try {
      const resp = await fetch(getApiUrl("research/runs"), {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_ids: runIds }),
      });
      if (resp.ok) {
        const idSet = new Set(runIds);
        setHistoryRuns((prev) => prev.filter((r) => !idSet.has(r.run_id) && !idSet.has(r.id)));
        if (idSet.has(selectedHistoryId)) {
          setResult(null);
          setSelectedHistoryId(null);
        }
      }
    } catch (err) {
      console.error("Failed to bulk delete research runs:", err);
    }
  };

  const handleDeleteAllRuns = async () => {
    try {
      const resp = await fetch(getApiUrl("research/runs"), {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (resp.ok) {
        setHistoryRuns([]);
        setResult(null);
        setSelectedHistoryId(null);
      }
    } catch (err) {
      console.error("Failed to delete all research runs:", err);
    }
  };

  // Prompt shortcut from a document
  const handlePromptFromDoc = (title) => {
    setObjective(`Explain the core concepts and architecture described in "${title}".`);
    setResult(null);
    setSelectedHistoryId(null);
    setActiveTab("research");
  };

  // Document Ingestion Handler
  const handleAddDocument = async (payloadOrFormData, isMultipart = false) => {
    const opts = {
      method: "POST",
      body: isMultipart ? payloadOrFormData : JSON.stringify(payloadOrFormData),
    };
    if (!isMultipart) {
      opts.headers = { "Content-Type": "application/json" };
    }

    const response = await fetch(getApiUrl("documents"), opts);
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || data.detail || `HTTP ${response.status}`);
    }
    await fetchDocuments();
    return data;
  };

  // Document Deletion Handler
  const handleDeleteDocument = async (docId) => {
    const response = await fetch(`${getApiUrl("documents")}${docId}/`, {
      method: "DELETE",
    });
    if (!response.ok && response.status !== 204) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.error || `HTTP ${response.status}`);
    }
    setDocuments((prev) => prev.filter((d) => d.id !== docId));
  };

  // Model Profile Actions
  const handleActivateProfile = async (id) => {
    try {
      const resp = await fetch(`${getApiUrl("models")}${id}/activate/`, { method: "POST" });
      if (resp.ok) {
        await fetchModelProfiles();
        setSelectedModelProfileId(id);
      }
    } catch (err) {
      console.error("Failed to activate profile:", err);
    }
  };

  const handleDeleteProfile = async (id) => {
    if (!confirm("Are you sure you want to delete this model profile?")) return;
    try {
      const resp = await fetch(`${getApiUrl("models")}${id}/`, { method: "DELETE" });
      if (resp.ok) {
        await fetchModelProfiles();
        if (selectedModelProfileId === id) {
          setSelectedModelProfileId("");
        }
      }
    } catch (err) {
      console.error("Failed to delete profile:", err);
    }
  };

  const handleTestProfile = async (id) => {
    setTestResults((prev) => ({ ...prev, [id]: { loading: true } }));
    try {
      const resp = await fetch(`${getApiUrl("models")}${id}/test/`, { method: "POST" });
      const data = await resp.json();
      setTestResults((prev) => ({
        ...prev,
        [id]: {
          loading: false,
          success: Boolean(data.success),
          message: data.message,
          error: data.error,
          latency_ms: data.latency_ms,
        },
      }));
    } catch (err) {
      setTestResults((prev) => ({
        ...prev,
        [id]: { loading: false, success: false, error: err.message },
      }));
    }
  };

  const handleTestDraftProfile = async (draftPayload) => {
    setTestResults((prev) => ({ ...prev, draft: { loading: true } }));
    try {
      const resp = await fetch(getApiUrl("models/test"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(draftPayload),
      });
      const data = await resp.json();
      setTestResults((prev) => ({
        ...prev,
        draft: {
          loading: false,
          success: Boolean(data.success),
          message: data.message,
          error: data.error,
          latency_ms: data.latency_ms,
        },
      }));
    } catch (err) {
      setTestResults((prev) => ({
        ...prev,
        draft: { loading: false, success: false, error: err.message },
      }));
    }
  };

  const handleSaveProfile = async (id, payload) => {
    const url = id ? `${getApiUrl("models")}${id}/` : getApiUrl("models");
    const method = id ? "PATCH" : "POST";

    const resp = await fetch(url, {
      method,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.error || `HTTP ${resp.status}`);
    }
    await fetchModelProfiles();
    return data;
  };

  // Compute dynamic contextual suggestions based on history and documents
  const contextualSuggestions = getContextualSuggestions(historyRuns, documents);

  return (
    <div className={`aura-shell ${isSidebarCollapsed ? "sidebar-collapsed" : ""}`}>
      {/* Persistent Sidebar */}
      <Sidebar
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        onNewResearch={handleNewResearch}
        historyRuns={historyRuns}
        activeRunId={activeRunId}
        selectedHistoryId={selectedHistoryId}
        onSelectHistoryRun={handleSelectHistoryRun}
        onDeleteRun={handleDeleteRun}
        onDeleteRuns={handleDeleteRuns}
        onDeleteAllRuns={handleDeleteAllRuns}
        documentsCount={documents.length}
        modelsCount={modelProfiles.length}
        activeProfile={activeProfile}
        isOpen={sidebarOpen}
        onCloseMobile={() => setSidebarOpen(false)}
        isCollapsed={isSidebarCollapsed}
        onToggleCollapse={() => setIsSidebarCollapsed((prev) => !prev)}
      />

      {/* Main Content Area */}
      <div className="aura-main-wrapper">
        {/* Top Navbar Header */}
        <header className="aura-topbar">
          <div className="topbar-left">
            <button
              type="button"
              className="btn-sidebar-toggle"
              onClick={() => {
                if (typeof window !== "undefined" && window.innerWidth <= 768) {
                  setSidebarOpen(!sidebarOpen);
                } else {
                  setIsSidebarCollapsed(!isSidebarCollapsed);
                }
              }}
              title={isSidebarCollapsed ? "Expand sidebar (⌘B)" : "Collapse sidebar (⌘B)"}
              aria-label={isSidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            >
              <SidebarToggleIcon size={16} />
            </button>
            <div className="topbar-title-group">
              <span className="topbar-logo">AURA</span>
              <span className="topbar-crumb-sep">/</span>
              <span className="topbar-tab-label">
                {activeTab === "research"
                  ? "Research Workspace"
                  : activeTab === "knowledge"
                  ? "Knowledge Base"
                  : "Model Profiles"}
              </span>
            </div>
          </div>

          <div className="topbar-right">
            {/* Quick tab switcher pill without emojis */}
            <div className="topbar-nav-pills" role="tablist">
              <button
                type="button"
                id="tab-research"
                className={`topbar-tab-btn ${activeTab === "research" ? "active" : ""}`}
                onClick={() => setActiveTab("research")}
              >
                Research
                {activeRunId && <span className="tab-live-dot" title="Running" />}
              </button>
              <button
                type="button"
                id="tab-knowledge"
                className={`topbar-tab-btn ${activeTab === "knowledge" ? "active" : ""}`}
                onClick={() => setActiveTab("knowledge")}
              >
                Knowledge ({documents.length})
              </button>
              <button
                type="button"
                id="tab-models"
                className={`topbar-tab-btn ${activeTab === "models" ? "active" : ""}`}
                onClick={() => setActiveTab("models")}
              >
                Models ({modelProfiles.length})
              </button>
            </div>

            {/* Active Model Indicator */}
            {activeProfile ? (
              <button
                type="button"
                className="topbar-model-indicator"
                onClick={() => setActiveTab("models")}
                title={`Inference Model: ${activeProfile.name} (${activeProfile.provider})`}
              >
                <span className="indicator-dot" />
                <span className="indicator-text">{activeProfile.name}</span>
              </button>
            ) : (
              <button
                type="button"
                className="topbar-model-indicator warning"
                onClick={() => setActiveTab("models")}
              >
                <span className="indicator-dot" />
                <span className="indicator-text">No Model Active</span>
              </button>
            )}
          </div>
        </header>

        {/* Dynamic Workspace Body */}
        <main className="aura-content-area page-fade-in">
          {/* TAB 1: RESEARCH VIEW */}
          {activeTab === "research" && (
            <div className="research-tab-layout">
              {/* Error banner if present */}
              {error && (
                <div className="error-banner" role="alert">
                  <WarningIcon size={16} className="error-icon-svg" />
                  <div className="error-content">
                    <p className="error-title">Research Notice</p>
                    <p className="error-message">{error}</p>
                    {(error.includes("No research model configured") ||
                      error.includes("Configure a model profile") ||
                      error.includes("No active model profile")) && (
                      <div className="error-cta-row">
                        <button
                          type="button"
                          className="btn-error-cta"
                          id="btn-error-configure-model"
                          onClick={() => setActiveTab("models")}
                        >
                          Configure Model Profile →
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* State A: Active Research Run */}
              {activeRunId ? (
                <div className="research-active-container">
                  <ResearchActiveView
                    activeRunId={activeRunId}
                    activeRunStatus={activeRunStatus}
                    activeRunMode={activeRunMode || researchMode}
                    elapsedSeconds={activeRunElapsed}
                    documents={documents}
                    onCancel={handleCancelResearch}
                  />
                </div>
              ) : result ? (
                /* State B: Research Result View + Docked Follow-Up Composer */
                <div className="research-result-container">
                  <ResearchResultView
                    result={result}
                    objective={objective}
                    onNewResearch={handleNewResearch}
                  />

                  {/* Docked Follow-Up Composer */}
                  <div className="docked-composer-wrap">
                    <ResearchComposer
                      mode="bottom"
                      objective={objective}
                      setObjective={setObjective}
                      researchMode={researchMode}
                      setResearchMode={setResearchMode}
                      modelProfiles={modelProfiles}
                      selectedModelProfileId={selectedModelProfileId}
                      setSelectedModelProfileId={setSelectedModelProfileId}
                      activeProfile={activeProfile}
                      loading={loading}
                      onSubmit={handleResearchSubmit}
                      onConfigureModels={() => setActiveTab("models")}
                      firstDocTitle={documents[0]?.title}
                      contextualSuggestions={contextualSuggestions}
                    />
                  </div>
                </div>
              ) : (
                /* State C: Hero Composer (Empty / New Research) */
                <div className="research-hero-container">
                  <ResearchComposer
                    mode="hero"
                    objective={objective}
                    setObjective={setObjective}
                    researchMode={researchMode}
                    setResearchMode={setResearchMode}
                    modelProfiles={modelProfiles}
                    selectedModelProfileId={selectedModelProfileId}
                    setSelectedModelProfileId={setSelectedModelProfileId}
                    activeProfile={activeProfile}
                    loading={loading}
                    onSubmit={handleResearchSubmit}
                    onConfigureModels={() => setActiveTab("models")}
                    firstDocTitle={documents[0]?.title}
                    contextualSuggestions={contextualSuggestions}
                  />
                </div>
              )}
            </div>
          )}

          {/* TAB 2: KNOWLEDGE BASE VIEW */}
          {activeTab === "knowledge" && (
            <KnowledgeBaseView
              documents={documents}
              loading={docsLoading}
              error={docsError}
              onRefresh={fetchDocuments}
              onAddDocument={handleAddDocument}
              onDeleteDocument={handleDeleteDocument}
              onPromptFromDoc={handlePromptFromDoc}
              getApiUrl={getApiUrl}
            />
          )}

          {/* TAB 3: MODEL PROFILES VIEW */}
          {activeTab === "models" && (
            <ModelsView
              modelProfiles={modelProfiles}
              loading={modelsLoading}
              error={modelsError}
              onRefresh={fetchModelProfiles}
              onActivateProfile={handleActivateProfile}
              onDeleteProfile={handleDeleteProfile}
              onSaveProfile={handleSaveProfile}
              onTestProfile={handleTestProfile}
              onTestDraftProfile={handleTestDraftProfile}
              testResults={testResults}
            />
          )}
        </main>
      </div>
    </div>
  );
}
