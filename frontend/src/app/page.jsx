"use client";

import { useEffect, useRef, useState } from "react";

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
- **M9 Async Engine**: Persistent research run tracking via Celery and Redis with complete historical trace auditability.

## Ingestion Pipeline
When a document is ingested, it is segmented into chunks, embedded with 384-dimensional vectors, and stored in PostgreSQL with pgvector for instant vector similarity retrieval.`,
};

export default function Home() {
  // Navigation
  const [activeTab, setActiveTab] = useState("research"); // 'research' | 'knowledge' | 'history'

  // Research State
  const [objective, setObjective] = useState("");
  const [researchMode, setResearchMode] = useState("knowledge_base"); // 'model_knowledge' | 'knowledge_base' | 'web' | 'web_knowledge_base'
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  // Active Run Tracking (M9 Polling)
  const [activeRunId, setActiveRunId] = useState(null);
  const [activeRunStatus, setActiveRunStatus] = useState(null); // 'queued' | 'running'
  const [activeRunMode, setActiveRunMode] = useState(null);
  const [activeRunElapsed, setActiveRunElapsed] = useState(0);
  const pollTimerRef = useRef(null);
  const elapsedTimerRef = useRef(null);

  const formatMode = (m) => {
    switch (m) {
      case "model_knowledge":
        return "Model Knowledge";
      case "knowledge_base":
        return "Knowledge Base";
      case "web":
        return "Web";
      case "web_knowledge_base":
        return "Web + KB";
      default:
        return m || "Knowledge Base";
    }
  };

  // Research History State
  const [historyRuns, setHistoryRuns] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  // Knowledge Base State
  const [documents, setDocuments] = useState([]);
  const [docsLoading, setDocsLoading] = useState(false);
  const [docsError, setDocsError] = useState(null);

  // Add Document State
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
  const fileInputRef = useRef(null);

  const formatFileSize = (bytes) => {
    if (bytes == null || isNaN(bytes)) return "";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
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

  const formatScore = (score) => {
    if (typeof score !== "number" || isNaN(score)) return "";
    if (score === 0) return "0.000";
    if (score >= 0.01) return score.toFixed(3);
    if (score >= 0.0001) return score.toFixed(4);
    return score.toFixed(6);
  };

  // Document Interaction State
  const [expandedDocId, setExpandedDocId] = useState(null);
  const [deletingId, setDeletingId] = useState(null);
  const [deleteConfirmId, setDeleteConfirmId] = useState(null);

  // Model Profiles State (M15)
  const [modelProfiles, setModelProfiles] = useState([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelsError, setModelsError] = useState(null);
  const [selectedModelProfileId, setSelectedModelProfileId] = useState("");
  const [showModelModal, setShowModelModal] = useState(false);
  const [editingProfileId, setEditingProfileId] = useState(null);
  const [testResults, setTestResults] = useState({});

  // Model Form State
  const [formName, setFormName] = useState("");
  const [formProvider, setFormProvider] = useState("nvidia");
  const [formEndpoint, setFormEndpoint] = useState("https://integrate.api.nvidia.com/v1");
  const [formModel, setFormModel] = useState("nvidia/nemotron-3-ultra-550b-a55b");
  const [formApiKey, setFormApiKey] = useState("");
  const [formTimeout, setFormTimeout] = useState("30");
  const [formTemperature, setFormTemperature] = useState("");
  const [formMaxTokens, setFormMaxTokens] = useState("");
  const [formIsActive, setFormIsActive] = useState(false);
  const [formError, setFormError] = useState(null);
  const [formSaving, setFormSaving] = useState(false);

  const activeProfile = modelProfiles.find((p) => p.is_active) || null;

  const handleProviderPresetChange = (provider) => {
    setFormProvider(provider);
    if (provider === "nvidia") {
      setFormEndpoint("https://integrate.api.nvidia.com/v1");
      setFormModel("nvidia/nemotron-3-ultra-550b-a55b");
    } else if (provider === "ollama") {
      setFormEndpoint("http://localhost:11434");
      setFormModel("llama3.1:8b");
    } else if (provider === "openai_compatible") {
      setFormEndpoint("http://localhost:8000/v1");
      setFormModel("Qwen/Qwen2.5-7B");
    } else if (provider === "mock") {
      setFormEndpoint("");
      setFormModel("mock-model");
    }
  };

  const openCreateModelModal = () => {
    setEditingProfileId(null);
    setFormName("");
    setFormProvider("nvidia");
    setFormEndpoint("https://integrate.api.nvidia.com/v1");
    setFormModel("nvidia/nemotron-3-ultra-550b-a55b");
    setFormApiKey("");
    setFormTimeout("30");
    setFormTemperature("");
    setFormMaxTokens("");
    setFormIsActive(modelProfiles.length === 0);
    setFormError(null);
    setShowModelModal(true);
  };

  const openEditModelModal = (profile) => {
    setEditingProfileId(profile.id);
    setFormName(profile.name || "");
    setFormProvider(profile.provider || "nvidia");
    setFormEndpoint(profile.endpoint || "");
    setFormModel(profile.model || "");
    setFormApiKey("");
    setFormTimeout(profile.timeout != null ? String(profile.timeout) : "30");
    setFormTemperature(profile.temperature != null ? String(profile.temperature) : "");
    setFormMaxTokens(profile.max_tokens != null ? String(profile.max_tokens) : "");
    setFormIsActive(Boolean(profile.is_active));
    setFormError(null);
    setShowModelModal(true);
  };

  const fetchModelProfiles = async () => {
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
  };

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

  const handleTestDraftProfile = async () => {
    setTestResults((prev) => ({ ...prev, draft: { loading: true } }));
    try {
      const payload = {
        provider: formProvider,
        model: formModel,
        endpoint: formEndpoint,
        api_key: formApiKey,
        timeout: formTimeout ? parseFloat(formTimeout) : 15.0,
      };
      const resp = await fetch(getApiUrl("models/test"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
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

  const handleSaveProfile = async (e) => {
    e.preventDefault();
    setFormError(null);
    setFormSaving(true);
    try {
      const payload = {
        name: formName.trim(),
        provider: formProvider,
        model: formModel.trim(),
        endpoint: formEndpoint.trim(),
        timeout: formTimeout ? parseFloat(formTimeout) : 30.0,
        is_active: formIsActive,
      };
      if (formApiKey.trim()) {
        payload.api_key = formApiKey.trim();
      }
      if (formTemperature !== "") {
        payload.temperature = parseFloat(formTemperature);
      }
      if (formMaxTokens !== "") {
        payload.max_tokens = parseInt(formMaxTokens, 10);
      }

      const url = editingProfileId ? `${getApiUrl("models")}${editingProfileId}/` : getApiUrl("models");
      const method = editingProfileId ? "PATCH" : "POST";

      const resp = await fetch(url, {
        method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.error || `HTTP ${resp.status}`);
      }

      setShowModelModal(false);
      setEditingProfileId(null);
      await fetchModelProfiles();
    } catch (err) {
      setFormError(err.message || "Failed to save model profile.");
    } finally {
      setFormSaving(false);
    }
  };

  const getApiUrl = (endpoint) => {
    const base = process.env.NEXT_PUBLIC_API_URL
      ? process.env.NEXT_PUBLIC_API_URL.replace(/\/$/, "")
      : "";
    const cleanEndpoint = endpoint.replace(/^\/|\/$/g, "");
    return `${base}/api/${cleanEndpoint}/`;
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

  // Fetch Research History
  const fetchHistory = async () => {
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
  };

  useEffect(() => {
    fetchDocuments();
    fetchHistory();
    fetchModelProfiles();
  }, []);

  // Timer for elapsed seconds during active run
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
  }, [activeRunId]);

  // Submit Research Objective (Async POST -> 202)
  const handleResearchSubmit = async (e) => {
    e.preventDefault();
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

      // 202 Accepted: Initialize polling
      setActiveRunId(data.run_id);
      setActiveRunStatus(data.status || "queued");
      setActiveRunMode(data.mode || researchMode);
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

  // File Selection Handler
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

  // Ingest New Document
  const handleAddDocument = async (e) => {
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

        const response = await fetch(getApiUrl("documents"), {
          method: "POST",
          body: formData,
        });

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || data.detail || `HTTP ${response.status}`);
        }

        setIngestSuccess({
          id: data.id,
          title: data.title,
          chunkCount: data.chunk_count,
          sourceType: data.source_type,
        });

        setSelectedFile(null);
        if (fileInputRef.current) fileInputRef.current.value = "";
        setNewTitle("");
        setNewContent("");
        setNewSource("");
        setShowAddForm(false);
        await fetchDocuments();
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
        if (newTitle.trim()) {
          payload.title = newTitle.trim();
        }

        const response = await fetch(getApiUrl("documents"), {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        });

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || data.detail || `HTTP ${response.status}`);
        }

        setIngestSuccess({
          id: data.id,
          title: data.title,
          chunkCount: data.chunk_count,
          sourceType: data.source_type,
          url: data.url,
        });

        setNewUrl("");
        setNewTitle("");
        setShowAddForm(false);
        await fetchDocuments();
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
          sourceType: data.source_type,
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
            <span className="phase-pill">M15 — Model Profiles</span>
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
            {activeRunId && (
              <span className="tab-pulse-dot" title="Research in progress" />
            )}
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
          <button
            type="button"
            id="tab-history"
            className={`nav-tab ${activeTab === "history" ? "active" : ""}`}
            onClick={() => {
              setActiveTab("history");
              fetchHistory();
            }}
          >
            <span className="tab-icon" aria-hidden="true">⏱</span>
            <span>Research History</span>
            <span className="tab-count-badge" id="history-count-badge">
              {historyRuns.length}
            </span>
          </button>
          <button
            type="button"
            id="tab-models"
            className={`nav-tab ${activeTab === "models" ? "active" : ""}`}
            onClick={() => {
              setActiveTab("models");
              fetchModelProfiles();
            }}
          >
            <span className="tab-icon" aria-hidden="true">🤖</span>
            <span>Models</span>
            <span className="tab-count-badge" id="models-count-badge">
              {modelProfiles.length}
            </span>
          </button>
        </nav>

        {/* ================================================================= */}
        {/* TAB 1: AUTONOMOUS RESEARCH VIEW */}
        {/* ================================================================= */}
        {activeTab === "research" && (
          <div className="research-view-container">
            {/* Input Form Card */}
            <section className="card" aria-label="Research input">
              <form onSubmit={handleResearchSubmit} className="form-group">
                {modelProfiles.length === 0 && (
                  <div className="warning-banner" id="no-model-warning-banner">
                    <div className="warning-banner-text">
                      <span className="warning-banner-title">⚠️ No research model configured</span>
                      <span>Configure a model profile before starting research.</span>
                    </div>
                    <button
                      type="button"
                      className="btn btn-secondary btn-sm"
                      id="btn-configure-model-banner"
                      onClick={() => setActiveTab("models")}
                    >
                      Configure Model
                    </button>
                  </div>
                )}

                {/* Model Profile Selector (M15) */}
                <div className="model-selector-container">
                  <div className="mode-selector-header">
                    <label htmlFor="model-profile-select" className="mode-selector-label">
                      Inference Model Profile
                    </label>
                    <span className="mode-selector-desc">
                      {activeProfile
                        ? `${activeProfile.name} • ${activeProfile.provider} (${activeProfile.model})`
                        : "No active model profile configured"}
                    </span>
                  </div>
                  <div className="mode-select-wrapper">
                    <select
                      id="model-profile-select"
                      className="mode-select-dropdown"
                      value={selectedModelProfileId || (activeProfile ? activeProfile.id : "")}
                      onChange={(e) => setSelectedModelProfileId(e.target.value)}
                      disabled={loading || modelProfiles.length === 0}
                    >
                      {modelProfiles.length === 0 ? (
                        <option value="">No model profile configured</option>
                      ) : (
                        modelProfiles.map((p) => (
                          <option key={p.id} value={p.id}>
                            {p.name} ({p.provider} / {p.model}){p.is_active ? " — Active" : ""}
                          </option>
                        ))
                      )}
                    </select>
                  </div>
                </div>

                {/* Research Mode Selector */}
                <div className="mode-selector-container">
                  <div className="mode-selector-header">
                    <label htmlFor="mode-select" className="mode-selector-label">
                      Research Mode
                    </label>
                    <span className="mode-selector-desc">
                      {researchMode === "model_knowledge" && "Pretrained LLM knowledge only (no RAG or web)."}
                      {researchMode === "knowledge_base" && "Grounded search strictly against indexed Knowledge Base."}
                      {researchMode === "web" && "Live web search only (no Knowledge Base)."}
                      {researchMode === "web_knowledge_base" && "Hybrid: Autonomous planner uses both KB and Web."}
                    </span>
                  </div>
                  <div className="mode-select-wrapper">
                    <select
                      id="mode-select"
                      className="mode-select-dropdown"
                      value={researchMode ?? "knowledge_base"}
                      onChange={(e) => setResearchMode(e.target.value)}
                      disabled={loading}
                    >
                      <option value="knowledge_base">Knowledge Base (Default)</option>
                      <option value="web">Web</option>
                      <option value="web_knowledge_base">Web + Knowledge Base</option>
                      <option value="model_knowledge">Model Knowledge</option>
                    </select>
                  </div>
                </div>

                <div className="label">
                  <label htmlFor="objective-input">Research Objective</label>
                  <span className="label-hint">
                    Async persistent run • {documents.length} doc(s) indexed
                  </span>
                </div>

                <textarea
                  id="objective-input"
                  className="textarea"
                  placeholder="Enter a research objective (e.g. How do the AURA Model Gateway, RAG pipeline, and Agent Runtime work together?)..."
                  value={objective ?? ""}
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
                        <span>Queuing...</span>
                      </>
                    ) : (
                      <>
                        <span>Start Research</span>
                        <span aria-hidden="true">→</span>
                      </>
                    )}
                  </button>
                </div>
              </form>
            </section>

            {/* Active Research Polling Card */}
            {activeRunId && (
              <div className="active-run-card" role="status" aria-live="polite">
                <div className="active-run-header">
                  <div className="active-run-status-badge">
                    <span className="status-spinner-small" aria-hidden="true" />
                    <span className="active-status-text">
                      Status: {activeRunStatus || "queued"}
                    </span>
                  </div>
                  <span className="mode-pill mode-pill-active">
                    Mode: {formatMode(activeRunMode || researchMode)}
                  </span>
                  <span className="active-run-timer">{activeRunElapsed}s elapsed</span>
                </div>

                <div className="active-run-body">
                  <p className="active-run-title">Autonomous Research in Progress</p>
                  <p className="active-run-sub">
                    Celery worker is executing iterative planning, information retrieval, grounded synthesis, and citation verification in the background.
                  </p>
                  <div className="active-run-meta">
                    <span>Run ID: <code>{activeRunId}</code></span>
                  </div>
                </div>

                <div className="active-run-actions">
                  <button
                    type="button"
                    className="btn-cancel-run"
                    onClick={handleCancelResearch}
                  >
                    ✕ Cancel Research Run
                  </button>
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
                  {(error.includes("No research model configured") || error.includes("Configure a model profile")) && (
                    <div style={{ marginTop: "0.6rem" }}>
                      <button
                        type="button"
                        className="btn btn-secondary btn-sm"
                        id="btn-error-configure-model"
                        onClick={() => {
                          setActiveTab("models");
                          fetchModelProfiles();
                        }}
                      >
                        Configure Model Profile →
                      </button>
                    </div>
                  )}
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
                  <span className="badge badge-mode">
                    Mode: {formatMode(result.mode || activeRunMode || researchMode)}
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
                  {(typeof result.duration_seconds === "number" || typeof result.duration_ms === "number") && (
                    <span className="badge badge-info">
                      Duration: {typeof result.duration_seconds === "number"
                        ? `${result.duration_seconds.toFixed(2)}s`
                        : `${(result.duration_ms / 1000).toFixed(2)}s`}
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
                          {src.url && (
                            <a
                              href={src.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="source-url-link"
                            >
                              🌐 {src.url}
                            </a>
                          )}
                          <div className="source-meta">
                            <span className="label-hint">
                              {src.chunk_count || src.chunk_ids?.length || 0} chunk(s)
                            </span>
                            {src.pages && src.pages.length > 0 && (
                              <span className="source-pages-tag">
                                Pages: {src.pages.join(", ")}
                              </span>
                            )}
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
                                Score: {formatScore(ev.score)}
                              </span>
                            )}
                          </div>
                          <div className="evidence-content">{ev.content}</div>
                          <div className="evidence-footer">
                            <span>Origin: {ev.document_source || "unknown"}</span>
                            {ev.url && (
                              <a
                                href={ev.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="evidence-url-link"
                              >
                                🌐 {ev.url}
                              </a>
                            )}
                            {(ev.page != null || ev.metadata?.page != null) && (
                              <span className="evidence-page-tag">
                                Page: {ev.page != null ? ev.page : ev.metadata.page}
                              </span>
                            )}
                            {ev.document_id ? <span className="evidence-id-tag">ID: {ev.document_id}</span> : ""}
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

        {/* ================================================================= */}
        {/* TAB 2: KNOWLEDGE BASE VIEW (M8) */}
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
                  {ingestMode === "paste" && (
                    <button
                      type="button"
                      className="btn-secondary-sm"
                      onClick={loadSampleMarkdown}
                      disabled={ingesting}
                    >
                      📄 Load Sample Markdown
                    </button>
                  )}
                </div>

                {/* Mode Selector: Upload File vs Web URL vs Paste Text */}
                <div className="kb-mode-toggle" role="tablist">
                  <button
                    type="button"
                    role="tab"
                    aria-selected={ingestMode === "file"}
                    className={`kb-mode-tab ${ingestMode === "file" ? "active" : ""}`}
                    onClick={() => {
                      setIngestMode("file");
                      setAddError(null);
                    }}
                    disabled={ingesting}
                  >
                    📁 Upload File (.txt, .md, .pdf, .docx)
                  </button>
                  <button
                    type="button"
                    role="tab"
                    aria-selected={ingestMode === "url"}
                    className={`kb-mode-tab ${ingestMode === "url" ? "active" : ""}`}
                    onClick={() => {
                      setIngestMode("url");
                      setAddError(null);
                    }}
                    disabled={ingesting}
                  >
                    🌐 Ingest Web URL
                  </button>
                  <button
                    type="button"
                    role="tab"
                    aria-selected={ingestMode === "paste"}
                    className={`kb-mode-tab ${ingestMode === "paste" ? "active" : ""}`}
                    onClick={() => {
                      setIngestMode("paste");
                      setAddError(null);
                    }}
                    disabled={ingesting}
                  >
                    ✍️ Paste Text / Markdown
                  </button>
                </div>

                <form onSubmit={handleAddDocument} className="form-group">
                  {ingestMode === "file" ? (
                    <>
                      <div key="field-file-upload">
                        <label htmlFor="doc-file-input" className="label-text">
                          Document File <span className="required-star">*</span>
                          <span className="label-hint"> (TXT, Markdown, PDF, DOCX up to 20MB)</span>
                        </label>
                        <input
                          key="input-doc-file"
                          id="doc-file-input"
                          ref={fileInputRef}
                          type="file"
                          accept=".txt,.md,.pdf,.docx"
                          className="file-input"
                          onChange={handleFileChange}
                          disabled={ingesting}
                          required
                        />
                        {selectedFile && (
                          <div className="selected-file-info">
                            <span className="selected-file-name">📄 {selectedFile.name}</span>
                            <span className="selected-file-size">({formatFileSize(selectedFile.size)})</span>
                          </div>
                        )}
                      </div>

                      <div key="field-file-title">
                        <label htmlFor="doc-title-input" className="label-text">
                          Document Title <span className="label-hint">(optional, defaults to filename)</span>
                        </label>
                        <input
                          key="input-file-title"
                          id="doc-title-input"
                          type="text"
                          className="text-input"
                          placeholder="e.g. Quantum Processor Architecture"
                          value={newTitle ?? ""}
                          onChange={(e) => setNewTitle(e.target.value)}
                          disabled={ingesting}
                        />
                      </div>

                      <div key="field-file-source">
                        <label htmlFor="doc-source-input" className="label-text">
                          Source Identifier <span className="label-hint">(optional origin or path)</span>
                        </label>
                        <input
                          key="input-file-source"
                          id="doc-source-input"
                          type="text"
                          className="text-input"
                          placeholder="e.g. specs/quantum.pdf"
                          value={newSource ?? ""}
                          onChange={(e) => setNewSource(e.target.value)}
                          disabled={ingesting}
                        />
                      </div>
                    </>
                  ) : ingestMode === "url" ? (
                    <>
                      <div key="field-url-input">
                        <label htmlFor="doc-url-input" className="label-text">
                          Public Web Page URL <span className="required-star">*</span>
                          <span className="label-hint"> (HTTP/HTTPS article or documentation page)</span>
                        </label>
                        <input
                          key="input-web-url"
                          id="doc-url-input"
                          type="url"
                          className="text-input"
                          placeholder="https://example.com/article"
                          value={newUrl ?? ""}
                          onChange={(e) => setNewUrl(e.target.value)}
                          disabled={ingesting}
                          required
                        />
                      </div>

                      <div key="field-url-title">
                        <label htmlFor="doc-url-title-input" className="label-text">
                          Document Title <span className="label-hint">(optional, defaults to page title)</span>
                        </label>
                        <input
                          key="input-url-title"
                          id="doc-url-title-input"
                          type="text"
                          className="text-input"
                          placeholder="e.g. Quantum Computing Breakthrough"
                          value={newTitle ?? ""}
                          onChange={(e) => setNewTitle(e.target.value)}
                          disabled={ingesting}
                        />
                      </div>
                    </>
                  ) : (
                    <>
                      <div key="field-paste-title">
                        <label htmlFor="doc-paste-title-input" className="label-text">
                          Title <span className="required-star">*</span>
                        </label>
                        <input
                          key="input-paste-title"
                          id="doc-paste-title-input"
                          type="text"
                          className="text-input"
                          placeholder="e.g. AURA Architecture Specifications"
                          value={newTitle ?? ""}
                          onChange={(e) => setNewTitle(e.target.value)}
                          disabled={ingesting}
                          required
                        />
                      </div>

                      <div key="field-paste-source">
                        <label htmlFor="doc-paste-source-input" className="label-text">
                          Source Identifier <span className="label-hint">(optional file path, URL, or tag)</span>
                        </label>
                        <input
                          key="input-paste-source"
                          id="doc-paste-source-input"
                          type="text"
                          className="text-input"
                          placeholder="e.g. docs/architecture.md"
                          value={newSource ?? ""}
                          onChange={(e) => setNewSource(e.target.value)}
                          disabled={ingesting}
                        />
                      </div>

                      <div key="field-paste-content">
                        <label htmlFor="doc-content-input" className="label-text">
                          Content (Plain Text or Markdown) <span className="required-star">*</span>
                        </label>
                        <textarea
                          key="textarea-paste-content"
                          id="doc-content-input"
                          className="textarea"
                          placeholder="Paste or write Markdown/text content here..."
                          value={newContent ?? ""}
                          onChange={(e) => setNewContent(e.target.value)}
                          disabled={ingesting}
                          rows={8}
                          required
                        />
                      </div>
                    </>
                  )}

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
                      className="btn-ingest"
                      disabled={
                        ingesting ||
                        (ingestMode === "file" && !selectedFile) ||
                        (ingestMode === "url" && !newUrl.trim()) ||
                        (ingestMode === "paste" && (!newTitle.trim() || !newContent.trim()))
                      }
                    >
                      {ingesting ? (
                        <>
                          <span className="spinner" aria-hidden="true" />
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
                  const typeBadge = getDocTypeBadge(doc.source_type, doc.filename || doc.source);

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
                            <span className={`doc-type-badge ${typeBadge.className}`}>
                              {typeBadge.icon} {typeBadge.label}
                            </span>
                            <span className="chunk-count-badge">
                              {doc.chunk_count} chunk{doc.chunk_count === 1 ? "" : "s"}
                            </span>
                            {doc.file_size != null && (
                              <span className="doc-size-tag">
                                {formatFileSize(doc.file_size)}
                              </span>
                            )}
                            {doc.domain && (
                              <span className="doc-domain-tag">
                                🔗 {doc.domain}
                              </span>
                            )}
                            {doc.source && (
                              <span className="doc-source-tag">
                                {doc.source_type === "web_page" ? "🌐 " : "📁 "}{doc.source}
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
                              <span className="preview-label">Format:</span>
                              <span className="preview-val">{typeBadge.label}</span>
                            </div>
                            {doc.url && (
                              <div>
                                <span className="preview-label">URL:</span>
                                <a
                                  href={doc.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="preview-link"
                                >
                                  {doc.url}
                                </a>
                              </div>
                            )}
                            {doc.domain && (
                              <div>
                                <span className="preview-label">Domain:</span>
                                <span className="preview-val">{doc.domain}</span>
                              </div>
                            )}
                            {doc.metadata?.author && (
                              <div>
                                <span className="preview-label">Author:</span>
                                <span className="preview-val">{doc.metadata.author}</span>
                              </div>
                            )}
                            {doc.metadata?.date && (
                              <div>
                                <span className="preview-label">Date:</span>
                                <span className="preview-val">{doc.metadata.date}</span>
                              </div>
                            )}
                            {doc.filename && (
                              <div>
                                <span className="preview-label">Filename:</span>
                                <span className="preview-val">{doc.filename}</span>
                              </div>
                            )}
                            {doc.metadata?.page_count && (
                              <div>
                                <span className="preview-label">Pages:</span>
                                <span className="preview-val">{doc.metadata.page_count}</span>
                              </div>
                            )}
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
        {/* TAB 3: RESEARCH HISTORY VIEW (M9) */}
        {/* ================================================================= */}
        {activeTab === "history" && (
          <div className="history-view-container">
            <div className="history-header-bar">
              <div>
                <h2 className="history-title">Research History</h2>
                <p className="history-subtitle">
                  Persistent log of autonomous research runs and verified evidence traces
                </p>
              </div>
              <button
                type="button"
                className="btn-refresh-history"
                onClick={fetchHistory}
                disabled={historyLoading}
              >
                🔄 Refresh
              </button>
            </div>

            {historyLoading && historyRuns.length === 0 ? (
              <div className="loading-card" role="status" aria-live="polite">
                <div className="loading-spinner-large" aria-hidden="true" />
                <p className="loading-title">Loading Research History...</p>
              </div>
            ) : historyRuns.length === 0 ? (
              <div className="history-empty-state">
                <div className="history-empty-icon" aria-hidden="true">⏱</div>
                <h3 className="history-empty-title">No research history yet</h3>
                <p className="history-empty-desc">
                  Start an objective in the Autonomous Research tab. Your executed runs and evidence results will be safely recorded here.
                </p>
                <button
                  type="button"
                  className="btn-add-doc"
                  onClick={() => setActiveTab("research")}
                >
                  🔬 Start First Research Run
                </button>
              </div>
            ) : (
              <div className="history-runs-list" id="history-runs-list">
                {historyRuns.map((run) => (
                  <article key={run.run_id} className="history-run-card" id={`run-card-${run.run_id}`}>
                    <div className="history-run-main">
                      <div className="history-run-info">
                        <h3 className="history-run-objective">{run.objective}</h3>
                        <div className="history-run-meta-row">
                          <span
                            className={`status-pill ${
                              run.status === "completed"
                                ? "status-ready"
                                : run.status === "failed"
                                ? "status-error"
                                : run.status === "running"
                                ? "status-pending"
                                : run.status === "cancelled"
                                ? "status-cancelled"
                                : "status-queued"
                            }`}
                          >
                            ● {run.status}
                          </span>
                          <span className="mode-pill">
                            {formatMode(run.mode)}
                          </span>
                          {run.is_grounded && (
                            <span className="grounded-pill">✓ Grounded</span>
                          )}
                          {((typeof run.duration_seconds === "number" && run.duration_seconds > 0) ||
                            (typeof run.duration_ms === "number" && run.duration_ms > 0)) && (
                            <span className="duration-pill">
                              ⏱ {typeof run.duration_seconds === "number" && run.duration_seconds > 0
                                ? `${run.duration_seconds.toFixed(1)}s`
                                : `${(run.duration_ms / 1000).toFixed(1)}s`}
                            </span>
                          )}
                          {run.citation_count > 0 && (
                            <span className="citations-pill">
                              🔖 {run.citation_count} citation{run.citation_count === 1 ? "" : "s"}
                            </span>
                          )}
                          {run.created_at && (
                            <span className="history-date">
                              {new Date(run.created_at).toLocaleString(undefined, {
                                month: "short",
                                day: "numeric",
                                hour: "2-digit",
                                minute: "2-digit",
                              })}
                            </span>
                          )}
                        </div>
                      </div>

                      <div className="history-run-actions">
                        <button
                          type="button"
                          className="btn-view-run"
                          onClick={() => handleSelectHistoryRun(run.run_id)}
                        >
                          View Result →
                        </button>
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ================================================================= */}
        {/* TAB 4: MODEL PROFILES VIEW (M15) */}
        {/* ================================================================= */}
        {activeTab === "models" && (
          <div className="models-view-container">
            <div className="section-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div>
                <h2 className="section-title">Model Profiles</h2>
                <p className="section-subtitle">
                  Configure local and hosted LLM inference profiles for autonomous research.
                </p>
              </div>
              <button
                type="button"
                className="btn-add-doc"
                id="btn-add-model"
                onClick={openCreateModelModal}
              >
                <span>🤖 + Add Model Profile</span>
              </button>
            </div>

            {modelsLoading && modelProfiles.length === 0 ? (
              <div className="loading-card" role="status" aria-live="polite">
                <div className="loading-spinner-large" aria-hidden="true" />
                <p className="loading-title">Loading Model Profiles...</p>
              </div>
            ) : modelProfiles.length === 0 ? (
              <div className="history-empty-state">
                <div className="history-empty-icon" aria-hidden="true">🤖</div>
                <h3 className="history-empty-title">No Model Profiles Configured</h3>
                <p className="history-empty-desc">
                  AURA requires an active model profile to execute research generation. Add a local Ollama model, NVIDIA NIM, or generic OpenAI-compatible server.
                </p>
                <button
                  type="button"
                  className="btn-add-doc"
                  id="btn-add-first-model"
                  onClick={openCreateModelModal}
                >
                  🤖 Add First Model Profile
                </button>
              </div>
            ) : (
              <div className="models-grid" id="models-grid">
                {modelProfiles.map((profile) => (
                  <div
                    key={profile.id}
                    className={`model-card ${profile.is_active ? "active-model-card" : ""}`}
                    id={`model-card-${profile.id}`}
                  >
                    <div>
                      <div className="model-card-header">
                        <div className="model-title-group">
                          <h3 className="model-card-title">{profile.name}</h3>
                          <div className="model-badge-group">
                            {profile.is_active && (
                              <span className="badge-active">● Active</span>
                            )}
                            <span className="badge-provider">
                              {profile.provider === "nvidia"
                                ? "NVIDIA NIM"
                                : profile.provider === "ollama"
                                ? "Ollama"
                                : profile.provider === "openai_compatible"
                                ? "OpenAI-compatible"
                                : profile.provider}
                            </span>
                          </div>
                        </div>
                      </div>

                      <div className="model-meta-table">
                        <div className="model-meta-row">
                          <span className="model-meta-key">Model</span>
                          <span className="model-meta-val">{profile.model}</span>
                        </div>
                        <div className="model-meta-row">
                          <span className="model-meta-key">Endpoint</span>
                          <span className="model-meta-val">{profile.endpoint || "Default"}</span>
                        </div>
                        <div className="model-meta-row">
                          <span className="model-meta-key">Credentials</span>
                          <span className="model-meta-val">
                            {profile.has_api_key ? `Configured (${profile.api_key_masked})` : "None required"}
                          </span>
                        </div>
                        <div className="model-meta-row">
                          <span className="model-meta-key">Parameters</span>
                          <span className="model-meta-val">
                            {profile.temperature != null ? `T: ${profile.temperature}` : "T: default"} • {profile.timeout}s timeout
                          </span>
                        </div>
                      </div>

                      {testResults[profile.id] && (
                        <div
                          className={`test-connection-banner ${
                            testResults[profile.id].loading
                              ? ""
                              : testResults[profile.id].success
                              ? "success"
                              : "error"
                          }`}
                        >
                          {testResults[profile.id].loading ? (
                            <span>Testing endpoint connection...</span>
                          ) : testResults[profile.id].success ? (
                            <span>✓ {testResults[profile.id].message} ({testResults[profile.id].latency_ms}ms)</span>
                          ) : (
                            <span>✗ {testResults[profile.id].error}</span>
                          )}
                        </div>
                      )}
                    </div>

                    <div className="model-card-actions">
                      <div style={{ display: "flex", gap: "0.5rem" }}>
                        {!profile.is_active && (
                          <button
                            type="button"
                            className="btn btn-secondary btn-sm"
                            id={`btn-activate-${profile.id}`}
                            onClick={() => handleActivateProfile(profile.id)}
                          >
                            Set Active
                          </button>
                        )}
                        <button
                          type="button"
                          className="btn btn-secondary btn-sm"
                          id={`btn-test-${profile.id}`}
                          onClick={() => handleTestProfile(profile.id)}
                          disabled={testResults[profile.id]?.loading}
                        >
                          Test
                        </button>
                      </div>
                      <div style={{ display: "flex", gap: "0.5rem" }}>
                        <button
                          type="button"
                          className="btn btn-secondary btn-sm"
                          id={`btn-edit-${profile.id}`}
                          onClick={() => openEditModelModal(profile)}
                        >
                          Edit
                        </button>
                        <button
                          type="button"
                          className="btn btn-secondary btn-sm btn-danger"
                          id={`btn-delete-${profile.id}`}
                          onClick={() => handleDeleteProfile(profile.id)}
                        >
                          Delete
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Modal for Add / Edit Model Profile */}
            {showModelModal && (
              <div className="modal-backdrop" onClick={() => setShowModelModal(false)}>
                <div className="modal-content-card" onClick={(e) => e.stopPropagation()}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.25rem" }}>
                    <h3 style={{ fontSize: "1.25rem", fontWeight: "600" }}>
                      {editingProfileId ? "Edit Model Profile" : "Add Model Profile"}
                    </h3>
                    <button
                      type="button"
                      className="btn btn-secondary btn-sm"
                      onClick={() => setShowModelModal(false)}
                    >
                      ✕
                    </button>
                  </div>

                  {formError && (
                    <div className="test-connection-banner error" style={{ marginBottom: "1rem" }}>
                      <span>✗ {formError}</span>
                    </div>
                  )}

                  <form onSubmit={handleSaveProfile} style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                    <div>
                      <label className="label" htmlFor="model-form-name">Profile Name *</label>
                      <input
                        id="model-form-name"
                        type="text"
                        className="mode-select-dropdown"
                        style={{ width: "100%", padding: "0.6rem 0.8rem", marginTop: "0.25rem" }}
                        placeholder="e.g. My Local Llama"
                        value={formName}
                        onChange={(e) => setFormName(e.target.value)}
                        required
                      />
                    </div>

                    <div>
                      <label className="label" htmlFor="model-form-provider">Provider *</label>
                      <select
                        id="model-form-provider"
                        className="mode-select-dropdown"
                        style={{ width: "100%", padding: "0.6rem 0.8rem", marginTop: "0.25rem" }}
                        value={formProvider}
                        onChange={(e) => handleProviderPresetChange(e.target.value)}
                      >
                        <option value="nvidia">NVIDIA NIM</option>
                        <option value="ollama">Ollama (Local / Native HTTP)</option>
                        <option value="openai_compatible">OpenAI-compatible (vLLM, LM Studio, etc.)</option>
                        <option value="mock">Mock Provider (Offline Testing)</option>
                      </select>
                    </div>

                    <div>
                      <label className="label" htmlFor="model-form-model">Model Identifier *</label>
                      <input
                        id="model-form-model"
                        type="text"
                        className="mode-select-dropdown"
                        style={{ width: "100%", padding: "0.6rem 0.8rem", marginTop: "0.25rem" }}
                        placeholder="e.g. nvidia/nemotron-3-ultra-550b-a55b or llama3.1:8b"
                        value={formModel}
                        onChange={(e) => setFormModel(e.target.value)}
                        required
                      />
                    </div>

                    <div>
                      <label className="label" htmlFor="model-form-endpoint">API Endpoint URL</label>
                      <input
                        id="model-form-endpoint"
                        type="text"
                        className="mode-select-dropdown"
                        style={{ width: "100%", padding: "0.6rem 0.8rem", marginTop: "0.25rem" }}
                        placeholder="e.g. http://localhost:11434"
                        value={formEndpoint}
                        onChange={(e) => setFormEndpoint(e.target.value)}
                      />
                    </div>

                    <div>
                      <label className="label" htmlFor="model-form-apikey">
                        API Key / Token {editingProfileId && "(leave blank to preserve existing key)"}
                      </label>
                      <input
                        id="model-form-apikey"
                        type="password"
                        className="mode-select-dropdown"
                        style={{ width: "100%", padding: "0.6rem 0.8rem", marginTop: "0.25rem" }}
                        placeholder={formProvider === "ollama" ? "Not required for local Ollama" : "sk-... or nvapi-..."}
                        value={formApiKey}
                        onChange={(e) => setFormApiKey(e.target.value)}
                      />
                    </div>

                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "0.75rem" }}>
                      <div>
                        <label className="label" htmlFor="model-form-temp">Temperature</label>
                        <input
                          id="model-form-temp"
                          type="number"
                          step="0.05"
                          min="0"
                          max="2"
                          className="mode-select-dropdown"
                          style={{ width: "100%", padding: "0.6rem 0.8rem", marginTop: "0.25rem" }}
                          placeholder="e.g. 0.7"
                          value={formTemperature}
                          onChange={(e) => setFormTemperature(e.target.value)}
                        />
                      </div>
                      <div>
                        <label className="label" htmlFor="model-form-tokens">Max Tokens</label>
                        <input
                          id="model-form-tokens"
                          type="number"
                          min="1"
                          className="mode-select-dropdown"
                          style={{ width: "100%", padding: "0.6rem 0.8rem", marginTop: "0.25rem" }}
                          placeholder="e.g. 1024"
                          value={formMaxTokens}
                          onChange={(e) => setFormMaxTokens(e.target.value)}
                        />
                      </div>
                      <div>
                        <label className="label" htmlFor="model-form-timeout">Timeout (s)</label>
                        <input
                          id="model-form-timeout"
                          type="number"
                          min="1"
                          className="mode-select-dropdown"
                          style={{ width: "100%", padding: "0.6rem 0.8rem", marginTop: "0.25rem" }}
                          placeholder="30"
                          value={formTimeout}
                          onChange={(e) => setFormTimeout(e.target.value)}
                        />
                      </div>
                    </div>

                    <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginTop: "0.25rem" }}>
                      <input
                        id="model-form-active"
                        type="checkbox"
                        checked={formIsActive}
                        onChange={(e) => setFormIsActive(e.target.checked)}
                      />
                      <label htmlFor="model-form-active" style={{ fontSize: "0.85rem", cursor: "pointer" }}>
                        Set as Active Model Profile for Research
                      </label>
                    </div>

                    {testResults["draft"] && (
                      <div
                        className={`test-connection-banner ${
                          testResults["draft"].loading
                            ? ""
                            : testResults["draft"].success
                            ? "success"
                            : "error"
                        }`}
                      >
                        {testResults["draft"].loading ? (
                          <span>Testing draft connection...</span>
                        ) : testResults["draft"].success ? (
                          <span>✓ {testResults["draft"].message} ({testResults["draft"].latency_ms}ms)</span>
                        ) : (
                          <span>✗ {testResults["draft"].error}</span>
                        )}
                      </div>
                    )}

                    <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", marginTop: "0.5rem" }}>
                      <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={handleTestDraftProfile}
                        disabled={testResults["draft"]?.loading || !formModel.trim()}
                      >
                        {testResults["draft"]?.loading ? "Testing..." : "Test Connection"}
                      </button>
                      <div style={{ display: "flex", gap: "0.5rem" }}>
                        <button
                          type="button"
                          className="btn btn-secondary"
                          onClick={() => setShowModelModal(false)}
                        >
                          Cancel
                        </button>
                        <button
                          type="submit"
                          className="btn-add-doc"
                          disabled={formSaving}
                        >
                          {formSaving ? "Saving..." : "Save Profile"}
                        </button>
                      </div>
                    </div>
                  </form>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
