/**
 * AURA Frontend Utility & Formatting Helpers
 */

export const VALID_RESEARCH_MODES = [
  "model_knowledge",
  "knowledge_base",
  "web",
  "web_knowledge_base",
];

export const formatMode = (m) => {
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

export const getModeDescription = (m) => {
  switch (m) {
    case "model_knowledge":
      return "Direct pretrained LLM knowledge without external context or RAG retrieval.";
    case "knowledge_base":
      return "Grounded research strictly against indexed internal documents.";
    case "web":
      return "Live web research via SearXNG search engine and crawled pages.";
    case "web_knowledge_base":
      return "Autonomous hybrid research synthesizing internal documents and web sources.";
    default:
      return "Grounded research against indexed knowledge.";
  }
};

export const formatFileSize = (bytes) => {
  if (bytes === null || bytes === undefined || isNaN(bytes)) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

export const getDocTypeBadge = (sourceType, filename) => {
  const t = (sourceType || "").toLowerCase();
  const ext = (filename || "").split(".").pop().toLowerCase();
  if (t === "web_page" || t === "url") {
    return { label: "Web Page", iconType: "web", className: "badge-web" };
  }
  if (t === "pdf" || ext === "pdf") {
    return { label: "PDF", iconType: "pdf", className: "badge-pdf" };
  }
  if (t === "docx" || ext === "docx") {
    return { label: "DOCX", iconType: "docx", className: "badge-docx" };
  }
  if (t === "markdown" || ext === "md" || ext === "markdown") {
    return { label: "Markdown", iconType: "md", className: "badge-md" };
  }
  return { label: "TXT", iconType: "txt", className: "badge-txt" };
};

export const formatScore = (score) => {
  if (typeof score !== "number" || isNaN(score)) return "";
  if (score === 0) return "0.000";
  if (score >= 0.01) return score.toFixed(3);
  if (score >= 0.0001) return score.toFixed(4);
  return score.toFixed(6);
};

export const formatDuration = (runOrResult) => {
  if (!runOrResult) return null;
  if (typeof runOrResult.duration_seconds === "number" && runOrResult.duration_seconds > 0) {
    return `${runOrResult.duration_seconds.toFixed(2)}s`;
  }
  if (typeof runOrResult.duration_ms === "number" && runOrResult.duration_ms > 0) {
    return `${(runOrResult.duration_ms / 1000).toFixed(2)}s`;
  }
  return null;
};

export const formatHistoryDuration = (run) => {
  if (!run) return null;
  if (typeof run.duration_seconds === "number" && run.duration_seconds > 0) {
    return `${run.duration_seconds.toFixed(1)}s`;
  }
  if (typeof run.duration_ms === "number" && run.duration_ms > 0) {
    return `${(run.duration_ms / 1000).toFixed(1)}s`;
  }
  return null;
};

export const formatProviderBadge = (provider) => {
  switch (provider) {
    case "nvidia":
      return "NVIDIA NIM";
    case "ollama":
      return "Ollama";
    case "openai_compatible":
      return "OpenAI Compatible";
    case "mock":
      return "Mock Provider";
    default:
      return provider || "Custom";
  }
};

export const resolveActiveProfile = (profiles) => {
  if (!Array.isArray(profiles) || profiles.length === 0) return null;
  return profiles.find((p) => p.is_active) || profiles[0] || null;
};

export const validateWebUrl = (urlStr) => {
  if (!urlStr || typeof urlStr !== "string") return false;
  const trimmed = urlStr.trim();
  if (!trimmed) return false;
  try {
    const parsed = new URL(trimmed);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch (_) {
    return false;
  }
};

export const validateFileExtension = (filename) => {
  if (!filename || typeof filename !== "string") return false;
  const lower = filename.toLowerCase();
  const supported = [".txt", ".md", ".pdf", ".docx"];
  return supported.some((ext) => lower.endsWith(ext));
};

export const groupHistoryByDate = (runs) => {
  if (!Array.isArray(runs) || runs.length === 0) return {};
  
  const groups = {
    Today: [],
    Yesterday: [],
    Earlier: [],
  };

  const now = new Date();
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const yesterdayStart = todayStart - 86400000;

  runs.forEach((run) => {
    if (!run.created_at) {
      groups.Earlier.push(run);
      return;
    }
    const itemTime = new Date(run.created_at).getTime();
    if (itemTime >= todayStart) {
      groups.Today.push(run);
    } else if (itemTime >= yesterdayStart) {
      groups.Yesterday.push(run);
    } else {
      groups.Earlier.push(run);
    }
  });

  return groups;
};

/**
 * Generate contextual research prompt suggestions based on available history and knowledge.
 * Avoids generic static prompts and gracefully returns [] if no context exists.
 */
export const getContextualSuggestions = (historyRuns = [], documents = []) => {
  const suggestions = [];

  // 1. Suggestions from recent research history
  if (Array.isArray(historyRuns) && historyRuns.length > 0) {
    const latestRun = historyRuns[0];
    const obj = (latestRun.objective || "").trim();

    if (obj) {
      // Extract main topic phrase
      const cleanObj = obj.replace(/^(what is|how does|tell me about|explain|compare)\s+/i, "").replace(/[.?]+$/, "");
      const shortTopic = cleanObj.length > 35 ? cleanObj.slice(0, 32) + "..." : cleanObj;

      suggestions.push({
        label: `Trade-offs of ${shortTopic}`,
        text: `Analyze the critical trade-offs, constraints, and alternative approaches to: ${cleanObj}.`,
        type: "history",
      });

      if (historyRuns.length > 1) {
        const prevRun = historyRuns[1];
        const prevClean = (prevRun.objective || "").replace(/^(what is|how does|tell me about|explain|compare)\s+/i, "").replace(/[.?]+$/, "");
        if (prevClean && prevClean !== cleanObj) {
          const prevShort = prevClean.length > 25 ? prevClean.slice(0, 22) + "..." : prevClean;
          suggestions.push({
            label: `Compare ${shortTopic} & ${prevShort}`,
            text: `Compare and synthesize findings between "${cleanObj}" and "${prevClean}".`,
            type: "history",
          });
        }
      }
    }
  }

  // 2. Suggestions from indexed knowledge documents
  if (Array.isArray(documents) && documents.length > 0) {
    const doc1 = documents[0];
    if (doc1 && doc1.title) {
      const shortTitle = doc1.title.length > 28 ? doc1.title.slice(0, 25) + "..." : doc1.title;
      suggestions.push({
        label: `Synthesize "${shortTitle}"`,
        text: `Summarize the core architectural concepts, key evidence, and implications described in "${doc1.title}".`,
        type: "document",
      });
    }

    if (documents.length > 1 && suggestions.length < 3) {
      const doc2 = documents[1];
      if (doc2 && doc2.title) {
        const short1 = documents[0].title.slice(0, 18);
        const short2 = doc2.title.slice(0, 18);
        suggestions.push({
          label: `Compare ${short1} & ${short2}`,
          text: `Compare findings, architectural decisions, and constraints between "${documents[0].title}" and "${doc2.title}".`,
          type: "document",
        });
      }
    }
  }

  return suggestions.slice(0, 3);
};

/**
 * Return user-facing truthful dynamic activity text during research based on active mode and elapsed time.
 */
export const getDynamicActivityText = (mode, elapsedSeconds = 0, documents = []) => {
  const m = mode || "knowledge_base";

  if (m === "model_knowledge") {
    if (elapsedSeconds <= 3) return "Working from model knowledge";
    if (elapsedSeconds <= 7) return "Analyzing reasoning pathways";
    return "Synthesizing research response";
  }

  if (m === "knowledge_base") {
    if (elapsedSeconds <= 2) {
      if (documents.length > 0 && documents[0]?.title) {
        const name = documents[0].title.length > 30 ? documents[0].title.slice(0, 28) + "..." : documents[0].title;
        return `Analyzing indexed knowledge (${name})`;
      }
      return "Analyzing indexed documents";
    }
    if (elapsedSeconds <= 6) return "Retrieving relevant passages";
    return "Synthesizing knowledge base findings";
  }

  if (m === "web") {
    if (elapsedSeconds <= 3) return "Searching the web";
    if (elapsedSeconds <= 7) return "Reviewing retrieved web sources";
    return "Synthesizing findings";
  }

  if (m === "web_knowledge_base") {
    if (elapsedSeconds <= 3) return "Searching the web";
    if (elapsedSeconds <= 6) {
      return documents.length > 0
        ? `Comparing web sources with ${documents.length} knowledge base documents`
        : "Comparing web sources with indexed knowledge";
    }
    return "Cross-checking sources & synthesizing";
  }

  return "Analyzing research context";
};
