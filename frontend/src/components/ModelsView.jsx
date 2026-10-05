"use client";

import { useState } from "react";
import { formatProviderBadge } from "../utils/helpers";
import {
  RefreshIcon,
  WarningIcon,
  CloseIcon,
  CheckIcon,
} from "./Icons";

/**
 * ModelsView
 *
 * Workspace for managing local and hosted LLM inference profiles (M15).
 * Supports NVIDIA NIM, local Ollama, vLLM/OpenAI-compatible inference, and offline mock models.
 */
export default function ModelsView({
  modelProfiles = [],
  loading = false,
  error = null,
  onRefresh,
  onActivateProfile,
  onDeleteProfile,
  onSaveProfile,
  onTestProfile,
  onTestDraftProfile,
  testResults = {},
}) {
  const [showModal, setShowModal] = useState(false);
  const [editingProfileId, setEditingProfileId] = useState(null);

  // Form State
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
    setShowModal(true);
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
    setShowModal(true);
  };

  const handleSubmitProfile = async (e) => {
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

      await onSaveProfile(editingProfileId, payload);
      setShowModal(false);
      setEditingProfileId(null);
    } catch (err) {
      setFormError(err.message || "Failed to save model profile.");
    } finally {
      setFormSaving(false);
    }
  };

  const handleTestDraft = async () => {
    const payload = {
      provider: formProvider,
      model: formModel.trim(),
      endpoint: formEndpoint.trim(),
      api_key: formApiKey.trim(),
      timeout: formTimeout ? parseFloat(formTimeout) : 15.0,
    };
    await onTestDraftProfile(payload);
  };

  return (
    <div className="models-workspace-container">
      {/* Header */}
      <header className="workspace-header">
        <div>
          <h2 className="workspace-title">Model Profiles</h2>
          <p className="workspace-subtitle">
            Configure open, local, and hosted inference engines for AURA autonomous research.
          </p>
        </div>
        <div className="workspace-header-actions">
          <button
            type="button"
            className="btn-workspace-refresh"
            onClick={onRefresh}
            disabled={loading}
            title="Refresh models"
          >
            <RefreshIcon size={13} />
            <span>Refresh</span>
          </button>
          <button
            type="button"
            id="btn-add-model"
            className="btn-primary-action"
            onClick={openCreateModelModal}
          >
            <span>+ Add Model Profile</span>
          </button>
        </div>
      </header>

      {/* Global Error */}
      {error && (
        <div className="error-notice-banner" role="alert">
          <WarningIcon size={14} />
          <span>{error}</span>
        </div>
      )}

      {/* Models List */}
      {loading && modelProfiles.length === 0 ? (
        <div className="workspace-loading-card" role="status" aria-live="polite">
          <div className="spinner-large" aria-hidden="true" />
          <p className="loading-caption">Loading Model Profiles...</p>
        </div>
      ) : modelProfiles.length === 0 ? (
        <div className="models-empty-box">
          <div className="empty-box-icon empty-box-text-tag" aria-hidden="true">LLM</div>
          <h3 className="empty-box-title">No Model Profiles Configured</h3>
          <p className="empty-box-desc">
            AURA requires at least one active model profile to execute research reasoning and synthesis. Add a local Ollama instance, NVIDIA NIM, or OpenAI-compatible server.
          </p>
          <button
            type="button"
            id="btn-add-first-model"
            className="btn-primary-action"
            onClick={openCreateModelModal}
          >
            + Add First Model Profile
          </button>
        </div>
      ) : (
        <div className="models-cards-grid" id="models-grid">
          {modelProfiles.map((profile) => {
            const test = testResults[profile.id];
            return (
              <article
                key={profile.id}
                className={`model-profile-card ${
                  profile.is_active ? "card-is-active" : ""
                }`}
                id={`model-card-${profile.id}`}
              >
                <div className="model-card-top">
                  <div className="model-title-wrap">
                    <h3 className="model-name">{profile.name}</h3>
                    <div className="model-badges-group">
                      {profile.is_active && (
                        <span className="badge-active">● Active</span>
                      )}
                      <span className="badge-provider">
                        {formatProviderBadge(profile.provider)}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="model-specs-table">
                  <div className="spec-row">
                    <span className="spec-k">Model</span>
                    <span className="spec-v spec-v-mono">{profile.model}</span>
                  </div>
                  <div className="spec-row">
                    <span className="spec-k">Endpoint</span>
                    <span className="spec-v spec-v-mono">
                      {profile.endpoint || "Default"}
                    </span>
                  </div>
                  <div className="spec-row">
                    <span className="spec-k">Credentials</span>
                    <span className="spec-v">
                      {profile.has_api_key
                        ? `Configured (${profile.api_key_masked})`
                        : "None required"}
                    </span>
                  </div>
                  <div className="spec-row">
                    <span className="spec-k">Parameters</span>
                    <span className="spec-v">
                      {profile.temperature != null
                        ? `T: ${profile.temperature}`
                        : "T: default"}{" "}
                      • {profile.timeout}s timeout
                    </span>
                  </div>
                </div>

                {/* Connection Test Banner */}
                {test && (
                  <div
                    className={`test-result-banner ${
                      test.loading
                        ? "testing"
                        : test.success
                        ? "success"
                        : "error"
                    }`}
                  >
                    {test.loading ? (
                      <span>Testing endpoint connection...</span>
                    ) : test.success ? (
                      <span className="test-banner-content">
                        <CheckIcon size={13} />
                        <span>{test.message} ({test.latency_ms}ms)</span>
                      </span>
                    ) : (
                      <span className="test-banner-content">
                        <WarningIcon size={13} />
                        <span>{test.error}</span>
                      </span>
                    )}
                  </div>
                )}

                {/* Card Action Buttons */}
                <div className="model-actions-row">
                  <div className="model-actions-left">
                    {!profile.is_active && (
                      <button
                        type="button"
                        id={`btn-activate-${profile.id}`}
                        className="btn-model-action btn-activate"
                        onClick={() => onActivateProfile(profile.id)}
                      >
                        Set Active
                      </button>
                    )}
                    <button
                      type="button"
                      id={`btn-test-${profile.id}`}
                      className="btn-model-action"
                      onClick={() => onTestProfile(profile.id)}
                      disabled={test?.loading}
                    >
                      {test?.loading ? "Testing..." : "Test Connection"}
                    </button>
                  </div>

                  <div className="model-actions-right">
                    <button
                      type="button"
                      id={`btn-edit-${profile.id}`}
                      className="btn-model-action"
                      onClick={() => openEditModelModal(profile)}
                    >
                      Edit
                    </button>
                    <button
                      type="button"
                      id={`btn-delete-${profile.id}`}
                      className="btn-model-action btn-action-danger"
                      onClick={() => onDeleteProfile(profile.id)}
                    >
                      Delete
                    </button>
                  </div>
                </div>
              </article>
            );
          })}
        </div>
      )}

      {/* Add / Edit Profile Modal */}
      {showModal && (
        <div className="modal-overlay" onClick={() => setShowModal(false)}>
          <div
            className="modal-dialog-card"
            onClick={(e) => e.stopPropagation()}
            role="dialog"
            aria-modal="true"
            aria-labelledby="model-modal-title"
          >
            <div className="modal-header">
              <h3 id="model-modal-title" className="modal-title">
                {editingProfileId ? "Edit Model Profile" : "Add Model Profile"}
              </h3>
              <button
                type="button"
                className="btn-modal-close"
                onClick={() => setShowModal(false)}
                aria-label="Close modal"
              >
                <CloseIcon size={14} />
              </button>
            </div>

            {formError && (
              <div className="modal-error-banner" role="alert">
                <WarningIcon size={13} />
                <span>{formError}</span>
              </div>
            )}

            <form onSubmit={handleSubmitProfile} className="modal-form-body">
              <div className="form-field">
                <label className="form-label" htmlFor="model-form-name">
                  Profile Name <span className="field-required">*</span>
                </label>
                <input
                  id="model-form-name"
                  type="text"
                  className="text-input-field"
                  placeholder="e.g. My Local Llama 3.1"
                  value={formName}
                  onChange={(e) => setFormName(e.target.value)}
                  required
                />
              </div>

              <div className="form-field">
                <label className="form-label" htmlFor="model-form-provider">
                  Provider Preset <span className="field-required">*</span>
                </label>
                <select
                  id="model-form-provider"
                  className="select-input-field mode-select-dropdown"
                  value={formProvider}
                  onChange={(e) => handleProviderPresetChange(e.target.value)}
                >
                  <option value="nvidia">NVIDIA NIM</option>
                  <option value="ollama">Ollama (Local / Native HTTP)</option>
                  <option value="openai_compatible">OpenAI-compatible (vLLM, LM Studio, etc.)</option>
                  <option value="mock">Mock Provider (Offline Testing)</option>
                </select>
              </div>

              <div className="form-field">
                <label className="form-label" htmlFor="model-form-model">
                  Model Identifier <span className="field-required">*</span>
                </label>
                <input
                  id="model-form-model"
                  type="text"
                  className="text-input-field"
                  placeholder="e.g. nvidia/nemotron-3-ultra-550b-a55b or llama3.1:8b"
                  value={formModel}
                  onChange={(e) => setFormModel(e.target.value)}
                  required
                />
              </div>

              <div className="form-field">
                <label className="form-label" htmlFor="model-form-endpoint">
                  API Endpoint URL
                </label>
                <input
                  id="model-form-endpoint"
                  type="text"
                  className="text-input-field"
                  placeholder="e.g. http://localhost:11434"
                  value={formEndpoint}
                  onChange={(e) => setFormEndpoint(e.target.value)}
                />
              </div>

              <div className="form-field">
                <label className="form-label" htmlFor="model-form-apikey">
                  API Key / Token{" "}
                  {editingProfileId && (
                    <span className="field-hint">(leave blank to preserve existing key)</span>
                  )}
                </label>
                <input
                  id="model-form-apikey"
                  type="password"
                  className="text-input-field"
                  placeholder={
                    formProvider === "ollama"
                      ? "Not required for local Ollama"
                      : "sk-... or nvapi-..."
                  }
                  value={formApiKey}
                  onChange={(e) => setFormApiKey(e.target.value)}
                />
              </div>

              <div className="form-grid-3">
                <div className="form-field">
                  <label className="form-label" htmlFor="model-form-temp">
                    Temperature
                  </label>
                  <input
                    id="model-form-temp"
                    type="number"
                    step="0.05"
                    min="0"
                    max="2"
                    className="text-input-field"
                    placeholder="0.7"
                    value={formTemperature}
                    onChange={(e) => setFormTemperature(e.target.value)}
                  />
                </div>
                <div className="form-field">
                  <label className="form-label" htmlFor="model-form-tokens">
                    Max Tokens
                  </label>
                  <input
                    id="model-form-tokens"
                    type="number"
                    min="1"
                    className="text-input-field"
                    placeholder="1024"
                    value={formMaxTokens}
                    onChange={(e) => setFormMaxTokens(e.target.value)}
                  />
                </div>
                <div className="form-field">
                  <label className="form-label" htmlFor="model-form-timeout">
                    Timeout (s)
                  </label>
                  <input
                    id="model-form-timeout"
                    type="number"
                    min="1"
                    className="text-input-field"
                    placeholder="30"
                    value={formTimeout}
                    onChange={(e) => setFormTimeout(e.target.value)}
                  />
                </div>
              </div>

              <div className="form-checkbox-row">
                <input
                  id="model-form-active"
                  type="checkbox"
                  checked={formIsActive}
                  onChange={(e) => setFormIsActive(e.target.checked)}
                />
                <label htmlFor="model-form-active" className="checkbox-label">
                  Set as Active Model Profile for Research
                </label>
              </div>

              {testResults["draft"] && (
                <div
                  className={`test-result-banner ${
                    testResults["draft"].loading
                      ? "testing"
                      : testResults["draft"].success
                      ? "success"
                      : "error"
                  }`}
                >
                  {testResults["draft"].loading ? (
                    <span>Testing draft connection...</span>
                  ) : testResults["draft"].success ? (
                    <span className="test-banner-content">
                      <CheckIcon size={13} />
                      <span>{testResults["draft"].message} ({testResults["draft"].latency_ms}ms)</span>
                    </span>
                  ) : (
                    <span className="test-banner-content">
                      <WarningIcon size={13} />
                      <span>{testResults["draft"].error}</span>
                    </span>
                  )}
                </div>
              )}

              <div className="modal-actions-bar">
                <button
                  type="button"
                  className="btn-modal-test"
                  onClick={handleTestDraft}
                  disabled={testResults["draft"]?.loading || !formModel.trim()}
                >
                  {testResults["draft"]?.loading ? "Testing..." : "Test Connection"}
                </button>
                <div className="modal-save-group">
                  <button
                    type="button"
                    className="btn-modal-cancel"
                    onClick={() => setShowModal(false)}
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="btn-modal-save"
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
  );
}
