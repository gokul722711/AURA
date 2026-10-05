"use client";

import { useMemo, useState } from "react";
import { formatMode, groupHistoryByDate } from "../utils/helpers";
import {
  SidebarToggleIcon,
  TrashIcon,
  MoreHorizontalIcon,
  CheckIcon,
  CloseIcon,
} from "./Icons";

/**
 * Sidebar Component
 *
 * Persistent navigation, multi-select chat management, individual deletion, and desktop collapsing.
 * Uses clean SVG icons and restrained typography with zero decorative emojis.
 */
export default function Sidebar({
  activeTab,
  onSelectTab,
  onNewResearch,
  historyRuns = [],
  activeRunId,
  selectedHistoryId,
  onSelectHistoryRun,
  onDeleteRun,
  onDeleteRuns,
  onDeleteAllRuns,
  documentsCount = 0,
  modelsCount = 0,
  activeProfile = null,
  isCollapsed = false,
  onToggleCollapse,
  isOpen = false,
  onCloseMobile,
}) {
  // Multi-select & management state
  const [selectMode, setSelectMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState(new Set());
  const [menuOpen, setMenuOpen] = useState(false);
  const [confirmDeleteAll, setConfirmDeleteAll] = useState(false);
  const [confirmDeleteId, setConfirmDeleteId] = useState(null);

  const groupedHistory = useMemo(() => {
    return groupHistoryByDate(historyRuns);
  }, [historyRuns]);

  const hasHistory = historyRuns.length > 0;

  // Toggle selection of a single item
  const handleToggleSelect = (runId, e) => {
    e.stopPropagation();
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(runId)) {
        next.delete(runId);
      } else {
        next.add(runId);
      }
      return next;
    });
  };

  // Select all or deselect all
  const handleToggleSelectAll = () => {
    if (selectedIds.size === historyRuns.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(historyRuns.map((r) => r.run_id)));
    }
  };

  // Delete selected batch
  const handleDeleteSelected = async () => {
    if (selectedIds.size === 0) return;
    const ids = Array.from(selectedIds);
    await onDeleteRuns(ids);
    setSelectedIds(new Set());
    setSelectMode(false);
  };

  // Individual deletion
  const handleDeleteSingle = async (runId, e) => {
    e.stopPropagation();
    await onDeleteRun(runId);
    setConfirmDeleteId(null);
  };

  // Delete all
  const handleExecuteDeleteAll = async () => {
    await onDeleteAllRuns();
    setConfirmDeleteAll(false);
    setMenuOpen(false);
    setSelectMode(false);
    setSelectedIds(new Set());
  };

  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div
          className="sidebar-backdrop"
          onClick={onCloseMobile}
          aria-hidden="true"
        />
      )}

      {/* Delete All Confirmation Dialog */}
      {confirmDeleteAll && (
        <div
          className="modal-overlay"
          onClick={() => setConfirmDeleteAll(false)}
        >
          <div
            className="modal-dialog-card delete-confirm-modal"
            onClick={(e) => e.stopPropagation()}
            role="dialog"
            aria-modal="true"
          >
            <h3 className="modal-title">Delete all research history?</h3>
            <p className="delete-modal-warning">
              This will permanently remove all {historyRuns.length} research runs, evidence logs, and synthesized findings. This action cannot be undone.
            </p>
            <div className="modal-actions-bar" style={{ marginTop: "1rem" }}>
              <button
                type="button"
                className="btn-modal-cancel"
                onClick={() => setConfirmDeleteAll(false)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn-modal-save btn-action-danger"
                onClick={handleExecuteDeleteAll}
              >
                Delete All History
              </button>
            </div>
          </div>
        </div>
      )}

      <aside
        className={`aura-sidebar ${isOpen ? "sidebar-open" : ""} ${
          isCollapsed ? "sidebar-collapsed" : ""
        }`}
        aria-label="Main Navigation"
      >
        {/* Brand Header & Toggle */}
        <div className="sidebar-header">
          <div className="sidebar-brand">
            <span className="brand-name">AURA</span>
            <span className="brand-badge">RESEARCH</span>
          </div>

          <div className="sidebar-header-actions">
            {onToggleCollapse && (
              <button
                type="button"
                className="btn-sidebar-collapse"
                onClick={onToggleCollapse}
                title="Collapse sidebar (⌘S)"
                aria-label="Collapse sidebar"
              >
                <SidebarToggleIcon size={16} />
              </button>
            )}
            {onCloseMobile && (
              <button
                type="button"
                className="btn-sidebar-close"
                onClick={onCloseMobile}
                aria-label="Close sidebar"
              >
                <CloseIcon size={14} />
              </button>
            )}
          </div>
        </div>

        {/* Primary Action: New Research */}
        <div className="sidebar-action-wrap">
          <button
            type="button"
            className="btn-new-research"
            onClick={() => {
              onNewResearch();
              if (onCloseMobile) onCloseMobile();
            }}
          >
            <span className="action-plus">+</span>
            <span>New Research</span>
            <kbd className="action-kbd">⌘K</kbd>
          </button>
        </div>

        {/* Scrollable Center Section */}
        <div className="sidebar-scroll-area">
          {/* Research History Section */}
          <div className="sidebar-section">
            <div className="sidebar-section-header">
              <span className="sidebar-section-title">Research</span>

              {/* History Overflow Menu & Actions */}
              {hasHistory && (
                <div className="history-header-actions">
                  <span className="sidebar-count">{historyRuns.length}</span>
                  <div className="history-menu-wrap">
                    <button
                      type="button"
                      className="btn-history-menu-trigger"
                      onClick={() => setMenuOpen(!menuOpen)}
                      title="Research history options"
                      aria-label="History options"
                    >
                      <MoreHorizontalIcon size={15} />
                    </button>

                    {menuOpen && (
                      <div
                        className="history-menu-dropdown"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <button
                          type="button"
                          className="history-menu-item"
                          onClick={() => {
                            setSelectMode(!selectMode);
                            setSelectedIds(new Set());
                            setMenuOpen(false);
                          }}
                        >
                          {selectMode ? "Cancel selection" : "Select conversations"}
                        </button>
                        <button
                          type="button"
                          className="history-menu-item item-danger"
                          onClick={() => {
                            setConfirmDeleteAll(true);
                            setMenuOpen(false);
                          }}
                        >
                          Delete all history...
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Multi-Selection Control Bar (only visible when in select mode) */}
            {selectMode && (
              <div className="selection-action-bar">
                <div className="selection-stats">
                  <span>{selectedIds.size} selected</span>
                  <button
                    type="button"
                    className="btn-select-all"
                    onClick={handleToggleSelectAll}
                  >
                    {selectedIds.size === historyRuns.length ? "Deselect all" : "Select all"}
                  </button>
                </div>
                <div className="selection-btns">
                  <button
                    type="button"
                    className="btn-delete-selected"
                    disabled={selectedIds.size === 0}
                    onClick={handleDeleteSelected}
                  >
                    Delete ({selectedIds.size})
                  </button>
                  <button
                    type="button"
                    className="btn-cancel-selection"
                    onClick={() => {
                      setSelectMode(false);
                      setSelectedIds(new Set());
                    }}
                  >
                    Done
                  </button>
                </div>
              </div>
            )}

            {!hasHistory ? (
              <div className="sidebar-empty-state">
                <span>No research runs yet</span>
              </div>
            ) : (
              <div className="sidebar-history-groups">
                {Object.entries(groupedHistory).map(([period, items]) => {
                  if (items.length === 0) return null;
                  return (
                    <div key={period} className="history-group">
                      <div className="history-group-title">{period}</div>
                      <div className="history-group-list">
                        {items.map((run) => {
                          const isSelected =
                            selectedHistoryId === run.run_id ||
                            activeRunId === run.run_id;
                          const isChecked = selectedIds.has(run.run_id);
                          const isConfirming = confirmDeleteId === run.run_id;

                          return (
                            <div
                              key={run.run_id}
                              className={`history-item-row-wrapper ${
                                isSelected ? "selected" : ""
                              }`}
                            >
                              {selectMode && (
                                <input
                                  type="checkbox"
                                  className="history-checkbox"
                                  checked={isChecked}
                                  onChange={(e) => handleToggleSelect(run.run_id, e)}
                                />
                              )}

                              <button
                                type="button"
                                className={`history-item-btn ${
                                  isSelected ? "selected" : ""
                                }`}
                                onClick={() => {
                                  if (selectMode) {
                                    handleToggleSelect(run.run_id, { stopPropagation: () => {} });
                                  } else {
                                    onSelectHistoryRun(run.run_id);
                                    if (onCloseMobile) onCloseMobile();
                                  }
                                }}
                                title={run.objective}
                              >
                                <div className="history-item-row">
                                  <span className="history-item-title">
                                    {run.objective}
                                  </span>
                                </div>
                                <div className="history-item-sub">
                                  <span className="history-mode-tag">
                                    {formatMode(run.mode)}
                                  </span>
                                  {run.is_grounded && (
                                    <span
                                      className="history-grounded-dot"
                                      title="Grounded in evidence"
                                    >
                                      <CheckIcon size={10} />
                                    </span>
                                  )}
                                  {run.status === "running" && (
                                    <span
                                      className="history-running-dot"
                                      title="In progress"
                                    >
                                      ●
                                    </span>
                                  )}
                                </div>
                              </button>

                              {/* Hover Delete Action (Normal Mode) */}
                              {!selectMode && (
                                <div className="history-item-actions">
                                  {isConfirming ? (
                                    <div className="history-delete-confirm">
                                      <button
                                        type="button"
                                        className="btn-confirm-del-yes"
                                        onClick={(e) => handleDeleteSingle(run.run_id, e)}
                                        title="Confirm delete"
                                      >
                                        Delete
                                      </button>
                                      <button
                                        type="button"
                                        className="btn-confirm-del-no"
                                        onClick={(e) => {
                                          e.stopPropagation();
                                          setConfirmDeleteId(null);
                                        }}
                                        title="Cancel"
                                      >
                                        <CloseIcon size={11} />
                                      </button>
                                    </div>
                                  ) : (
                                    <button
                                      type="button"
                                      className="btn-history-delete"
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        setConfirmDeleteId(run.run_id);
                                      }}
                                      title="Delete conversation"
                                      aria-label="Delete research"
                                    >
                                      <TrashIcon size={13} />
                                    </button>
                                  )}
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Workspace Section */}
          <div className="sidebar-section sidebar-workspace-section">
            <div className="sidebar-section-header">
              <span className="sidebar-section-title">Workspace</span>
            </div>
            <nav className="sidebar-nav">
              <button
                type="button"
                id="sidebar-nav-kb"
                className={`sidebar-nav-item ${
                  activeTab === "knowledge" ? "active" : ""
                }`}
                onClick={() => {
                  onSelectTab("knowledge");
                  if (onCloseMobile) onCloseMobile();
                }}
              >
                <span className="nav-item-label">Knowledge Base</span>
                <span className="nav-item-count">{documentsCount}</span>
              </button>

              <button
                type="button"
                id="sidebar-nav-models"
                className={`sidebar-nav-item ${
                  activeTab === "models" ? "active" : ""
                }`}
                onClick={() => {
                  onSelectTab("models");
                  if (onCloseMobile) onCloseMobile();
                }}
              >
                <span className="nav-item-label">Models</span>
                <span className="nav-item-count">{modelsCount}</span>
              </button>
            </nav>
          </div>
        </div>

        {/* Footer: Active Model Status */}
        <div className="sidebar-footer">
          <button
            type="button"
            className="sidebar-status-card"
            onClick={() => {
              onSelectTab("models");
              if (onCloseMobile) onCloseMobile();
            }}
            title="Configure model profiles"
          >
            <div className="status-indicator">
              <span
                className={`status-dot ${
                  activeProfile ? "status-dot-active" : "status-dot-warning"
                }`}
              />
            </div>
            <div className="status-text">
              <span className="status-label">Inference Model</span>
              <span className="status-model-name">
                {activeProfile ? activeProfile.name : "No Model Active"}
              </span>
            </div>
            <span className="status-chevron">→</span>
          </button>
        </div>
      </aside>
    </>
  );
}
