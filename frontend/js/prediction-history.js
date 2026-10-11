/**
 * ==========================================================================
 * Crop Recommendation System - Prediction History & Audit Trail Controller
 * Principles of Data Science (PoDS) Project
 * ==========================================================================
 */

class PredictionHistoryController {
  constructor() {
    this.container = document.getElementById('history-content-container');
    this.countBadge = document.getElementById('history-count-badge');
    this.clearBtn = document.getElementById('btn-clear-history');
    this.announcer = document.getElementById('aria-live-announcer');

    this.historyRecords = [];
    this.isLoading = false;

    this.init();
  }

  init() {
    if (this.clearBtn) {
      this.clearBtn.addEventListener('click', () => this.handleClearHistory());
    }

    // Load initial history records on initialization
    this.loadHistory();
  }

  async loadHistory() {
    if (!this.container) return;

    try {
      this.isLoading = true;
      const res = await fetch('/api/prediction-history');
      if (!res.ok) {
        throw new Error(`Failed to load history (HTTP ${res.status})`);
      }

      const data = await res.json();
      this.historyRecords = Array.isArray(data.history) ? data.history : [];
      this.render();
    } catch (err) {
      this.renderError(err.message);
    } finally {
      this.isLoading = false;
    }
  }

  render() {
    if (!this.container) return;

    const count = this.historyRecords.length;

    // Update count badge
    if (this.countBadge) {
      if (count > 0) {
        this.countBadge.className = 'badge badge-slate';
        this.countBadge.textContent = `${count} ${count === 1 ? 'Inference Logged' : 'Inferences Logged'}`;
      } else {
        this.countBadge.className = 'badge badge-neutral';
        this.countBadge.textContent = '0 Inferences Logged';
      }
    }

    // Toggle clear button visibility
    if (this.clearBtn) {
      this.clearBtn.style.display = count > 0 ? 'inline-flex' : 'none';
    }

    if (count === 0) {
      this.renderEmptyState();
      return;
    }

    // Render interactive responsive audit log table
    let tableHtml = `
      <div class="table-responsive" style="overflow-x: auto; -webkit-overflow-scrolling: touch;">
        <table class="table" style="width: 100%; border-collapse: collapse; font-size: var(--font-size-sm);" aria-label="Prediction History Audit Log">
          <thead>
            <tr style="border-bottom: 2px solid var(--border-color-default); text-align: left; background-color: var(--color-alabaster-surface);">
              <th style="padding: var(--space-3) var(--space-4); font-weight: 700; color: var(--color-dark-emerald);">#</th>
              <th style="padding: var(--space-3) var(--space-4); font-weight: 700; color: var(--color-dark-emerald);">Timestamp</th>
              <th style="padding: var(--space-3) var(--space-4); font-weight: 700; color: var(--color-dark-emerald);">Recommended Crop</th>
              <th style="padding: var(--space-3) var(--space-4); font-weight: 700; color: var(--color-dark-emerald);">Champion Model</th>
              <th style="padding: var(--space-3) var(--space-4); font-weight: 700; color: var(--color-dark-emerald);">N</th>
              <th style="padding: var(--space-3) var(--space-4); font-weight: 700; color: var(--color-dark-emerald);">P</th>
              <th style="padding: var(--space-3) var(--space-4); font-weight: 700; color: var(--color-dark-emerald);">K</th>
              <th style="padding: var(--space-3) var(--space-4); font-weight: 700; color: var(--color-dark-emerald);">Temp (°C)</th>
              <th style="padding: var(--space-3) var(--space-4); font-weight: 700; color: var(--color-dark-emerald);">Humidity (%)</th>
              <th style="padding: var(--space-3) var(--space-4); font-weight: 700; color: var(--color-dark-emerald);">Soil pH</th>
              <th style="padding: var(--space-3) var(--space-4); font-weight: 700; color: var(--color-dark-emerald);">Rainfall (mm)</th>
              <th style="padding: var(--space-3) var(--space-4); font-weight: 700; color: var(--color-dark-emerald);">Fingerprint</th>
            </tr>
          </thead>
          <tbody>
    `;

    this.historyRecords.forEach((item, idx) => {
      let formattedDate = '—';
      try {
        if (item.timestamp) {
          const d = new Date(item.timestamp);
          formattedDate = d.toLocaleString(undefined, {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
          });
        }
      } catch (e) {
        formattedDate = item.timestamp || '—';
      }

      const crop = item.predicted_crop || '—';
      const model = item.champion_model || '—';
      const fp = item.dataset_fingerprint ? item.dataset_fingerprint.substring(0, 10) : '—';

      const nVal = (item.n !== undefined && item.n !== null) ? Number(item.n).toFixed(0) : ((item.N !== undefined && item.N !== null) ? Number(item.N).toFixed(0) : '—');
      const pVal = (item.p !== undefined && item.p !== null) ? Number(item.p).toFixed(0) : ((item.P !== undefined && item.P !== null) ? Number(item.P).toFixed(0) : '—');
      const kVal = (item.k !== undefined && item.k !== null) ? Number(item.k).toFixed(0) : ((item.K !== undefined && item.K !== null) ? Number(item.K).toFixed(0) : '—');
      const tempVal = (item.temperature !== undefined && item.temperature !== null) ? Number(item.temperature).toFixed(1) : '—';
      const humidVal = (item.humidity !== undefined && item.humidity !== null) ? Number(item.humidity).toFixed(1) : '—';
      const phVal = (item.ph !== undefined && item.ph !== null) ? Number(item.ph).toFixed(2) : '—';
      const rainVal = (item.rainfall !== undefined && item.rainfall !== null) ? Number(item.rainfall).toFixed(1) : '—';

      tableHtml += `
        <tr style="border-bottom: 1px solid var(--border-color-default); transition: background-color 0.15s ease;">
          <td style="padding: var(--space-3) var(--space-4); font-weight: 600; color: var(--text-muted);">${count - idx}</td>
          <td style="padding: var(--space-3) var(--space-4); white-space: nowrap; color: var(--text-secondary); font-size: var(--font-size-xs);">${this.escapeHtml(formattedDate)}</td>
          <td style="padding: var(--space-3) var(--space-4);">
            <span class="badge badge-moss" style="font-weight: 700; text-transform: capitalize;">
              🌱 ${this.escapeHtml(crop)}
            </span>
          </td>
          <td style="padding: var(--space-3) var(--space-4);">
            <span class="badge badge-slate">${this.escapeHtml(model)}</span>
          </td>
          <td style="padding: var(--space-3) var(--space-4); font-family: var(--font-family-mono); font-weight: 600;">${nVal}</td>
          <td style="padding: var(--space-3) var(--space-4); font-family: var(--font-family-mono); font-weight: 600;">${pVal}</td>
          <td style="padding: var(--space-3) var(--space-4); font-family: var(--font-family-mono); font-weight: 600;">${kVal}</td>
          <td style="padding: var(--space-3) var(--space-4); font-family: var(--font-family-mono);">${tempVal}</td>
          <td style="padding: var(--space-3) var(--space-4); font-family: var(--font-family-mono);">${humidVal}</td>
          <td style="padding: var(--space-3) var(--space-4); font-family: var(--font-family-mono);">${phVal}</td>
          <td style="padding: var(--space-3) var(--space-4); font-family: var(--font-family-mono);">${rainVal}</td>
          <td style="padding: var(--space-3) var(--space-4);">
            <code style="font-size: 11px; background-color: var(--color-alabaster-surface); border: 1px solid var(--border-color-default); border-radius: 4px; padding: 2px 6px;">${this.escapeHtml(fp)}</code>
          </td>
        </tr>
      `;
    });

    tableHtml += `
          </tbody>
        </table>
      </div>
      <div style="margin-top: var(--space-3); font-size: var(--font-size-xs); color: var(--text-muted); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: var(--space-2);">
        <span>Showing all ${count} query audit log record(s) persisted in <code>data/prediction_history.db</code>.</span>
        <span>Deterministic inference records survive browser refreshes.</span>
      </div>
    `;

    this.container.innerHTML = tableHtml;
  }

  renderEmptyState() {
    this.container.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon" data-icon="history" aria-hidden="true"></div>
        <span class="empty-state-tag">Audit Trail</span>
        <h3 class="empty-state-title">No Predictions Recorded Yet</h3>
        <p class="empty-state-description">
          Queries executed through the Crop Prediction module will be automatically logged here with exact input vectors, timestamps, model identifiers, and dataset fingerprints.
        </p>
        <div style="margin-top: var(--space-4);">
          <button class="btn btn-primary btn-sm" type="button" data-navigate-to="prediction">
            <span data-icon="sprout" aria-hidden="true"></span>
            <span>Go to Crop Prediction</span>
          </button>
        </div>
      </div>
    `;

    if (window.AppIcons && typeof window.AppIcons.render === 'function') {
      window.AppIcons.render();
    }
  }

  renderError(msg) {
    if (!this.container) return;
    this.container.innerHTML = `
      <div class="callout form-banner-error" style="margin-bottom: 0;">
        <div class="callout-icon" style="color: #A93226;">
          ${window.AppIcons ? window.AppIcons.get('alertCircle') : ''}
        </div>
        <div class="callout-content">
          <div class="callout-title" style="color: #A93226;">History Load Error</div>
          <div class="callout-body">${this.escapeHtml(msg)}</div>
        </div>
      </div>
    `;
  }

  async handleClearHistory() {
    if (this.historyRecords.length === 0) return;

    const confirmed = window.confirm(
      "Are you sure you want to clear the entire prediction history audit trail? This will permanently delete all logged inference records."
    );

    if (!confirmed) return;

    try {
      const res = await fetch('/api/prediction-history/clear', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      const data = await res.json();
      if (!res.ok || !data.success) {
        throw new Error(data.message || 'Server failed to clear history records.');
      }

      this.historyRecords = [];
      this.render();

      if (this.announcer) {
        this.announcer.textContent = 'Prediction history audit trail cleared successfully.';
      }
    } catch (err) {
      alert(`Error clearing history: ${err.message}`);
    }
  }

  escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
}

// Instantiate on DOMContentLoaded
document.addEventListener('DOMContentLoaded', () => {
  window.predictionHistory = new PredictionHistoryController();
});
