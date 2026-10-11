/**
 * ==========================================================================
 * Crop Recommendation System - Prediction Form & Inference Controller
 * Principles of Data Science (PoDS) Project
 * ==========================================================================
 */

class PredictionFormController {
  constructor() {
    this.form = document.getElementById('crop-prediction-form');
    this.validationBanner = document.getElementById('form-validation-banner');
    this.submitBtn = document.getElementById('btn-validate-inputs');
    this.resetBtn = document.getElementById('btn-reset-form');

    // Prediction output containers
    this.resultBadge = document.getElementById('prediction-status-badge');
    this.resultBody = document.getElementById('prediction-result-body');

    // Session inferences counter
    this.sessionInferences = 0;
    this.isPredicting = false;

    this.fields = [
      {
        id: 'input-n',
        apiKey: 'N',
        name: 'Nitrogen (N)',
        unit: 'mg/kg',
        min: 0,
        max: 500,
        isInteger: true,
        errorId: 'error-n'
      },
      {
        id: 'input-p',
        apiKey: 'P',
        name: 'Phosphorus (P)',
        unit: 'mg/kg',
        min: 0,
        max: 500,
        isInteger: true,
        errorId: 'error-p'
      },
      {
        id: 'input-k',
        apiKey: 'K',
        name: 'Potassium (K)',
        unit: 'mg/kg',
        min: 0,
        max: 500,
        isInteger: true,
        errorId: 'error-k'
      },
      {
        id: 'input-temp',
        apiKey: 'temperature',
        name: 'Temperature',
        unit: '°C',
        min: -10,
        max: 60,
        isInteger: false,
        errorId: 'error-temp'
      },
      {
        id: 'input-humidity',
        apiKey: 'humidity',
        name: 'Relative Humidity',
        unit: '%',
        min: 0,
        max: 100,
        isInteger: false,
        errorId: 'error-humidity'
      },
      {
        id: 'input-ph',
        apiKey: 'ph',
        name: 'Soil pH',
        unit: 'pH',
        min: 0,
        max: 14,
        isInteger: false,
        errorId: 'error-ph'
      },
      {
        id: 'input-rainfall',
        apiKey: 'rainfall',
        name: 'Rainfall',
        unit: 'mm',
        min: 0,
        max: 2000,
        isInteger: false,
        errorId: 'error-rainfall'
      }
    ];

    this.init();
  }

  init() {
    if (!this.form) return;

    // Handle form submit
    this.form.addEventListener('submit', (e) => {
      e.preventDefault();
      if (this.isPredicting) return;
      this.handleFormSubmission();
    });

    // Handle form reset
    this.form.addEventListener('reset', () => {
      setTimeout(() => {
        this.clearAllErrors();
        this.resetResultView();
      }, 10);
    });

    // Inline validation clearing on input
    this.fields.forEach(field => {
      const inputEl = document.getElementById(field.id);
      if (inputEl) {
        inputEl.addEventListener('input', () => {
          this.clearFieldError(field);
        });
      }
    });

    // Listen to dataset updates and resets across modules
    window.addEventListener('dataset:updated', () => {
      this.checkModelStatus();
    });
    window.addEventListener('dataset:reset', () => {
      this.checkModelStatus();
    });

    // Check model status on page load to set initial banner/output state
    this.checkModelStatus();
  }

  async checkModelStatus() {
    try {
      const res = await fetch('/api/model-status');
      if (res.ok) {
        const data = await res.json();
        if (data.trained && data.artifact_valid && data.model_compatible) {
          // Clear error banner if it was showing a model status warning
          if (this.validationBanner && (
            this.validationBanner.textContent.includes('different dataset') ||
            this.validationBanner.textContent.includes('No valid trained model') ||
            this.validationBanner.textContent.includes('No active dataset')
          )) {
            this.validationBanner.style.display = 'none';
          }

          if (this.resultBadge) {
            this.resultBadge.className = 'badge badge-slate';
            this.resultBadge.textContent = 'Model Ready';
          }
          if (this.resultBody && (!this.resultBody.querySelector('.badge-moss') || this.resultBody.querySelector('.result-unavailable-panel') || this.resultBody.querySelector('.callout-title')?.textContent.includes('Incompatible') || this.resultBody.querySelector('.callout-title')?.textContent.includes('Re-training'))) {
            this.resultBody.innerHTML = `
              <div class="callout" style="margin-bottom: 0;">
                <div class="callout-icon" data-icon="info" aria-hidden="true"></div>
                <div class="callout-content">
                  <div class="callout-title">Model Pipeline Ready for Inference</div>
                  <div class="callout-body">
                    Champion model <strong>${this.escapeHtml(data.champion_model || 'Random Forest')}</strong> is active and deserialized. Populate the 7 agronomic parameters above and click <em>Run Crop Prediction</em> to generate a real-time recommendation.
                  </div>
                </div>
              </div>
            `;
          }
          if (this.submitBtn) {
            const btnSpan = this.submitBtn.querySelector('span:last-child');
            if (btnSpan) btnSpan.textContent = 'Run Crop Prediction';
          }
        } else if (data.artifact_present && data.model_compatible === false) {
          // Stale / Incompatible model
          this.showErrorBanner(data.message || 'Model was trained on a different dataset. Re-training required.');
          if (this.resultBadge) {
            this.resultBadge.className = 'badge badge-slate';
            this.resultBadge.textContent = 'Model Incompatible';
          }
          if (this.resultBody && !this.resultBody.querySelector('.badge-moss')) {
            this.resultBody.innerHTML = `
              <div class="callout form-banner-error" style="margin-bottom: 0;">
                <div class="callout-icon" style="color: #A93226;">
                  ${window.AppIcons ? window.AppIcons.get('alertCircle') : ''}
                </div>
                <div class="callout-content">
                  <div class="callout-title" style="color: #A93226;">Model Re-training Required</div>
                  <div class="callout-body">
                    The active model was trained on a different dataset or fingerprint than the one currently ingested. Navigate to the <strong>Model Evaluation</strong> module to retrain the classifier for the active dataset.
                  </div>
                </div>
              </div>
            `;
          }
        } else {
          // Untrained / No model
          if (this.resultBadge) {
            this.resultBadge.className = 'badge badge-neutral';
            this.resultBadge.textContent = 'Pipeline Inactive';
          }
          if (this.resultBody && !this.resultBody.querySelector('.badge-moss')) {
            this.renderNoModelView();
          }
        }
      }
    } catch (e) {
      // Backend offline or unreachable
    }
  }

  checkModelStatusInitial() {
    return this.checkModelStatus();
  }

  async handleFormSubmission() {
    // 1. Validate inputs client-side
    const validationResult = this.validateInputs();
    if (!validationResult.isValid) {
      this.showErrorBanner('Please correct the highlighted input errors before submitting.');
      if (validationResult.firstInvalidEl) {
        validationResult.firstInvalidEl.focus();
      }
      return;
    }

    const payload = validationResult.payload;

    // 2. Prevent duplicate submissions
    this.isPredicting = true;
    this.setSubmittingState(true);

    try {
      // 3. Verify backend model readiness before running inference
      const statusRes = await fetch('/api/model-status');
      const statusData = await statusRes.json();

      if (!statusData.trained || !statusData.artifact_valid || !statusData.model_compatible) {
        this.isPredicting = false;
        this.setSubmittingState(false);
        this.showErrorBanner(
          statusData.message || 'No valid trained model found. You must train an approved model in the Model Evaluation module before requesting predictions.'
        );
        if (statusData.artifact_present && !statusData.model_compatible) {
          // Incompatibility detected
          if (this.resultBadge) {
            this.resultBadge.className = 'badge badge-slate';
            this.resultBadge.textContent = 'Model Incompatible';
          }
        } else {
          this.renderNoModelView();
        }
        return;
      }

      // 4. Send inference request to backend
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const data = await res.json();

      if (!res.ok || !data.success) {
        this.isPredicting = false;
        this.setSubmittingState(false);
        this.showErrorBanner(data.message || 'Inference execution failed on the server.');
        this.renderErrorResultView(data.message || 'Inference failed');
        return;
      }

      // 5. Successful prediction
      this.sessionInferences++;
      this.updateDashboardInferenceTelemetry(data.predicted_crop);
      this.showSuccessBanner(data.predicted_crop, data.champion_model);
      this.renderRecommendationResult(data);

      // 6. Refresh prediction history audit log in background
      if (window.predictionHistory && typeof window.predictionHistory.loadHistory === 'function') {
        window.predictionHistory.loadHistory();
      }

      // Screen reader announcement
      const announcer = document.getElementById('aria-live-announcer');
      if (announcer) {
        announcer.textContent = `Crop prediction complete. Recommended crop: ${data.predicted_crop}.`;
      }

    } catch (err) {
      this.showErrorBanner(`Network or server communication error: ${err.message}`);
      this.renderErrorResultView(err.message);
    } finally {
      this.isPredicting = false;
      this.setSubmittingState(false);
    }
  }

  validateInputs() {
    let isValid = true;
    let firstInvalidEl = null;
    const payload = {};

    this.fields.forEach(field => {
      const inputEl = document.getElementById(field.id);
      if (!inputEl) return;

      const rawVal = inputEl.value.trim();

      // Check required
      if (rawVal === '') {
        this.setFieldError(field, `${field.name} is required.`);
        isValid = false;
        if (!firstInvalidEl) firstInvalidEl = inputEl;
        return;
      }

      // Check numeric
      const numVal = Number(rawVal);
      if (isNaN(numVal) || !isFinite(numVal)) {
        this.setFieldError(field, `${field.name} must be a valid finite number.`);
        isValid = false;
        if (!firstInvalidEl) firstInvalidEl = inputEl;
        return;
      }

      // Check bounds
      if (field.min !== undefined && numVal < field.min) {
        this.setFieldError(field, `${field.name} cannot be less than ${field.min} ${field.unit}.`);
        isValid = false;
        if (!firstInvalidEl) firstInvalidEl = inputEl;
        return;
      }

      if (field.max !== undefined && numVal > field.max) {
        this.setFieldError(field, `${field.name} cannot exceed ${field.max} ${field.unit}.`);
        isValid = false;
        if (!firstInvalidEl) firstInvalidEl = inputEl;
        return;
      }

      this.clearFieldError(field);
      payload[field.apiKey] = numVal;
    });

    return { isValid, firstInvalidEl, payload };
  }

  setFieldError(field, message) {
    const inputEl = document.getElementById(field.id);
    const errorEl = document.getElementById(field.errorId);

    if (inputEl) {
      inputEl.classList.add('is-invalid');
      inputEl.setAttribute('aria-invalid', 'true');
    }

    if (errorEl) {
      errorEl.textContent = message;
      errorEl.style.display = 'block';
    }
  }

  clearFieldError(field) {
    const inputEl = document.getElementById(field.id);
    const errorEl = document.getElementById(field.errorId);

    if (inputEl) {
      inputEl.classList.remove('is-invalid');
      inputEl.removeAttribute('aria-invalid');
    }

    if (errorEl) {
      errorEl.textContent = '';
      errorEl.style.display = 'none';
    }
  }

  clearAllErrors() {
    this.fields.forEach(field => this.clearFieldError(field));
    if (this.validationBanner) {
      this.validationBanner.style.display = 'none';
      this.validationBanner.innerHTML = '';
      this.validationBanner.className = '';
    }
  }

  setSubmittingState(isSubmitting) {
    if (this.submitBtn) {
      this.submitBtn.disabled = isSubmitting;
      const btnSpan = this.submitBtn.querySelector('span:last-child');
      if (btnSpan) {
        btnSpan.textContent = isSubmitting ? 'Evaluating Model...' : 'Run Crop Prediction';
      }
    }
    if (this.resultBadge) {
      if (isSubmitting) {
        this.resultBadge.className = 'badge badge-slate';
        this.resultBadge.textContent = 'Running Inference...';
      }
    }
  }

  updateDashboardInferenceTelemetry(cropName) {
    const dashInferencesVal = document.getElementById('dash-inferences-value');
    if (dashInferencesVal) {
      dashInferencesVal.textContent = String(this.sessionInferences);
    }

    const dashInferencesBadge = document.getElementById('dash-inferences-badge');
    if (dashInferencesBadge) {
      dashInferencesBadge.className = 'badge badge-moss';
      dashInferencesBadge.textContent = `${this.sessionInferences} Run${this.sessionInferences > 1 ? 's' : ''}`;
    }

    const dashInferencesSub = document.getElementById('dash-inferences-subtext');
    if (dashInferencesSub) {
      const timeStr = new Date().toLocaleTimeString();
      dashInferencesSub.textContent = `Last: ${cropName} (${timeStr})`;
    }
  }

  showSuccessBanner(predictedCrop, championModel) {
    if (!this.validationBanner) return;

    this.validationBanner.className = 'callout form-banner-success';
    this.validationBanner.style.display = 'flex';
    this.validationBanner.innerHTML = `
      <div class="callout-icon" style="color: var(--color-moss-green-dark);">
        ${window.AppIcons ? window.AppIcons.get('checkCircle') : ''}
      </div>
      <div class="callout-content">
        <div class="callout-title" style="color: var(--color-moss-green-dark);">
          Recommendation Generated: ${this.escapeHtml(predictedCrop.toUpperCase())}
        </div>
        <div class="callout-body">
          Validated against 7 input bounds and classified via <strong>${this.escapeHtml(championModel)}</strong>.
        </div>
      </div>
    `;
  }

  showErrorBanner(message) {
    if (!this.validationBanner) return;

    this.validationBanner.className = 'callout form-banner-error';
    this.validationBanner.style.display = 'flex';
    this.validationBanner.innerHTML = `
      <div class="callout-icon" style="color: #A93226;">
        ${window.AppIcons ? window.AppIcons.get('alertCircle') : ''}
      </div>
      <div class="callout-content">
        <div class="callout-title" style="color: #A93226;">Inference Action Required</div>
        <div class="callout-body">${this.escapeHtml(message)}</div>
      </div>
    `;
  }

  renderRecommendationResult(data) {
    if (this.resultBadge) {
      this.resultBadge.className = 'badge badge-moss';
      this.resultBadge.textContent = 'Recommendation Ready';
    }

    if (!this.resultBody) return;

    const crop = data.predicted_crop;
    const champion = data.champion_model;
    const inputs = data.inputs || {};
    const timestamp = data.inference_timestamp ? new Date(data.inference_timestamp).toLocaleTimeString() : 'Just now';

    this.resultBody.innerHTML = `
      <div style="display: flex; flex-direction: column; gap: var(--space-5);">
        <!-- Champion Crop Recommendation Banner -->
        <div style="background-color: var(--color-alabaster-surface); border: 2px solid var(--color-moss-green); border-radius: var(--radius-md); padding: var(--space-5); display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: var(--space-4);">
          <div style="display: flex; align-items: center; gap: var(--space-4);">
            <div style="width: 56px; height: 56px; border-radius: var(--radius-md); background-color: var(--color-dark-emerald); color: var(--color-alabaster); display: flex; align-items: center; justify-content: center; font-size: 24px;">
              ${window.AppIcons ? window.AppIcons.get('sprout') : '🌱'}
            </div>
            <div>
              <div style="font-size: var(--font-size-xs); color: var(--color-slate-green); font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Recommended Crop Species</div>
              <h3 style="font-size: 28px; font-weight: 700; color: var(--color-dark-emerald); margin: 2px 0 0 0; text-transform: capitalize;">
                ${this.escapeHtml(crop)}
              </h3>
            </div>
          </div>
          <div style="display: flex; flex-direction: column; align-items: flex-end; gap: var(--space-1);">
            <span class="badge badge-moss">${this.escapeHtml(champion)}</span>
            <span style="font-size: var(--font-size-xs); color: var(--text-muted);">Inference executed at ${timestamp}</span>
          </div>
        </div>

        <!-- Verified Input Vector Summary -->
        <div>
          <h4 style="font-size: var(--font-size-sm); font-weight: 600; color: var(--text-primary); margin-bottom: var(--space-2);">
            Verified Agronomic Input Vector
          </h4>
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: var(--space-2);">
            ${this.fields.map(f => `
              <div style="background-color: var(--bg-surface); border: 1px solid var(--border-color-default); border-radius: var(--radius-sm); padding: var(--space-2) var(--space-3);">
                <div style="font-size: 11px; color: var(--text-muted); font-weight: 600;">${f.name}</div>
                <div style="font-size: var(--font-size-base); font-weight: 700; color: var(--color-dark-emerald); margin-top: 2px;">
                  ${inputs[f.apiKey] !== undefined ? inputs[f.apiKey] : '—'} <span style="font-size: 11px; font-weight: 400; color: var(--text-secondary);">${f.unit}</span>
                </div>
              </div>
            `).join('')}
          </div>
        </div>

        <!-- Methodological Reliability Notice -->
        <div style="font-size: var(--font-size-xs); color: var(--text-muted); border-top: 1px solid var(--border-color-default); padding-top: var(--space-3); display: flex; justify-content: space-between; flex-wrap: wrap; gap: var(--space-2);">
          <span>• Deterministic prediction from saved artifact <code>models/best_crop_model.json</code>.</span>
          <span>• Evaluated with leak-free StandardScaler preprocessing.</span>
        </div>
      </div>
    `;
  }

  renderNoModelView() {
    if (this.resultBadge) {
      this.resultBadge.className = 'badge badge-neutral';
      this.resultBadge.textContent = 'Pipeline Inactive';
    }
    if (this.resultBody) {
      this.resultBody.innerHTML = `
        <div class="result-unavailable-panel">
          <div class="result-unavailable-icon" data-icon="alertCircle" aria-hidden="true"></div>
          <div class="result-unavailable-content">
            <div class="result-unavailable-title">
              No trained model connected. Complete model training in Model Evaluation before requesting a crop prediction.
            </div>
            <div class="result-unavailable-text">
              The machine learning inference backend requires a trained and approved model pipeline. Navigate to the Model Evaluation tab to fit the champion classifier on the validated agricultural dataset.
            </div>
          </div>
        </div>
      `;
    }
  }

  renderErrorResultView(msg) {
    if (this.resultBadge) {
      this.resultBadge.className = 'badge badge-slate';
      this.resultBadge.textContent = 'Inference Error';
    }
    if (this.resultBody) {
      this.resultBody.innerHTML = `
        <div class="callout form-banner-error" style="margin-bottom: 0;">
          <div class="callout-icon" style="color: #A93226;">
            ${window.AppIcons ? window.AppIcons.get('alertCircle') : ''}
          </div>
          <div class="callout-content">
            <div class="callout-title" style="color: #A93226;">Inference Execution Failed</div>
            <div class="callout-body">${this.escapeHtml(msg)}</div>
          </div>
        </div>
      `;
    }
  }

  resetResultView() {
    this.checkModelStatusInitial();
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
  window.predictionForm = new PredictionFormController();
});
