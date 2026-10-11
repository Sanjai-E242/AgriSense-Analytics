/**
 * ==========================================================================
 * Crop Recommendation System - Model Training & Evaluation Controller
 * Principles of Data Science (PoDS) Project
 * ==========================================================================
 */

class ModelEvaluationController {
  constructor() {
    this.badge = document.getElementById('model-pipeline-badge');
    this.statusBanner = document.getElementById('training-status-banner');
    this.datasetInfo = document.getElementById('training-dataset-info');
    this.confirmCheckbox = document.getElementById('chk-confirm-training');
    this.trainBtn = document.getElementById('btn-start-training');
    this.spinnerText = document.getElementById('training-spinner-text');

    this.benchmarkContainer = document.getElementById('model-benchmark-container');
    this.championContainer = document.getElementById('champion-deepdive-container');
    this.confusionContainer = document.getElementById('confusion-matrix-container');
    this.warningsContainer = document.getElementById('model-warnings-container');
    this.storageContainer = document.getElementById('model-storage-container');

    // Dashboard Telemetry Metric Elements (Explicit stable IDs)
    this.dashModelBadge = document.getElementById('dash-model-badge');
    this.dashModelValue = document.getElementById('dash-model-value');
    this.dashModelSubtext = document.getElementById('dash-model-subtext');

    this.activeResults = null;
    this.isTraining = false;

    this.init();
  }

  init() {
    // 1. Checkbox authorization toggle
    if (this.confirmCheckbox) {
      this.confirmCheckbox.addEventListener('change', () => {
        this.updateTrainButtonState();
      });
    }

    // 2. Train button click handler
    if (this.trainBtn) {
      this.trainBtn.addEventListener('click', () => {
        this.executeTraining();
      });
    }

    // 3. Check existing trained model status on load
    this.checkExistingModel();

    // 4. Update dataset eligibility state on tab navigation
    document.querySelectorAll('[data-nav-item], [data-navigate-to]').forEach(el => {
      el.addEventListener('click', () => {
        setTimeout(() => this.syncDatasetEligibility(), 80);
      });
    });

    // 5. Reactive dataset lifecycle event listeners (No repeated 2s polling loops)
    window.addEventListener('dataset:updated', () => {
      this.syncDatasetEligibility();
    });

    window.addEventListener('dataset:reset', () => {
      this.syncDatasetEligibility();
    });
  }

  async checkExistingModel() {
    try {
      const res = await fetch('/api/model-status');
      const contentType = res.headers.get('content-type') || '';
      if (res.ok && contentType.includes('application/json')) {
        const data = await res.json();
        if (data.trained && data.artifact_valid && data.metadata) {
          const champion = data.champion_model || data.metadata.champion_model || 'Champion Model';
          this.renderExistingMetadata(data.metadata, champion);
        } else if (data.artifact_present && !data.artifact_valid) {
          // Model artifact present but invalid/corrupted
          if (this.dashModelBadge) {
            this.dashModelBadge.className = 'badge badge-slate';
            this.dashModelBadge.textContent = 'Artifact Error';
          }
          if (this.dashModelValue) {
            this.dashModelValue.className = 'metric-value';
            this.dashModelValue.textContent = 'Invalid';
          }
          if (this.dashModelSubtext) {
            this.dashModelSubtext.textContent = data.error || 'Saved artifact failed validation';
          }
        } else {
          // No model trained
          if (this.dashModelBadge) {
            this.dashModelBadge.className = 'badge badge-neutral';
            this.dashModelBadge.textContent = 'Untrained';
          }
          if (this.dashModelValue) {
            this.dashModelValue.className = 'metric-value unavailable';
            this.dashModelValue.textContent = 'None';
          }
          if (this.dashModelSubtext) {
            this.dashModelSubtext.textContent = 'Awaiting model training pipeline';
          }
        }
      }
    } catch (e) {
      // Offline / basic server mode
    }
  }

  syncDatasetEligibility() {
    if (this.isTraining) return;

    const ingestion = window.datasetIngestion;
    const activeDataset = ingestion ? ingestion.activeDataset : null;

    if (!this.datasetInfo) return;

    if (!activeDataset) {
      this.datasetInfo.innerHTML = `
        <div class="callout" role="note" style="margin-bottom: var(--space-4);">
          <div class="callout-icon" data-icon="info" aria-hidden="true"></div>
          <div class="callout-content">
            <div class="callout-title">No Active Dataset Ingested</div>
            <div class="callout-body">
              Please navigate to the <a href="#dataset" style="color: var(--color-slate-green); font-weight: 600; text-decoration: underline;" data-navigate-to="dataset">Dataset Explorer</a> and upload the verified agricultural CSV dataset before initiating model training.
            </div>
          </div>
        </div>
      `;
      if (this.confirmCheckbox) {
        this.confirmCheckbox.checked = false;
        this.confirmCheckbox.disabled = true;
        this.confirmCheckbox.closest('.form-checkbox-label')?.classList.add('disabled');
      }
      this.updateTrainButtonState();
      return;
    }

    // Check explicitly designated prohibited test fixtures
    const filename = (activeDataset.filename || '').trim();
    const filenameLower = filename.toLowerCase();
    const prohibitedFixtures = [
      'crop_recommendation_sample_test.csv',
      'crop_recommendation_5000_simulated_demo.csv',
      'synthetic_dataset.csv',
      'fixture_data.csv'
    ];
    const isFixture = prohibitedFixtures.includes(filenameLower) ||
                      filenameLower.startsWith('synthetic_') ||
                      filenameLower.startsWith('fixture_') ||
                      Boolean(activeDataset.is_simulated);

    if (isFixture) {
      this.datasetInfo.innerHTML = `
        <div class="callout form-banner-error" style="margin-bottom: var(--space-4);">
          <div class="callout-icon" style="color: #A93226;">
            ${window.AppIcons ? window.AppIcons.get('alertCircle') : ''}
          </div>
          <div class="callout-content">
            <div class="callout-title" style="color: #A93226;">Training Prohibited on Test Fixture Dataset</div>
            <div class="callout-body">
              The currently ingested file (<code>${this.escapeHtml(filename)}</code>) is a designated test fixture. 
              In compliance with data safety protocols, machine learning models may not be trained on synthetic fixtures. 
              Please upload the project owner's verified crop dataset.
            </div>
          </div>
        </div>
      `;
      if (this.confirmCheckbox) {
        this.confirmCheckbox.checked = false;
        this.confirmCheckbox.disabled = true;
        this.confirmCheckbox.closest('.form-checkbox-label')?.classList.add('disabled');
      }
      this.updateTrainButtonState();
      return;
    }

    // Verified dataset ready
    this.datasetInfo.innerHTML = `
      <div class="callout form-banner-success" style="margin-bottom: var(--space-4);">
        <div class="callout-icon" style="color: var(--color-moss-green-dark);">
          ${window.AppIcons ? window.AppIcons.get('checkCircle') : ''}
        </div>
        <div class="callout-content">
          <div class="callout-title" style="color: var(--color-moss-green-dark);">
            Verified Agricultural Dataset Ingested & Eligible for Training
          </div>
          <div class="callout-body">
            File: <strong>${this.escapeHtml(filename)}</strong> • 
            <strong>${activeDataset.rowCount.toLocaleString()} sample records</strong> across 
            <strong>${activeDataset.columnCount} columns</strong>. 
            All required continuous features and target labels have been validated.
          </div>
        </div>
      </div>
    `;

    if (this.confirmCheckbox) {
      this.confirmCheckbox.disabled = false;
      this.confirmCheckbox.closest('.form-checkbox-label')?.classList.remove('disabled');
    }
    this.updateTrainButtonState();
  }

  updateTrainButtonState() {
    if (!this.trainBtn) return;
    const canTrain = this.confirmCheckbox && this.confirmCheckbox.checked && !this.confirmCheckbox.disabled && !this.isTraining;
    this.trainBtn.disabled = !canTrain;
  }

  async executeTraining() {
    const ingestion = window.datasetIngestion;
    if (!ingestion || !ingestion.activeDataset) {
      this.showError('No active dataset found. Please upload a verified dataset first.');
      return;
    }

    if (!this.confirmCheckbox || !this.confirmCheckbox.checked) {
      this.showError('Project owner authorization required. Please check the confirmation checkbox.');
      return;
    }

    const dataset = ingestion.activeDataset;

    // Set UI to training state
    this.isTraining = true;
    this.updateTrainButtonState();
    if (this.spinnerText) this.spinnerText.style.display = 'inline-block';
    if (this.badge) {
      this.badge.className = 'badge badge-slate';
      this.badge.textContent = 'Training in Progress...';
    }

    // Update Dashboard Active Model card to Training state using explicit IDs
    if (this.dashModelBadge) {
      this.dashModelBadge.className = 'badge badge-slate';
      this.dashModelBadge.textContent = 'Training';
    }
    if (this.dashModelValue) {
      this.dashModelValue.className = 'metric-value';
      this.dashModelValue.textContent = 'Fitting...';
    }
    if (this.dashModelSubtext) {
      this.dashModelSubtext.textContent = 'Fitting Decision Tree, Random Forest & KNN';
    }

    // Hide error banner if visible
    if (this.statusBanner) this.statusBanner.style.display = 'none';

    try {
      // Reconstruct CSV text from parsed rows or active memory
      let csvText = '';
      if (ingestion.rawCsvText) {
        csvText = ingestion.rawCsvText;
      } else if (dataset.rawCsvText) {
        csvText = dataset.rawCsvText;
      } else if (Array.isArray(dataset.rows) && dataset.rows.length > 0) {
        const header = dataset.columns.join(',');
        const rows = dataset.rows.map(r => r.join(','));
        csvText = [header, ...rows].join('\n');
      }

      const res = await fetch('/api/train-models', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_name: dataset.filename,
          dataset_fingerprint: dataset.dataset_fingerprint,
          approved_by_user: true,
          csv_text: csvText
        })
      });

      const contentType = res.headers.get('content-type') || '';
      let data = null;

      if (contentType.includes('application/json')) {
        data = await res.json();
      } else {
        const rawText = await res.text();
        const cleanSnippet = rawText.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
        let errorDetail = `Server returned HTTP ${res.status} (${res.statusText || 'Unexpected response'}).`;
        if (res.status === 504 || res.status === 502) {
          errorDetail = `Training timed out on host gateway (HTTP ${res.status}).`;
        } else if (res.status === 500) {
          errorDetail = `Server encountered an internal error (HTTP 500): ${cleanSnippet.slice(0, 160) || 'Internal Server Error'}.`;
        } else if (cleanSnippet) {
          errorDetail += ` Response: ${cleanSnippet.slice(0, 160)}`;
        }
        throw new Error(errorDetail);
      }

      if (!res.ok || !data.success) {
        this.showError(data.message || 'Failed to train classification pipeline.');
        this.isTraining = false;
        this.updateTrainButtonState();
        if (this.spinnerText) this.spinnerText.style.display = 'none';
        if (this.badge) {
          this.badge.className = 'badge badge-neutral';
          this.badge.textContent = 'Pipeline Inactive';
        }
        // Restore dashboard active model card
        if (this.dashModelBadge) {
          this.dashModelBadge.className = 'badge badge-neutral';
          this.dashModelBadge.textContent = 'Untrained';
        }
        if (this.dashModelValue) {
          this.dashModelValue.className = 'metric-value unavailable';
          this.dashModelValue.textContent = 'None';
        }
        if (this.dashModelSubtext) {
          this.dashModelSubtext.textContent = 'Awaiting model training pipeline';
        }
        return;
      }

      // Training Successful
      this.activeResults = data;
      this.renderTrainingResults(data);

      if (this.badge) {
        this.badge.className = 'badge badge-moss';
        this.badge.textContent = `Trained (${data.champion_model})`;
      }

      // Explicitly update Dashboard Active Model card (Stable IDs)
      if (this.dashModelBadge) {
        this.dashModelBadge.className = 'badge badge-moss';
        this.dashModelBadge.textContent = 'Trained';
      }
      if (this.dashModelValue) {
        this.dashModelValue.className = 'metric-value';
        this.dashModelValue.textContent = data.champion_model;
      }
      if (this.dashModelSubtext) {
        const timeStr = data.trained_at ? new Date(data.trained_at).toLocaleTimeString() : 'Just now';
        this.dashModelSubtext.textContent = `Champion model active • ${timeStr}`;
      }

      // Automatically synchronize Crop Prediction inference view state
      if (window.predictionForm && typeof window.predictionForm.checkModelStatus === 'function') {
        window.predictionForm.checkModelStatus();
      }

      // Screen reader announcement
      const announcer = document.getElementById('aria-live-announcer');
      if (announcer) {
        announcer.textContent = `Machine learning models successfully trained. Champion model: ${data.champion_model}.`;
      }

    } catch (err) {
      this.showError(`Training execution error: ${err.message}`);
      if (this.badge) {
        this.badge.className = 'badge badge-neutral';
        this.badge.textContent = 'Pipeline Inactive';
      }
      if (this.dashModelBadge) {
        this.dashModelBadge.className = 'badge badge-neutral';
        this.dashModelBadge.textContent = 'Untrained';
      }
      if (this.dashModelValue) {
        this.dashModelValue.className = 'metric-value unavailable';
        this.dashModelValue.textContent = 'None';
      }
    } finally {
      this.isTraining = false;
      this.updateTrainButtonState();
      if (this.spinnerText) this.spinnerText.style.display = 'none';
    }
  }

  showError(message) {
    if (this.statusBanner) {
      this.statusBanner.className = 'callout form-banner-error';
      this.statusBanner.style.display = 'flex';
      this.statusBanner.innerHTML = `
        <div class="callout-icon" style="color: #A93226;">
          ${window.AppIcons ? window.AppIcons.get('alertCircle') : ''}
        </div>
        <div class="callout-content">
          <div class="callout-title" style="color: #A93226;">Training Halted</div>
          <div class="callout-body">${this.escapeHtml(message)}</div>
        </div>
      `;
    }
  }

  renderTrainingResults(data) {
    const results = data.benchmark_results;
    const champion = data.champion_model;

    // 1. Render Benchmark Comparison Table
    if (this.benchmarkContainer && results) {
      this.benchmarkContainer.innerHTML = `
        <div class="card-header">
          <div class="card-title-group">
            <h2 class="card-title">Comparative Classifier Benchmark Table</h2>
            <div class="card-subtitle">Empirical performance across 80% train / 20% held-out test splits</div>
          </div>
          <span class="badge badge-moss">3 Classifiers Evaluated</span>
        </div>
        <div class="card-body">
          <div class="table-responsive">
            <table class="data-table" aria-label="Classifier Performance Benchmark Table">
              <thead>
                <tr>
                  <th scope="col">Classifier Algorithm</th>
                  <th scope="col" title="Accuracy on training split">Train Accuracy</th>
                  <th scope="col" title="Accuracy on held-out test split">Held-out Test Acc</th>
                  <th scope="col" title="Macro-averaged precision">Macro Precision</th>
                  <th scope="col" title="Macro-averaged recall">Macro Recall</th>
                  <th scope="col" title="Macro-averaged F1 score">Macro F1-Score</th>
                  <th scope="col" title="Stratified K-Fold cross validation">5-Fold CV (Macro-F1 μ ± σ)</th>
                </tr>
              </thead>
              <tbody>
                ${Object.keys(results).map(name => {
                  const m = results[name];
                  const isChamp = (name === champion);
                  const cvText = m.cross_validation && m.cross_validation.supported && m.cross_validation.mean !== null
                    ? `${(m.cross_validation.mean * 100).toFixed(1)}% ± ${(m.cross_validation.std * 100).toFixed(1)}%`
                    : '<span style="color: var(--text-muted);">Too small for CV</span>';

                  return `
                    <tr class="${isChamp ? 'benchmark-champion-row' : ''}">
                      <td>
                        <strong>${name}</strong>
                        ${isChamp ? '<span class="badge badge-moss" style="margin-left: var(--space-2); font-size: 10px;">Champion</span>' : ''}
                      </td>
                      <td class="metric-num" style="color: var(--text-secondary);">${(m.train_accuracy * 100).toFixed(1)}%</td>
                      <td class="metric-num" style="color: var(--color-dark-emerald); font-weight: 700;">${(m.test_accuracy * 100).toFixed(1)}%</td>
                      <td class="metric-num">${(m.macro_precision * 100).toFixed(1)}%</td>
                      <td class="metric-num">${(m.macro_recall * 100).toFixed(1)}%</td>
                      <td class="metric-num" style="font-weight: 700; color: ${isChamp ? 'var(--color-moss-green-dark)' : 'var(--text-primary)'};">${(m.macro_f1 * 100).toFixed(1)}%</td>
                      <td class="metric-num">${cvText}</td>
                    </tr>
                  `;
                }).join('')}
              </tbody>
            </table>
          </div>
          <div style="margin-top: var(--space-3); font-size: var(--font-size-xs); color: var(--text-secondary); display: flex; gap: var(--space-4); flex-wrap: wrap;">
            <span>• <strong>Held-out split:</strong> Evaluated on strictly unseen 20% observations.</span>
            <span>• <strong>Leakage prevention:</strong> StandardScaler fit exclusively on each training fold.</span>
            <span>• <strong>Champion rule:</strong> Highest mean cross-validation Macro-F1 score (evaluated strictly on 5-fold CV within the 80% training partition). Untouched held-out test split is evaluated post-selection.</span>
          </div>
        </div>
      `;
    }

    // 2. Render Champion Model Deep-Dive & Per-Class Table
    if (this.championContainer && results && results[champion]) {
      const champData = results[champion];
      const perClass = champData.per_class_metrics;

      this.championContainer.innerHTML = `
        <div class="card-header">
          <div class="card-title-group">
            <h2 class="card-title">Champion Model: ${champion}</h2>
            <div class="card-subtitle">Per-class precision, recall, and F1-score breakdowns on held-out test data</div>
          </div>
          <span class="badge badge-moss">${(champData.macro_f1 * 100).toFixed(1)}% Macro F1</span>
        </div>
        <div class="card-body">
          <div class="table-responsive">
            <table class="data-table" aria-label="Per-Class Crop Recommendation Metrics">
              <thead>
                <tr>
                  <th scope="col">Target Crop Species</th>
                  <th scope="col">Precision</th>
                  <th scope="col">Recall</th>
                  <th scope="col">F1-Score</th>
                  <th scope="col">Held-Out Test Support</th>
                </tr>
              </thead>
              <tbody>
                ${Object.keys(perClass).map(cls => {
                  const cm = perClass[cls];
                  return `
                    <tr>
                      <td style="text-transform: capitalize;"><strong>${this.escapeHtml(cls)}</strong></td>
                      <td class="metric-num">${(cm.precision * 100).toFixed(1)}%</td>
                      <td class="metric-num">${(cm.recall * 100).toFixed(1)}%</td>
                      <td class="metric-num" style="font-weight: 700; color: var(--color-dark-emerald);">${(cm.f1_score * 100).toFixed(1)}%</td>
                      <td class="metric-num" style="color: var(--text-secondary);">${cm.support} samples</td>
                    </tr>
                  `;
                }).join('')}
              </tbody>
            </table>
          </div>
        </div>
      `;
    }

    // 3. Render Multiclass Confusion Matrix
    if (this.confusionContainer && results && results[champion]) {
      const cmData = results[champion].confusion_matrix;
      const classes = cmData.classes;
      const matrix = cmData.matrix;

      this.confusionContainer.innerHTML = `
        <div class="card-header">
          <div class="card-title-group">
            <h2 class="card-title">Multiclass Confusion Matrix Visualizer</h2>
            <div class="card-subtitle">True Species (Rows) vs Predicted Species (Columns) on Held-Out Test Set</div>
          </div>
          <span class="badge badge-slate">${classes.length} × ${classes.length} Matrix</span>
        </div>
        <div class="card-body">
          <div class="table-responsive">
            <table class="data-table" style="font-size: 11px;">
              <thead>
                <tr>
                  <th scope="col" style="background-color: var(--color-alabaster-surface);">Actual \\ Predicted</th>
                  ${classes.map(c => `<th scope="col" style="text-align: center; text-transform: capitalize;">${this.escapeHtml(c)}</th>`).join('')}
                </tr>
              </thead>
              <tbody>
                ${classes.map((trueCls, rowIdx) => `
                  <tr>
                    <td style="text-transform: capitalize;"><strong>${this.escapeHtml(trueCls)}</strong></td>
                    ${classes.map((predCls, colIdx) => {
                      const count = matrix[rowIdx][colIdx];
                      const isMatch = (rowIdx === colIdx);
                      const isError = (!isMatch && count > 0);
                      const cellClass = isMatch ? 'confusion-cell-match' : (isError ? 'confusion-cell-error' : '');

                      return `
                        <td class="confusion-cell ${cellClass}">
                          ${count}
                        </td>
                      `;
                    }).join('')}
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
          <div style="margin-top: var(--space-2); font-size: var(--font-size-xs); color: var(--text-muted);">
            Green cells represent accurate classifications along the principal diagonal. Tinted orange cells highlight misclassifications.
          </div>
        </div>
      `;
    }

    // 4. Render Statistical Reliability Warnings
    if (this.warningsContainer) {
      const warnings = data.reliability_warnings || [];
      if (warnings.length > 0) {
        this.warningsContainer.innerHTML = `
          <div class="card-header">
            <div class="card-title-group">
              <h2 class="card-title">Statistical Reliability & Sample Limitations</h2>
              <div class="card-subtitle">Empirical validation caveats and pre-deployment warnings</div>
            </div>
            <span class="badge badge-slate">${warnings.length} Caveat(s)</span>
          </div>
          <div class="card-body">
            <div style="display: flex; flex-direction: column; gap: var(--space-2);">
              ${warnings.map(w => `
                <div class="callout form-banner-warning" style="margin-bottom: 0;">
                  <div class="callout-icon" style="color: var(--color-slate-green);">
                    ${window.AppIcons ? window.AppIcons.get('info') : ''}
                  </div>
                  <div class="callout-content">
                    <div class="callout-title" style="color: var(--color-slate-green);">Statistical Caveat</div>
                    <div class="callout-body">${this.escapeHtml(w)}</div>
                  </div>
                </div>
              `).join('')}
            </div>
          </div>
        `;
        this.warningsContainer.style.display = 'block';
      } else {
        this.warningsContainer.style.display = 'none';
      }
    }

    // 5. Render Model Storage & Integration Info
    if (this.storageContainer) {
      const storage = data.storage || {
        model_path: "models/best_crop_model.json",
        metadata_path: "models/model_metadata.json"
      };
      this.storageContainer.innerHTML = `
        <div class="card-header">
          <div class="card-title-group">
            <h2 class="card-title">Model Serialization & Schema Storage</h2>
            <div class="card-subtitle">Safe local pipeline artifact and inference integration metadata</div>
          </div>
          <span class="badge badge-moss">Artifact Saved</span>
        </div>
        <div class="card-body">
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: var(--space-4);">
            <div style="padding: var(--space-3) var(--space-4); background-color: var(--bg-surface); border: 1px solid var(--border-color-default); border-radius: var(--radius-sm);">
              <div style="font-size: var(--font-size-xs); color: var(--text-muted); font-weight: 600;">MODEL ARTIFACT</div>
              <div style="font-family: var(--font-family-mono); font-size: var(--font-size-sm); color: var(--color-dark-emerald); font-weight: 600; margin-top: 4px;">
                ${storage.model_path}
              </div>
              <div style="font-size: var(--font-size-xs); color: var(--text-secondary); margin-top: 4px;">
                Serialized champion pipeline including fitted StandardScaler and ${champion} estimator.
              </div>
            </div>

            <div style="padding: var(--space-3) var(--space-4); background-color: var(--bg-surface); border: 1px solid var(--border-color-default); border-radius: var(--radius-sm);">
              <div style="font-size: var(--font-size-xs); color: var(--text-muted); font-weight: 600;">METADATA SCHEMA</div>
              <div style="font-family: var(--font-family-mono); font-size: var(--font-size-sm); color: var(--color-dark-emerald); font-weight: 600; margin-top: 4px;">
                ${storage.metadata_path}
              </div>
              <div style="font-size: var(--font-size-xs); color: var(--text-secondary); margin-top: 4px;">
                Contains feature column order, class labels, training timestamp, and held-out benchmark scores.
              </div>
            </div>
          </div>
        </div>
      `;
    }
  }

  renderExistingMetadata(meta, championName) {
    const champ = championName || meta.champion_model || 'Champion Model';
    if (this.badge) {
      this.badge.className = 'badge badge-moss';
      this.badge.textContent = `Trained (${champ})`;
    }

    if (this.dashModelBadge) {
      this.dashModelBadge.className = 'badge badge-moss';
      this.dashModelBadge.textContent = 'Trained';
    }
    if (this.dashModelValue) {
      this.dashModelValue.className = 'metric-value';
      this.dashModelValue.textContent = champ;
    }
    if (this.dashModelSubtext) {
      const timeStr = meta.trained_at ? new Date(meta.trained_at).toLocaleTimeString() : 'Saved';
      this.dashModelSubtext.textContent = `Champion model active • ${timeStr}`;
    }

    // If metadata contains benchmark results, render them
    if (meta.benchmark_results) {
      this.renderTrainingResults({
        champion_model: champ,
        benchmark_results: meta.benchmark_results,
        reliability_warnings: meta.reliability_warnings || [],
        storage: {
          model_path: "models/best_crop_model.json",
          metadata_path: "models/model_metadata.json"
        }
      });
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
  window.modelEvaluation = new ModelEvaluationController();
});
