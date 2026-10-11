# Models Directory (`models/`)

This directory stores serialized model artifacts and training metadata produced by the machine learning pipeline:

1. **`best_crop_model.json`**:
   - JSON-serialized weights, decision tree node structures, or neighbor indexes for the selected champion classifier (Random Forest, CART Decision Tree, or KNN).

2. **`model_metadata.json`**:
   - Comprehensive model provenance metadata, including:
     - `champion_model`: Identifier of selected candidate algorithm.
     - `selection_rule`: Evaluated strictly within training partition via Stratified 5-Fold CV (Training Partition Only).
     - `selection_score_f1`: Mean cross-validation Macro-F1 score across training folds.
     - `dataset_fingerprint`: 16-character SHA-256 hash linking the trained model to the active dataset schema.
     - `classes`: Complete array of recognized crop classes.
     - `scaler`: Trained `StandardScalerModel` parameters (`mean_`, `scale_`) fitted exclusively on training data.

### Clean Startup Policy
To prevent serving stale simulated weights, clean releases start with no pre-trained model artifacts. Models are dynamically generated and validated upon user-authorized training.
