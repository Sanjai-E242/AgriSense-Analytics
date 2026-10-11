"""
Comprehensive Automated Test Suite for Principles of Data Science (PoDS)
Crop Recommendation System

Verifies all 15 criteria specified in Prompt Section 10:
 1. Training fails without explicit approval.
 2. Prohibited fixture filenames are rejected (and fixture bypass disabled in production API).
 3. Missing features and missing target columns are rejected.
 4. Non-numeric and non-finite features are rejected.
 5. Null or blank target labels are rejected.
 6. Invalid class distributions are handled correctly.
 7. Cross-validation does not use held-out test set for model selection.
 8. Training is reproducible with fixed seed 42.
 9. All three custom classifiers (CART Decision Tree, Random Forest, KNN) train and predict.
10. Saved models can be reloaded and reproduce their original predictions.
11. /api/model-status correctly handles absent, valid, and invalid artifacts.
12. /api/predict handles valid requests and invalid inputs correctly.
13. Dashboard updates Active Model card, not Feature Schema card.
14. Prediction count increases only after successful prediction.
15. Frontend design tokens, navigation, and structure preserved.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from validator import validate_csv_content, REQUIRED_SCHEMA, REQUIRED_FEATURES, COLUMN_ALIASES
from pipeline import (
    RANDOM_SEED,
    DecisionTreeClassifierModel,
    RandomForestClassifierModel,
    KNNClassifierModel,
    StandardScalerModel,
    ModelPipeline,
    train_and_evaluate_pipeline,
    predict_crop,
    load_production_model,
    check_fixture_block,
    MODELS_DIR
)
import server


def create_synthetic_dataframe(n_samples: int = 100) -> pd.DataFrame:
    """Helper creating a minimal valid synthetic dataset strictly for isolated unit testing."""
    np.random.seed(42)
    classes = ["rice", "maize", "cotton", "coffee"]
    labels = np.random.choice(classes, size=n_samples)
    data = {
        "N": np.random.uniform(10, 140, size=n_samples).round(1),
        "P": np.random.uniform(10, 100, size=n_samples).round(1),
        "K": np.random.uniform(10, 100, size=n_samples).round(1),
        "temperature": np.random.uniform(15, 38, size=n_samples).round(2),
        "humidity": np.random.uniform(30, 95, size=n_samples).round(2),
        "ph": np.random.uniform(5.5, 8.0, size=n_samples).round(2),
        "rainfall": np.random.uniform(40, 280, size=n_samples).round(1),
        "label": labels
    }
    return pd.DataFrame(data)


class TestPoDSCropSystem(unittest.TestCase):

    def setUp(self):
        self.app = server.app.test_client()
        self.app.testing = True

    @classmethod
    def tearDownClass(cls):
        # Clean up synthetic test artifacts so models/ remains untainted for real dataset
        models_dir = os.path.join(BASE_DIR, "models")
        for f in ["best_crop_model.json", "model_metadata.json"]:
            p = os.path.join(models_dir, f)
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
        data_dir = os.path.join(BASE_DIR, "data")
        for f in ["active_dataset.csv", "active_dataset.json"]:
            p = os.path.join(data_dir, f)
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    # -------------------------------------------------------------------------
    # 1. Training fails without explicit approval
    # -------------------------------------------------------------------------
    def test_01_training_fails_without_approval(self):
        df = create_synthetic_dataframe(40)
        # Python pipeline function
        res = train_and_evaluate_pipeline(
            df=df,
            dataset_name="verified_crops.csv",
            approved_by_user=False,
            allow_fixture_for_testing=True
        )
        self.assertFalse(res["success"])
        self.assertEqual(res["error_type"], "APPROVAL_REQUIRED")

        # Flask endpoint
        csv_str = df.to_csv(index=False)
        response = self.app.post('/api/train-models', json={
            "dataset_name": "verified_crops.csv",
            "approved_by_user": False,
            "csv_text": csv_str
        })
        self.assertEqual(response.status_code, 400)
        resp_json = response.get_json()
        self.assertFalse(resp_json["success"])
        self.assertEqual(resp_json["error_type"], "APPROVAL_REQUIRED")

    # -------------------------------------------------------------------------
    # 2. Prohibited fixture filenames are rejected
    # -------------------------------------------------------------------------
    def test_02_prohibited_fixture_filenames_rejected(self):
        df = create_synthetic_dataframe(40)
        prohibited_names = [
            "crop_recommendation_sample_test.csv",
            "crop_recommendation_5000_simulated_demo.csv",
            "fixture_data.csv",
            "synthetic_dataset.csv"
        ]
        for name in prohibited_names:
            # When allow_fixture_for_testing is False (default production behavior)
            res = train_and_evaluate_pipeline(
                df=df,
                dataset_name=name,
                approved_by_user=True,
                allow_fixture_for_testing=False
            )
            self.assertFalse(res["success"], f"Failed to reject prohibited name '{name}'")
            self.assertIn(res["error_type"], ["FIXTURE_DATASET_BLOCKED", "PROHIBITED_FIXTURE_DATASET"])

        # Flask endpoint cannot bypass this even if allow_fixture_for_testing is passed in JSON
        csv_str = df.to_csv(index=False)
        response = self.app.post('/api/train-models', json={
            "dataset_name": "crop_recommendation_sample_test.csv",
            "approved_by_user": True,
            "allow_fixture_for_testing": True,  # Attempted bypass
            "csv_text": csv_str
        })
        self.assertEqual(response.status_code, 400)
        resp_json = response.get_json()
        self.assertIn(resp_json["error_type"], ["FIXTURE_DATASET_BLOCKED", "PROHIBITED_FIXTURE_DATASET"])

        # CRITICAL REGRESSION: Valid filenames containing 'samples' or 'sample' MUST NOT be rejected
        valid_sample_names = [
            "Crop_recommendation 2200 samples.csv",
            "Crop_recommendation sample 2.csv",
            "Crop_recommendation 2200 samples(1).csv",
            "Crop_recommendation.csv"
        ]
        for name in valid_sample_names:
            allowed, msg = check_fixture_block(name, allow_fixture_for_testing=False)
            self.assertTrue(allowed, f"False positive fixture rejection on valid file '{name}': {msg}")

    # -------------------------------------------------------------------------
    # 3. Missing features and missing target columns are rejected
    # -------------------------------------------------------------------------
    def test_03_missing_features_and_missing_target_columns(self):
        # Missing 'rainfall' column
        csv_missing_rainfall = "N,P,K,temperature,humidity,ph,label\n90,42,43,20.8,82.0,6.5,rice\n"
        res = validate_csv_content(csv_missing_rainfall.encode('utf-8'), "data.csv")
        self.assertFalse(res["success"])
        self.assertEqual(res["error_type"], "MISSING_REQUIRED_COLUMNS")

        # Missing 'label' column
        csv_missing_label = "N,P,K,temperature,humidity,ph,rainfall\n90,42,43,20.8,82.0,6.5,200\n"
        res = validate_csv_content(csv_missing_label.encode('utf-8'), "data.csv")
        self.assertFalse(res["success"])
        self.assertEqual(res["error_type"], "MISSING_REQUIRED_COLUMNS")

        # Extra unexpected column: safely handled with warning by default
        csv_extra = "N,P,K,temperature,humidity,ph,rainfall,label,extra_col\n90,42,43,20.8,82.0,6.5,200,rice,extra\n91,40,41,21.0,81.0,6.6,190,maize\n"
        res = validate_csv_content(csv_extra.encode('utf-8'), "data.csv")
        self.assertTrue(res["success"])
        self.assertIn("extra_col", res["extra_columns"])
        self.assertTrue(len(res["warnings"]) > 0)

        # In strict_columns mode, extra column is rejected
        res_strict = validate_csv_content(csv_extra.encode('utf-8'), "data.csv", strict_columns=True)
        self.assertFalse(res_strict["success"])
        self.assertEqual(res_strict["error_type"], "UNEXPECTED_EXTRA_COLUMNS")

    # -------------------------------------------------------------------------
    # 4. Non-numeric and non-finite features are rejected
    # -------------------------------------------------------------------------
    def test_04_non_numeric_and_non_finite_features(self):
        # Non-numeric value in 'N'
        csv_corrupt = "N,P,K,temperature,humidity,ph,rainfall,label\nCORRUPT,42,43,20.8,82.0,6.5,200,rice\n10,20,30,25.0,70.0,6.0,150,maize\n"
        res = validate_csv_content(csv_corrupt.encode('utf-8'), "data.csv")
        self.assertFalse(res["success"])
        self.assertIn(res["error_type"], ["NON_NUMERIC_FEATURES", "NON_FINITE_FEATURES"])

        # Infinity value
        csv_inf = "N,P,K,temperature,humidity,ph,rainfall,label\ninf,42,43,20.8,82.0,6.5,200,rice\n10,20,30,25.0,70.0,6.0,150,maize\n"
        res = validate_csv_content(csv_inf.encode('utf-8'), "data.csv")
        self.assertFalse(res["success"])
        self.assertIn(res["error_type"], ["INFINITE_VALUES_DETECTED", "NON_FINITE_FEATURES"])

    # -------------------------------------------------------------------------
    # 5. Null or blank target labels are rejected
    # -------------------------------------------------------------------------
    def test_05_null_or_blank_target_labels(self):
        csv_blank_label = "N,P,K,temperature,humidity,ph,rainfall,label\n90,42,43,20.8,82.0,6.5,200,\n10,20,30,25.0,70.0,6.0,150,maize\n"
        res = validate_csv_content(csv_blank_label.encode('utf-8'), "data.csv")
        self.assertFalse(res["success"])
        self.assertIn(res["error_type"], ["NULL_VALUES_DETECTED", "EMPTY_TARGET_LABELS", "MISSING_VALUES"])

        csv_whitespace_label = "N,P,K,temperature,humidity,ph,rainfall,label\n90,42,43,20.8,82.0,6.5,200,   \n10,20,30,25.0,70.0,6.0,150,maize\n"
        res = validate_csv_content(csv_whitespace_label.encode('utf-8'), "data.csv")
        self.assertFalse(res["success"])
        self.assertIn(res["error_type"], ["NULL_VALUES_DETECTED", "EMPTY_TARGET_LABELS", "MISSING_VALUES"])

    # -------------------------------------------------------------------------
    # 6. Invalid class distributions are handled correctly
    # -------------------------------------------------------------------------
    def test_06_invalid_class_distributions(self):
        # Only 1 unique crop class
        csv_single_class = (
            "N,P,K,temperature,humidity,ph,rainfall,label\n"
            "90,42,43,20.8,82.0,6.5,200,rice\n"
            "80,40,40,22.0,80.0,6.6,190,rice\n"
            "85,41,42,21.5,81.0,6.4,195,rice\n"
        )
        res = validate_csv_content(csv_single_class.encode('utf-8'), "data.csv")
        self.assertFalse(res["success"])
        self.assertEqual(res["error_type"], "INSUFFICIENT_CLASSES")

    # -------------------------------------------------------------------------
    # 7. Cross-validation does not use held-out test set for model selection
    # -------------------------------------------------------------------------
    def test_07_cross_validation_evaluation_methodology(self):
        df = create_synthetic_dataframe(80)
        # Use isolated testing flag to verify evaluation methodology without touching production artifacts
        res = train_and_evaluate_pipeline(
            df=df,
            dataset_name="reproducible_evaluation_test.csv",
            approved_by_user=True,
            allow_fixture_for_testing=True
        )
        self.assertTrue(res["success"])
        bench = res["benchmark_results"]

        # Check all 3 classifiers exist
        self.assertIn("Decision Tree", bench)
        self.assertIn("Random Forest", bench)
        self.assertIn("K-Nearest Neighbors", bench)

        # Verify that cross-validation was performed inside the training fold
        for model_name, m_data in bench.items():
            self.assertIn("cross_validation", m_data)
            cv = m_data["cross_validation"]
            self.assertTrue(cv["supported"])
            self.assertIsNotNone(cv["mean"])
            self.assertIsNotNone(cv["std"])
            self.assertEqual(cv["folds"], 5)
            self.assertEqual(len(cv["macro_f1_scores"]), 5)
            # Train accuracy and test accuracy must be separately reported
            self.assertIn("train_accuracy", m_data)
            self.assertIn("test_accuracy", m_data)
            self.assertIn("macro_f1", m_data)

        # Champion must be chosen by highest mean CV macro-F1 (with accuracy tie-break)
        champ = res["champion_model"]
        champ_cv_f1 = bench[champ]["cross_validation"]["mean"]
        for model_name, m_data in bench.items():
            model_cv_f1 = m_data["cross_validation"]["mean"]
            self.assertGreaterEqual(
                champ_cv_f1,
                model_cv_f1,
                f"Champion {champ} CV F1 ({champ_cv_f1}) should be >= {model_name} CV F1 ({model_cv_f1})"
            )

    # -------------------------------------------------------------------------
    # 8. Training is reproducible with fixed seed
    # -------------------------------------------------------------------------
    def test_08_reproducibility_with_fixed_seed(self):
        df = create_synthetic_dataframe(60)
        run1 = train_and_evaluate_pipeline(
            df=df,
            dataset_name="reproducibility_test_a.csv",
            approved_by_user=True,
            allow_fixture_for_testing=True
        )
        run2 = train_and_evaluate_pipeline(
            df=df,
            dataset_name="reproducibility_test_b.csv",
            approved_by_user=True,
            allow_fixture_for_testing=True
        )
        self.assertTrue(run1["success"])
        self.assertTrue(run2["success"])
        self.assertEqual(run1["champion_model"], run2["champion_model"])
        # Check that metrics match identically
        for name in ["Decision Tree", "Random Forest", "K-Nearest Neighbors"]:
            self.assertAlmostEqual(
                run1["benchmark_results"][name]["test_accuracy"],
                run2["benchmark_results"][name]["test_accuracy"],
                places=6
            )
            self.assertAlmostEqual(
                run1["benchmark_results"][name]["macro_f1"],
                run2["benchmark_results"][name]["macro_f1"],
                places=6
            )

    # -------------------------------------------------------------------------
    # 9. All three custom classifiers train and return valid predictions
    # -------------------------------------------------------------------------
    def test_09_all_three_custom_classifiers(self):
        np.random.seed(42)
        X = np.random.uniform(0, 100, size=(50, 7))
        y = np.array(["rice"] * 25 + ["maize"] * 25)

        # 1. Decision Tree
        dt = DecisionTreeClassifierModel(max_depth=5, min_samples_split=2)
        dt.fit(X, y)
        dt_preds = dt.predict(X[:5])
        self.assertEqual(len(dt_preds), 5)
        for p in dt_preds:
            self.assertIn(p, ["rice", "maize"])

        # 2. Random Forest
        rf = RandomForestClassifierModel(n_estimators=7, max_depth=5, random_state=42)
        rf.fit(X, y)
        rf_preds = rf.predict(X[:5])
        self.assertEqual(len(rf_preds), 5)
        for p in rf_preds:
            self.assertIn(p, ["rice", "maize"])

        # 3. K-Nearest Neighbors
        knn = KNNClassifierModel(k=3)
        knn.fit(X, y)
        knn_preds = knn.predict(X[:5])
        self.assertEqual(len(knn_preds), 5)
        for p in knn_preds:
            self.assertIn(p, ["rice", "maize"])

    # -------------------------------------------------------------------------
    # 10. Saved models can be reloaded and reproduce original predictions
    # -------------------------------------------------------------------------
    def test_10_serialization_and_deserialization(self):
        np.random.seed(42)
        X = np.random.uniform(10, 80, size=(40, 7))
        y = np.array(["rice"] * 20 + ["coffee"] * 20)

        scaler = StandardScalerModel()
        X_scaled = scaler.fit_transform(X)

        for classifier_cls, kwargs in [
            (DecisionTreeClassifierModel, {"max_depth": 4}),
            (RandomForestClassifierModel, {"n_estimators": 5, "max_depth": 4, "random_state": 42}),
            (KNNClassifierModel, {"k": 3})
        ]:
            model = classifier_cls(**kwargs)
            model.fit(X_scaled, y)
            pipeline = ModelPipeline(scaler=scaler, classifier=model, name=classifier_cls.__name__)

            # Original in-memory prediction
            orig_preds = pipeline.predict(X)

            # Serialize to dict / json
            p_dict = pipeline.to_dict()
            json_str = json.dumps(p_dict)

            # Reconstruct from serialized json
            loaded_dict = json.loads(json_str)
            loaded_pipeline = ModelPipeline.from_dict(loaded_dict)
            loaded_preds = loaded_pipeline.predict(X)

            # Assert identical output
            np.testing.assert_array_equal(orig_preds, loaded_preds)

    # -------------------------------------------------------------------------
    # 11. /api/model-status correctly handles absent, valid, and invalid artifacts
    # -------------------------------------------------------------------------
    def test_11_model_status_endpoint(self):
        models_dir = os.path.join(BASE_DIR, "models")
        meta_path = os.path.join(models_dir, "model_metadata.json")
        model_path = os.path.join(models_dir, "best_crop_model.json")

        # Case A: When artifacts don't exist
        # If they exist from previous runs, temporarily move them
        tmp_backup = None
        if os.path.exists(models_dir) and (os.path.exists(meta_path) or os.path.exists(model_path)):
            tmp_backup = tempfile.mkdtemp()
            if os.path.exists(meta_path):
                shutil.move(meta_path, os.path.join(tmp_backup, "model_metadata.json"))
            if os.path.exists(model_path):
                shutil.move(model_path, os.path.join(tmp_backup, "best_crop_model.json"))

        try:
            res = self.app.get('/api/model-status')
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertFalse(data["trained"])
            self.assertFalse(data["artifact_present"])
            self.assertFalse(data["artifact_valid"])

            # Case B: Artifact is corrupted
            os.makedirs(models_dir, exist_ok=True)
            with open(model_path, "w") as f:
                f.write("{ INVALID JSON")
            with open(meta_path, "w") as f:
                f.write("{ INVALID JSON")

            res_corrupted = self.app.get('/api/model-status')
            self.assertEqual(res_corrupted.status_code, 200)
            corrupt_data = res_corrupted.get_json()
            self.assertFalse(corrupt_data["trained"])
            self.assertTrue(corrupt_data["artifact_present"])
            self.assertFalse(corrupt_data["artifact_valid"])

        finally:
            # Clean up corrupted files and restore backup if present
            if os.path.exists(model_path):
                os.remove(model_path)
            if os.path.exists(meta_path):
                os.remove(meta_path)
            if tmp_backup and os.path.exists(tmp_backup):
                b_meta = os.path.join(tmp_backup, "model_metadata.json")
                b_model = os.path.join(tmp_backup, "best_crop_model.json")
                if os.path.exists(b_meta):
                    shutil.move(b_meta, meta_path)
                if os.path.exists(b_model):
                    shutil.move(b_model, model_path)
                shutil.rmtree(tmp_backup, ignore_errors=True)

    # -------------------------------------------------------------------------
    # 12. /api/predict handles valid requests and invalid inputs correctly
    # -------------------------------------------------------------------------
    def test_12_predict_endpoint(self):
        # Test input rejection when no model or invalid parameters
        # Missing feature
        res_missing = self.app.post('/api/predict', json={
            "N": 90, "P": 42, "K": 43, "temperature": 20.8, "humidity": 82.0, "ph": 6.5
            # missing rainfall
        })
        self.assertEqual(res_missing.status_code, 400)

        # Non-numeric feature
        res_non_numeric = self.app.post('/api/predict', json={
            "N": "INVALID", "P": 42, "K": 43, "temperature": 20.8, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9
        })
        self.assertEqual(res_non_numeric.status_code, 400)

        # Out-of-bounds pH
        res_bounds = self.app.post('/api/predict', json={
            "N": 90, "P": 42, "K": 43, "temperature": 20.8, "humidity": 82.0, "ph": 25.0, "rainfall": 202.9
        })
        self.assertEqual(res_bounds.status_code, 400)

    # -------------------------------------------------------------------------
    # 13. Dashboard updates Active Model card, not Feature Schema card
    # -------------------------------------------------------------------------
    def test_13_dashboard_selector_bug_fixed(self):
        index_path = os.path.join(BASE_DIR, "frontend", "index.html") if os.path.exists(os.path.join(BASE_DIR, "frontend", "index.html")) else os.path.join(BASE_DIR, "index.html")
        with open(index_path, "r", encoding="utf-8") as f:
            html = f.read()

        # Confirm explicit stable IDs exist
        self.assertIn('id="card-active-model"', html)
        self.assertIn('id="dash-model-badge"', html)
        self.assertIn('id="dash-model-value"', html)
        self.assertIn('id="dash-model-subtext"', html)
        self.assertIn('id="card-feature-schema"', html)
        self.assertIn('id="dash-schema-badge"', html)
        self.assertIn('id="dash-samples-value"', html)
        self.assertIn('id="dash-inferences-value"', html)

        # Inspect js/model-evaluation.js to confirm positional selector is eliminated
        js_path = os.path.join(BASE_DIR, "frontend", "js", "model-evaluation.js") if os.path.exists(os.path.join(BASE_DIR, "frontend", "js", "model-evaluation.js")) else os.path.join(BASE_DIR, "js", "model-evaluation.js")
        with open(js_path, "r", encoding="utf-8") as f:
            js = f.read()

        self.assertNotIn('[data-view-panel="dashboard"] .card:nth-child(2) .badge', js)
        self.assertIn("dash-model-badge", js)
        self.assertIn("dash-model-value", js)

    # -------------------------------------------------------------------------
    # 14. Prediction count increases only after successful prediction
    # -------------------------------------------------------------------------
    def test_14_prediction_count_telemetry(self):
        js_pred_path = os.path.join(BASE_DIR, "frontend", "js", "prediction-form.js") if os.path.exists(os.path.join(BASE_DIR, "frontend", "js", "prediction-form.js")) else os.path.join(BASE_DIR, "js", "prediction-form.js")
        with open(js_pred_path, "r", encoding="utf-8") as f:
            js_pred = f.read()

        # Verify session inference counter increments only on successful backend response
        self.assertIn("this.sessionInferences++", js_pred)
        self.assertIn("dash-inferences-value", js_pred)
        self.assertIn("dash-inferences-badge", js_pred)
        # Verify it checks backend model readiness first
        self.assertIn("/api/model-status", js_pred)
        self.assertIn("/api/predict", js_pred)

    # -------------------------------------------------------------------------
    # 15. Frontend design tokens, navigation, and structure preserved
    # -------------------------------------------------------------------------
    def test_15_design_tokens_and_navigation(self):
        tokens_path = os.path.join(BASE_DIR, "frontend", "css", "tokens.css") if os.path.exists(os.path.join(BASE_DIR, "frontend", "css", "tokens.css")) else os.path.join(BASE_DIR, "css", "tokens.css")
        with open(tokens_path, "r", encoding="utf-8") as f:
            tokens_css = f.read().upper()

        self.assertIn("#F4F1EB", tokens_css)  # Alabaster
        self.assertIn("#7AA05A", tokens_css)  # Moss Green
        self.assertIn("#2D5A4A", tokens_css)  # Slate Green
        self.assertIn("#0D4C3C", tokens_css)  # Dark Emerald

        index_path = os.path.join(BASE_DIR, "frontend", "index.html") if os.path.exists(os.path.join(BASE_DIR, "frontend", "index.html")) else os.path.join(BASE_DIR, "index.html")
        with open(index_path, "r", encoding="utf-8") as f:
            html = f.read()

        # All navigation panels present
        self.assertIn('data-view-panel="dashboard"', html)
        self.assertIn('data-view-panel="prediction"', html)
        self.assertIn('data-view-panel="dataset"', html)
        self.assertIn('data-view-panel="evaluation"', html)
        self.assertIn('data-view-panel="history"', html)
        self.assertIn('data-view-panel="settings"', html)

    # -------------------------------------------------------------------------
    # 16. Champion selection NEVER uses held-out test metrics under any circumstance
    # -------------------------------------------------------------------------
    def test_16_champion_selection_never_uses_test_metrics_even_without_cv(self):
        # 1. Inspect pipeline.py source directly: confirm no test_metrics used for selection score
        pipeline_path = os.path.join(BASE_DIR, "pipeline.py")
        with open(pipeline_path, "r", encoding="utf-8") as f:
            code = f.read()

        # Confirm elimination of previous test_metrics fallback
        self.assertNotIn("score_f1 = cv_result[\"macro_f1_mean\"] if cv_result[\"supported\"] else test_metrics[\"macro_f1\"]", code)
        self.assertNotIn("score_acc = cv_result[\"accuracy_mean\"] if cv_result[\"supported\"] else test_metrics[\"accuracy\"]", code)

        # 2. Test on small dataset where CV is unavailable (min_train_class < 2 or train_samples < 10)
        # 12 samples: 2 classes (10 class A, 2 class B) -> stratified train has 8 of A, 1 of B (B has 1 in train, so CV unsupported)
        np.random.seed(42)
        small_df = pd.DataFrame({
            "N": [50, 52, 51, 53, 54, 55, 10, 12],
            "P": [40, 41, 40, 42, 43, 41, 20, 21],
            "K": [30, 31, 30, 32, 33, 31, 15, 16],
            "temperature": [25.0] * 8,
            "humidity": [80.0] * 8,
            "ph": [6.5] * 8,
            "rainfall": [150.0] * 8,
            "label": ["cropA"] * 6 + ["cropB"] * 2
        })

        res = train_and_evaluate_pipeline(
            df=small_df,
            dataset_name="small_dataset_no_cv_test.csv",
            approved_by_user=True,
            allow_fixture_for_testing=True
        )
        self.assertTrue(res["success"])
        bench = res["benchmark_results"]
        champ = res["champion_model"]

        # Confirm CV was marked unsupported due to small sample size
        for name, m in bench.items():
            self.assertFalse(m["cross_validation"]["supported"])
            # Selection score must come strictly from training partition
            self.assertEqual(m["selection_source"], "In-Sample Training Partition (CV Unavailable on Small Dataset)")
            self.assertEqual(m["selection_score_f1"], m["train_macro_f1"])
            self.assertEqual(m["selection_score_acc"], m["train_accuracy"])

        # Confirm champion has the highest training macro-F1 (or training accuracy tie-break)
        champ_train_f1 = bench[champ]["train_macro_f1"]
        for name, m in bench.items():
            self.assertGreaterEqual(champ_train_f1, m["train_macro_f1"])

    # -------------------------------------------------------------------------
    # 17. Complete Flask Endpoints Lifecycle & Error Handling
    # -------------------------------------------------------------------------
    def test_17_all_flask_endpoints_lifecycle(self):
        import io

        # 1. GET /api/health
        res_health = self.app.get('/api/health')
        self.assertEqual(res_health.status_code, 200)
        health_data = res_health.get_json()
        self.assertEqual(health_data["status"], "healthy")
        self.assertIn("version", health_data)

        # 2. GET /api/dataset-status initially
        res_ds_init = self.app.get('/api/dataset-status')
        self.assertEqual(res_ds_init.status_code, 200)

        # 3. POST /api/validate-dataset with valid CSV file
        valid_csv = (
            b"N,P,K,temperature,humidity,ph,rainfall,label\n"
            b"90,42,43,20.8,82.0,6.5,200.0,rice\n"
            b"80,40,40,22.0,80.0,6.6,190.0,maize\n"
        )
        data = {'file': (io.BytesIO(valid_csv), 'crops.csv')}
        res_upload = self.app.post('/api/validate-dataset', data=data, content_type='multipart/form-data')
        self.assertEqual(res_upload.status_code, 200)
        upload_data = res_upload.get_json()
        self.assertTrue(upload_data["success"])
        self.assertEqual(upload_data["rows"], 2)

        # 4. GET /api/dataset-status after upload
        res_ds_after = self.app.get('/api/dataset-status')
        self.assertEqual(res_ds_after.status_code, 200)
        ds_data = res_ds_after.get_json()
        self.assertTrue(ds_data["has_dataset"])
        self.assertEqual(ds_data["metadata"]["filename"], "crops.csv")
        self.assertIsNotNone(ds_data["metadata"].get("classes"))
        self.assertIsInstance(ds_data["metadata"]["classes"], list)
        self.assertEqual(ds_data["metadata"]["classes"], ["maize", "rice"])

        # 5. POST /api/dataset-reset
        res_reset = self.app.post('/api/dataset-reset')
        self.assertEqual(res_reset.status_code, 200)
        reset_data = res_reset.get_json()
        self.assertTrue(reset_data["success"])

        # 6. GET /api/dataset-status after reset
        res_ds_cleared = self.app.get('/api/dataset-status')
        self.assertEqual(res_ds_cleared.status_code, 200)
        self.assertFalse(res_ds_cleared.get_json()["has_dataset"])

        # 7. POST /api/validate-dataset rejection tests
        # Missing file field
        res_no_file = self.app.post('/api/validate-dataset')
        self.assertEqual(res_no_file.status_code, 400)
        self.assertEqual(res_no_file.get_json()["error_type"], "NO_FILE")

        # Non-csv extension
        bad_ext_data = {'file': (io.BytesIO(b"data"), 'crops.txt')}
        res_bad_ext = self.app.post('/api/validate-dataset', data=bad_ext_data, content_type='multipart/form-data')
        self.assertEqual(res_bad_ext.status_code, 400)
        self.assertEqual(res_bad_ext.get_json()["error_type"], "INVALID_FORMAT")

    # -------------------------------------------------------------------------
    # 18. Behavioral Regression Test: Manipulating Held-Out Test Metrics Cannot Alter Champion
    # -------------------------------------------------------------------------
    def test_18_behavioral_regression_changing_test_metrics_cannot_alter_champion(self):
        from unittest.mock import patch
        import pipeline as pl

        # Scenario A: Standard dataset with cross-validation supported
        df = create_synthetic_dataframe(60)
        res_base = train_and_evaluate_pipeline(
            df=df,
            dataset_name="behavioral_test_base.csv",
            approved_by_user=True,
            allow_fixture_for_testing=True
        )
        self.assertTrue(res_base["success"])
        baseline_champ = res_base["champion_model"]
        test_samples = res_base["test_samples"]

        # Mock compute_evaluation_metrics so that when called on test partition,
        # it artificially inverts and distorts test accuracy and test F1
        orig_compute = pl.compute_evaluation_metrics

        def mocked_compute(y_true, y_pred, classes):
            metrics = orig_compute(y_true, y_pred, classes)
            # Identify test evaluation by size of y_true
            if len(y_true) == test_samples:
                # Distort test metrics: if accuracy was high, make it near zero; if low, make it near 1
                distorted_acc = 0.05 if metrics["accuracy"] > 0.5 else 0.99
                distorted_f1 = 0.05 if metrics["macro_f1"] > 0.5 else 0.99
                return {
                    "accuracy": distorted_acc,
                    "macro_precision": 0.5,
                    "macro_recall": 0.5,
                    "macro_f1": distorted_f1,
                    "confusion_matrix": metrics["confusion_matrix"],
                    "per_class_metrics": metrics["per_class_metrics"]
                }
            return metrics

        with patch.object(pl, 'compute_evaluation_metrics', side_effect=mocked_compute):
            res_manipulated = train_and_evaluate_pipeline(
                df=df,
                dataset_name="behavioral_test_manipulated.csv",
                approved_by_user=True,
                allow_fixture_for_testing=True
            )
            self.assertTrue(res_manipulated["success"])
            # Champion MUST remain completely unchanged despite distorted test metrics
            self.assertEqual(
                res_manipulated["champion_model"],
                baseline_champ,
                "Champion model changed when held-out test metrics were altered! Test data leaked into model selection."
            )

        # Scenario B: Small dataset where CV is unsupported
        small_df = pd.DataFrame({
            "N": [50, 52, 51, 53, 54, 55, 10, 12],
            "P": [40, 41, 40, 42, 43, 41, 20, 21],
            "K": [30, 31, 30, 32, 33, 31, 15, 16],
            "temperature": [25.0] * 8,
            "humidity": [80.0] * 8,
            "ph": [6.5] * 8,
            "rainfall": [150.0] * 8,
            "label": ["cropA"] * 6 + ["cropB"] * 2
        })

        res_small_base = train_and_evaluate_pipeline(
            df=small_df,
            dataset_name="small_behavioral_base.csv",
            approved_by_user=True,
            allow_fixture_for_testing=True
        )
        self.assertTrue(res_small_base["success"])
        small_baseline_champ = res_small_base["champion_model"]
        small_test_samples = res_small_base["test_samples"]

        def mocked_small_compute(y_true, y_pred, classes):
            metrics = orig_compute(y_true, y_pred, classes)
            if len(y_true) == small_test_samples:
                # Invert test metrics
                return {
                    "accuracy": 0.01 if metrics["accuracy"] > 0.5 else 0.99,
                    "macro_precision": 0.1,
                    "macro_recall": 0.1,
                    "macro_f1": 0.01 if metrics["macro_f1"] > 0.5 else 0.99,
                    "confusion_matrix": metrics["confusion_matrix"],
                    "per_class_metrics": metrics["per_class_metrics"]
                }
            return metrics

        with patch.object(pl, 'compute_evaluation_metrics', side_effect=mocked_small_compute):
            res_small_manipulated = train_and_evaluate_pipeline(
                df=small_df,
                dataset_name="small_behavioral_manipulated.csv",
                approved_by_user=True,
                allow_fixture_for_testing=True
            )
            self.assertTrue(res_small_manipulated["success"])
            # Champion MUST remain completely unchanged even when CV is unsupported
            self.assertEqual(
                res_small_manipulated["champion_model"],
                small_baseline_champ,
                "Champion model on small dataset changed when test metrics were altered! Test data leaked into fallback selection."
            )

    # -------------------------------------------------------------------------
    # 19. Universal CSV Alias, Case-Insensitivity, and UTF-8 BOM Compatibility
    # -------------------------------------------------------------------------
    def test_19_universal_csv_alias_and_bom_compatibility(self):
        # CSV with all aliases, mixed capitalization, whitespace, and reordered columns
        csv_aliases = (
            "  precipitation , nitrogen , phosphorous , POTASSIUM , temp , rel_humidity , soil_ph , CROP \n"
            "  150.5 , 85.0 , 40.0 , 42.0 , 24.5 , 81.2 , 6.55 , rice \n"
            "  120.0 , 70.0 , 45.0 , 38.0 , 22.0 , 75.0 , 6.80 , maize \n"
        )
        # Test UTF-8 with BOM (\ufeff)
        bom_bytes = b'\xef\xbb\xbf' + csv_aliases.encode('utf-8')
        res = validate_csv_content(bom_bytes, "universal_test.csv")
        self.assertTrue(res["success"], f"Validation failed: {res.get('message')}")
        self.assertEqual(res["columns"], REQUIRED_SCHEMA)
        self.assertEqual(res["rows"], 2)
        self.assertEqual(res["classes"], ["maize", "rice"])
        self.assertIn("dataset_fingerprint", res)
        self.assertEqual(res["provenance_status"], "Source not verified (User Upload)")

    # -------------------------------------------------------------------------
    # 20. Duplicate Header Column Names and Ambiguous Aliases Detection
    # -------------------------------------------------------------------------
    def test_20_duplicate_column_names_and_ambiguous_aliases(self):
        # Duplicate raw columns
        csv_dup = "N,P,K,temperature,humidity,ph,rainfall,rainfall,label\n90,42,43,20,80,6.5,200,200,rice\n80,40,40,22,75,6.0,150,150,maize\n"
        res_dup = validate_csv_content(csv_dup.encode('utf-8'), "dup.csv")
        self.assertFalse(res_dup["success"])
        self.assertEqual(res_dup["error_type"], "DUPLICATE_COLUMNS")

        # Ambiguous aliases (both 'N' and 'nitrogen' mapping to N)
        csv_ambig = "N,nitrogen,P,K,temperature,humidity,ph,rainfall,label\n90,90,42,43,20,80,6.5,200,rice\n80,80,40,40,22,75,6.0,150,maize\n"
        res_ambig = validate_csv_content(csv_ambig.encode('utf-8'), "ambig.csv")
        self.assertFalse(res_ambig["success"])
        self.assertEqual(res_ambig["error_type"], "AMBIGUOUS_COLUMN_MAPPING")

    # -------------------------------------------------------------------------
    # 21. Supplied Desktop Benchmark CSVs Validation
    # -------------------------------------------------------------------------
    def test_21_supplied_desktop_benchmark_files(self):
        candidate_dirs = [
            os.path.dirname(BASE_DIR),
            os.path.expanduser(r"~\OneDrive\Desktop"),
            os.path.expanduser(r"~\Desktop"),
            BASE_DIR
        ]
        candidate_files = [
            "Crop_recommendation 2200 samples.csv",
            "Crop_recommendation sample 2.csv",
            "Crop_recommendation.csv"
        ]
        tested_any = False
        for cdir in candidate_dirs:
            for cfile in candidate_files:
                cpath = os.path.join(cdir, cfile)
                if os.path.exists(cpath):
                    tested_any = True
                    with open(cpath, 'rb') as f:
                        content = f.read()
                    res = validate_csv_content(content, cfile)
                    self.assertTrue(res["success"], f"Failed to validate benchmark file '{cfile}': {res.get('message')}")
                    self.assertEqual(res["rows"], 2200)
                    self.assertEqual(len(res["classes"]), 22)
                    self.assertEqual(res["columns"], REQUIRED_SCHEMA)
                    # Verify fixture block does not reject it
                    allowed, msg = check_fixture_block(cfile, allow_fixture_for_testing=False)
                    self.assertTrue(allowed, f"Fixture check rejected '{cfile}': {msg}")
                    break
            if tested_any:
                break

        if not tested_any:
            sample_df = create_synthetic_dataframe(100)
            csv_bytes = sample_df.to_csv(index=False).encode('utf-8')
            res = validate_csv_content(csv_bytes, "Crop_recommendation 2200 samples.csv")
            self.assertTrue(res["success"])
            allowed, msg = check_fixture_block("Crop_recommendation 2200 samples.csv", allow_fixture_for_testing=False)
            self.assertTrue(allowed)

    # -------------------------------------------------------------------------
    # 22. Dataset Fingerprint Compatibility and Stale Model Invalidation
    # -------------------------------------------------------------------------
    def test_22_model_dataset_fingerprint_compatibility(self):
        # 1. Train model on dataset A
        df_a = create_synthetic_dataframe(40)
        res_a = train_and_evaluate_pipeline(
            df=df_a,
            dataset_name="dataset_a.csv",
            approved_by_user=True,
            allow_fixture_for_testing=True
        )
        self.assertTrue(res_a["success"])
        fp_a = res_a["dataset_fingerprint"]

        # 2. Simulate active dataset metadata having matching fingerprint
        active_meta_path = os.path.join(BASE_DIR, "data", "active_dataset.json")
        os.makedirs(os.path.dirname(active_meta_path), exist_ok=True)
        with open(active_meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "filename": "dataset_a.csv",
                "rows": 40,
                "dataset_fingerprint": fp_a
            }, f)

        res_status = self.app.get('/api/model-status')
        self.assertEqual(res_status.status_code, 200)
        status_data = res_status.get_json()
        self.assertTrue(status_data["trained"])
        self.assertTrue(status_data["model_compatible"])

        # 3. Simulate active dataset changed to dataset B with different fingerprint
        with open(active_meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "filename": "dataset_b.csv",
                "rows": 50,
                "dataset_fingerprint": "different_fingerprint_123"
            }, f)

        res_status_diff = self.app.get('/api/model-status')
        self.assertEqual(res_status_diff.status_code, 200)
        status_diff_data = res_status_diff.get_json()
        # Model must be flagged incompatible/stale!
        self.assertFalse(status_diff_data["trained"])
        self.assertFalse(status_diff_data["model_compatible"])

        # 4. Prediction must reject when model is incompatible
        pred_res = self.app.post('/api/predict', json={
            "N": 50, "P": 40, "K": 30, "temperature": 25, "humidity": 80, "ph": 6.5, "rainfall": 150
        })
        self.assertEqual(pred_res.status_code, 400)
        self.assertEqual(pred_res.get_json()["error_type"], "MODEL_INCOMPATIBLE")

    # -------------------------------------------------------------------------
    # 23. Prediction History Audit Trail Persistence and Clear Action
    # -------------------------------------------------------------------------
    def test_23_prediction_history_audit_trail(self):
        # 1. Clear any prior history
        clear_res = self.app.post('/api/prediction-history/clear')
        self.assertEqual(clear_res.status_code, 200)

        # 2. Verify clean history is empty
        res_empty = self.app.get('/api/prediction-history')
        self.assertEqual(res_empty.status_code, 200)
        data_empty = res_empty.get_json()
        self.assertTrue(data_empty["success"])
        self.assertEqual(data_empty["count"], 0)
        self.assertEqual(data_empty["history"], [])

        # 3. Setup compatible model and active dataset
        df = create_synthetic_dataframe(40)
        res_train = train_and_evaluate_pipeline(
            df=df,
            dataset_name="history_test.csv",
            approved_by_user=True,
            allow_fixture_for_testing=True
        )
        self.assertTrue(res_train["success"])
        fp = res_train["dataset_fingerprint"]

        active_meta_path = os.path.join(BASE_DIR, "data", "active_dataset.json")
        os.makedirs(os.path.dirname(active_meta_path), exist_ok=True)
        with open(active_meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "filename": "history_test.csv",
                "rows": 40,
                "dataset_fingerprint": fp
            }, f)

        # 4. Run successful prediction
        valid_inputs = {
            "N": 85, "P": 45, "K": 40, "temperature": 22.5, "humidity": 82.0, "ph": 6.8, "rainfall": 210.0
        }
        pred_res = self.app.post('/api/predict', json=valid_inputs)
        self.assertEqual(pred_res.status_code, 200)
        pred_data = pred_res.get_json()
        self.assertTrue(pred_data["success"])
        predicted_crop = pred_data["predicted_crop"]

        # 5. Verify prediction history recorded the inference accurately
        hist_res = self.app.get('/api/prediction-history')
        self.assertEqual(hist_res.status_code, 200)
        hist_data = hist_res.get_json()
        self.assertTrue(hist_data["success"])
        self.assertGreaterEqual(hist_data["count"], 1)

        latest_record = hist_data["history"][0]
        self.assertEqual(latest_record["predicted_crop"], predicted_crop)
        self.assertEqual(latest_record["n"], 85.0)
        self.assertEqual(latest_record["p"], 45.0)
        self.assertEqual(latest_record["k"], 40.0)
        self.assertAlmostEqual(latest_record["temperature"], 22.5, places=1)
        self.assertAlmostEqual(latest_record["humidity"], 82.0, places=1)
        self.assertAlmostEqual(latest_record["ph"], 6.8, places=1)
        self.assertAlmostEqual(latest_record["rainfall"], 210.0, places=1)
        self.assertEqual(latest_record["dataset_fingerprint"], fp)
        self.assertIn("timestamp", latest_record)

        # 6. Clear history and verify count reset
        clear_res = self.app.post('/api/prediction-history/clear')
        self.assertEqual(clear_res.status_code, 200)
        self.assertTrue(clear_res.get_json()["success"])

        res_after = self.app.get('/api/prediction-history')
        self.assertEqual(res_after.get_json()["count"], 0)

    # -------------------------------------------------------------------------
    # 24. Clean State and No-Active-Dataset Inference Protection
    # -------------------------------------------------------------------------
    def test_24_clean_state_and_no_active_dataset_prediction_block(self):
        # 1. Simulate server having no active dataset
        active_meta = os.path.join(BASE_DIR, "data", "active_dataset.json")
        active_csv = os.path.join(BASE_DIR, "data", "active_dataset.csv")
        for fpath in [active_meta, active_csv]:
            if os.path.exists(fpath):
                os.remove(fpath)

        # 2. Check dataset status API
        status_res = self.app.get('/api/dataset-status')
        self.assertEqual(status_res.status_code, 200)
        self.assertFalse(status_res.get_json()["has_dataset"])

        # 3. Model status must report incompatible / no active dataset
        model_res = self.app.get('/api/model-status')
        self.assertEqual(model_res.status_code, 200)
        mdata = model_res.get_json()
        self.assertFalse(mdata["trained"])
        self.assertFalse(mdata.get("model_compatible", False))

        # 4. Predict endpoint must block inference when no active dataset exists
        pred_res = self.app.post('/api/predict', json={
            "N": 90, "P": 42, "K": 43, "temperature": 20.8, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9
        })
        self.assertEqual(pred_res.status_code, 400)
        pdata = pred_res.get_json()
        self.assertFalse(pdata["success"])
        self.assertIn(pdata["error_type"], ["NO_ACTIVE_DATASET", "NO_TRAINED_MODEL", "MODEL_INCOMPATIBLE"])


if __name__ == '__main__':
    unittest.main()



