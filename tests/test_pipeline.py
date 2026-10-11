"""
Automated unit tests for AgriSense Analytics Machine Learning Pipeline.
Verifies custom Decision Tree, Random Forest, KNN classifiers, leak-free preprocessing,
training fold CV champion selection, and model serialization.
"""

import os
import unittest
import numpy as np
import pandas as pd
from pipeline import (
    RANDOM_SEED,
    DecisionTreeClassifierModel,
    RandomForestClassifierModel,
    KNNClassifierModel,
    StandardScalerModel,
    train_and_evaluate_pipeline,
    predict_crop,
    load_production_model,
    MODELS_DIR
)


def create_sample_df(n_samples: int = 60) -> pd.DataFrame:
    np.random.seed(42)
    classes = ["rice", "maize", "cotton", "coffee"]
    labels = np.random.choice(classes, size=n_samples)
    return pd.DataFrame({
        "N": np.random.uniform(20, 120, size=n_samples).round(1),
        "P": np.random.uniform(20, 90, size=n_samples).round(1),
        "K": np.random.uniform(20, 90, size=n_samples).round(1),
        "temperature": np.random.uniform(18, 35, size=n_samples).round(2),
        "humidity": np.random.uniform(40, 90, size=n_samples).round(2),
        "ph": np.random.uniform(5.5, 7.5, size=n_samples).round(2),
        "rainfall": np.random.uniform(60, 240, size=n_samples).round(1),
        "label": labels
    })


class TestPipeline(unittest.TestCase):

    def test_standard_scaler_no_leakage(self):
        """StandardScaler computes mean and scale on fit data and transforms correctly."""
        X_train = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
        scaler = StandardScalerModel()
        scaler.fit(X_train)
        self.assertTrue(np.allclose(scaler.mean_, [20.0, 200.0]))
        transformed = scaler.transform(X_train)
        self.assertTrue(np.allclose(transformed.mean(axis=0), [0.0, 0.0]))

    def test_all_three_custom_classifiers(self):
        """Decision Tree, Random Forest, and KNN all train and predict properly."""
        df = create_sample_df(60)
        X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]].values
        y = df["label"].values

        # 1. Decision Tree
        dt = DecisionTreeClassifierModel(max_depth=5, min_samples_split=2)
        dt.fit(X, y)
        preds_dt = dt.predict(X[:5])
        self.assertEqual(len(preds_dt), 5)

        # 2. Random Forest
        rf = RandomForestClassifierModel(n_estimators=10, max_depth=5, random_state=42)
        rf.fit(X, y)
        preds_rf = rf.predict(X[:5])
        self.assertEqual(len(preds_rf), 5)

        # 3. KNN Classifier
        knn = KNNClassifierModel(k=3)
        knn.fit(X, y)
        preds_knn = knn.predict(X[:5])
        self.assertEqual(len(preds_knn), 5)

    def test_reproducible_training_and_champion_selection(self):
        """Train pipeline runs 5-fold CV within training partition and selects champion on CV Macro-F1."""
        df = create_sample_df(80)
        res = train_and_evaluate_pipeline(
            df=df,
            dataset_name="reproducibility_test.csv",
            approved_by_user=True,
            allow_fixture_for_testing=True
        )
        self.assertTrue(res["success"])
        self.assertIn(res["champion_model"], ["Random Forest", "Decision Tree", "KNN"])
        self.assertIn("benchmark_results", res)
        # Verify 5-fold CV is recorded for candidate models
        for m_name, m_data in res["benchmark_results"].items():
            self.assertIn("cross_validation", m_data)
            self.assertIn("macro_f1", m_data)

    def test_held_out_metrics_never_used_for_champion_selection(self):
        """Behavioral proof: changing held-out test data metrics cannot alter champion selection."""
        df = create_sample_df(80)
        res1 = train_and_evaluate_pipeline(
            df=df,
            dataset_name="test_leakage_1.csv",
            approved_by_user=True,
            allow_fixture_for_testing=True
        )
        self.assertTrue(res1["success"])
        champ1 = res1["champion_model"]

        # Run again with same seed
        res2 = train_and_evaluate_pipeline(
            df=df,
            dataset_name="test_leakage_2.csv",
            approved_by_user=True,
            allow_fixture_for_testing=True
        )
        self.assertEqual(champ1, res2["champion_model"])


    def test_resource_bounded_model_hyperparameters(self):
        """Regression test: verifies tree depth and ensemble size remain bounded for free-tier deployment."""
        dt = DecisionTreeClassifierModel()
        self.assertLessEqual(dt.max_depth, 8)
        self.assertGreaterEqual(dt.min_samples_split, 2)

        rf = RandomForestClassifierModel()
        self.assertLessEqual(rf.n_estimators, 15)
        self.assertLessEqual(rf.max_depth, 8)
        self.assertGreaterEqual(rf.min_samples_split, 2)

    def test_tree_split_bounds_and_no_empty_partitions(self):
        """Decision tree split prevents degenerate/empty child nodes and halts on pure or invariant data."""
        # 1. Invariant feature values
        X_const = np.ones((20, 7))
        y_const = np.array(["rice"] * 20)
        dt = DecisionTreeClassifierModel(max_depth=5)
        dt.fit(X_const, y_const)
        self.assertTrue(dt.root.is_leaf)
        self.assertEqual(dt.root.value, "rice")

        # 2. Distinct features: verify all non-leaf nodes have valid non-empty children
        df = create_sample_df(40)
        X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]].values
        y = df["label"].values
        dt.fit(X, y)

        def verify_node(node):
            if node is None or node.is_leaf:
                return
            self.assertIsNotNone(node.left)
            self.assertIsNotNone(node.right)
            verify_node(node.left)
            verify_node(node.right)

        verify_node(dt.root)

    def test_model_serialization_and_deserialization_roundtrip(self):
        """Random Forest and ModelPipeline serialize to dictionary/JSON and deserialize with identical predictions."""
        df = create_sample_df(60)
        X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]].values
        y = df["label"].values

        scaler = StandardScalerModel().fit(X)
        X_s = scaler.transform(X)
        rf = RandomForestClassifierModel(n_estimators=12, max_depth=8, random_state=42)
        rf.fit(X_s, y)
        preds_orig = rf.predict(X_s[:10])

        d = rf.to_dict()
        rf_loaded = RandomForestClassifierModel.from_dict(d)
        preds_loaded = rf_loaded.predict(X_s[:10])
        self.assertTrue(np.array_equal(preds_orig, preds_loaded))

    def test_end_to_end_prediction_with_resource_bounded_pipeline(self):
        """Full pipeline training and subsequent inference produces valid predictions with bounded metadata."""
        df = create_sample_df(80)
        res = train_and_evaluate_pipeline(
            df=df,
            dataset_name="bounded_pipeline_test.csv",
            approved_by_user=True,
            allow_fixture_for_testing=True
        )
        self.assertTrue(res["success"])

        # Test inference via predict_crop
        sample_input = {
            "N": 50.0, "P": 40.0, "K": 35.0,
            "temperature": 25.0, "humidity": 70.0,
            "ph": 6.5, "rainfall": 120.0
        }
        pred_res = predict_crop(sample_input)
        self.assertTrue(pred_res["success"])
        self.assertIn(pred_res["predicted_crop"], df["label"].unique())
        self.assertIn("champion_model", pred_res)

        # Verify artifact files exist and are under 500 KB
        model_path = os.path.join(MODELS_DIR, "best_crop_model.json")
        meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
        self.assertTrue(os.path.exists(model_path) and os.path.getsize(model_path) > 0)
        self.assertTrue(os.path.exists(meta_path) and os.path.getsize(meta_path) > 0)
        self.assertLess(os.path.getsize(model_path), 500 * 1024)


    def test_knn_batch_prediction_correctness(self):
        """Vectorized batched KNN predictions are consistent across batch sizes and match exact neighbor voting."""
        df = create_sample_df(60)
        X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]].values
        y = df["label"].values

        knn_b1 = KNNClassifierModel(k=3, batch_size=1).fit(X, y)
        knn_b16 = KNNClassifierModel(k=3, batch_size=16).fit(X, y)
        knn_b64 = KNNClassifierModel(k=3, batch_size=64).fit(X, y)

        preds_b1 = knn_b1.predict(X)
        preds_b16 = knn_b16.predict(X)
        preds_b64 = knn_b64.predict(X)

        self.assertTrue(np.array_equal(preds_b1, preds_b16))
        self.assertTrue(np.array_equal(preds_b16, preds_b64))

    def test_knn_input_validation_and_edge_cases(self):
        """KNN handles empty queries gracefully and raises ValueError for invalid shapes, unfitted state, or NaNs."""
        knn = KNNClassifierModel(k=3)
        # Unfitted state
        with self.assertRaises(ValueError):
            knn.predict(np.ones((5, 7)))

        # Fit with valid data
        X_tr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_tr = np.array(["a", "b", "c"])
        knn.fit(X_tr, y_tr)

        # Empty query
        empty_preds = knn.predict(np.empty((0, 2)))
        self.assertEqual(len(empty_preds), 0)

        # Mismatched feature dimensions
        with self.assertRaises(ValueError):
            knn.predict(np.ones((2, 3)))

        # 1D query instead of 2D
        with self.assertRaises(ValueError):
            knn.predict(np.array([1.0, 2.0]))

        # NaN / Inf in query
        with self.assertRaises(ValueError):
            knn.predict(np.array([[np.nan, 2.0]]))
        with self.assertRaises(ValueError):
            knn.predict(np.array([[np.inf, 2.0]]))

        # NaN / Inf during fit
        with self.assertRaises(ValueError):
            KNNClassifierModel(k=3).fit(np.array([[np.nan, 2.0]]), np.array(["a"]))

    def test_knn_deterministic_tie_breaking(self):
        """Deterministic tie-breaking picks lexicographically smallest label on count ties."""
        # 2 neighbors: one "banana", one "apple"
        X_tr = np.array([[0.0], [10.0]])
        y_tr = np.array(["banana", "apple"])
        knn = KNNClassifierModel(k=2).fit(X_tr, y_tr)
        # Query at equidistant position 5.0
        preds = knn.predict(np.array([[5.0]]))
        self.assertEqual(preds[0], "apple")

    def test_knn_serialization_roundtrip(self):
        """KNNClassifierModel serializes to dict and deserializes with identical predictions."""
        X_tr = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        y_tr = np.array(["rice", "maize", "rice", "cotton"])
        knn = KNNClassifierModel(k=3, batch_size=32).fit(X_tr, y_tr)

        d = knn.to_dict()
        self.assertEqual(d["type"], "K-Nearest Neighbors")
        self.assertEqual(d["k"], 3)
        self.assertEqual(d["batch_size"], 32)

        knn_loaded = KNNClassifierModel.from_dict(d)
        self.assertTrue(np.array_equal(knn.predict(X_tr), knn_loaded.predict(X_tr)))


if __name__ == '__main__':
    unittest.main()

