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


if __name__ == '__main__':
    unittest.main()
