"""
Automated unit tests for AgriSense Analytics Flask API.
Verifies health, dataset status, upload validation, training authorization,
model status, prediction, SQLite audit history, and reset lifecycle.
"""

import os
import json
import io
import unittest
import numpy as np
import pandas as pd
import server
from server import BASE_DIR, DATA_DIR, MODELS_DIR


class TestAgriSenseAPI(unittest.TestCase):

    def setUp(self):
        self.app = server.app.test_client()
        self.app.testing = True

    def tearDown(self):
        # Reset runtime state to preserve isolation
        self.app.post('/api/dataset-reset')

    def test_health_endpoint(self):
        """Health endpoint returns status healthy and HTTP 200."""
        res = self.app.get('/api/health')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "healthy")

    def test_clean_startup_status(self):
        """Initial state shows no active dataset and no trained model."""
        self.app.post('/api/dataset-reset')
        d_res = self.app.get('/api/dataset-status')
        self.assertEqual(d_res.status_code, 200)
        self.assertFalse(d_res.get_json().get("has_dataset"))

        m_res = self.app.get('/api/model-status')
        self.assertEqual(m_res.status_code, 200)
        self.assertFalse(m_res.get_json().get("trained"))

    def test_training_fails_without_user_approval(self):
        """Training requests without explicit confirmation return HTTP 400 APPROVAL_REQUIRED."""
        res = self.app.post('/api/train-models', json={
            "approved_by_user": False,
            "dataset_name": "crops.csv"
        })
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.get_json().get("error_type"), "APPROVAL_REQUIRED")

    def test_prediction_blocked_without_compatible_model(self):
        """Prediction without an active model returns HTTP 400."""
        self.app.post('/api/dataset-reset')
        res = self.app.post('/api/predict', json={
            "N": 90, "P": 42, "K": 43, "temperature": 20.87, "humidity": 82.00, "ph": 6.50, "rainfall": 202.93
        })
        self.assertEqual(res.status_code, 400)

    def test_prediction_history_audit_trail_and_clear(self):
        """Inference audit records persist in SQLite and clear on request."""
        # Clean history first
        self.app.post('/api/prediction-history/clear')
        h_init = self.app.get('/api/prediction-history').get_json()
        self.assertEqual(h_init.get("count"), 0)

        # Directly insert a test inference record
        server.record_prediction_history(
            inputs={"N": 90, "P": 42, "K": 43, "temperature": 20.87, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9},
            predicted_crop="rice",
            champion_model="Random Forest",
            dataset_name="Crop_recommendation 2200 samples.csv",
            dataset_fingerprint="7fdf53ec131d6875"
        )

        h_after = self.app.get('/api/prediction-history').get_json()
        self.assertEqual(h_after.get("count"), 1)
        rec = h_after["history"][0]
        self.assertEqual(rec["predicted_crop"], "rice")
        self.assertEqual(rec["champion_model"], "Random Forest")

        # Clear history
        c_res = self.app.post('/api/prediction-history/clear')
        self.assertEqual(c_res.status_code, 200)
        h_cleared = self.app.get('/api/prediction-history').get_json()
        self.assertEqual(h_cleared.get("count"), 0)

    def test_training_fails_missing_dataset_returns_json_400(self):
        """Training with approval but without dataset content returns useful JSON 400."""
        self.app.post('/api/dataset-reset')
        res = self.app.post('/api/train-models', json={
            "approved_by_user": True,
            "dataset_name": "missing.csv",
            "csv_text": ""
        })
        self.assertEqual(res.status_code, 400)
        self.assertIn("application/json", res.headers.get("content-type", ""))
        data = res.get_json()
        self.assertFalse(data.get("success"))
        self.assertEqual(data.get("error_type"), "NO_DATASET_CONTENT")

    def test_api_404_returns_json_not_html(self):
        """Unrecognized /api/* endpoints return JSON 404 rather than HTML error pages."""
        res = self.app.get('/api/nonexistent-endpoint')
        self.assertEqual(res.status_code, 404)
        self.assertIn("application/json", res.headers.get("content-type", ""))
        data = res.get_json()
        self.assertFalse(data.get("success"))
        self.assertEqual(data.get("error_type"), "NOT_FOUND")

    def test_api_405_method_not_allowed_returns_json(self):
        """Disallowed HTTP methods on API routes return JSON 405 rather than HTML error pages."""
        res = self.app.get('/api/train-models')
        self.assertEqual(res.status_code, 405)
        self.assertIn("application/json", res.headers.get("content-type", ""))
        data = res.get_json()
        self.assertFalse(data.get("success"))
        self.assertEqual(data.get("error_type"), "METHOD_NOT_ALLOWED")

    def test_training_successful_small_dataset(self):
        """Valid dataset training succeeds and persists matching active metadata and model."""
        self.app.post('/api/dataset-reset')
        # Create a valid synthetic dataset of 50 samples with 2 classes
        lines = ["N,P,K,temperature,humidity,ph,rainfall,label"]
        for i in range(25):
            lines.append(f"{50+i},{40+i},{30+i},25.0,80.0,6.5,150.0,crop_a")
        for i in range(25):
            lines.append(f"{80+i},{20+i},{10+i},30.0,60.0,7.0,100.0,crop_b")
        csv_content = "\n".join(lines)

        res = self.app.post('/api/train-models', json={
            "approved_by_user": True,
            "dataset_name": "test_crop_dataset.csv",
            "csv_text": csv_content
        })
        self.assertEqual(res.status_code, 200)
        self.assertIn("application/json", res.headers.get("content-type", ""))
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertIn("champion_model", data)
        self.assertIn("benchmark_results", data)

        # Model status should now reflect trained model
        m_res = self.app.get('/api/model-status')
        self.assertEqual(m_res.status_code, 200)
        m_data = m_res.get_json()
        self.assertTrue(m_data.get("trained"))
        self.assertTrue(m_data.get("artifact_valid"))


if __name__ == '__main__':
    unittest.main()
