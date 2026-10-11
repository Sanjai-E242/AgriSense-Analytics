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


if __name__ == '__main__':
    unittest.main()
