"""
Automated unit tests for AgriSense Analytics Dataset Validator.
Verifies canonical schema validation, alias normalization, encoding, and data quality checks.
"""

import unittest
import pandas as pd
from validator import validate_csv_content


class TestDatasetValidator(unittest.TestCase):

    def test_valid_canonical_schema(self):
        """Accepts a canonical 8-column CSV with valid values."""
        df = pd.DataFrame({
            "N": [90.0, 85.0, 60.0, 74.0],
            "P": [42.0, 58.0, 55.0, 35.0],
            "K": [43.0, 41.0, 44.0, 40.0],
            "temperature": [20.87, 21.77, 23.00, 26.49],
            "humidity": [82.00, 80.31, 82.32, 80.15],
            "ph": [6.50, 7.03, 7.84, 6.98],
            "rainfall": [202.93, 226.65, 263.96, 242.86],
            "label": ["rice", "rice", "maize", "maize"]
        })
        res = validate_csv_content(df.to_csv(index=False).encode('utf-8'), "canonical.csv")
        self.assertTrue(res["success"])
        self.assertEqual(res["rows"], 4)
        self.assertEqual(len(res["classes"]), 2)
        self.assertIsNotNone(res["dataset_fingerprint"])

    def test_alias_normalization_and_utf8_bom(self):
        """Normalizes aliases and strips UTF-8 BOM encoding."""
        csv_bom = (
            "\ufeffprecipitation,soil_nitrogen,phosphorous,soil_potassium,ambient_temperature,rel_humidity,ph_level,crop\n"
            "200.5,90,42,43,21.5,82.0,6.5,rice\n"
            "180.2,80,40,40,22.0,80.0,6.7,maize\n"
        )
        res = validate_csv_content(csv_bom.encode('utf-8-sig'), "bom_aliases.csv")
        self.assertTrue(res["success"])
        self.assertEqual(res["rows"], 2)
        self.assertIn("rice", res["classes"])
        self.assertIn("maize", res["classes"])

    def test_missing_required_column(self):
        """Rejects datasets missing any required feature column."""
        csv_missing = "N,P,temperature,humidity,ph,rainfall,label\n90,42,20.8,82.0,6.5,200.0,rice\n"
        res = validate_csv_content(csv_missing.encode('utf-8'), "missing_k.csv")
        self.assertFalse(res["success"])
        self.assertEqual(res["error_type"], "MISSING_REQUIRED_COLUMNS")

    def test_non_numeric_and_infinite_values(self):
        """Rejects non-numeric text and infinite feature values."""
        csv_text = "N,P,K,temperature,humidity,ph,rainfall,label\n90,INVALID,43,20.8,82.0,6.5,200.0,rice\n"
        res_text = validate_csv_content(csv_text.encode('utf-8'), "text.csv")
        self.assertFalse(res_text["success"])

        csv_inf = "N,P,K,temperature,humidity,ph,rainfall,label\n90,inf,43,20.8,82.0,6.5,200.0,rice\n"
        res_inf = validate_csv_content(csv_inf.encode('utf-8'), "inf.csv")
        self.assertFalse(res_inf["success"])

    def test_blank_target_label(self):
        """Rejects empty or whitespace-only target labels."""
        csv_blank = "N,P,K,temperature,humidity,ph,rainfall,label\n90,42,43,20.8,82.0,6.5,200.0,   \n"
        res = validate_csv_content(csv_blank.encode('utf-8'), "blank.csv")
        self.assertFalse(res["success"])

    def test_duplicate_headers_and_ambiguous_aliases(self):
        """Detects duplicate column names and ambiguous alias mappings."""
        csv_dup = "N,N,K,temperature,humidity,ph,rainfall,label\n90,42,43,20.8,82.0,6.5,200.0,rice\n"
        res_dup = validate_csv_content(csv_dup.encode('utf-8'), "dup.csv")
        self.assertFalse(res_dup["success"])

        csv_ambig = "N,soil_nitrogen,P,K,temperature,humidity,ph,rainfall,label\n90,90,42,43,20.8,82.0,6.5,200.0,rice\n"
        res_ambig = validate_csv_content(csv_ambig.encode('utf-8'), "ambig.csv")
        self.assertFalse(res_ambig["success"])


if __name__ == '__main__':
    unittest.main()
