"""
Automated unit tests for AgriSense Analytics Frontend.
Verifies static asset routing, HTML landmarks, CSS design tokens, and JS modules.
"""

import os
import unittest
import server
from server import BASE_DIR, FRONTEND_DIR


class TestFrontend(unittest.TestCase):

    def setUp(self):
        self.app = server.app.test_client()
        self.app.testing = True

    def test_root_route_serves_index_html(self):
        """Root GET / returns HTML content with 200 OK."""
        res = self.app.get('/')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"AgriSense Analytics", res.data)
        self.assertIn(b"Skip to main content", res.data)

    def test_static_css_assets_served(self):
        """All required CSS stylesheets return HTTP 200."""
        css_files = ["tokens.css", "base.css", "layout.css", "components.css", "responsive.css"]
        for css in css_files:
            res = self.app.get(f'/css/{css}')
            self.assertEqual(res.status_code, 200, f"Failed serving /css/{css}")
            self.assertGreater(len(res.data), 0)

    def test_static_js_assets_served(self):
        """All required JavaScript modules return HTTP 200."""
        js_files = [
            "icons.js", "navigation.js", "prediction-form.js",
            "dataset-parser.js", "model-evaluation.js",
            "prediction-history.js", "app.js"
        ]
        for js in js_files:
            res = self.app.get(f'/js/{js}')
            self.assertEqual(res.status_code, 200, f"Failed serving /js/{js}")
            self.assertGreater(len(res.data), 0)

    def test_design_tokens_present(self):
        """CSS design tokens include the 4 core approved palette colors."""
        tokens_path = os.path.join(FRONTEND_DIR, "css", "tokens.css") if os.path.exists(os.path.join(FRONTEND_DIR, "css", "tokens.css")) else os.path.join(BASE_DIR, "css", "tokens.css")
        with open(tokens_path, "r", encoding="utf-8") as f:
            tokens_css = f.read().upper()

        self.assertIn("#F4F1EB", tokens_css)  # Alabaster
        self.assertIn("#7AA05A", tokens_css)  # Moss Green
        self.assertIn("#2D5A4A", tokens_css)  # Slate Green
        self.assertIn("#0D4C3C", tokens_css)  # Dark Emerald

    def test_navigation_panels_present_in_html(self):
        """All 6 core navigation view panels are defined in HTML."""
        index_path = os.path.join(FRONTEND_DIR, "index.html") if os.path.exists(os.path.join(FRONTEND_DIR, "index.html")) else os.path.join(BASE_DIR, "index.html")
        with open(index_path, "r", encoding="utf-8") as f:
            html = f.read()

        for panel in ["dashboard", "prediction", "dataset", "evaluation", "history", "settings"]:
            self.assertIn(f'data-view-panel="{panel}"', html)


if __name__ == '__main__':
    unittest.main()
