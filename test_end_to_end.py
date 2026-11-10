import unittest
import pandas as pd
from streamlit_integration import build_ai_panel

class EndToEnd(unittest.TestCase):
    def test_stale_data_never_shows_validated(self):
        t = pd.date_range("2020-01-01", periods=110, freq="min", tz="UTC")
        html = build_ai_panel(pd.DataFrame({"time": t, "close": [100.0]*110}))
        self.assertIn("SIN DATOS", html)
        self.assertNotIn("VALIDADO", html)
    def test_valid_fresh_data_stays_unvalidated(self):
        t = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("min"), periods=110, freq="min")
        html = build_ai_panel(pd.DataFrame({"time": t, "close": [100.0]*110}))
        self.assertIn("EN EVALUACIÓN", html)
        self.assertNotIn("nan%", html)
    def test_missing_data_stays_safe(self):
        html = build_ai_panel(None)
        self.assertEqual(html.count("ESPERAR"), 2)

if __name__ == "__main__":
    unittest.main()
