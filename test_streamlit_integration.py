import unittest
import pandas as pd
from streamlit_integration import build_ai_panel

class IntegrationTests(unittest.TestCase):
    def test_missing_data_does_not_crash(self):
        html = build_ai_panel(pd.DataFrame())
        self.assertIn('IA 10M', html)
        self.assertIn('IA 15M', html)
        self.assertIn('ESPERAR', html)

    def test_no_fake_accuracy(self):
        html = build_ai_panel(pd.DataFrame({'close': [100.0] * 110}))
        self.assertNotIn('nan%', html)
        self.assertIn('EN EVALUACIÓN', html)

    def test_escapes_text(self):
        html = build_ai_panel(pd.DataFrame({'close': [100.0] * 110}))
        self.assertNotIn('<script>', html)
