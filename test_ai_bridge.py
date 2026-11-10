import unittest
import pandas as pd
from ai_bridge import get_ai_status

class TestBridge(unittest.TestCase):
    def test_missing_data(self):
        result = get_ai_status(None)
        self.assertEqual(result['signal'], 'ESPERAR')
        self.assertEqual(result['status'], 'SIN DATOS')

    def test_insufficient_data(self):
        result = get_ai_status(pd.DataFrame({'close':[100.0]*5}))
        self.assertEqual(result['signal'], 'ESPERAR')

    def test_valid_prices_still_unvalidated(self):
        result = get_ai_status(pd.DataFrame({'close':[100.0+i for i in range(120)]}))
        self.assertEqual(result['status'], 'EN EVALUACIÓN')
        self.assertEqual(result['signal'], 'ESPERAR')

    def test_unsupported_horizon(self):
        result = get_ai_status(pd.DataFrame({'close':[100.0]*120}), 3)
        self.assertEqual(result['status'], 'SIN DATOS')

if __name__ == '__main__':
    unittest.main()


class FreshnessIntegrationTests(unittest.TestCase):
    def test_stale_timestamp_suspends_signal(self):
        import pandas as pd
        from ai_bridge import get_ai_status
        t = pd.date_range('2020-01-01', periods=110, freq='min', tz='UTC')
        result = get_ai_status(pd.DataFrame({'time': t, 'close': [100.0]*110}), 15)
        self.assertEqual(result['signal'], 'ESPERAR')
        self.assertEqual(result['status'], 'SIN DATOS')

    def test_future_timestamp_suspends_signal(self):
        import pandas as pd
        from ai_bridge import get_ai_status
        t = pd.date_range(pd.Timestamp.now(tz='UTC') + pd.Timedelta(hours=2), periods=110, freq='min')
        result = get_ai_status(pd.DataFrame({'time': t, 'close': [100.0]*110}), 10)
        self.assertEqual(result['status'], 'SIN DATOS')
