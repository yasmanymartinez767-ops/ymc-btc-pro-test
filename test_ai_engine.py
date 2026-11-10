import unittest
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from ai_engine import estimate

class TestEngine(unittest.TestCase):
    def candles(self, n=150):
        t = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        return pd.DataFrame({'time': pd.date_range(end=t, periods=n, freq='min'), 'close': np.linspace(50000, 51000, n)})

    def test_safe_default(self):
        for h in (10, 15):
            r = estimate(self.candles(), h)
            self.assertEqual(r['signal'], 'ESPERAR')
            self.assertFalse(r['validated'])
            self.assertEqual(r['validation_samples'], 0)

    def test_bad_data(self):
        with self.assertRaises(ValueError): estimate(pd.DataFrame({'close': [1, 2, 3]}))

    def test_bad_horizon(self):
        with self.assertRaises(ValueError): estimate(self.candles(), 5)

    def test_missing_candle(self):
        df = self.candles().drop(index=42).reset_index(drop=True)
        with self.assertRaises(ValueError): estimate(df)

    def test_reversed_candles(self):
        with self.assertRaises(ValueError): estimate(self.candles().iloc[::-1].reset_index(drop=True))

    def test_invalid_price(self):
        df = self.candles(); df.loc[4, 'close'] = np.nan
        with self.assertRaises(ValueError): estimate(df)

    def test_stale_candles(self):
        df = self.candles(); df['time'] -= timedelta(hours=1)
        with self.assertRaisesRegex(ValueError, 'desactualizados'):
            estimate(df)

    def test_no_timestamps(self):
        r = estimate(self.candles()[['close']])
        self.assertIn('Sin marcas de tiempo', r['reason'])

if __name__ == '__main__': unittest.main()
