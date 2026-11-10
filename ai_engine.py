"""YMC BTC AI Pro: conservative experimental signal gate, no automated trades.

This module does not claim to be a trained or validated predictor.
"""
from datetime import datetime, timezone
import numpy as np
import pandas as pd


def estimate(df, horizon=15):
    """Validate chronological one-minute candles and return a safe signal contract.

    Accepts a time column or DatetimeIndex when available. If timestamps are
    absent, freshness cannot be verified and the result stays unvalidated.
    """
    if horizon not in (10, 15):
        raise ValueError('Unsupported horizon')
    if not isinstance(df, pd.DataFrame) or 'close' not in df:
        raise ValueError('Missing close prices')
    close = pd.to_numeric(df['close'], errors='coerce')
    if len(close) < 100 or not np.isfinite(close.to_numpy()).all() or (close <= 0).any():
        raise ValueError('Insufficient or invalid candles')

    reason = 'Modelo pendiente de validación temporal fuera de muestra'
    timestamps = None
    for key in ('time', 'timestamp', 'date', 'datetime'):
        if key in df.columns:
            values = df[key]
            if pd.api.types.is_numeric_dtype(values):
                unit = 'ms' if values.dropna().median() > 1e11 else 's'
                timestamps = pd.to_datetime(values, unit=unit, utc=True, errors='coerce')
            else:
                timestamps = pd.to_datetime(values, utc=True, errors='coerce')
            break
    if timestamps is None and isinstance(df.index, pd.DatetimeIndex):
        timestamps = pd.to_datetime(df.index, utc=True)
    if timestamps is not None:
        ts = pd.DatetimeIndex(timestamps)
        if ts.isna().any() or not ts.is_monotonic_increasing or ts.has_duplicates:
            raise ValueError('Invalid or unordered candle timestamps')
        gaps = pd.Series(ts).diff().dropna().dt.total_seconds()
        if (gaps > 90).any() or (gaps < 30).any():
            raise ValueError('Missing or irregular 1-minute candles')
        age = (datetime.now(timezone.utc) - ts[-1].to_pydatetime()).total_seconds()
        if age < -90 or age > 180:
            raise ValueError('Datos desactualizados o reloj desincronizado')
    else:
        reason = 'Sin marcas de tiempo verificables; modelo sin validar'

    return {
        'signal': 'ESPERAR', 'validated': False,
        'validation_accuracy': float('nan'), 'validation_samples': 0,
        'horizon_minutes': horizon, 'reason': reason,
    }
