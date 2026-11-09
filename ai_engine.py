"""Experimental short-horizon ML model. Never claims validated real-world accuracy."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def estimate(df, horizon=15, now=None):
    if horizon not in (10, 15):
        raise ValueError('Horizonte inválido')
    if now is None:
        import time
        now = time.time()
    frame = df.copy().sort_values('timestamp').drop_duplicates('timestamp')
    frame = frame[frame.timestamp + 60 <= now - 3].copy()
    if len(frame) < 190:
        raise ValueError('Historial de 1m insuficiente para IA')
    if (frame.timestamp.diff().tail(190).dropna() != 60).any():
        raise ValueError('Huecos recientes: entrenamiento suspendido')
    close = frame.close.astype(float)
    volume = frame.volume.astype(float)
    ret = close.pct_change()
    features = pd.DataFrame({
        'ret1': ret, 'ret3': close.pct_change(3), 'ret5': close.pct_change(5),
        'ret10': close.pct_change(10), 'ret15': close.pct_change(15),
        'vol5': ret.rolling(5).std(), 'vol15': ret.rolling(15).std(),
        'range': (frame.high - frame.low) / close,
        'body': (frame.close - frame.open) / close,
        'volume_ratio': volume / volume.rolling(15).mean().replace(0, np.nan),
    }, index=frame.index).replace([np.inf, -np.inf], np.nan)
    # Only labels whose future candles are already known are used.
    future = close.shift(-horizon)
    labels = (future > close).astype(int)
    eligible = features.notna().all(axis=1) & future.notna()
    x = features.loc[eligible]
    y = labels.loc[eligible]
    if len(x) < 140:
        raise ValueError('Muestras etiquetadas insuficientes')
    split = int(len(x) * .70)
    if split < 90 or len(x) - split < 35 or y.iloc[:split].nunique() < 2:
        raise ValueError('No hay diversidad o muestras suficientes')
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=400, C=.3))
    model.fit(x.iloc[:split], y.iloc[:split])
    predicted = model.predict(x.iloc[split:])
    score = float(accuracy_score(y.iloc[split:], predicted))
    baseline = float(max(y.iloc[split:].mean(), 1-y.iloc[split:].mean()))
    latest = features.iloc[[-1]]
    if latest.isna().any().any():
        raise ValueError('Última observación incompleta')
    probability = float(model.predict_proba(latest)[0, 1])
    # This is a small historical holdout, not proof of out-of-sample future success.
    reliable = score >= baseline + .03 and len(x) - split >= 40
    signal = 'ALCISTA' if probability >= .60 else 'BAJISTA' if probability <= .40 else 'ESPERAR'
    if not reliable:
        signal = 'EN EVALUACIÓN'
    return dict(horizon=horizon, signal=signal, probability=probability,
                validation_accuracy=score, baseline=baseline, validation_samples=len(x)-split,
                training_samples=split, validated=reliable,
                last_closed_timestamp=int(frame.timestamp.iloc[-1]))
