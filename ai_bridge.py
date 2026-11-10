"""Safe integration between YMC AI engine and Streamlit presentation."""
from ai_engine import estimate
from ai_display import present_estimate


def get_ai_status(candles, horizon=15):
    """Return UI-safe status even if data or model processing fails."""
    try:
        result = estimate(candles, horizon)
        view = present_estimate(result)
    except (ValueError, TypeError, KeyError, IndexError, OverflowError):
        view = {'signal': 'ESPERAR', 'status': 'SIN DATOS',
                'detail': 'No se pudo verificar la información de BTC'}
    return view
