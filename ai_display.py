"""Safe UI adapter for YMC experimental AI signals."""
import math


def present_estimate(result):
    """Return UI-safe labels; never imply unverified model performance."""
    if not isinstance(result, dict):
        return {'signal': 'ESPERAR', 'status': 'SIN DATOS', 'detail': 'Resultado no disponible'}
    validated = result.get('validated') is True
    signal = result.get('signal')
    accuracy = result.get('validation_accuracy')
    samples = result.get('validation_samples')
    valid_accuracy = isinstance(accuracy, (int, float)) and not isinstance(accuracy, bool) and math.isfinite(accuracy) and 0 <= accuracy <= 1
    valid_samples = isinstance(samples, int) and not isinstance(samples, bool) and samples >= 100
    if validated and signal in ('ALCISTA', 'BAJISTA', 'ESPERAR') and valid_accuracy and valid_samples:
        return {'signal': signal, 'status': 'VALIDADO', 'detail': f'{accuracy:.1%} prueba · {samples} muestras'}
    reason = str(result.get('reason') or 'Modelo pendiente de validación')
    return {'signal': 'ESPERAR', 'status': 'EN EVALUACIÓN', 'detail': reason[:120]}
