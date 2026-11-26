"""Forecast provider for the app.

Uses Member 2's predict_next_24h(model, history) once available, otherwise a dummy profile.
Assumed contract: returns 24 floats (kW), for the 24 hours after the end of `history`.
"""
import numpy as np


def dummy_forecast() -> np.ndarray:
    """Typical household daily shape: morning bump, evening peak."""
    h = np.arange(24)
    base = 0.8 + 0.4 * np.exp(-((h - 7.5) ** 2) / 4) + 1.6 * np.exp(-((h - 19.5) ** 2) / 5)
    return np.round(base, 3)


def get_forecast(model=None, history=None) -> np.ndarray:
    if model is not None and history is not None:
        try:
            from src.model import predict_next_24h  # TODO: adjust to Member 2's module/signature
            out = np.asarray(predict_next_24h(model, history), dtype=float)
            if out.shape == (24,):
                return out
        except ImportError:
            pass
    return dummy_forecast()
