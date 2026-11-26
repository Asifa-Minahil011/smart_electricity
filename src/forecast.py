"""Forecast provider for the app.

Member 2 contract: predict_next_24h(model, history) -> pandas Series of 24 hourly kW
values indexed by future timestamps (in src/model.py).

get_forecast() returns a Series indexed by hour of day (0..23), so it lines up with
the tariff, which is defined per clock hour. Falls back to a dummy profile if the
model or src/model.py is not available yet.
"""
from pathlib import Path
import numpy as np
import pandas as pd

MODEL_PATH = Path("models/best_model.joblib")  # TODO: confirm path/name with Member 2


def dummy_forecast() -> pd.Series:
    """Typical household daily shape: morning bump, evening peak."""
    h = np.arange(24)
    base = 0.8 + 0.4 * np.exp(-((h - 7.5) ** 2) / 4) + 1.6 * np.exp(-((h - 19.5) ** 2) / 5)
    return pd.Series(np.round(base, 3), index=h, name="kw")


def to_hour_of_day(s) -> pd.Series:
    """Re-index 24 consecutive hourly values by clock hour (0..23)."""
    s = pd.Series(s)
    if isinstance(s.index, pd.DatetimeIndex):
        s = pd.Series(s.to_numpy(dtype=float), index=s.index.hour, name="kw").sort_index()
    else:
        s = pd.Series(s.to_numpy(dtype=float), index=range(24), name="kw")
    if len(s) != 24 or s.index.duplicated().any():
        raise ValueError("expected 24 consecutive hourly values")
    return s


def load_model():
    """Trained model from models/best_model.joblib, or None if unavailable."""
    try:
        import joblib
        return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    except Exception:
        return None


def load_history():
    """Hourly DataFrame with column 'kw' (from data_prep.load_data), or None."""
    try:
        from src.data_prep import load_data
        return load_data()
    except Exception:
        return None


def get_forecast(model=None, history=None):
    """Return (forecast Series by hour of day, source string)."""
    try:
        from src.model import predict_next_24h
        model = load_model() if model is None else model
        history = load_history() if history is None else history
        if model is not None and history is not None:
            return to_hour_of_day(predict_next_24h(model, history)), "model"
    except ImportError:
        pass
    return dummy_forecast(), "dummy"


def backtest(model, history, day):
    """Forecast `day` using only data before it; return DataFrame(forecast, actual) by hour, or None."""
    from src.model import predict_next_24h
    start = pd.Timestamp(day).normalize()
    past = history[history.index < start]
    actual = history.loc[start: start + pd.Timedelta(hours=23), "kw"]
    if past.empty or len(actual) != 24:
        return None
    pred = to_hour_of_day(predict_next_24h(model, past))
    return pd.DataFrame({"Forecast": pred, "Actual": to_hour_of_day(actual)})
