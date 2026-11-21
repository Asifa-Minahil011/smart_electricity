"""Tests for src/model.py. Synthetic series are used only to check logic;
the training test uses the real data/hourly.csv."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import model as m
from src.data_prep import HOURLY_PATH, load_data


def counter_series(start="2009-11-01", end="2010-02-01"):
    """kw = 1, 2, 3, ... so every target value is easy to predict by hand."""
    idx = pd.date_range(start, end, freq="h")
    return pd.DataFrame({"kw": np.arange(1, len(idx) + 1, dtype=float)}, index=idx)


def tiny_model():
    return RandomForestRegressor(n_estimators=5, random_state=0)


def test_target_alignment():
    hourly = counter_series()
    X, y = m.build_dataset(hourly)
    assert list(y.columns) == m.target_columns()
    t = X.index[10]
    for h in range(1, 25):
        assert y.loc[t, f"kw_t+{h}"] == hourly.loc[t + pd.Timedelta(hours=h), "kw"]


def test_targets_nan_when_hour_missing():
    hourly = counter_series().drop(pd.Timestamp("2009-12-10 05:00"))
    X, y = m.build_dataset(hourly)
    # origins that need the missing hour must have been dropped
    for h in range(1, 25):
        bad_origin = pd.Timestamp("2009-12-10 05:00") - pd.Timedelta(hours=h)
        assert bad_origin not in y.index


def test_no_chronological_leakage():
    X, y = m.build_dataset(counter_series())
    train_X, train_y, test_X, test_y = m.split_train_test(X, y)
    split = pd.Timestamp("2010-01-01")
    assert train_X.index.max() + pd.Timedelta(hours=24) < split   # all 24 targets before split
    assert test_X.index.min() >= split
    assert train_X.index.is_monotonic_increasing and test_X.index.is_monotonic_increasing
    assert train_X.index.max() < test_X.index.min()
    assert (train_y.index == train_X.index).all() and (test_y.index == test_X.index).all()


def fitted_tiny_model():
    X, y = m.build_dataset(counter_series("2009-06-01", "2010-02-01"))
    train_X, train_y, _, _ = m.split_train_test(X, y)
    model = tiny_model().fit(train_X, train_y.to_numpy())
    return model


def test_predict_output_shape_and_index():
    model = fitted_tiny_model()
    history = counter_series("2009-12-01", "2010-01-15 06:00")
    out = m.predict_next_24h(model, history)
    assert isinstance(out, pd.Series) and len(out) == 24
    expected = pd.date_range("2010-01-15 07:00", periods=24, freq="h")
    assert (out.index == expected).all()
    assert (out >= 0).all()


def test_predict_works_with_xgboost_wrapper():
    X, y = m.build_dataset(counter_series("2009-06-01", "2010-02-01"))
    train_X, train_y, _, _ = m.split_train_test(X, y)
    xgb = MultiOutputRegressor(XGBRegressor(n_estimators=3, random_state=0))
    xgb.fit(train_X, train_y.to_numpy())
    out = m.predict_next_24h(xgb, counter_series("2009-12-01", "2010-01-15"))
    assert len(out) == 24


def test_invalid_history_errors():
    model = fitted_tiny_model()
    good = counter_series("2009-12-01", "2010-01-15")
    with pytest.raises(TypeError):
        m.predict_next_24h(model, [1, 2, 3])
    with pytest.raises(ValueError, match="DatetimeIndex"):
        m.predict_next_24h(model, good.reset_index(drop=True))
    with pytest.raises(ValueError, match="kw"):
        m.predict_next_24h(model, good.rename(columns={"kw": "power"}))
    with pytest.raises(ValueError, match="at least"):
        m.predict_next_24h(model, good.iloc[-100:])
    with_nan = good.copy()
    with_nan.iloc[-5, 0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        m.predict_next_24h(model, with_nan)
    gap = good.drop(good.index[-50])
    with pytest.raises(ValueError, match="consecutive"):
        m.predict_next_24h(model, gap)
    with pytest.raises(ValueError, match="sorted"):
        m.predict_next_24h(model, good.iloc[::-1])


@pytest.mark.skipif(not os.path.exists(HOURLY_PATH), reason="data/hourly.csv not found")
def test_training_and_evaluation_on_real_data(tmp_path, monkeypatch):
    hourly = load_data()
    hourly = hourly[hourly.index >= "2009-01-01"]          # smaller slice to keep the test fast
    models = {
        "random_forest": tiny_model(),
        "xgboost": MultiOutputRegressor(XGBRegressor(n_estimators=5, random_state=0)),
    }
    fitted, metrics, per_h, best = m.train_and_evaluate(hourly, models)
    assert set(metrics["model"]) == {"random_forest", "xgboost"}
    assert {"MAE", "RMSE", "R2"} <= set(metrics.columns)
    assert len(per_h) == 2 * 24
    assert best in fitted
    assert (metrics[["MAE", "RMSE"]] > 0).all().all()

    monkeypatch.setattr(m, "MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(m, "REPORTS_DIR", str(tmp_path / "reports"))
    m.save_outputs(fitted, metrics, per_h, best)
    for f in ["models/random_forest.joblib", "models/xgboost.joblib", "models/best_model.joblib",
              "reports/model_metrics.csv", "reports/per_horizon_metrics.csv"]:
        assert (tmp_path / f).exists()
