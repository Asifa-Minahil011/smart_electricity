"""model.py - 24-hour-ahead multi-output forecasting (Random Forest + XGBoost).

For every forecast origin t (an hour with a full set of features) the model
predicts kw at t+1, t+2, ..., t+24 in one shot (multi-output).

Run training + evaluation:   python -m src.model
Use a trained model:         predict_next_24h(model, history)
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from src.data_prep import SPLIT_DATE, load_data, make_features, train_test_split_time

HORIZON = 24                  # hours ahead to forecast
MIN_HISTORY_HOURS = 169       # lag_168 needs 168 past hours + the current hour
RANDOM_STATE = 42

MODELS_DIR = "models"
REPORTS_DIR = "reports"


# ---------------------------------------------------------------- dataset ---

def target_columns(horizon=HORIZON):
    """Names of the target columns: kw_t+1 ... kw_t+24."""
    return [f"kw_t+{h}" for h in range(1, horizon + 1)]


def build_targets(kw, origins, horizon=HORIZON):
    """Targets for each origin t: kw at t+1 ... t+horizon.

    Looked up by timestamp (not by row position), so gaps in the data can
    never shift a target onto the wrong hour. Missing hours become NaN.
    """
    targets = {}
    for h, name in enumerate(target_columns(horizon), start=1):
        future_times = origins + pd.Timedelta(hours=h)
        targets[name] = kw.reindex(future_times).to_numpy()
    return pd.DataFrame(targets, index=origins)


def build_dataset(hourly, horizon=HORIZON):
    """Return (X, y): features at each origin t and the next-24h targets.

    X comes from make_features() (past information only). Origins whose
    24 future hours are not all available are dropped.
    """
    feats = make_features(hourly)
    y = build_targets(hourly["kw"], feats.index, horizon)
    keep = y.notna().all(axis=1)
    return feats[keep], y[keep]


def split_train_test(X, y, split_date=SPLIT_DATE, horizon=HORIZON):
    """Chronological split with no leakage.

    Train: origins whose LAST target hour (t+24) is strictly before split_date.
    Test:  origins from split_date onward.
    """
    split = pd.Timestamp(split_date)
    train_X, test_X = train_test_split_time(X, split_date)
    train_X = train_X[train_X.index + pd.Timedelta(hours=horizon) < split]
    return train_X, y.loc[train_X.index], test_X, y.loc[test_X.index]


# ----------------------------------------------------------------- models ---

def build_models():
    """Primary model (Random Forest) and comparison model (XGBoost)."""
    rf = RandomForestRegressor(
        n_estimators=100, min_samples_leaf=20, max_features=0.5,
        n_jobs=-1, random_state=RANDOM_STATE)
    xgb = MultiOutputRegressor(XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=RANDOM_STATE))
    return {"random_forest": rf, "xgboost": xgb}


# ------------------------------------------------------------- evaluation ---

def evaluate(y_true, y_pred):
    """MAE, RMSE, R2 over all horizons together, plus a per-horizon table."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    overall = {
        "MAE": mean_absolute_error(y_true.ravel(), y_pred.ravel()),
        "RMSE": float(np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel()))),
        "R2": r2_score(y_true.ravel(), y_pred.ravel()),
    }
    rows = []
    for i in range(y_true.shape[1]):
        rows.append({
            "horizon": i + 1,
            "MAE": mean_absolute_error(y_true[:, i], y_pred[:, i]),
            "RMSE": float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))),
            "R2": r2_score(y_true[:, i], y_pred[:, i]),
        })
    return overall, pd.DataFrame(rows)


def train_and_evaluate(hourly, models=None):
    """Train every model on the training origins, score on the test origins.

    Returns (fitted_models, metrics_df, per_horizon_df, best_name).
    Best model = lowest aggregate RMSE on the test set.
    """
    X, y = build_dataset(hourly)
    train_X, train_y, test_X, test_y = split_train_test(X, y)
    if len(train_X) == 0 or len(test_X) == 0:
        raise ValueError("Train or test set is empty - check the data range.")

    models = build_models() if models is None else models
    fitted, summary, per_horizon = {}, [], []
    for name, model in models.items():
        print(f"Training {name} on {len(train_X):,} origins ...")
        model.fit(train_X, train_y.to_numpy())          # no shuffling anywhere
        pred = model.predict(test_X)
        overall, by_h = evaluate(test_y.to_numpy(), pred)
        summary.append({"model": name, **overall})
        by_h.insert(0, "model", name)
        per_horizon.append(by_h)
        fitted[name] = model

    metrics_df = pd.DataFrame(summary)
    per_horizon_df = pd.concat(per_horizon, ignore_index=True)
    best_name = metrics_df.loc[metrics_df["RMSE"].idxmin(), "model"]
    return fitted, metrics_df, per_horizon_df, best_name


def save_outputs(fitted, metrics_df, per_horizon_df, best_name):
    """Save models with joblib and metrics as CSV (folders are created)."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    for name, model in fitted.items():
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.joblib"), compress=3)
    joblib.dump(fitted[best_name], os.path.join(MODELS_DIR, "best_model.joblib"), compress=3)
    metrics_df.to_csv(os.path.join(REPORTS_DIR, "model_metrics.csv"), index=False)
    per_horizon_df.to_csv(os.path.join(REPORTS_DIR, "per_horizon_metrics.csv"), index=False)


# ------------------------------------------------------------- prediction ---

def predict_next_24h(model, history):
    """Forecast the next 24 hours of consumption.

    model:   a trained model with .predict() (e.g. models/best_model.joblib)
    history: hourly DataFrame, DatetimeIndex, column 'kw'. The last
             MIN_HISTORY_HOURS (169) rows must be consecutive hours with no gaps.
    Returns: pd.Series of 24 kW values indexed by the next 24 hourly timestamps.
    """
    if not hasattr(model, "predict"):
        raise TypeError("model must be a trained model with a predict() method.")
    if not isinstance(history, pd.DataFrame):
        raise TypeError("history must be a pandas DataFrame.")
    if not isinstance(history.index, pd.DatetimeIndex):
        raise ValueError("history must have a DatetimeIndex.")
    if "kw" not in history.columns:
        raise ValueError("history must have a 'kw' column.")
    if not history.index.is_monotonic_increasing or history.index.has_duplicates:
        raise ValueError("history index must be sorted with no duplicate timestamps.")
    if len(history) < MIN_HISTORY_HOURS:
        raise ValueError(
            f"Need at least {MIN_HISTORY_HOURS} hourly rows of history, got {len(history)}.")

    kw = pd.to_numeric(history["kw"], errors="coerce")
    recent = kw.iloc[-MIN_HISTORY_HOURS:].to_frame("kw")
    if recent["kw"].isna().any():
        raise ValueError(
            f"The last {MIN_HISTORY_HOURS} hours of 'kw' contain missing or non-numeric values.")
    gaps = recent.index.to_series().diff().dropna()
    if not (gaps == pd.Timedelta(hours=1)).all():
        raise ValueError(
            f"The last {MIN_HISTORY_HOURS} rows must be consecutive hourly timestamps.")

    feats = make_features(recent)        # same features as training
    if feats.empty:
        raise ValueError("Could not build features from the given history.")
    latest = feats.iloc[[-1]]            # newest valid row = forecast origin

    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        latest = latest[list(names)]
    values = np.asarray(model.predict(latest)).reshape(-1)
    if values.shape[0] != HORIZON:
        raise ValueError(f"Model returned {values.shape[0]} values, expected {HORIZON}.")

    next_hours = pd.date_range(latest.index[0] + pd.Timedelta(hours=1),
                               periods=HORIZON, freq="h")
    return pd.Series(np.clip(values, 0, None), index=next_hours, name="forecast_kw")


# ------------------------------------------------------------- entry point ---

def main():
    hourly = load_data()
    fitted, metrics_df, per_horizon_df, best_name = train_and_evaluate(hourly)
    save_outputs(fitted, metrics_df, per_horizon_df, best_name)

    print("\nTest-set metrics (all 24 horizons, same test origins):")
    print(metrics_df.round(4).to_string(index=False))
    print("\nPer-horizon RMSE (hours ahead 1, 6, 12, 18, 24):")
    for name in fitted:
        sub = per_horizon_df[per_horizon_df["model"] == name].set_index("horizon")
        print(f"  {name}: " + ", ".join(f"{sub.loc[h, 'RMSE']:.3f}" for h in (1, 6, 12, 18, 24)))
    print(f"\nBest model (lowest RMSE): {best_name}")
    print(f"Saved models to {MODELS_DIR}/ and metrics to {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
