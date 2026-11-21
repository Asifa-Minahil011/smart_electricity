"""data_prep.py - load, clean, resample and engineer features for hourly forecasting."""
import os
import numpy as np
import pandas as pd

RAW_PATH = "data/household_power_consumption.txt"
HOURLY_PATH = "data/hourly.csv"
SPLIT_DATE = "2010-01-01"   # train: before this date, test: from this date


def load_data(path=RAW_PATH, use_cache=True, verbose=False):
    """Return an hourly DataFrame with one column 'kw' (mean active power in kW)."""
    if use_cache and os.path.exists(HOURLY_PATH):
        return pd.read_csv(HOURLY_PATH, index_col=0, parse_dates=True)

    df = pd.read_csv(path, sep=";", na_values=["?"], low_memory=False)
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df = df.set_index("datetime")
    power = df["Global_active_power"].astype(float)

    # Resample 1-minute readings to hourly means (mean kW over an hour = kWh)
    hourly = power.resample("h").mean().to_frame("kw")
    missing_before = int(hourly["kw"].isna().sum())

    # Fill only short gaps (up to 3 hours); long outages stay NaN
    hourly["kw"] = hourly["kw"].interpolate(limit=3)
    missing_after = int(hourly["kw"].isna().sum())

    if verbose:
        print(f"Raw minute rows:          {len(df):,}")
        print(f"Missing minute readings:  {power.isna().sum():,} "
              f"({power.isna().mean():.2%})")
        print(f"Hourly rows:              {len(hourly):,}")
        print(f"Missing hours before fill: {missing_before}")
        print(f"Missing hours after fill:  {missing_after}")
        print(f"Date range: {hourly.index.min()} to {hourly.index.max()}")

    hourly.to_csv(HOURLY_PATH)
    return hourly


def make_features(hourly):
    """Add lag, rolling and calendar features. Uses only past information."""
    df = hourly.copy()
    for lag in [1, 2, 3, 24, 48, 168]:            # 168 = same hour last week
        df[f"lag_{lag}"] = df["kw"].shift(lag)
    df["roll_mean_24"] = df["kw"].shift(1).rolling(24).mean()
    df["roll_mean_168"] = df["kw"].shift(1).rolling(168).mean()
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["month"] = df.index.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df.dropna()   # drops rows near gaps or without enough history


def train_test_split_time(df, split_date=SPLIT_DATE):
    """Chronological split. Never shuffle time series."""
    train = df[df.index < split_date]
    test = df[df.index >= split_date]
    return train, test


if __name__ == "__main__":
    hourly = load_data(use_cache=False, verbose=True)   # forces a fresh load + prints stats
    feats = make_features(hourly)
    train, test = train_test_split_time(feats)
    print(f"\nFeature rows: {len(feats):,}  |  columns: {list(feats.columns)}")
    print(f"Train: {len(train):,} rows ({train.index.min().date()} to {train.index.max().date()})")
    print(f"Test:  {len(test):,} rows ({test.index.min().date()} to {test.index.max().date()})")