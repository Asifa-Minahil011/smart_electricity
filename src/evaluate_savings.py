"""Cost savings of the optimized schedule vs the usual (naive) schedule on past test days.

Run from the repo root:
    python -m src.evaluate_savings                 # every 7th day of 2010
    python -m src.evaluate_savings --step 1        # all test days (slower)
    python -m src.evaluate_savings --limit 3.5

For each day the optimizer plans with the FORECAST; costs are then computed with the ACTUAL load,
so the numbers reflect what a household would really have paid.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.forecast import load_model, load_history, forecast_for_day
from src.optimizer import Appliance, DEFAULT_APPLIANCES, optimize, load_profile
from src.tariff import hourly_rates, daily_cost


def evaluate(model, history, days, limit_kw):
    rates = hourly_rates()
    apps = [Appliance(*a) for a in DEFAULT_APPLIANCES]
    naive = load_profile(apps, [a.preferred_start for a in apps])
    rows = []
    for day in days:
        try:
            fc, actual = forecast_for_day(model, history, day)
        except Exception:
            continue
        if actual is None:
            continue
        fc, act = fc.to_numpy(), actual.to_numpy()
        sched, _, _ = optimize(apps, rates, base_load=fc, limit_kw=limit_kw)
        opt = load_profile(apps, [r["optimal_start"] for r in sched])
        before, after = daily_cost(act + naive, rates), daily_cost(act + opt, rates)
        rows.append({"day": pd.Timestamp(day).date(), "cost_before": before, "cost_after": after,
                     "saving": before - after, "saving_pct": 100 * (before - after) / before,
                     "peak_before": (act + naive).max(), "peak_after": (act + opt).max(),
                     "over_limit_before": bool((act + naive).max() > limit_kw),
                     "over_limit_after": bool((act + opt).max() > limit_kw),
                     "forecast_mae": float(np.abs(fc - act).mean())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=float, default=6.0, help="household power limit in kW")
    ap.add_argument("--step", type=int, default=7, help="evaluate every N-th day")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--out", default="results/savings.csv")
    a = ap.parse_args()

    model, history = load_model(), load_history()
    if model is None or history is None:
        raise SystemExit("Need models/best_model.joblib and src/data_prep.load_data()")
    days = pd.date_range(a.start, history.index.max().normalize() - pd.Timedelta(days=1), freq=f"{a.step}D")
    df = evaluate(model, history, days, a.limit)
    if df.empty:
        raise SystemExit("No days could be evaluated")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)

    print(f"Days evaluated: {len(df)}  (limit {a.limit} kW)")
    print(f"Mean daily cost before: {df.cost_before.mean():.3f}  after: {df.cost_after.mean():.3f}")
    print(f"Mean daily saving: {df.saving.mean():.3f} ({df.saving_pct.mean():.1f}% of the bill), "
          f"about {df.saving.mean() * 365:.0f} per year")
    print(f"Mean peak load: {df.peak_before.mean():.2f} kW before, {df.peak_after.mean():.2f} kW after")
    print(f"Days above the limit: {df.over_limit_before.sum()} before, {df.over_limit_after.sum()} after")
    print(f"Mean forecast MAE on these days: {df.forecast_mae.mean():.3f} kW")
    print(f"Saved per-day results to {a.out}")


if __name__ == "__main__":
    main()
