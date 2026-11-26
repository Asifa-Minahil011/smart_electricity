"""Run:  streamlit run app.py"""
import numpy as np
import pandas as pd
import streamlit as st

from src.forecast import get_forecast, load_model, load_history, backtest
from src.optimizer import Appliance, optimize, load_profile
from src.tariff import TARIFF, CURRENCY, hourly_rates, daily_cost

st.set_page_config(page_title="Smart Electricity", layout="wide")
st.title("Smart Electricity: 24-hour forecast and cost optimizer")

rates = hourly_rates()
@st.cache_resource
def _model():
    return load_model()


@st.cache_data
def _history():
    return load_history()


model, history = _model(), _history()
forecast, source = get_forecast(model, history)
forecast = forecast.to_numpy()
hours = list(range(24))

with st.sidebar:
    st.header("Tariff (€/kWh)")
    for s, e, p in TARIFF:
        st.write(f"{s:02d}:00-{e:02d}:00: {p:.2f}")
    st.header("Power limit")
    limit_kw = st.slider("Max total load (kW)", 2.0, 12.0, 6.0, 0.5)
    st.header("Flexible appliances")
    defaults = [("Washing machine", 0.5, 2, 18), ("Dishwasher", 1.2, 2, 20), ("EV charger", 3.0, 4, 19)]
    appliances = []
    for name, kw, dur, pref in defaults:
        if st.checkbox(name, value=True):
            c1, c2 = st.columns(2)
            kw_ = c1.number_input(f"{name} kW", 0.1, 10.0, kw, 0.1, key=f"kw{name}")
            dur_ = c2.number_input(f"{name} hours", 1, 8, dur, key=f"d{name}")
            pref_ = st.slider(f"{name} usual start hour", 0, 23 - int(dur_) + 1, min(pref, 24 - int(dur_)), key=f"p{name}")
            appliances.append(Appliance(name, kw_, int(dur_), int(pref_)))

rows, base_app, opt_app = optimize(appliances, rates, base_load=forecast, limit_kw=limit_kw)
base_load = forecast + load_profile(appliances, [r["preferred_start"] for r in rows])
opt_load = forecast + load_profile(appliances, [r["optimal_start"] for r in rows])
cost_before, cost_after = daily_cost(base_load, rates), daily_cost(opt_load, rates)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Cost before", f"{CURRENCY}{cost_before:.2f}")
c2.metric("Cost after", f"{CURRENCY}{cost_after:.2f}")
c3.metric("Daily saving", f"{CURRENCY}{cost_before - cost_after:.2f}",
          f"{(cost_before - cost_after) / cost_before * 100:.1f}%" if cost_before else None)
c4.metric("Peak load after", f"{opt_load.max():.2f} kW", f"{opt_load.max() - base_load.max():+.2f} kW vs before",
          delta_color="inverse")
if any(r["note"] for r in rows):
    st.warning("Some appliances cannot fit under the power limit (see the note column). Raise the limit or shorten them.")

st.subheader("Load profile (kW)")
st.line_chart(pd.DataFrame({"Before": base_load, "After": opt_load, "Forecast only": forecast, "Limit": limit_kw}, index=hours))
st.subheader("Tariff (€/kWh)")
st.bar_chart(pd.Series(rates, index=hours, name="price"))
st.subheader("Recommended schedule")
if rows:
    st.dataframe(pd.DataFrame(rows), hide_index=True)
else:
    st.info("Select at least one appliance in the sidebar.")
if source == "model" and history is not None:
    st.subheader("Forecast vs actual (backtest on a past day)")
    last = history.index.max().normalize() - pd.Timedelta(days=1)
    first = max(pd.Timestamp("2010-01-01"), history.index.min().normalize() + pd.Timedelta(days=8))
    day = st.date_input("Day to test", value=min(pd.Timestamp("2010-06-01"), last).date(),
                        min_value=first.date(), max_value=last.date())
    try:
        bt = backtest(model, history, day)
    except Exception as e:  # keep the app alive if the model rejects the history
        bt, err = None, str(e)
    if bt is None:
        st.info("Not enough data for that day.")
    else:
        mae = (bt["Forecast"] - bt["Actual"]).abs().mean()
        rmse = ((bt["Forecast"] - bt["Actual"]) ** 2).mean() ** 0.5
        st.line_chart(bt)
        st.caption(f"MAE {mae:.3f} kW, RMSE {rmse:.3f} kW for {day}")
st.caption("Forecast source: " + ("trained model" if source == "model" else "dummy profile (real model not connected yet)"))
