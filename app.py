"""Run:  streamlit run app.py"""
import numpy as np
import pandas as pd
import streamlit as st

from src.forecast import get_forecast
from src.optimizer import Appliance, optimize, load_profile
from src.tariff import TARIFF, CURRENCY, hourly_rates, daily_cost

st.set_page_config(page_title="Smart Electricity", layout="wide")
st.title("Smart Electricity: 24-hour forecast and cost optimizer")

rates = hourly_rates()
forecast = get_forecast()  # TODO: pass (model, history) once Member 2's model is ready
hours = list(range(24))

with st.sidebar:
    st.header("Tariff (€/kWh)")
    for s, e, p in TARIFF:
        st.write(f"{s:02d}:00-{e:02d}:00: {p:.2f}")
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

rows, base_app, opt_app = optimize(appliances, rates)
base_load = forecast + load_profile(appliances, [r["preferred_start"] for r in rows])
opt_load = forecast + load_profile(appliances, [r["optimal_start"] for r in rows])
cost_before, cost_after = daily_cost(base_load, rates), daily_cost(opt_load, rates)

c1, c2, c3 = st.columns(3)
c1.metric("Cost before", f"{CURRENCY}{cost_before:.2f}")
c2.metric("Cost after", f"{CURRENCY}{cost_after:.2f}")
c3.metric("Daily saving", f"{CURRENCY}{cost_before - cost_after:.2f}",
          f"{(cost_before - cost_after) / cost_before * 100:.1f}%" if cost_before else None)

st.subheader("Load profile (kW)")
st.line_chart(pd.DataFrame({"Before": base_load, "After": opt_load, "Forecast only": forecast}, index=hours))
st.subheader("Tariff (€/kWh)")
st.bar_chart(pd.Series(rates, index=hours, name="price"))
st.subheader("Recommended schedule")
if rows:
    st.dataframe(pd.DataFrame(rows), hide_index=True)
else:
    st.info("Select at least one appliance in the sidebar.")
st.caption("Forecast: dummy profile until the real model is connected.")
