# %%
import os, sys
sys.path.append(os.getcwd())          # lets us import src/ when run from project root
import pandas as pd
import matplotlib.pyplot as plt
from src.data_prep import load_data, RAW_PATH

os.makedirs("reports/figures", exist_ok=True)
plt.rcParams.update({"figure.figsize": (9, 4.5), "axes.grid": True, "grid.alpha": 0.3})

def save(name):
    plt.tight_layout()
    plt.savefig(f"reports/figures/{name}.png", dpi=150)
    plt.show()

hourly = load_data()

# %% Plot 1: missing minute readings per month (raw data, before cleaning)
raw = pd.read_csv(RAW_PATH, sep=";", na_values=["?"], low_memory=False,
                  usecols=["Date", "Time", "Global_active_power"])
raw.index = pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%d/%m/%Y %H:%M:%S")
missing = raw["Global_active_power"].isna().resample("MS").sum()

plt.bar(range(len(missing)), missing.values)
plt.xticks(range(0, len(missing), 6), [d.strftime("%b %Y") for d in missing.index[::6]], rotation=45)
plt.title("Missing minute readings per month (before cleaning)")
plt.ylabel("Missing readings")
save("01_missing_per_month")

# %% Plot 2: average consumption by hour of day
hourly.groupby(hourly.index.hour)["kw"].mean().plot(marker="o")
plt.title("Average consumption by hour of day")
plt.xlabel("Hour"); plt.ylabel("Mean power (kW)")
plt.xticks(range(0, 24, 2))
save("02_by_hour")

# %% Plot 3: average consumption by day of week
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
by_day = hourly.groupby(hourly.index.day_of_week)["kw"].mean()
plt.bar(days, by_day.values)
plt.title("Average consumption by day of week")
plt.ylabel("Mean power (kW)")
save("03_by_weekday")

# %% Plot 4: seasonal pattern (average by month of year)
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
by_month = hourly.groupby(hourly.index.month)["kw"].mean()
plt.plot(months[:len(by_month)] if len(by_month) < 12 else months, by_month.values, marker="o")
plt.title("Average consumption by month of year")
plt.ylabel("Mean power (kW)")
save("04_by_month")

# %% Plot 5: one sample week of hourly consumption
hourly.loc["2010-01-11":"2010-01-17", "kw"].plot()
plt.title("Sample week of hourly consumption (11-17 Jan 2010)")
plt.ylabel("Power (kW)")
save("05_sample_week")