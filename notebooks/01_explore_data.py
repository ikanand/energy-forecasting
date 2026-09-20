# %% [markdown]
# # 01 - Explore the Kaggle energy data (runs on your laptop, no Databricks)
#
# Put the two Kaggle CSVs in `data/` (git-ignored), then run the cells in VS Code
# ("Run Cell") or with `uv run python notebooks/01_explore_data.py`.
# This reuses the package's own standardize functions, so what you see here is
# exactly what the pipeline will load.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from energy_forecasting.data_prep import aggregate_weather, standardize_energy, standardize_weather

DATA = Path(__file__).resolve().parents[1] / "data" if "__file__" in globals() else Path("data")
energy = standardize_energy(pd.read_csv(DATA / "energy_dataset.csv"))
city_weather = standardize_weather(pd.read_csv(DATA / "weather_features.csv"))
weather = aggregate_weather(city_weather)

# %% Basic health checks - these become the data quality rules in quality.py
print(f"Energy:  {len(energy):,} hours  {energy.timestamp.min()} -> {energy.timestamp.max()} (UTC)")
print(f"Weather: {len(city_weather):,} city-hours, cities = {sorted(city_weather.city.unique())}")
print(f"Missing demand hours: {energy.demand_mw.isna().sum()}")
expected = pd.date_range(energy.timestamp.min(), energy.timestamp.max(), freq="h")
print(f"Gaps in the hourly timeline: {len(expected.difference(energy.timestamp))}")
print(energy.demand_mw.describe().round(0))

# %% The benchmark to beat: the grid operator's own day-ahead forecast
err = (energy.tso_forecast_mw - energy.demand_mw).abs() / energy.demand_mw * 100
print(f"Grid operator (TSO) forecast MAPE over the whole dataset: {err.mean():.2f}%")

# %% Demand over time: yearly seasonality (winter heating + summer cooling peaks)
daily = energy.set_index("timestamp").demand_mw.resample("D").mean()
daily.plot(figsize=(12, 3), title="Daily average demand (MW)")
plt.show()

# %% Hourly profile: why we forecast 24 values per day, and why weekends differ
local = energy.assign(local=energy.timestamp.dt.tz_localize("UTC").dt.tz_convert("Europe/Madrid"))
local["hour"] = local.local.dt.hour
local["weekend"] = local.local.dt.dayofweek >= 5
local.pivot_table(index="hour", columns="weekend", values="demand_mw").plot(
    figsize=(8, 3), title="Average demand by local hour (weekday vs weekend)"
)
plt.show()

# %% Temperature vs demand: the U-shape (both cold and heat increase demand)
merged = energy.merge(weather, on="timestamp")
merged.plot.scatter(x="temp_c", y="demand_mw", s=1, alpha=0.2, figsize=(6, 4), title="Temperature vs demand")
plt.show()

# %% How much does yesterday predict today? (justifies lag_24h and lag_168h)
s = energy.set_index("timestamp").demand_mw
for h in (24, 48, 168):
    print(f"corr(demand, demand {h}h earlier) = {s.corr(s.shift(h)):.3f}")
