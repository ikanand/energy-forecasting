"""Day-ahead feature construction (pure pandas).

A forecast for day D+1 is produced after day D closes. So for any target hour t,
only demand up to t - 24h is known. Every demand feature here is built from
demand shifted by at least `min_lag_hours`; tests enforce this.

The dataset is ~35k hourly rows, so pandas is the simple, testable choice.
At larger scale the same logic moves to Spark window functions.
"""

from __future__ import annotations

import holidays
import pandas as pd

from energy_forecasting.config import FeatureSettings


def build_features(
    energy: pd.DataFrame,
    weather: pd.DataFrame,
    horizon_end: pd.Timestamp,
    settings: FeatureSettings,
    location_id: str,
    timezone: str,
) -> pd.DataFrame:
    """Build one feature row per hour from the first known hour up to horizon_end (inclusive).

    energy:  timestamp, demand_mw          (actuals known so far)
    weather: timestamp, <weather columns>  (city-averaged; may extend into the forecast day)
    Rows after the last known actual are the hours we will forecast.
    """
    start = energy["timestamp"].min()
    spine = pd.date_range(start, horizon_end, freq="h")
    demand = energy.set_index("timestamp")["demand_mw"].reindex(spine)

    df = pd.DataFrame({"timestamp": spine})
    for h in settings.lags_hours:
        df[f"lag_{h}h"] = demand.shift(h).to_numpy()

    known = demand.shift(settings.min_lag_hours)  # latest value usable at forecast time
    for w in settings.rolling_windows_hours:
        df[f"rolling_mean_{w}h"] = known.rolling(w, min_periods=int(w * 0.8)).mean().to_numpy()

    df = df.merge(weather[["timestamp", *settings.weather_columns]], on="timestamp", how="left")
    df = add_calendar_features(df, timezone)
    df.insert(0, "location_id", location_id)
    return df


def add_calendar_features(df: pd.DataFrame, timezone: str) -> pd.DataFrame:
    local = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(timezone)
    years = range(local.dt.year.min(), local.dt.year.max() + 1)
    national_holidays = holidays.country_holidays("ES", years=years)
    out = df.copy()
    out["hour_of_day"] = local.dt.hour
    out["day_of_week"] = local.dt.dayofweek  # 0 = Monday
    out["month"] = local.dt.month
    out["is_weekend"] = (local.dt.dayofweek >= 5).astype(int)
    out["is_holiday"] = local.dt.date.map(lambda d: int(d in national_holidays))
    return out


def forecast_hours(day: pd.Timestamp) -> pd.DatetimeIndex:
    """The 24 UTC hours of a calendar day."""
    start = pd.Timestamp(day).normalize()
    return pd.date_range(start, periods=24, freq="h")
