"""Pure pandas transforms for the Kaggle files. No Spark here, so it is unit-testable.

All timestamps leave this module as timezone-naive UTC. Spark jobs set
spark.sql.session.timeZone=UTC so naive values are read the same way.
"""

from __future__ import annotations

import pandas as pd

KELVIN_OFFSET = 273.15


def _to_utc_naive(values: pd.Series) -> pd.Series:
    # Source strings carry +01:00 / +02:00 offsets (CET/CEST); utc=True normalises both.
    return pd.to_datetime(values, utc=True).dt.tz_localize(None)


def standardize_energy(raw: pd.DataFrame) -> pd.DataFrame:
    """energy_dataset.csv -> timestamp, demand_mw, tso_forecast_mw.

    tso_forecast_mw is the grid operator's own forecast. It is kept only as a
    benchmark to beat, never used as a model feature.
    """
    out = pd.DataFrame(
        {
            "timestamp": _to_utc_naive(raw["time"]),
            "demand_mw": pd.to_numeric(raw["total load actual"], errors="coerce"),
            "tso_forecast_mw": pd.to_numeric(raw["total load forecast"], errors="coerce"),
        }
    )
    return out.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def standardize_weather(raw: pd.DataFrame) -> pd.DataFrame:
    """weather_features.csv -> timestamp, city, temp_c, humidity, wind_speed, clouds_all (one row per city-hour)."""
    out = pd.DataFrame(
        {
            "timestamp": _to_utc_naive(raw["dt_iso"]),
            "city": raw["city_name"].astype(str).str.strip(),  # source has " Barcelona" with a leading space
            "temp_c": pd.to_numeric(raw["temp"], errors="coerce") - KELVIN_OFFSET,
            "humidity": pd.to_numeric(raw["humidity"], errors="coerce"),
            "wind_speed": pd.to_numeric(raw["wind_speed"], errors="coerce"),
            "clouds_all": pd.to_numeric(raw["clouds_all"], errors="coerce"),
        }
    )
    # The source repeats some city-hours (one row per weather condition); average them.
    out = out.groupby(["timestamp", "city"], as_index=False).mean(numeric_only=True)
    return out.sort_values(["timestamp", "city"]).reset_index(drop=True)


def aggregate_weather(city_weather: pd.DataFrame) -> pd.DataFrame:
    """Per-city hourly weather -> one national average row per hour."""
    return (
        city_weather.drop(columns=["city"])
        .groupby("timestamp", as_index=False)
        .mean(numeric_only=True)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def clean_energy(bronze: pd.DataFrame) -> pd.DataFrame:
    """Bronze energy -> silver actuals: one row per hour, sorted, no duplicate hours."""
    return bronze.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
