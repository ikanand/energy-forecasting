"""Data quality gate. Runs before features/training so a broken feed never trains a model."""

from __future__ import annotations

import pandas as pd


def check_energy(df: pd.DataFrame, max_null_fraction: float, demand_min: float, demand_max: float) -> list[str]:
    """Return a list of human-readable failures (empty list = pass)."""
    failures: list[str] = []
    if df.empty:
        return ["No energy rows in the checked window."]

    null_frac = df["demand_mw"].isna().mean()
    if null_frac > max_null_fraction:
        failures.append(f"demand_mw null fraction {null_frac:.1%} > {max_null_fraction:.1%}")

    valid = df["demand_mw"].dropna()
    out_of_range = ((valid < demand_min) | (valid > demand_max)).sum()
    if out_of_range:
        failures.append(f"{out_of_range} demand values outside [{demand_min:.0f}, {demand_max:.0f}] MW")

    dupes = df["timestamp"].duplicated().sum()
    if dupes:
        failures.append(f"{dupes} duplicate timestamps")

    per_day = df.groupby(df["timestamp"].dt.normalize()).size()
    incomplete = per_day[per_day < 24]
    if len(incomplete):
        failures.append(f"{len(incomplete)} day(s) with fewer than 24 hourly rows: {list(incomplete.index.date)}")
    return failures


def check_weather(df: pd.DataFrame, expected_cities: int, max_null_fraction: float) -> list[str]:
    failures: list[str] = []
    if df.empty:
        return ["No weather rows in the checked window."]
    cities_per_hour = df.groupby("timestamp")["city"].nunique()
    short = (cities_per_hour < expected_cities).sum()
    if short:
        failures.append(f"{short} hour(s) with fewer than {expected_cities} cities reporting")
    null_frac = df["temp_c"].isna().mean()
    if null_frac > max_null_fraction:
        failures.append(f"temp_c null fraction {null_frac:.1%} > {max_null_fraction:.1%}")
    return failures
