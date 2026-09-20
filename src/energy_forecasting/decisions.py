"""Promotion and retraining rules as pure functions, so the rules are unit-tested, not buried in jobs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mape(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    """Mean absolute percentage error in %, ignoring rows with missing or zero actuals."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(p) & (a != 0)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)


def should_promote(challenger_mape: float, champion_mape: float | None, min_improvement_pct: float) -> bool:
    """Promote if there is no champion yet, or the challenger is better by at least min_improvement_pct (relative)."""
    if champion_mape is None or np.isnan(champion_mape):
        return True
    return challenger_mape <= champion_mape * (1 - min_improvement_pct / 100)


def needs_retrain(daily_mape: pd.Series, baseline_mape: float, degradation_ratio: float, consecutive_days: int) -> bool:
    """True when the last `consecutive_days` daily MAPEs all exceed baseline x ratio.

    Requiring several bad days in a row stops a single odd day (a holiday, a
    storm) from triggering a retrain.
    """
    recent = daily_mape.dropna().tail(consecutive_days)
    if len(recent) < consecutive_days:
        return False
    return bool((recent > baseline_mape * degradation_ratio).all())
