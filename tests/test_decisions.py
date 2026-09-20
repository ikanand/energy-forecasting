import numpy as np
import pandas as pd

from energy_forecasting.decisions import mape, needs_retrain, should_promote


def test_mape_ignores_missing_and_zero():
    assert mape([100, 200, np.nan, 0], [110, 180, 5, 5]) == 10.0


def test_first_model_is_always_promoted():
    assert should_promote(5.0, None, 1.0)


def test_promotion_needs_meaningful_improvement():
    assert should_promote(4.9, 5.0, 1.0)  # 2% better
    assert not should_promote(4.97, 5.0, 1.0)  # 0.6% better: noise, keep champion
    assert not should_promote(5.5, 5.0, 1.0)


def test_single_bad_day_does_not_retrain():
    assert not needs_retrain(pd.Series([3, 3, 3, 9]), baseline_mape=3, degradation_ratio=1.25, consecutive_days=3)


def test_sustained_degradation_triggers_retrain():
    assert needs_retrain(pd.Series([3, 4, 4, 4]), baseline_mape=3, degradation_ratio=1.25, consecutive_days=3)


def test_not_enough_history_does_not_retrain():
    assert not needs_retrain(pd.Series([9, 9]), baseline_mape=3, degradation_ratio=1.25, consecutive_days=3)
