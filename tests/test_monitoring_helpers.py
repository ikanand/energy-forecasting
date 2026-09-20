import datetime as dt

import pandas as pd
import pytest

pytest.importorskip("pyspark")  # monitoring.py imports Spark helpers; skipped in CI without Spark
from energy_forecasting.monitoring import daily_metrics  # noqa: E402


def test_daily_metrics_one_row_per_day_with_benchmark():
    hours = pd.date_range("2018-01-02", periods=48, freq="h")
    joined = pd.DataFrame(
        {
            "forecast_date": [h.date() for h in hours],
            "target_hour": hours,
            "actual_mw": 30000.0,
            "predicted_mw": [33000.0] * 24 + [30000.0] * 24,  # day 1: 10% off, day 2: perfect
            "tso_forecast_mw": 30300.0,  # 1% off
            "model_version": "9",
        }
    )
    daily = daily_metrics(joined)
    assert list(daily.forecast_date) == [dt.date(2018, 1, 2), dt.date(2018, 1, 3)]
    assert daily.mape.round(6).tolist() == [10.0, 0.0]
    assert daily.tso_mape.round(6).tolist() == [1.0, 1.0]
    assert daily.hours.tolist() == [24, 24]


def test_inference_log_appends_only_new_days_with_wallclock_time():
    from energy_forecasting.monitoring import rows_to_log

    hours = pd.date_range("2018-01-02", periods=72, freq="h")
    joined = pd.DataFrame({"forecast_date": [h.date() for h in hours], "target_hour": hours, "predicted_mw": 1.0})
    now = pd.Timestamp("2026-09-20 10:00")
    new = rows_to_log(joined, already_logged_days={dt.date(2018, 1, 2)}, scored_at=now)
    assert sorted(set(new.forecast_date)) == [dt.date(2018, 1, 3), dt.date(2018, 1, 4)]
    assert len(new) == 48
    assert (new.scored_at == now).all()  # real time, so the monitor's 30-day window includes it
