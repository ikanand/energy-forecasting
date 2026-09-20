import numpy as np
import pandas as pd

from energy_forecasting.features import build_features, forecast_hours


def _build(cfg, energy, weather):
    horizon_end = energy.timestamp.max() + pd.Timedelta(hours=24)
    return build_features(energy, weather, horizon_end, cfg.features, cfg.location_id, cfg.timezone)


def test_output_has_all_configured_features(cfg, hourly_energy, hourly_weather):
    df = _build(cfg, hourly_energy, hourly_weather)
    assert set(cfg.feature_columns) <= set(df.columns)
    assert {"location_id", "timestamp"} <= set(df.columns)


def test_lags_equal_demand_exactly_h_hours_earlier(cfg, hourly_energy, hourly_weather):
    df = _build(cfg, hourly_energy, hourly_weather).set_index("timestamp")
    demand = hourly_energy.set_index("timestamp").demand_mw
    for h in cfg.features.lags_hours:
        t = hourly_energy.timestamp.iloc[500]
        assert df.loc[t, f"lag_{h}h"] == demand[t - pd.Timedelta(hours=h)]


def test_no_feature_uses_demand_from_the_last_24_hours(cfg, hourly_energy, hourly_weather):
    """Poison the 24h before t: features at t must not change. This is the day-ahead leakage test."""
    t = hourly_energy.timestamp.iloc[1000]
    clean = _build(cfg, hourly_energy, hourly_weather).set_index("timestamp").loc[t]

    poisoned = hourly_energy.copy()
    recent = (poisoned.timestamp > t - pd.Timedelta(hours=24)) & (poisoned.timestamp <= t)
    poisoned.loc[recent, "demand_mw"] = 1e9
    dirty = _build(cfg, poisoned, hourly_weather).set_index("timestamp").loc[t]

    demand_features = [c for c in cfg.feature_columns if c.startswith(("lag_", "rolling_"))]
    pd.testing.assert_series_equal(clean[demand_features], dirty[demand_features])


def test_tomorrow_rows_exist_and_have_lags(cfg, hourly_energy, hourly_weather):
    df = _build(cfg, hourly_energy, hourly_weather)
    tomorrow = df[df.timestamp > hourly_energy.timestamp.max()]
    assert len(tomorrow) == 24
    assert tomorrow["lag_24h"].notna().all()  # yesterday's actuals are known


def test_calendar_uses_local_time(cfg):
    energy = pd.DataFrame({"timestamp": pd.date_range("2017-07-03 22:00", periods=2, freq="h"), "demand_mw": 1.0})
    weather = pd.DataFrame(
        {"timestamp": energy.timestamp, "temp_c": 1.0, "humidity": 1.0, "wind_speed": 1.0, "clouds_all": 1.0}
    )
    df = build_features(energy, weather, energy.timestamp.max(), cfg.features, "ES", "Europe/Madrid")
    assert list(df.hour_of_day) == [0, 1]  # 22:00 UTC = 00:00 in Madrid summer time


def test_spanish_holiday_flag(cfg):
    energy = pd.DataFrame({"timestamp": pd.date_range("2017-12-24 23:00", periods=2, freq="h"), "demand_mw": 1.0})
    weather = energy.assign(temp_c=1.0, humidity=1.0, wind_speed=1.0, clouds_all=1.0).drop(columns="demand_mw")
    df = build_features(energy, weather, energy.timestamp.max(), cfg.features, "ES", "Europe/Madrid")
    assert list(df.is_holiday) == [1, 1]  # 00:00 and 01:00 on 25 Dec, Madrid time


def test_forecast_hours_are_24_utc_hours():
    hours = forecast_hours(pd.Timestamp("2018-01-02 13:45"))
    assert len(hours) == 24 and hours[0] == pd.Timestamp("2018-01-02") and np.all(np.diff(hours.hour) == 1)
