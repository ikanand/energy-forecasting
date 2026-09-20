import pandas as pd

from energy_forecasting.quality import check_energy, check_weather


def _day(value=30000.0):
    return pd.DataFrame({"timestamp": pd.date_range("2018-01-01", periods=24, freq="h"), "demand_mw": value})


def test_clean_day_passes():
    assert check_energy(_day(), 0.02, 10000, 50000) == []


def test_unit_bug_is_caught():
    # e.g. a feed switching from MW to kW
    failures = check_energy(_day(30_000_000.0), 0.02, 10000, 50000)
    assert any("outside" in f for f in failures)


def test_missing_hours_and_nulls_caught():
    df = _day().iloc[:20].copy()
    df.loc[0:5, "demand_mw"] = None
    failures = check_energy(df, 0.02, 10000, 50000)
    assert any("null" in f for f in failures) and any("fewer than 24" in f for f in failures)


def test_missing_city_caught():
    ts = pd.Timestamp("2018-01-01")
    w = pd.DataFrame({"timestamp": [ts] * 4, "city": list("ABCD"), "temp_c": 10.0})
    assert check_weather(w, expected_cities=5, max_null_fraction=0.02)
