import pandas as pd

from energy_forecasting.data_prep import aggregate_weather, standardize_energy, standardize_weather


def test_energy_offsets_converted_to_utc():
    raw = pd.DataFrame(
        {
            "time": ["2015-01-01 01:00:00+01:00", "2015-07-01 02:00:00+02:00"],
            "total load actual": [25000, 30000],
            "total load forecast": [24800, 30500],
        }
    )
    out = standardize_energy(raw)
    assert list(out.timestamp) == [pd.Timestamp("2015-01-01 00:00"), pd.Timestamp("2015-07-01 00:00")]
    assert out.timestamp.dt.tz is None
    assert list(out.columns) == ["timestamp", "demand_mw", "tso_forecast_mw"]


def test_weather_kelvin_city_names_and_duplicates():
    raw = pd.DataFrame(
        {
            "dt_iso": ["2015-01-01 00:00:00+01:00"] * 3,
            "city_name": [" Barcelona", " Barcelona", "Madrid"],
            "temp": [283.15, 285.15, 273.15],
            "humidity": [50, 70, 80],
            "wind_speed": [1, 3, 2],
            "clouds_all": [0, 20, 40],
        }
    )
    out = standardize_weather(raw)
    assert set(out.city) == {"Barcelona", "Madrid"}
    bcn = out[out.city == "Barcelona"].iloc[0]
    assert bcn.temp_c == 11.0  # duplicates averaged: (10 + 12) / 2
    national = aggregate_weather(out)
    assert len(national) == 1 and national.temp_c.iloc[0] == 5.5
