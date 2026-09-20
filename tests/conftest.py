from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_forecasting.config import ProjectConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg() -> ProjectConfig:
    return ProjectConfig.from_yaml(str(ROOT / "project_config.yml"), env="dev")


@pytest.fixture
def hourly_energy() -> pd.DataFrame:
    """60 days of synthetic hourly demand with a daily cycle; value encodes its own hour index."""
    ts = pd.date_range("2017-01-01", periods=24 * 60, freq="h")
    idx = np.arange(len(ts))
    demand = 28000 + 4000 * np.sin(2 * np.pi * (idx % 24) / 24) + idx * 0.01
    return pd.DataFrame({"timestamp": ts, "demand_mw": demand})


@pytest.fixture
def hourly_weather(hourly_energy) -> pd.DataFrame:
    ts = pd.date_range(hourly_energy.timestamp.min(), periods=len(hourly_energy) + 24, freq="h")
    return pd.DataFrame({"timestamp": ts, "temp_c": 15.0, "humidity": 60.0, "wind_speed": 3.0, "clouds_all": 20.0})
