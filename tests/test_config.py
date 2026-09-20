import pytest

from energy_forecasting.config import FeatureSettings, ProjectConfig


def test_names_follow_environment(cfg):
    assert cfg.table("energy_features") == "mlops_dev.energy_forecasting.energy_features"
    assert cfg.model_name == "mlops_dev.energy_forecasting.energy_demand_model"
    assert cfg.experiment_name == "/Shared/energy_forecasting_dev"
    assert cfg.endpoint_name == "energy-demand-dev"


def test_each_env_gets_its_own_catalog():
    from tests.conftest import ROOT

    catalogs = {e: ProjectConfig.from_yaml(str(ROOT / "project_config.yml"), e).catalog for e in ("dev", "uat", "prod")}
    assert len(set(catalogs.values())) == 3


def test_unknown_env_rejected():
    with pytest.raises(ValueError):
        ProjectConfig.from_yaml("project_config.yml", env="staging")


def test_lag_shorter_than_24h_rejected():
    with pytest.raises(ValueError, match="day-ahead"):
        FeatureSettings(lags_hours=[1, 24], rolling_windows_hours=[24], weather_columns=["temp_c"])
