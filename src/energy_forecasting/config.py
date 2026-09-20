"""Typed project configuration loaded from project_config.yml."""

from __future__ import annotations

from typing import Any

import yaml
from pydantic import BaseModel, model_validator

VALID_ENVS = ("dev", "uat", "prod")


class FeatureSettings(BaseModel):
    min_lag_hours: int = 24
    lags_hours: list[int]
    rolling_windows_hours: list[int]
    weather_columns: list[str]

    @model_validator(mode="after")
    def _day_ahead_safe(self) -> FeatureSettings:
        too_short = [h for h in self.lags_hours if h < self.min_lag_hours]
        if too_short:
            raise ValueError(
                f"Lags {too_short} are shorter than min_lag_hours={self.min_lag_hours}; "
                "they would not be known when a day-ahead forecast is made."
            )
        return self


class TrainingSettings(BaseModel):
    holdout_days: int
    validation_days: int
    n_trials: int
    random_state: int
    search_space: dict[str, list[float]]


class ProjectConfig(BaseModel):
    env: str
    catalog: str
    schema_name: str
    location_id: str
    timezone: str
    model_short_name: str
    experiment_name: str
    endpoint_name: str
    volume: str
    energy_file: str
    weather_file: str
    replay_enabled: bool
    replay_start_date: str
    features: FeatureSettings
    training: TrainingSettings
    min_improvement_pct: float
    degradation_ratio: float
    consecutive_days: int
    quality: dict[str, float]
    serving_enabled: bool

    @classmethod
    def from_yaml(cls, path: str, env: str) -> ProjectConfig:
        if env not in VALID_ENVS:
            raise ValueError(f"Invalid env '{env}'. Expected one of {VALID_ENVS}.")
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)
        return cls.from_dict(raw, env)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], env: str) -> ProjectConfig:
        e, p = raw["environments"][env], raw["project"]
        return cls(
            env=env,
            catalog=e["catalog"],
            schema_name=e["schema"],
            location_id=p["location_id"],
            timezone=p["timezone"],
            model_short_name=p["model_name"],
            experiment_name=p["experiment_name"].format(env=env),
            endpoint_name=p["endpoint_name"].format(env=env),
            volume=raw["data"]["volume"],
            energy_file=raw["data"]["energy_file"],
            weather_file=raw["data"]["weather_file"],
            replay_enabled=raw["replay"]["enabled"],
            replay_start_date=raw["replay"]["start_date"],
            features=FeatureSettings(**raw["features"]),
            training=TrainingSettings(**raw["training"]),
            min_improvement_pct=raw["promotion"]["min_improvement_pct"],
            degradation_ratio=raw["monitoring"]["degradation_ratio"],
            consecutive_days=raw["monitoring"]["consecutive_days"],
            quality=raw["quality"],
            serving_enabled=raw["serving"]["enabled"],
        )

    # ---- Derived names (one place, so naming stays consistent) -------------
    def table(self, name: str) -> str:
        return f"{self.catalog}.{self.schema_name}.{name}"

    @property
    def model_name(self) -> str:
        return self.table(self.model_short_name)

    @property
    def volume_path(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schema_name}/{self.volume}"

    @property
    def feature_columns(self) -> list[str]:
        f = self.features
        return (
            [f"lag_{h}h" for h in f.lags_hours]
            + [f"rolling_mean_{w}h" for w in f.rolling_windows_hours]
            + list(f.weather_columns)
            + CALENDAR_COLUMNS
        )


CALENDAR_COLUMNS = ["hour_of_day", "day_of_week", "month", "is_weekend", "is_holiday"]


class Tables:
    """Table short names. Layer prefix = medallion layer; gold tables are self-describing."""

    SOURCE_ENERGY = "source_energy_history"  # simulated source system (full Kaggle history)
    SOURCE_WEATHER = "source_weather_history"
    REPLAY_STATE = "replay_state"
    BRONZE_ENERGY = "bronze_energy"  # data "released" so far
    BRONZE_WEATHER = "bronze_weather"
    SILVER_ENERGY = "silver_energy"  # cleaned actuals = labels
    SILVER_WEATHER = "silver_weather"  # city-averaged hourly weather
    FEATURES = "energy_features"  # Feature Engineering table (UC)
    FORECAST = "demand_forecast"  # every forecast ever made
    FORECAST_VS_ACTUAL = "forecast_vs_actual"  # monitored table
    DAILY_METRICS = "forecast_daily_metrics"


class Tags(BaseModel):
    """Lineage tags attached to every MLflow run and model version."""

    git_sha: str
    branch: str
    job_run_id: str

    def to_dict(self) -> dict[str, str]:
        return self.model_dump()
