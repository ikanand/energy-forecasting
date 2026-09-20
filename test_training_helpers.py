"""Train -> log -> register -> compare with champion, using Feature Engineering in Unity Catalog.

Deploy-code pattern: this exact code runs in dev, uat and prod, each on its own
catalog's data, producing its own model versions.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
import pandas as pd
from loguru import logger
from mlflow import MlflowClient
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from energy_forecasting.config import ProjectConfig, Tables, Tags
from energy_forecasting.decisions import mape, should_promote
from energy_forecasting.modeling import tune_and_fit

TARGET = "demand_mw"
CHAMPION, CHALLENGER = "champion", "challenger"


@dataclass
class TrainResult:
    version: str
    challenger_mape: float
    champion_mape: float | None
    tso_mape: float
    promote: bool


def _lookups(cfg: ProjectConfig) -> list:
    from databricks.feature_engineering import FeatureLookup

    return [
        FeatureLookup(
            table_name=cfg.table(Tables.FEATURES),
            feature_names=cfg.feature_columns,
            lookup_key="location_id",
            timestamp_lookup_key="timestamp",  # point-in-time join on the time-series key
        )
    ]


def version_for_run(versions: list, run_id: str) -> str:
    """Pick the model version created by this run.

    Unity Catalog only allows searching versions by name, so we filter by run_id here.
    If MLflow 3 registered from a LoggedModel without a run_id, fall back to the newest version.
    """
    if not versions:
        raise RuntimeError("No model versions found right after registration.")
    matching = [v for v in versions if getattr(v, "run_id", None) == run_id]
    candidates = matching or versions
    return str(max(candidates, key=lambda v: int(v.version)).version)


def champion_version(cfg: ProjectConfig) -> str | None:
    try:
        return MlflowClient().get_model_version_by_alias(cfg.model_name, CHAMPION).version
    except Exception:
        return None


def score_keys(spark: SparkSession, model_uri: str, keys) -> pd.DataFrame:
    """Score (location_id, timestamp) keys with a feature-aware model; features are looked up automatically."""
    from databricks.feature_engineering import FeatureEngineeringClient

    return FeatureEngineeringClient().score_batch(model_uri=model_uri, df=keys).toPandas()


def train_and_register(spark: SparkSession, cfg: ProjectConfig, tags: Tags) -> TrainResult:
    from databricks.feature_engineering import FeatureEngineeringClient

    fe = FeatureEngineeringClient()
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(cfg.experiment_name)

    labels = spark.table(cfg.table(Tables.SILVER_ENERGY)).where(F.col(TARGET).isNotNull())
    labels = labels.select(F.lit(cfg.location_id).alias("location_id"), "timestamp", TARGET, "tso_forecast_mw")
    cutoff = labels.agg(F.max("timestamp")).first()[0] - pd.Timedelta(days=cfg.training.holdout_days)
    train_labels = labels.where(F.col("timestamp") <= cutoff).drop("tso_forecast_mw")
    holdout = labels.where(F.col("timestamp") > cutoff)

    # Two views of the same training set: one keeps timestamps (for time-ordered tuning),
    # one excludes keys so the logged model's inputs are exactly the feature columns.
    with_keys = fe.create_training_set(df=train_labels, feature_lookups=_lookups(cfg), label=TARGET)
    for_logging = fe.create_training_set(
        df=train_labels, feature_lookups=_lookups(cfg), label=TARGET, exclude_columns=["location_id", "timestamp"]
    )
    train_pdf = with_keys.load_df().toPandas().dropna(subset=[f"lag_{max(cfg.features.lags_hours)}h"])
    logger.info(f"Training rows: {len(train_pdf)} (up to {cutoff}); holdout: last {cfg.training.holdout_days} days")

    run_name = f"train_{pd.Timestamp.now(tz='UTC'):%Y-%m-%d_%H%M}"
    with mlflow.start_run(run_name=run_name, tags=tags.to_dict()) as run:
        mlflow.log_params({"cutoff": str(cutoff), "train_rows": len(train_pdf), **cfg.training.search_space})

        def log_trial(params: dict, score: float) -> None:
            with mlflow.start_run(nested=True, run_name="trial"):
                mlflow.log_params(params)
                mlflow.log_metric("val_mape", score)

        model, best_params, val_mape = tune_and_fit(
            train_pdf, cfg.feature_columns, TARGET, cfg.training, on_trial=log_trial
        )
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("val_mape", val_mape)

        fe.log_model(
            model=model,
            artifact_path="model",
            flavor=mlflow.lightgbm,
            training_set=for_logging,
            registered_model_name=cfg.model_name,
        )

    client = MlflowClient()
    version = version_for_run(client.search_model_versions(f"name='{cfg.model_name}'"), run.info.run_id)

    # Score challenger and champion on the same unseen holdout, the same way production scores.
    keys = holdout.select("location_id", "timestamp")
    actual = holdout.select("timestamp", TARGET, "tso_forecast_mw").toPandas()
    challenger_scored = score_keys(spark, f"models:/{cfg.model_name}/{version}", keys)
    challenger = actual.merge(challenger_scored[["timestamp", "prediction"]])
    challenger_mape = mape(challenger[TARGET], challenger["prediction"])
    tso_mape = mape(actual[TARGET], actual["tso_forecast_mw"])

    champ_version = champion_version(cfg)
    champion_mape = None
    if champ_version:
        champion_scored = score_keys(spark, f"models:/{cfg.model_name}@{CHAMPION}", keys)
        champ = actual.merge(champion_scored[["timestamp", "prediction"]])
        champion_mape = mape(champ[TARGET], champ["prediction"])

    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.log_metrics({"holdout_mape": challenger_mape, "tso_holdout_mape": tso_mape})
        if champion_mape is not None:
            mlflow.log_metric("champion_holdout_mape", champion_mape)

    client.set_registered_model_alias(cfg.model_name, CHALLENGER, version)
    version_tags = {**tags.to_dict(), "holdout_mape": f"{challenger_mape:.4f}", "tso_holdout_mape": f"{tso_mape:.4f}"}
    for k, v in version_tags.items():
        client.set_model_version_tag(cfg.model_name, version, k, v)

    go_live = should_promote(challenger_mape, champion_mape, cfg.min_improvement_pct)
    logger.info(
        f"v{version}: holdout MAPE {challenger_mape:.2f}% | champion {champion_mape} | "
        f"grid operator's own forecast {tso_mape:.2f}% | promote={go_live}"
    )
    return TrainResult(version, challenger_mape, champion_mape, tso_mape, go_live)


def promote(cfg: ProjectConfig, version: str) -> None:
    client = MlflowClient()
    client.set_registered_model_alias(cfg.model_name, CHAMPION, version)
    logger.info(f"{cfg.model_name} v{version} is now @{CHAMPION}")


def champion_baseline_mape(cfg: ProjectConfig) -> float | None:
    """The champion's holdout MAPE at training time: the yardstick for 'has it degraded?'."""
    try:
        mv = MlflowClient().get_model_version_by_alias(cfg.model_name, CHAMPION)
        return float(mv.tags["holdout_mape"])
    except Exception:
        return None
