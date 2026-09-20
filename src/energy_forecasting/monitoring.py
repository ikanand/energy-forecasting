"""Join forecasts with actuals, compute daily accuracy, refresh Lakehouse Monitoring, decide on retraining.

forecast_vs_actual is an append-only *inference log*: each daily run appends the
forecast days that just got their actuals, stamped with the real wall-clock time
they were scored (`scored_at`). Lakehouse Monitoring windows on `scored_at`, so it
works even though the replayed data itself is from 2018 (monitors only analyse
rows whose timestamp is within the last 30 days when they are created).
"""

from __future__ import annotations

import pandas as pd
from loguru import logger
from pyspark.sql import SparkSession

from energy_forecasting.config import ProjectConfig, Tables
from energy_forecasting.decisions import mape, needs_retrain
from energy_forecasting.lakehouse import read_pdf, write_pdf
from energy_forecasting.training import champion_baseline_mape

MONITOR_TIMESTAMP_COL = "scored_at"
MONITOR_GRANULARITIES = ["5 minutes", "1 day"]  # 5 min: back-to-back demo runs; 1 day: real daily schedule


def rows_to_log(joined: pd.DataFrame, already_logged_days: set, scored_at: pd.Timestamp) -> pd.DataFrame:
    """Forecast days that now have actuals and are not in the inference log yet, stamped with scored_at."""
    new = joined[~joined["forecast_date"].isin(already_logged_days)].copy()
    new[MONITOR_TIMESTAMP_COL] = scored_at
    return new


def backfill_actuals(spark: SparkSession, cfg: ProjectConfig) -> pd.DataFrame:
    """Append newly-scorable forecast days to the inference log and return daily accuracy for all days."""
    fc = read_pdf(spark, cfg, Tables.FORECAST)
    actual = read_pdf(spark, cfg, Tables.SILVER_ENERGY).rename(columns={"timestamp": "target_hour"})
    joined = fc.merge(actual, on="target_hour", how="inner").dropna(subset=["demand_mw"])
    joined = joined.rename(columns={"demand_mw": "actual_mw"})
    joined["abs_pct_error"] = (joined["predicted_mw"] - joined["actual_mw"]).abs() / joined["actual_mw"] * 100

    table = cfg.table(Tables.FORECAST_VS_ACTUAL)
    existing = _existing_log(spark, cfg)
    logged_days = set(existing["forecast_date"]) if not existing.empty else set()

    new = rows_to_log(joined, logged_days, pd.Timestamp.now(tz="UTC").tz_localize(None))
    if not new.empty:
        write_pdf(spark, cfg, new, Tables.FORECAST_VS_ACTUAL, mode="append")
        spark.sql(f"ALTER TABLE {table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
        logger.info(f"Logged {len(new)} newly scored forecast hours ({sorted(set(new['forecast_date']))})")

    log = pd.concat([existing, new], ignore_index=True) if not existing.empty else new
    if log.empty:
        logger.info("No forecast hours have actuals yet - nothing to score today.")
        return pd.DataFrame(columns=["forecast_date", "mape", "tso_mape", "hours", "model_version"])

    daily = daily_metrics(log)
    write_pdf(spark, cfg, daily, Tables.DAILY_METRICS)
    logger.info(f"{len(log)} forecast hours now have actuals across {len(daily)} day(s)")
    return daily


def _existing_log(spark: SparkSession, cfg: ProjectConfig) -> pd.DataFrame:
    """Current inference log. A table from the old (overwrite, no scored_at) layout is dropped and rebuilt."""
    table = cfg.table(Tables.FORECAST_VS_ACTUAL)
    if not spark.catalog.tableExists(table):
        return pd.DataFrame()
    if MONITOR_TIMESTAMP_COL not in spark.table(table).columns:
        logger.warning(f"{table} has the old layout without {MONITOR_TIMESTAMP_COL}; recreating it as an append log.")
        _drop_monitor_and_outputs(spark, cfg)
        spark.sql(f"DROP TABLE IF EXISTS {table}")
        return pd.DataFrame()
    return spark.table(table).toPandas()


def daily_metrics(joined: pd.DataFrame) -> pd.DataFrame:
    """One row per forecast day: our MAPE, the grid operator's MAPE, hours scored, model version."""
    rows = [
        {
            "forecast_date": day,
            "mape": mape(g["actual_mw"], g["predicted_mw"]),
            "tso_mape": mape(g["actual_mw"], g["tso_forecast_mw"]),
            "hours": len(g),
            "model_version": str(g["model_version"].iloc[-1]),
        }
        for day, g in joined.groupby("forecast_date")
    ]
    return pd.DataFrame(rows).sort_values("forecast_date").reset_index(drop=True)


def _drop_monitor_and_outputs(spark: SparkSession, cfg: ProjectConfig) -> None:
    table = cfg.table(Tables.FORECAST_VS_ACTUAL)
    try:
        from databricks.sdk import WorkspaceClient

        WorkspaceClient().quality_monitors.delete(table_name=table)
        logger.info(f"Deleted old monitor on {table}")
    except Exception as e:
        logger.info(f"No old monitor to delete ({e})")
    for suffix in ("_profile_metrics", "_drift_metrics"):
        spark.sql(f"DROP TABLE IF EXISTS {table}{suffix}")


def refresh_lakehouse_monitor(spark: SparkSession, cfg: ProjectConfig, has_data: bool = True) -> bool:
    """Create the monitor on first run, refresh afterwards. Returns False if the tier doesn't allow it."""
    if not has_data:
        logger.info("Skipping Lakehouse Monitoring: forecast_vs_actual has no rows yet.")
        return False
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.errors import NotFound
        from databricks.sdk.service.catalog import MonitorInferenceLog, MonitorInferenceLogProblemType

        w = WorkspaceClient()
        table = cfg.table(Tables.FORECAST_VS_ACTUAL)
        try:
            info = w.quality_monitors.get(table)
            if info.inference_log and info.inference_log.timestamp_col != MONITOR_TIMESTAMP_COL:
                logger.info("Existing monitor windows on the wrong column; recreating it on scored_at.")
                _drop_monitor_and_outputs(spark, cfg)
                raise NotFound("recreate")
            w.quality_monitors.run_refresh(table_name=table)
            logger.info("Lakehouse Monitoring refreshed")
        except NotFound:
            w.quality_monitors.create(
                table_name=table,
                assets_dir=f"/Workspace/Shared/lakehouse_monitoring/{table}",
                output_schema_name=f"{cfg.catalog}.{cfg.schema_name}",
                inference_log=MonitorInferenceLog(
                    problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_REGRESSION,
                    prediction_col="predicted_mw",
                    label_col="actual_mw",
                    timestamp_col=MONITOR_TIMESTAMP_COL,
                    model_id_col="model_version",
                    granularities=MONITOR_GRANULARITIES,
                ),
            )
            logger.info("Lakehouse Monitoring created (dashboard: table -> Quality tab -> View dashboard)")
        return True
    except Exception as e:  # trial tiers may not include it; daily metrics table still works
        logger.warning(f"Lakehouse Monitoring unavailable, using forecast_daily_metrics only: {e}")
        return False


def retrain_needed(cfg: ProjectConfig, daily: pd.DataFrame) -> bool:
    baseline = champion_baseline_mape(cfg)
    if baseline is None or daily.empty:
        return False
    decision = needs_retrain(daily["mape"], baseline, cfg.degradation_ratio, cfg.consecutive_days)
    logger.info(
        f"Baseline MAPE {baseline:.2f}%, last {cfg.consecutive_days} days: "
        f"{daily['mape'].tail(cfg.consecutive_days).round(2).tolist()} -> retrain={decision}"
    )
    return decision
