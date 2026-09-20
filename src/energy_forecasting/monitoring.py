"""Join forecasts with actuals, compute daily accuracy, refresh Lakehouse Monitoring, decide on retraining."""

from __future__ import annotations

import pandas as pd
from loguru import logger
from pyspark.sql import SparkSession

from energy_forecasting.config import ProjectConfig, Tables
from energy_forecasting.decisions import mape, needs_retrain
from energy_forecasting.lakehouse import read_pdf, write_pdf
from energy_forecasting.training import champion_baseline_mape


def backfill_actuals(spark: SparkSession, cfg: ProjectConfig) -> pd.DataFrame:
    """Forecasts whose hours have now happened, joined with what actually happened."""
    fc = read_pdf(spark, cfg, Tables.FORECAST)
    actual = read_pdf(spark, cfg, Tables.SILVER_ENERGY).rename(columns={"timestamp": "target_hour"})
    joined = fc.merge(actual, on="target_hour", how="inner").dropna(subset=["demand_mw"])
    joined = joined.rename(columns={"demand_mw": "actual_mw"})
    if joined.empty:
        # First daily run: tomorrow's forecast exists but no forecast hour has happened yet.
        logger.info("No forecast hours have actuals yet - nothing to score today.")
        return pd.DataFrame(columns=["forecast_date", "mape", "tso_mape", "hours", "model_version"])

    joined["abs_pct_error"] = (joined["predicted_mw"] - joined["actual_mw"]).abs() / joined["actual_mw"] * 100
    write_pdf(spark, cfg, joined, Tables.FORECAST_VS_ACTUAL)
    spark.sql(
        f"ALTER TABLE {cfg.table(Tables.FORECAST_VS_ACTUAL)} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
    )

    daily = daily_metrics(joined)
    write_pdf(spark, cfg, daily, Tables.DAILY_METRICS)
    logger.info(f"{len(joined)} forecast hours now have actuals across {len(daily)} day(s)")
    return daily


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


def refresh_lakehouse_monitor(cfg: ProjectConfig, has_data: bool = True) -> bool:
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
            w.quality_monitors.get(table)
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
                    timestamp_col="target_hour",
                    model_id_col="model_version",
                    granularities=["1 day"],
                ),
            )
            logger.info("Lakehouse Monitoring created (drift + accuracy dashboards appear in the table's Quality tab)")
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
