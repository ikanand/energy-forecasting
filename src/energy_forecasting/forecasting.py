"""Daily day-ahead forecast: 24 hourly values for tomorrow, scored with @champion."""

from __future__ import annotations

import pandas as pd
from loguru import logger
from pyspark.sql import SparkSession

from energy_forecasting.config import ProjectConfig, Tables
from energy_forecasting.features import forecast_hours
from energy_forecasting.lakehouse import current_day
from energy_forecasting.training import CHAMPION, champion_version, score_keys


def forecast_next_day(spark: SparkSession, cfg: ProjectConfig) -> pd.DataFrame:
    version = champion_version(cfg)
    if version is None:
        raise RuntimeError(f"No @{CHAMPION} for {cfg.model_name}. Run the training workflow first.")

    today = current_day(spark, cfg)
    target_day = today + pd.Timedelta(days=1)
    keys = pd.DataFrame({"location_id": cfg.location_id, "timestamp": forecast_hours(target_day)})
    scored = score_keys(spark, f"models:/{cfg.model_name}@{CHAMPION}", spark.createDataFrame(keys))

    out = pd.DataFrame(
        {
            "forecast_date": target_day.date(),
            "target_hour": scored["timestamp"],
            "predicted_mw": scored["prediction"].astype(float),
            "forecast_created_at": target_day,  # issued when day D closes (simulated clock)
            "model_version": str(version),
        }
    ).sort_values("target_hour")

    table = cfg.table(Tables.FORECAST)
    if spark.catalog.tableExists(table):
        spark.sql(f"DELETE FROM {table} WHERE forecast_date = '{target_day.date()}'")  # idempotent re-runs
    spark.createDataFrame(out).write.mode("append").saveAsTable(table)
    logger.info(f"Forecast for {target_day.date()} written with model v{version}: peak {out.predicted_mw.max():.0f} MW")
    return out
