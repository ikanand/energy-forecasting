"""Unity Catalog I/O: setup, replay, silver tables, feature table.

Everything that touches Spark lives here (and in training/forecasting/monitoring),
so the pure logic modules stay importable in CI without Spark.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger
from pyspark.sql import SparkSession

from energy_forecasting.config import ProjectConfig, Tables
from energy_forecasting.data_prep import aggregate_weather, clean_energy, standardize_energy, standardize_weather
from energy_forecasting.features import build_features


def get_spark() -> SparkSession:
    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")  # naive timestamps are UTC everywhere
    return spark


def read_pdf(spark: SparkSession, cfg: ProjectConfig, table: str) -> pd.DataFrame:
    return spark.table(cfg.table(table)).toPandas()


def write_pdf(spark: SparkSession, cfg: ProjectConfig, pdf: pd.DataFrame, table: str, mode: str = "overwrite") -> None:
    spark.createDataFrame(pdf).write.mode(mode).option("overwriteSchema", "true").saveAsTable(cfg.table(table))


# ---------------------------------------------------------------- setup ----
def ensure_schema_and_volume(spark: SparkSession, cfg: ProjectConfig) -> None:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.catalog}.{cfg.schema_name}")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {cfg.catalog}.{cfg.schema_name}.{cfg.volume}")


def load_source_history(spark: SparkSession, cfg: ProjectConfig) -> None:
    """Kaggle CSVs in the Volume -> standardized 'source system' tables (the replay feed)."""
    energy = standardize_energy(pd.read_csv(f"{cfg.volume_path}/{cfg.energy_file}"))
    weather = standardize_weather(pd.read_csv(f"{cfg.volume_path}/{cfg.weather_file}"))
    write_pdf(spark, cfg, energy, Tables.SOURCE_ENERGY)
    write_pdf(spark, cfg, weather, Tables.SOURCE_WEATHER)
    logger.info(f"Source history loaded: {len(energy)} energy rows, {len(weather)} weather rows")


def init_replay(spark: SparkSession, cfg: ProjectConfig) -> pd.Timestamp:
    """Reset bronze to 'everything before replay start'. Returns the last released day.

    Weather is released one day ahead of demand: at forecast time we pretend the
    next day's weather forecast is known. (Simplification: we use the actual
    weather as a perfect forecast; a production system would store forecasts.)
    """
    start = pd.Timestamp(cfg.replay_start_date)
    energy = read_pdf(spark, cfg, Tables.SOURCE_ENERGY)
    weather = read_pdf(spark, cfg, Tables.SOURCE_WEATHER)
    write_pdf(spark, cfg, energy[energy["timestamp"] < start], Tables.BRONZE_ENERGY)
    write_pdf(spark, cfg, weather[weather["timestamp"] < start + pd.Timedelta(days=1)], Tables.BRONZE_WEATHER)
    last_day = start - pd.Timedelta(days=1)
    _write_state(spark, cfg, last_day)
    logger.info(f"Replay initialised. Last released day: {last_day.date()}")
    return last_day


def release_next_day(spark: SparkSession, cfg: ProjectConfig) -> pd.Timestamp:
    """Append the next day of demand (and the day after's weather) to bronze. Returns the newly released day D."""
    day = current_day(spark, cfg) + pd.Timedelta(days=1)
    next_day = day + pd.Timedelta(days=1)
    src_e = spark.table(cfg.table(Tables.SOURCE_ENERGY))
    src_w = spark.table(cfg.table(Tables.SOURCE_WEATHER))
    new_e = src_e.where((src_e.timestamp >= day) & (src_e.timestamp < next_day))
    new_w = src_w.where((src_w.timestamp >= next_day) & (src_w.timestamp < next_day + pd.Timedelta(days=1)))
    if new_e.count() == 0:
        raise RuntimeError(f"No source data for {day.date()} - the replay has reached the end of the dataset.")
    # Idempotent: delete first so a re-run of the same day does not duplicate rows.
    spark.sql(f"DELETE FROM {cfg.table(Tables.BRONZE_ENERGY)} WHERE timestamp >= '{day}' AND timestamp < '{next_day}'")
    new_e.write.mode("append").saveAsTable(cfg.table(Tables.BRONZE_ENERGY))
    spark.sql(
        f"DELETE FROM {cfg.table(Tables.BRONZE_WEATHER)} "
        f"WHERE timestamp >= '{next_day}' AND timestamp < '{next_day + pd.Timedelta(days=1)}'"
    )
    new_w.write.mode("append").saveAsTable(cfg.table(Tables.BRONZE_WEATHER))
    _write_state(spark, cfg, day)
    logger.info(f"Released {day.date()} (demand) and {next_day.date()} (weather)")
    return day


def current_day(spark: SparkSession, cfg: ProjectConfig) -> pd.Timestamp:
    """Last fully released day = 'today' in the simulation."""
    row = spark.table(cfg.table(Tables.REPLAY_STATE)).collect()[0]
    return pd.Timestamp(row["last_released_day"])


def _write_state(spark: SparkSession, cfg: ProjectConfig, day: pd.Timestamp) -> None:
    write_pdf(spark, cfg, pd.DataFrame({"last_released_day": [day]}), Tables.REPLAY_STATE)


# --------------------------------------------------------- silver + gold ----
def build_silver(spark: SparkSession, cfg: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    energy = clean_energy(read_pdf(spark, cfg, Tables.BRONZE_ENERGY))
    weather = aggregate_weather(read_pdf(spark, cfg, Tables.BRONZE_WEATHER))
    write_pdf(spark, cfg, energy, Tables.SILVER_ENERGY)
    write_pdf(spark, cfg, weather, Tables.SILVER_WEATHER)
    return energy, weather


def build_feature_table(spark: SparkSession, cfg: ProjectConfig) -> None:
    """Recompute features up to the end of tomorrow and merge into the UC feature table."""
    from databricks.feature_engineering import FeatureEngineeringClient

    energy, weather = build_silver(spark, cfg)
    today = current_day(spark, cfg)
    horizon_end = today + pd.Timedelta(days=2) - pd.Timedelta(hours=1)  # last hour of tomorrow
    pdf = build_features(
        energy[["timestamp", "demand_mw"]], weather, horizon_end, cfg.features, cfg.location_id, cfg.timezone
    )
    sdf = spark.createDataFrame(pdf)

    fe = FeatureEngineeringClient()
    name = cfg.table(Tables.FEATURES)
    if not spark.catalog.tableExists(name):
        fe.create_table(
            name=name,
            primary_keys=["location_id", "timestamp"],
            timeseries_column="timestamp",
            df=sdf,
            description="Day-ahead demand features (lags >= 24h, rolling means, weather, calendar). "
            "Built by scripts/03_build_features.py.",
        )
        spark.sql(f"ALTER TABLE {name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    else:
        fe.write_table(name=name, df=sdf, mode="merge")
    logger.info(f"Feature table {name}: {len(pdf)} rows up to {horizon_end}")
