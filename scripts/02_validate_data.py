"""Data quality gate. Fails the job (so nothing downstream runs) if the recent data looks broken."""

import pandas as pd
from loguru import logger

from energy_forecasting.config import Tables
from energy_forecasting.lakehouse import current_day, get_spark, read_pdf
from energy_forecasting.quality import check_energy, check_weather
from energy_forecasting.runtime import load_config, parse_args

args = parse_args([("--window_days", {"type": int, "default": 7})])
cfg = load_config(args)
spark = get_spark()

end = current_day(spark, cfg) + pd.Timedelta(days=1)
start = end - pd.Timedelta(days=args.window_days)
energy = read_pdf(spark, cfg, Tables.BRONZE_ENERGY)
weather = read_pdf(spark, cfg, Tables.BRONZE_WEATHER)
energy = energy[(energy.timestamp >= start) & (energy.timestamp < end)]
weather = weather[(weather.timestamp >= start) & (weather.timestamp < end)]

q = cfg.quality
failures = check_energy(energy, q["max_null_fraction"], q["demand_min_mw"], q["demand_max_mw"])
failures += check_weather(weather, int(q["expected_cities"]), q["max_null_fraction"])
if failures:
    for f in failures:
        logger.error(f)
    raise SystemExit(f"Data quality gate failed ({len(failures)} issue(s)) for {start.date()}..{end.date()}")
logger.info(f"Data quality OK for {start.date()}..{end.date()}")
