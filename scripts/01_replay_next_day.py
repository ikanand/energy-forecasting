"""Simulated live feed: release the next day of history into bronze."""

from energy_forecasting.lakehouse import get_spark, release_next_day
from energy_forecasting.runtime import load_config, parse_args

args = parse_args()
cfg = load_config(args)
if not cfg.replay_enabled:
    raise SystemExit("replay.enabled is false - a real ingestion job should feed bronze instead.")
release_next_day(get_spark(), cfg)
