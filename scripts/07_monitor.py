"""Backfill actuals, compute daily accuracy, refresh Lakehouse Monitoring, decide whether to retrain."""

from energy_forecasting.lakehouse import get_spark
from energy_forecasting.monitoring import backfill_actuals, refresh_lakehouse_monitor, retrain_needed
from energy_forecasting.runtime import load_config, parse_args, set_task_value

args = parse_args()
cfg = load_config(args)
spark = get_spark()

daily = backfill_actuals(spark, cfg)
refresh_lakehouse_monitor(spark, cfg, has_data=not daily.empty)
set_task_value(spark, "needs_retrain", 1 if retrain_needed(cfg, daily) else 0)
