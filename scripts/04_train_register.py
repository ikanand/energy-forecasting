"""Train a challenger, register it in Unity Catalog, compare with @champion on the same holdout."""

from energy_forecasting.lakehouse import get_spark
from energy_forecasting.runtime import load_config, parse_args, set_task_value, tags_from
from energy_forecasting.training import train_and_register

args = parse_args()
cfg = load_config(args)
spark = get_spark()

result = train_and_register(spark, cfg, tags_from(args))
set_task_value(spark, "challenger_version", result.version)
set_task_value(spark, "model_improved", 1 if result.promote else 0)
