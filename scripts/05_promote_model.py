"""Runs only when the challenger won: move @champion, and update the endpoint if serving is enabled."""

from pyspark.dbutils import DBUtils

from energy_forecasting.lakehouse import get_spark
from energy_forecasting.runtime import load_config, parse_args
from energy_forecasting.training import promote

args = parse_args()
cfg = load_config(args)
spark = get_spark()

version = DBUtils(spark).jobs.taskValues.get(taskKey="train_register", key="challenger_version")
promote(cfg, version)
if cfg.serving_enabled:
    from energy_forecasting.serving import deploy_champion

    deploy_champion(cfg, version)
