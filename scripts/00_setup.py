"""One-time per environment: schema + volume, load Kaggle history, initialise the replay clock."""

from energy_forecasting.lakehouse import ensure_schema_and_volume, get_spark, init_replay, load_source_history
from energy_forecasting.runtime import load_config, parse_args

args = parse_args()
cfg = load_config(args)
spark = get_spark()

ensure_schema_and_volume(spark, cfg)
load_source_history(spark, cfg)
init_replay(spark, cfg)
