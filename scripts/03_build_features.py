"""bronze -> silver -> Feature Engineering table (through the end of tomorrow)."""

from energy_forecasting.lakehouse import build_feature_table, get_spark
from energy_forecasting.runtime import load_config, parse_args

args = parse_args()
cfg = load_config(args)
build_feature_table(get_spark(), cfg)
