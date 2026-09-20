"""Day-ahead forecast: 24 hourly values for tomorrow using @champion."""

from energy_forecasting.forecasting import forecast_next_day
from energy_forecasting.lakehouse import get_spark
from energy_forecasting.runtime import load_config, parse_args

args = parse_args()
cfg = load_config(args)
forecast_next_day(get_spark(), cfg)
