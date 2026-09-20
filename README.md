# Energy Forecasting on Databricks

Day-ahead hourly electricity demand forecasting for Spain, built as an end-to-end
MLOps demo on Databricks: Unity Catalog, Feature Engineering, MLflow, Model
Registry aliases, Workflows, Asset Bundles CI/CD, and Lakehouse Monitoring.

**Data:** [Kaggle - Hourly energy demand, generation and weather (Spain, 2015-2018)](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather).
The history is replayed one day at a time to simulate a live feed.

## What it does every day

```
replay next day -> data quality gate -> features -> forecast tomorrow (24 hours, @champion)
      -> join past forecasts with actuals -> daily MAPE + Lakehouse Monitoring
      -> if accuracy degraded 3 days in a row: retrain -> compare -> promote only if better
```

The model is benchmarked against the grid operator's own published forecast
(`tso_forecast_mw`), which is never used as a feature.

## Layout

```
databricks.yml               bundle: dev / uat / prod targets, wheel artifact
project_config.yml           every name and setting, per environment
src/energy_forecasting/      the package (all logic lives here, unit-tested)
  config.py                  typed config + table names + lineage tags
  data_prep.py               Kaggle files -> clean UTC tables        (pure pandas)
  features.py                day-ahead-safe features                 (pure pandas)
  quality.py                 data quality rules                      (pure pandas)
  modeling.py                time split, Optuna tuning, LightGBM      (pure pandas)
  decisions.py               promote / retrain rules, MAPE           (pure)
  lakehouse.py               Unity Catalog I/O, replay, feature table (Spark)
  training.py                MLflow + Feature Engineering + registry  (Databricks)
  forecasting.py             daily 24-hour forecast                  (Databricks)
  monitoring.py              actuals backfill, metrics, monitor      (Databricks)
  serving.py                 optional real-time endpoint             (Databricks)
scripts/00..07_*.py          thin job entry points (args -> package calls)
resources/workflows.yml      setup, training, daily_forecast jobs
notebooks/                   exploration only; jobs never run notebooks
tests/                       unit tests (run in CI on every PR)
.github/workflows/           ci.yml (PR checks), cd.yml (uat -> approval -> prod)
```

## Naming

| Object | dev | uat | prod |
|---|---|---|---|
| Catalog | `mlops_dev` | `mlops_uat` | `mlops_prod` |
| Schema | `energy_forecasting` | same | same |
| Feature table | `<catalog>.energy_forecasting.energy_features` | | |
| Model | `<catalog>.energy_forecasting.energy_demand_model` (`@champion`, `@challenger`) | | |
| Experiment | `/Shared/energy_forecasting_dev` | `..._uat` | `..._prod` |
| Endpoint (optional) | `energy-demand-dev` | `energy-demand-uat` | `energy-demand-prod` |

## Local development

```bash
uv sync --extra dev --extra test     # create .venv with Python 3.12
uv run pre-commit install            # lint + format on every commit
uv run pytest                        # unit tests
uv build --wheel                     # what the bundle deploys
```

Put the Kaggle CSVs in `data/` (git-ignored) to run `notebooks/01_explore_data.py`.

## Known simplifications (deliberate, for a demo)

- **Perfect weather forecast:** the replay uses actual weather for tomorrow as if
  it were the forecast. A production system would store real weather forecasts.
- **Forecast issued at midnight UTC:** real day-ahead markets close around noon
  the day before; that would need lags of 36h+ instead of 24h.
- **pandas for features:** fine at 35k rows; the same logic moves to Spark at scale.
