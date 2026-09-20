"""Model fitting and tuning (pure pandas + LightGBM + Optuna, no Databricks calls).

Splits are always by time. A random split would put future hours in training
and make the model look better than it is.
"""

from __future__ import annotations

from collections.abc import Callable

import optuna
import pandas as pd
from lightgbm import LGBMRegressor

from energy_forecasting.config import TrainingSettings
from energy_forecasting.decisions import mape


def time_split(df: pd.DataFrame, holdout_days: int, ts_col: str = "timestamp") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Last `holdout_days` of data -> holdout; everything earlier -> train."""
    df = df.sort_values(ts_col)
    cutoff = df[ts_col].max() - pd.Timedelta(days=holdout_days)
    return df[df[ts_col] <= cutoff], df[df[ts_col] > cutoff]


def tune_and_fit(
    train: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    settings: TrainingSettings,
    on_trial: Callable[[dict, float], None] | None = None,
) -> tuple[LGBMRegressor, dict, float]:
    """Tune on the last `validation_days` of `train`, then refit the best params on all of `train`.

    Returns (fitted_model, best_params, best_validation_mape).
    """
    fit_part, val_part = time_split(train, settings.validation_days)
    space = settings.search_space

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", *map(int, space["n_estimators"])),
            "num_leaves": trial.suggest_int("num_leaves", *map(int, space["num_leaves"])),
            "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", *map(int, space["min_child_samples"])),
        }
        model = _make_model(params, settings.random_state)
        model.fit(fit_part[feature_cols], fit_part[target])
        score = mape(val_part[target], model.predict(val_part[feature_cols]))
        if on_trial:
            on_trial(params, score)
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=settings.random_state))
    study.optimize(objective, n_trials=settings.n_trials)

    best_model = _make_model(study.best_params, settings.random_state)
    best_model.fit(train[feature_cols], train[target])
    return best_model, study.best_params, study.best_value


def _make_model(params: dict, random_state: int) -> LGBMRegressor:
    return LGBMRegressor(**params, random_state=random_state, verbose=-1)
