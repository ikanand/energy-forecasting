from energy_forecasting.features import build_features
from energy_forecasting.modeling import time_split, tune_and_fit


def test_time_split_is_chronological(hourly_energy):
    train, holdout = time_split(hourly_energy, holdout_days=7)
    assert train.timestamp.max() < holdout.timestamp.min()
    assert len(holdout) == 7 * 24


def test_tune_and_fit_learns_the_daily_cycle(cfg, hourly_energy, hourly_weather):
    horizon_end = hourly_energy.timestamp.max()
    feats = build_features(hourly_energy, hourly_weather, horizon_end, cfg.features, "ES", cfg.timezone)
    data = feats.merge(hourly_energy, on="timestamp").dropna(subset=["lag_168h"])
    settings = cfg.training.model_copy(update={"n_trials": 3, "validation_days": 7})
    train, holdout = time_split(data, 7)

    model, params, val_mape = tune_and_fit(train, cfg.feature_columns, "demand_mw", settings)
    preds = model.predict(holdout[cfg.feature_columns])
    holdout_mape = (abs(preds - holdout.demand_mw) / holdout.demand_mw).mean() * 100
    assert set(params) == set(cfg.training.search_space)
    assert holdout_mape < 2.0  # synthetic series is very regular
