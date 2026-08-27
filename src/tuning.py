"""Optuna hyperparameter search for the CatBoost volatility model.

Tunes on a **single chronological holdout split** (last `val_fraction` of
the series), not the full 5-fold walk-forward CV -- running an Optuna trial
per fold per trial would multiply the tuning cost by `n_splits` for a
search that, in practice, mostly finds the same handful of well-generalizing
regions. This is a pragmatic, explicitly-documented choice (see the README's
methodology section): the best hyperparameters found here are then evaluated
across all 5 walk-forward folds for the final leaderboard, so the *reported*
comparison against GARCH/HAR-RV is still a full walk-forward evaluation --
only the *search* itself uses one split.
"""

from __future__ import annotations

import numpy as np
import optuna
import polars as pl
from catboost import CatBoostRegressor, Pool

SEED = 42


def run_optuna_search(
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str,
    n_trials: int = 30,
    val_fraction: float = 0.2,
    seed: int = SEED,
) -> optuna.Study:
    X = df.select(feature_cols).to_numpy()
    y = df.select(target_col).to_numpy().ravel()

    split = int(len(X) * (1 - val_fraction))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_pool = Pool(X_train, y_train, feature_names=feature_cols)
    val_pool = Pool(X_val, y_val, feature_names=feature_cols)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 3, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        }
        model = CatBoostRegressor(loss_function="RMSE", random_seed=seed, verbose=False, **params)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)
        preds = model.predict(X_val)
        return float(np.sqrt(np.mean((preds - y_val) ** 2)))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study
