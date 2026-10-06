"""Optuna hyperparameter search for the CatBoost volatility model.

The search never sees the data a model is scored on. Inside the walk-forward comparison
(`src/modeling.py`) it runs once per fold, on that fold's training rows only: the last
``EARLY_STOP_FRACTION`` of them is the holdout that both Optuna and CatBoost's early
stopping look at, and the validation fold stays untouched until the final score. An
earlier version tuned once on the last 20% of the whole series and early-stopped on the
validation fold itself, which let the scored data choose the model; that is gone.

`run_optuna_search` (one search on the full series) is only used for the descriptive
SHAP model, which is not scored.
"""

from __future__ import annotations

import numpy as np
import optuna
import polars as pl
from catboost import CatBoostRegressor, Pool

SEED = 42
EARLY_STOP_FRACTION = 0.2
EARLY_STOPPING_ROUNDS = 50


def split_for_early_stopping(n_rows: int, fraction: float = EARLY_STOP_FRACTION) -> int:
    """Index where the chronological holdout of a training block starts."""
    return int(n_rows * (1 - fraction))


def tune_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    feature_cols: list[str],
    n_trials: int = 30,
    seed: int = SEED,
) -> optuna.Study:
    train_pool = Pool(X_train, y_train, feature_names=feature_cols)
    holdout_pool = Pool(X_holdout, y_holdout, feature_names=feature_cols)

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
        model.fit(train_pool, eval_set=holdout_pool, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        preds = model.predict(X_holdout)
        return float(np.sqrt(np.mean((preds - y_holdout) ** 2)))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def run_optuna_search(
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str,
    n_trials: int = 30,
    val_fraction: float = EARLY_STOP_FRACTION,
    seed: int = SEED,
) -> optuna.Study:
    X = df.select(feature_cols).to_numpy()
    y = df.select(target_col).to_numpy().ravel()
    split = split_for_early_stopping(len(X), val_fraction)
    return tune_catboost(X[:split], y[:split], X[split:], y[split:], feature_cols, n_trials=n_trials, seed=seed)
