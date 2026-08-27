"""Walk-forward evaluation: tuned CatBoost vs. GARCH(1,1) vs. HAR-RV.

All three models are evaluated on the identical `TimeSeriesSplit` folds, so
every validation fold is chronologically later than its training fold for
all three -- no future information ever informs a model evaluated on the
past, for either the ML model or the econometric baselines.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import TimeSeriesSplit

from src.baselines import garch_predict_fold, har_rv_predict_fold

SEED = 42
N_CV_SPLITS = 5


def _rmse(preds: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((preds - actual) ** 2)))


def _mae(preds: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(preds - actual)))


def run_walk_forward_comparison(
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str,
    catboost_params: dict,
    horizon: int,
    n_splits: int = N_CV_SPLITS,
    seed: int = SEED,
) -> dict:
    X = df.select(feature_cols).to_numpy()
    y = df.select(target_col).to_numpy().ravel()
    log_returns = df.select("log_return").to_numpy().ravel()

    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = {"catboost": {"rmse": [], "mae": []}, "garch": {"rmse": [], "mae": []}, "har_rv": {"rmse": [], "mae": []}}
    fold_sizes = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        y_val = y[val_idx]

        train_pool = Pool(X[train_idx], y[train_idx], feature_names=feature_cols)
        val_pool = Pool(X[val_idx], y_val, feature_names=feature_cols)
        cb_model = CatBoostRegressor(loss_function="RMSE", random_seed=seed, verbose=False, **catboost_params)
        cb_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)
        cb_preds = cb_model.predict(X[val_idx])

        garch_preds = garch_predict_fold(log_returns, train_idx, val_idx, horizon=horizon)
        har_preds = har_rv_predict_fold(df, train_idx, val_idx, target_col)

        for name, preds in [("catboost", cb_preds), ("garch", garch_preds), ("har_rv", har_preds)]:
            results[name]["rmse"].append(_rmse(preds, y_val))
            results[name]["mae"].append(_mae(preds, y_val))

        fold_sizes.append((len(train_idx), len(val_idx)))
        print(
            f"[fold {fold}/{n_splits}] train={len(train_idx):>5d} val={len(val_idx):>5d} | "
            f"CatBoost RMSE={results['catboost']['rmse'][-1]:.6f}  "
            f"GARCH RMSE={results['garch']['rmse'][-1]:.6f}  "
            f"HAR-RV RMSE={results['har_rv']['rmse'][-1]:.6f}"
        )

    summary = {
        name: {
            "rmse_mean": float(np.mean(vals["rmse"])),
            "rmse_std": float(np.std(vals["rmse"])),
            "mae_mean": float(np.mean(vals["mae"])),
            "mae_std": float(np.std(vals["mae"])),
            "rmse_per_fold": vals["rmse"],
            "mae_per_fold": vals["mae"],
        }
        for name, vals in results.items()
    }
    summary["fold_sizes"] = fold_sizes
    return summary


def fit_final_model(
    df: pl.DataFrame, feature_cols: list[str], target_col: str, catboost_params: dict, seed: int = SEED
) -> CatBoostRegressor:
    """Fits CatBoost on the full series (chronological 90/10 train/eval
    split purely for early stopping) -- this is the model used for SHAP
    explainability, not for the walk-forward leaderboard above."""
    X = df.select(feature_cols).to_numpy()
    y = df.select(target_col).to_numpy().ravel()
    split = int(len(X) * 0.9)

    train_pool = Pool(X[:split], y[:split], feature_names=feature_cols)
    eval_pool = Pool(X[split:], y[split:], feature_names=feature_cols)

    model = CatBoostRegressor(loss_function="RMSE", random_seed=seed, verbose=False, **catboost_params)
    model.fit(train_pool, eval_set=eval_pool, early_stopping_rounds=50)
    return model
