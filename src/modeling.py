"""Walk-forward evaluation: tuned CatBoost vs. GARCH(1,1), HAR-RV, EWMA and persistence.

Every model is evaluated on the identical `TimeSeriesSplit` folds, so every validation fold
is chronologically later than its training fold for all of them. Besides the per-fold
metrics, the out-of-sample predictions of every fold are kept (`oos`), so the README's
pooled metrics and Diebold-Mariano tests run on exactly the forecasts scored here.

CatBoost is re-tuned inside every fold on that fold's training rows only (see
`src/tuning.py`): the validation fold never chooses its hyperparameters or its number of
trees.

Two variants measure what the macro inputs are worth out of sample, instead of reading it
off SHAP: CatBoost without the VIX and dollar-index features, and HAR-RV with two of them.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import TimeSeriesSplit

from src.baselines import ewma_forecast, garch_predict_fold, har_rv_predict_fold, naive_predict_fold
from src.features import HAR_COLUMNS
from src.metrics import mae, qlike, rmse
from src.tuning import EARLY_STOPPING_ROUNDS, split_for_early_stopping, tune_catboost

SEED = 42
N_CV_SPLITS = 5
MODELS = ("catboost", "catboost_no_macro", "garch", "har_rv", "har_rv_macro", "ewma", "naive")
HAR_MACRO_COLUMNS = ["vix_level", "dxy_change_std_20d"]  # fixed before looking at any result
N_TRIALS_PER_FOLD = 30


def fit_catboost_on_fold(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    feature_cols: list[str],
    params: dict | None = None,
    n_trials: int = N_TRIALS_PER_FOLD,
    seed: int = SEED,
) -> tuple[CatBoostRegressor, dict]:
    """Tunes (unless ``params`` is given) and fits CatBoost using ``train_idx`` only: the
    last part of the training block is the holdout for Optuna and for early stopping."""
    cut = split_for_early_stopping(len(train_idx))
    core, holdout = train_idx[:cut], train_idx[cut:]
    if params is None:
        study = tune_catboost(X[core], y[core], X[holdout], y[holdout], feature_cols, n_trials=n_trials, seed=seed)
        params = study.best_params
    model = CatBoostRegressor(loss_function="RMSE", random_seed=seed, verbose=False, **params)
    model.fit(
        Pool(X[core], y[core], feature_names=feature_cols),
        eval_set=Pool(X[holdout], y[holdout], feature_names=feature_cols),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    return model, params


def run_walk_forward_comparison(
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str,
    catboost_params: dict | None,
    horizon: int,
    n_splits: int = N_CV_SPLITS,
    seed: int = SEED,
    n_trials: int = N_TRIALS_PER_FOLD,
) -> dict:
    """``catboost_params=None`` re-tunes CatBoost inside every fold (what the README
    reports); fixed params skip the search, which keeps the unit tests fast."""
    X = df.select(feature_cols).to_numpy()
    y = df.select(target_col).to_numpy().ravel()
    log_returns = df.select("log_return").to_numpy().ravel()
    ewma_all = ewma_forecast(log_returns)
    no_macro_cols = [c for c in feature_cols if not c.startswith(("vix_", "dxy_"))]
    X_no_macro = df.select(no_macro_cols).to_numpy()

    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = {name: {"rmse": [], "mae": [], "qlike": []} for name in MODELS}
    oos_idx: list[np.ndarray] = []
    oos_pred: dict[str, list[np.ndarray]] = {name: [] for name in MODELS}
    fold_sizes = []
    catboost_folds: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        y_val = y[val_idx]

        cb_model, fold_params = fit_catboost_on_fold(
            X, y, train_idx, feature_cols, params=catboost_params, n_trials=n_trials, seed=seed
        )
        cb_no_macro, no_macro_params = fit_catboost_on_fold(
            X_no_macro, y, train_idx, no_macro_cols, params=catboost_params, n_trials=n_trials, seed=seed
        )
        catboost_folds.append(
            {
                "params": fold_params,
                "best_iteration": int(cb_model.get_best_iteration()),
                "no_macro_params": no_macro_params,
                "no_macro_best_iteration": int(cb_no_macro.get_best_iteration()),
            }
        )

        preds = {
            "catboost": cb_model.predict(X[val_idx]),
            "catboost_no_macro": cb_no_macro.predict(X_no_macro[val_idx]),
            "garch": garch_predict_fold(log_returns, train_idx, val_idx, horizon=horizon),
            "har_rv": har_rv_predict_fold(df, train_idx, val_idx, target_col),
            "har_rv_macro": har_rv_predict_fold(
                df, train_idx, val_idx, target_col, columns=HAR_COLUMNS + HAR_MACRO_COLUMNS
            ),
            "ewma": ewma_all[val_idx],
            "naive": naive_predict_fold(df, val_idx),
        }
        for name, p in preds.items():
            results[name]["rmse"].append(rmse(p, y_val))
            results[name]["mae"].append(mae(p, y_val))
            results[name]["qlike"].append(qlike(p, y_val))
            oos_pred[name].append(np.asarray(p, dtype=float))
        oos_idx.append(val_idx)

        fold_sizes.append((len(train_idx), len(val_idx)))
        print(
            f"[fold {fold}/{n_splits}] train={len(train_idx):>5d} val={len(val_idx):>5d} | "
            + "  ".join(f"{name} RMSE={results[name]['rmse'][-1]:.6f}" for name in MODELS)
        )

    summary = {
        name: {
            "rmse_mean": float(np.mean(vals["rmse"])),
            "rmse_std": float(np.std(vals["rmse"])),
            "mae_mean": float(np.mean(vals["mae"])),
            "mae_std": float(np.std(vals["mae"])),
            "qlike_mean": float(np.mean(vals["qlike"])),
            "qlike_std": float(np.std(vals["qlike"])),
            "rmse_per_fold": vals["rmse"],
            "mae_per_fold": vals["mae"],
            "qlike_per_fold": vals["qlike"],
        }
        for name, vals in results.items()
    }
    summary["fold_sizes"] = fold_sizes
    summary["catboost_folds"] = catboost_folds
    idx = np.concatenate(oos_idx)
    summary["oos"] = {
        "row_idx": idx,
        "actual": y[idx],
        "pred": {name: np.concatenate(parts) for name, parts in oos_pred.items()},
    }
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
    model.fit(train_pool, eval_set=eval_pool, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    return model
