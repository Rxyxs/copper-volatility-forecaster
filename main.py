"""
Copper price volatility forecaster, on real data.

Forecasts the realized volatility of the LME copper price over the next 5 trading days with
an Optuna-tuned CatBoost model, and compares it on identical walk-forward folds with
GARCH(1,1), HAR-RV (Corsi 2009), RiskMetrics EWMA, a persistence baseline and a PyTorch MLP.
Data: the daily copper price published by mindicador.cl (October 2012 onward), and the VIX
and the Federal Reserve's broad dollar index from FRED, each one used only once it had been
published.

    python main.py --download   # fetch the raw files into data/raw/ (needed once)
    python main.py              # run the analysis on the cached files
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from src.data import load_market
from src.deep_learning import run_activation_comparison
from src.explainability import compute_shap_values, global_importance_by_feature, global_importance_by_group
from src.features import TARGET_COL, VOL_TARGET_HORIZON, build_features_and_target
from src.metrics import diebold_mariano, mae, qlike, qlike_losses, rmse, squared_errors
from src.modeling import MODELS, N_CV_SPLITS, fit_final_model, run_walk_forward_comparison
from src.persistence import get_connection, persist_comparison
from src.plots import (
    PLOTS_DIR,
    plot_forecasts_last_fold,
    plot_mlp_loss_curves,
    plot_mlp_loss_curves_animated,
    plot_model_comparison,
    plot_predicted_vs_actual,
    plot_price_and_vol,
    plot_residual_distribution,
)
from src.sources import download_all
from src.tuning import run_optuna_search

ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
N_OPTUNA_TRIALS = 30


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    return value


def mincer_zarnowitz_r2(preds: np.ndarray, actual: np.ndarray) -> float:
    """R^2 of regressing the realized volatility on the forecast."""
    return float(np.corrcoef(preds, actual)[0, 1] ** 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true", help="download the raw files that are missing")
    parser.add_argument("--refresh", action="store_true", help="also re-download the files that can still change")
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if args.download or args.refresh:
        written = download_all(refresh=args.refresh)
        print(f"Downloaded {len(written)} raw files into data/raw/")

    market = load_market()
    report = market.report
    print(
        f"Copper: {report['copper_days']} daily prices, {report['copper_first']} to {report['copper_last']} "
        f"({report['duplicates_collapsed']} duplicate collapsed, holes: {len(report['holes'])}). "
        f"VIX: {report['vix_days']} days. Dollar index: {report['dxy_days']} days."
    )
    years = (market.copper["date"].max() - market.copper["date"].min()).days / 365.25
    periods_per_year = market.copper.height / years
    annualize = np.sqrt(periods_per_year) * 100.0  # daily vol -> annualized %

    df, feature_cols, feature_groups = build_features_and_target(market)
    print(f"Rows after feature/target construction: {df.height} ({df['date'].min()} to {df['date'].max()})")
    print(f"Features used ({len(feature_cols)}): {feature_cols}")

    print(f"\nWalk-forward comparison, {N_CV_SPLITS} folds, CatBoost re-tuned inside every fold...")
    comparison = run_walk_forward_comparison(df, feature_cols, TARGET_COL, None, horizon=VOL_TARGET_HORIZON)

    print(f"\nPyTorch MLP activation comparison (ReLU/GELU/Swish), {N_CV_SPLITS} folds...")
    mlp = run_activation_comparison(df, feature_cols, TARGET_COL, n_splits=N_CV_SPLITS)
    best_activation = mlp["best_activation"]

    # ---- pooled out-of-sample metrics and Diebold-Mariano tests -------------------------
    oos = comparison["oos"]
    actual = oos["actual"]
    preds = dict(oos["pred"])
    for activation, p in mlp["oos_pred"].items():
        preds[f"mlp_{activation}"] = p
    assert all(len(p) == len(actual) for p in preds.values())

    pooled = {}
    for name, p in preds.items():
        pooled[name] = {
            "rmse": rmse(p, actual),
            "rmse_annual_pp": rmse(p, actual) * annualize,
            "mae_annual_pp": mae(p, actual) * annualize,
            "qlike": qlike(p, actual),
            "mz_r2": mincer_zarnowitz_r2(p, actual),
        }
    for name in pooled:
        pooled[name]["rmse_vs_naive"] = pooled[name]["rmse"] / pooled["naive"]["rmse"]
        pooled[name]["qlike_vs_naive"] = pooled[name]["qlike"] / pooled["naive"]["qlike"]

    dm = {}
    pairs = [("catboost", "catboost_no_macro"), ("har_rv_macro", "har_rv")]  # what the macro inputs add
    for name, reference in pairs:
        dm[f"{name}_vs_{reference}"] = {
            "mse": diebold_mariano(
                squared_errors(preds[name], actual), squared_errors(preds[reference], actual), VOL_TARGET_HORIZON
            ),
            "qlike": diebold_mariano(
                qlike_losses(preds[name], actual), qlike_losses(preds[reference], actual), VOL_TARGET_HORIZON
            ),
        }
    for reference in ("garch", "naive"):
        for name in preds:
            if name == reference:
                continue
            dm[f"{name}_vs_{reference}"] = {
                "mse": diebold_mariano(
                    squared_errors(preds[name], actual), squared_errors(preds[reference], actual), VOL_TARGET_HORIZON
                ),
                "qlike": diebold_mariano(
                    qlike_losses(preds[name], actual), qlike_losses(preds[reference], actual), VOL_TARGET_HORIZON
                ),
            }

    print("\n" + "-" * 86)
    print(f"{'Model':<14}{'RMSE (ann. pp)':>16}{'QLIKE':>10}{'RMSE/naive':>12}{'QLIKE/naive':>13}{'MZ R2':>8}")
    for name, m in sorted(pooled.items(), key=lambda kv: kv[1]["qlike"]):
        print(
            f"{name:<14}{m['rmse_annual_pp']:>16.2f}{m['qlike']:>10.4f}{m['rmse_vs_naive']:>12.3f}"
            f"{m['qlike_vs_naive']:>13.3f}{m['mz_r2']:>8.3f}"
        )

    # ---- descriptive SHAP model (not scored) ----------------------------------------------
    print(f"\nOptuna search on the full series for the SHAP model ({N_OPTUNA_TRIALS} trials)...")
    study = run_optuna_search(df, feature_cols, TARGET_COL, n_trials=N_OPTUNA_TRIALS)
    final_model = fit_final_model(df, feature_cols, TARGET_COL, study.best_params)
    _, shap_values, X_df = compute_shap_values(final_model, df, feature_cols)
    importance_by_feature = global_importance_by_feature(shap_values, feature_cols)
    importance_by_group = global_importance_by_group(shap_values, feature_cols, feature_groups)
    print("\nSHAP importance by feature group:")
    print(importance_by_group.to_string(index=False))
    print("\nTop 10 SHAP features:")
    print(importance_by_feature.head(10).to_string(index=False))

    # ---- figures -------------------------------------------------------------------------
    print("\nWriting figures...")
    trailing_vol = market.copper["log_return"].pow(2).rolling_mean(window_size=20).sqrt().to_numpy()
    plot_price_and_vol(market.copper["date"].to_numpy(), market.copper["price"].to_numpy(), trailing_vol * annualize)
    last_fold = slice(len(actual) - comparison["fold_sizes"][-1][1], len(actual))
    last_dates = df["date"].to_numpy()[oos["row_idx"][last_fold]]
    plot_forecasts_last_fold(
        last_dates,
        actual[last_fold] * annualize,
        {name: preds[name][last_fold] * annualize for name in ("garch", "har_rv_macro", "catboost")},
    )
    per_fold = {name: comparison[name] for name in MODELS}
    per_fold = {name: {"rmse": v["rmse_per_fold"], "qlike": v["qlike_per_fold"]} for name, v in per_fold.items()}
    best_mlp = mlp["by_activation"][best_activation]
    per_fold[f"mlp_{best_activation}"] = {"rmse": best_mlp["rmse_per_fold"], "qlike": best_mlp["qlike_per_fold"]}
    plot_model_comparison(per_fold)
    scatter = {
        name: (preds[name][last_fold] * annualize, actual[last_fold] * annualize)
        for name in ("garch", "catboost", f"mlp_{best_activation}")
    }
    plot_predicted_vs_actual(scatter)
    plot_residual_distribution(scatter)
    loss_histories = {
        act: {
            "train_loss_history": d["train_loss_history"],
            "val_loss_history": d["val_loss_history"],
            "best_epoch": d["best_epoch"],
        }
        for act, d in mlp["all_detail"].items()
    }
    plot_mlp_loss_curves(loss_histories)
    plot_mlp_loss_curves_animated(loss_histories)
    for figure in PLOTS_DIR.iterdir():
        shutil.copy2(figure, FIGURES_DIR / figure.name)

    # ---- artifacts -----------------------------------------------------------------------
    con = get_connection()
    all_metrics = {name: comparison[name] for name in MODELS}
    for act, r in mlp["by_activation"].items():
        all_metrics[f"mlp_{act}"] = r
    persist_comparison(con, dt.datetime.now(), all_metrics, {name: (preds[name], actual) for name in preds})
    con.close()

    importance_by_feature.to_csv(OUTPUTS_DIR / "shap_feature_importance.csv", index=False)
    importance_by_group.to_csv(OUTPUTS_DIR / "shap_group_importance.csv", index=False)
    np.save(OUTPUTS_DIR / "shap_values.npy", shap_values)
    X_df.to_parquet(OUTPUTS_DIR / "shap_features.parquet")
    final_model.save_model(str(OUTPUTS_DIR / "final_catboost_model.cbm"))
    with open(OUTPUTS_DIR / "optuna_best_params.json", "w", encoding="utf-8") as f:
        json.dump({"best_params": study.best_params, "best_holdout_rmse": study.best_value}, f, indent=2)
    with open(OUTPUTS_DIR / "optuna_trial_history.json", "w", encoding="utf-8") as f:
        trials = [t for t in study.trials if t.value is not None]
        history = [{"trial": t.number, "value": t.value, "params": t.params} for t in trials]
        json.dump(history, f, indent=2)
    with open(OUTPUTS_DIR / "walk_forward_comparison.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable({k: v for k, v in comparison.items() if k != "oos"}), f, indent=2)

    results = {
        "data": report,
        "sample": {
            "rows": df.height,
            "first_row": df["date"].min(),
            "last_row": df["date"].max(),
            "oos_rows": len(actual),
            "oos_first": df["date"][int(oos["row_idx"][0])],
            "periods_per_year": periods_per_year,
            "folds": comparison["fold_sizes"],
            "horizon_days": VOL_TARGET_HORIZON,
        },
        "realized_vol_annual_pct": {
            "mean": float(np.sqrt(np.mean(actual**2)) * annualize),
            "min": float(actual.min() * annualize),
            "max": float(actual.max() * annualize),
        },
        "pooled": pooled,
        "per_fold": {name: {k: comparison[name][k] for k in ("rmse_per_fold", "qlike_per_fold")} for name in MODELS},
        "diebold_mariano": dm,
        "catboost_folds": comparison["catboost_folds"],
        "mlp": {
            act: {k: v for k, v in r.items() if k != "latency_ms_per_sample"}  # timing varies run to run
            for act, r in mlp["by_activation"].items()
        },
        "mlp_best_activation": best_activation,
        "shap_groups_pct": dict(zip(importance_by_group["group"], importance_by_group["share_pct"])),
        "shap_top_features": importance_by_feature.head(10).to_dict(orient="records"),
    }
    with open(REPORTS_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(results), f, indent=2, ensure_ascii=False, allow_nan=False)

    print(f"\nArtifacts written to {OUTPUTS_DIR} and {REPORTS_DIR}")


if __name__ == "__main__":
    main()
