"""
Copper price volatility forecaster.

Predicts forward realized volatility of copper prices with an Optuna-tuned
CatBoost model, benchmarked against two econometric baselines -- GARCH(1,1)
and HAR-RV (Corsi 2009) -- on identical walk-forward (`TimeSeriesSplit`)
folds. Explainability (global + local) is computed with SHAP to quantify
how much predictive weight the macro (USD index, global risk proxy) and
volume features actually carry, versus the pure return-based features.

Data note: this pipeline runs on a SYNTHETIC daily copper price/volume/macro
series (GARCH-X volatility clustering with documented, injected macro
leading indicators -- see `src/data.py`) as a placeholder so the full
pipeline can be built and validated end to end. It is explicitly labeled as
synthetic throughout and is meant to be swapped for a real sourced series
(e.g. LME/COMEX copper futures + real macro data) before any real
forecasting is done on it.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from src.data import N_DAYS, generate_synthetic_copper_market
from src.deep_learning import run_activation_comparison
from src.explainability import compute_shap_values, global_importance_by_group, global_importance_by_feature
from src.features import VOL_TARGET_HORIZON, build_features_and_target
from src.modeling import N_CV_SPLITS, fit_final_model, run_walk_forward_comparison
from src.persistence import get_connection, persist_comparison
from src.plots import plot_mlp_loss_curves, plot_predicted_vs_actual, plot_residual_distribution
from src.tuning import run_optuna_search

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
N_OPTUNA_TRIALS = 30


def main() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)

    print(f"Generating synthetic copper market series ({N_DAYS} days: price, volume, usd_index, risk_proxy)...")
    market = generate_synthetic_copper_market()

    print("Building lookahead-safe features and forward-volatility target...")
    df, feature_cols, feature_groups = build_features_and_target(market)
    print(f"Rows after feature/target construction: {df.height}")
    print(f"Features used ({len(feature_cols)}): {feature_cols}")

    print(f"\nRunning Optuna search ({N_OPTUNA_TRIALS} trials, single chronological holdout split)...")
    study = run_optuna_search(df, feature_cols, target_col="target_fwd_realized_vol", n_trials=N_OPTUNA_TRIALS)
    best_params = study.best_params
    print(f"Best params: {best_params}")
    print(f"Best single-split holdout RMSE: {study.best_value:.6f}")

    print(f"\nRunning walk-forward comparison (CatBoost tuned vs. GARCH(1,1) vs. HAR-RV), {N_CV_SPLITS} folds...")
    comparison = run_walk_forward_comparison(
        df, feature_cols, target_col="target_fwd_realized_vol",
        catboost_params=best_params, horizon=VOL_TARGET_HORIZON,
    )

    print("\n" + "-" * 70)
    print(f"{'Model':<12}{'RMSE (mean)':>16}{'RMSE (std)':>14}{'MAE (mean)':>14}{'MAE (std)':>13}")
    for name in ("catboost", "garch", "har_rv"):
        r = comparison[name]
        print(f"{name:<12}{r['rmse_mean']:>16.6f}{r['rmse_std']:>14.6f}{r['mae_mean']:>14.6f}{r['mae_std']:>13.6f}")

    print(f"\nRunning PyTorch MLP activation comparison (ReLU/GELU/Swish), {N_CV_SPLITS} folds...")
    mlp_comparison = run_activation_comparison(
        df, feature_cols, target_col="target_fwd_realized_vol", n_splits=N_CV_SPLITS,
    )
    best_activation = mlp_comparison["best_activation"]
    print(f"\nBest MLP activation: {best_activation}")
    for name, r in mlp_comparison["by_activation"].items():
        print(f"mlp[{name:<6}]  RMSE={r['rmse_mean']:.6f}  MAE={r['mae_mean']:.6f}  "
              f"latency={r['latency_ms_per_sample']:.4f} ms/sample")

    print("\nWriting comparison plots to outputs/plots/...")
    best_preds = mlp_comparison["best_detail"]["preds"]
    best_actual = mlp_comparison["best_detail"]["actual"]
    predictions_for_plots = {f"mlp_{best_activation}": (best_preds, best_actual)}
    plot_predicted_vs_actual(predictions_for_plots)
    plot_residual_distribution(predictions_for_plots)
    loss_histories = {
        act: {
            "train_loss_history": detail["train_loss_history"],
            "val_loss_history": detail["val_loss_history"],
        }
        for act, detail in mlp_comparison["all_detail"].items()
    }
    plot_mlp_loss_curves(loss_histories)

    print("Persisting comparative metrics to DuckDB (outputs/comparison_metrics.duckdb)...")
    con = get_connection()
    run_id = dt.datetime.now()
    all_metrics = {
        "catboost": comparison["catboost"],
        "garch": comparison["garch"],
        "har_rv": comparison["har_rv"],
    }
    for act, r in mlp_comparison["by_activation"].items():
        all_metrics[f"mlp_{act}"] = r
    persist_comparison(
        con, run_id, all_metrics,
        {f"mlp_{best_activation}": (best_preds, best_actual)},
    )
    con.close()

    print("\nFitting final CatBoost model (full series) for SHAP explainability...")
    final_model = fit_final_model(df, feature_cols, "target_fwd_realized_vol", best_params)

    print("Computing SHAP values (TreeExplainer)...")
    _, shap_values, X_df = compute_shap_values(final_model, df, feature_cols)
    importance_by_feature = global_importance_by_feature(shap_values, feature_cols)
    importance_by_group = global_importance_by_group(shap_values, feature_cols, feature_groups)

    print("\nSHAP importance by feature group:")
    print(importance_by_group.to_string(index=False))
    print("\nTop 10 SHAP features:")
    print(importance_by_feature.head(10).to_string(index=False))

    importance_by_feature.to_csv(OUTPUTS_DIR / "shap_feature_importance.csv", index=False)
    importance_by_group.to_csv(OUTPUTS_DIR / "shap_group_importance.csv", index=False)
    np.save(OUTPUTS_DIR / "shap_values.npy", shap_values)
    X_df.to_parquet(OUTPUTS_DIR / "shap_features.parquet")
    final_model.save_model(str(OUTPUTS_DIR / "final_catboost_model.cbm"))

    with open(OUTPUTS_DIR / "optuna_best_params.json", "w", encoding="utf-8") as f:
        json.dump({"best_params": best_params, "best_holdout_rmse": study.best_value}, f, indent=2)

    with open(OUTPUTS_DIR / "walk_forward_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    optuna_history = [
        {"trial": t.number, "value": t.value, "params": t.params}
        for t in study.trials if t.value is not None
    ]
    with open(OUTPUTS_DIR / "optuna_trial_history.json", "w", encoding="utf-8") as f:
        json.dump(optuna_history, f, indent=2)

    print(f"\nArtifacts written to: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
