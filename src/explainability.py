"""SHAP explainability for the final CatBoost volatility model.

CatBoost is a tree ensemble, so `shap.TreeExplainer` gives exact (not
approximate) Shapley values efficiently -- no KernelSHAP/DeepSHAP tradeoffs
needed here. This module computes both global importance (mean |SHAP| per
feature and per feature *group* -- return/volume/macro/calendar, to directly
answer "how much weight do macro and volume variables carry?") and local
(single-instance) attribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import shap
from catboost import CatBoostRegressor


def compute_shap_values(model: CatBoostRegressor, df: pl.DataFrame, feature_cols: list[str]) -> tuple[shap.Explainer, np.ndarray, pd.DataFrame]:
    X_df = df.select(feature_cols).to_pandas()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)
    return explainer, shap_values, X_df


def global_importance_by_feature(shap_values: np.ndarray, feature_cols: list[str]) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def global_importance_by_group(
    shap_values: np.ndarray, feature_cols: list[str], feature_groups: dict[str, list[str]]
) -> pd.DataFrame:
    """Sums mean |SHAP| within each feature group (return/volume/macro/
    calendar) and reports each group's share of total importance -- the
    direct answer to "how much weight do macro and volume variables carry?"
    """
    importance = global_importance_by_feature(shap_values, feature_cols).set_index("feature")["mean_abs_shap"]
    rows = []
    for group, cols in feature_groups.items():
        cols_present = [c for c in cols if c in importance.index]
        rows.append({"group": group, "total_mean_abs_shap": float(importance.loc[cols_present].sum())})
    group_df = pd.DataFrame(rows).sort_values("total_mean_abs_shap", ascending=False).reset_index(drop=True)
    total = group_df["total_mean_abs_shap"].sum()
    group_df["share_pct"] = (group_df["total_mean_abs_shap"] / total * 100.0).round(2)
    return group_df


def local_explanation(
    shap_values: np.ndarray, X_df: pd.DataFrame, row_idx: int, top_n: int = 8
) -> pd.DataFrame:
    """SHAP attribution for one specific prediction (row `row_idx`),
    sorted by absolute contribution -- the "why did the model predict this
    much volatility for this specific day" view."""
    row_shap = shap_values[row_idx]
    row_values = X_df.iloc[row_idx]
    local_df = pd.DataFrame(
        {"feature": X_df.columns, "value": row_values.to_numpy(), "shap_value": row_shap}
    ).sort_values("shap_value", key=np.abs, ascending=False)
    return local_df.head(top_n).reset_index(drop=True)
