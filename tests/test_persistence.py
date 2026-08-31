import datetime as dt

import numpy as np

from src.persistence import get_connection, load_latest_comparison, persist_comparison


def test_persist_and_load_comparison_roundtrip(tmp_path):
    db_path = tmp_path / "test_metrics.duckdb"
    con = get_connection(db_path)

    run_id = dt.datetime(2026, 1, 1, 12, 0, 0)
    metrics = {
        "catboost": {"rmse_mean": 0.008, "rmse_std": 0.001, "mae_mean": 0.006, "mae_std": 0.0009},
        "garch": {"rmse_mean": 0.0078, "rmse_std": 0.0009, "mae_mean": 0.0058, "mae_std": 0.0008},
        "mlp_relu": {
            "rmse_mean": 0.0082,
            "rmse_std": 0.0011,
            "mae_mean": 0.0062,
            "mae_std": 0.0009,
            "latency_ms_per_sample": 0.05,
        },
    }
    predictions = {
        "catboost": (np.array([0.01, 0.02]), np.array([0.011, 0.019])),
        "garch": (np.array([0.009, 0.021]), np.array([0.011, 0.019])),
    }

    persist_comparison(con, run_id, metrics, predictions)

    result = load_latest_comparison(con)
    con.close()

    assert set(result["model"]) == {"catboost", "garch", "mlp_relu"}
    assert result.iloc[0]["rmse_mean"] <= result.iloc[-1]["rmse_mean"]  # sorted ascending by RMSE
    garch_row = result[result["model"] == "garch"].iloc[0]
    assert abs(garch_row["rmse_mean"] - 0.0078) < 1e-9


def test_persist_comparison_appends_across_runs(tmp_path):
    db_path = tmp_path / "test_metrics2.duckdb"
    con = get_connection(db_path)

    run1 = dt.datetime(2026, 1, 1)
    run2 = dt.datetime(2026, 1, 2)
    metrics1 = {"catboost": {"rmse_mean": 0.01, "rmse_std": 0.001, "mae_mean": 0.008, "mae_std": 0.001}}
    metrics2 = {"catboost": {"rmse_mean": 0.009, "rmse_std": 0.001, "mae_mean": 0.007, "mae_std": 0.001}}

    persist_comparison(con, run1, metrics1, {})
    persist_comparison(con, run2, metrics2, {})

    latest = load_latest_comparison(con)
    con.close()

    assert len(latest) == 1
    assert abs(latest.iloc[0]["rmse_mean"] - 0.009) < 1e-9  # only the most recent run_id
