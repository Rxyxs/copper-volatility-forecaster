"""DuckDB persistence for the three-way model comparison (CatBoost, GARCH,
HAR-RV, MLP-by-activation). Writes comparative walk-forward metrics to a
local `outputs/comparison_metrics.duckdb` file so results from separate runs
can be queried/joined without re-running the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

DB_PATH = Path(__file__).resolve().parent.parent / "outputs" / "comparison_metrics.duckdb"

_CREATE_METRICS_TABLE = """
CREATE TABLE IF NOT EXISTS model_metrics (
    run_id TIMESTAMP,
    model VARCHAR,
    rmse_mean DOUBLE,
    rmse_std DOUBLE,
    mae_mean DOUBLE,
    mae_std DOUBLE,
    latency_ms_per_sample DOUBLE
)
"""

_CREATE_PREDICTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS fold_predictions (
    run_id TIMESTAMP,
    model VARCHAR,
    row_idx INTEGER,
    predicted DOUBLE,
    actual DOUBLE
)
"""


def get_connection(db_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(_CREATE_METRICS_TABLE)
    con.execute(_CREATE_PREDICTIONS_TABLE)
    return con


def persist_comparison(
    con: duckdb.DuckDBPyConnection,
    run_id,
    metrics: dict[str, dict],
    predictions: dict[str, tuple],
) -> None:
    """`metrics` maps model name -> dict with rmse_mean/rmse_std/mae_mean/
    mae_std (and optionally latency_ms_per_sample). `predictions` maps model
    name -> (preds, actual) arrays for one representative fold."""
    for model, m in metrics.items():
        con.execute(
            "INSERT INTO model_metrics VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                run_id,
                model,
                m.get("rmse_mean"),
                m.get("rmse_std"),
                m.get("mae_mean"),
                m.get("mae_std"),
                m.get("latency_ms_per_sample"),
            ],
        )

    for model, (preds, actual) in predictions.items():
        rows = [(run_id, model, int(i), float(p), float(a)) for i, (p, a) in enumerate(zip(preds, actual))]
        con.executemany("INSERT INTO fold_predictions VALUES (?, ?, ?, ?, ?)", rows)


def load_latest_comparison(con: duckdb.DuckDBPyConnection):
    """Returns the metrics table for the most recent `run_id`, ordered by
    RMSE (best first) -- the query behind the README comparison table."""
    return con.execute(
        """
        SELECT model, rmse_mean, rmse_std, mae_mean, mae_std, latency_ms_per_sample
        FROM model_metrics
        WHERE run_id = (SELECT max(run_id) FROM model_metrics)
        ORDER BY rmse_mean ASC
        """
    ).df()
