import numpy as np

from src.data import generate_synthetic_copper_market


def test_generated_series_has_no_nulls_or_overflow():
    df = generate_synthetic_copper_market(n_days=1000, seed=1)
    assert df.height == 1000
    assert df.null_count().sum_horizontal().item() == 0
    for col in ("price", "volume", "usd_index", "risk_proxy"):
        values = df[col].to_numpy()
        assert np.isfinite(values).all(), f"{col} has non-finite values"


def test_price_and_volume_stay_within_sane_bounds():
    df = generate_synthetic_copper_market(n_days=3000, seed=42)
    assert 0.5 < df["price"].min() < 10.0
    assert df["price"].max() < 500.0
    assert df["volume"].min() > 0
    assert df["volume"].max() < 1e10


def test_deterministic_given_seed():
    df1 = generate_synthetic_copper_market(n_days=200, seed=7)
    df2 = generate_synthetic_copper_market(n_days=200, seed=7)
    assert (df1["price"].to_numpy() == df2["price"].to_numpy()).all()
