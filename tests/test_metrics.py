import numpy as np

from src.metrics import diebold_mariano, qlike, qlike_losses, rmse


def test_qlike_is_zero_for_a_perfect_forecast_and_positive_otherwise():
    actual = np.array([0.01, 0.02, 0.015])
    assert qlike(actual, actual) == 0.0
    assert qlike(actual * 1.2, actual) > 0


def test_qlike_punishes_under_forecasting_risk_more_than_over_forecasting():
    actual = np.full(100, 0.01)
    assert qlike(actual * 0.5, actual) > qlike(actual * 1.5, actual)
    # RMSE treats both misses alike
    assert np.isclose(rmse(actual * 0.5, actual), rmse(actual * 1.5, actual))


def test_qlike_losses_floor_avoids_infinite_loss_on_zero_forecasts():
    losses = qlike_losses(np.array([0.0, 0.01]), np.array([0.01, 0.01]))
    assert np.isfinite(losses).all()


def test_diebold_mariano_detects_a_clearly_better_model():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, 2000)
    good = (0.5 * noise) ** 2
    bad = (1.5 * noise) ** 2
    result = diebold_mariano(good, bad, max_lags=5)
    assert result["statistic"] < 0
    assert result["p_value"] < 0.01
    assert result["n"] == 2000


def test_diebold_mariano_does_not_reject_two_equivalent_models():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 3000) ** 2
    b = rng.normal(0, 1, 3000) ** 2
    assert diebold_mariano(a, b, max_lags=5)["p_value"] > 0.01
