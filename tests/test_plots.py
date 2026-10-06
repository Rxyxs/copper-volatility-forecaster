import numpy as np

from src.plots import plot_mlp_loss_curves, plot_predicted_vs_actual, plot_residual_distribution


def _fake_predictions():
    rng = np.random.default_rng(0)
    actual = np.abs(rng.normal(0.01, 0.002, size=200))
    preds_a = actual + rng.normal(0, 0.0005, size=200)
    preds_b = actual + rng.normal(0, 0.001, size=200)
    return {"catboost": (preds_a, actual), "mlp": (preds_b, actual)}


def test_plot_predicted_vs_actual_writes_file(tmp_path, monkeypatch):
    import src.plots as plots_module

    monkeypatch.setattr(plots_module, "PLOTS_DIR", tmp_path)
    path = plot_predicted_vs_actual(_fake_predictions(), filename="pva.png")
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_residual_distribution_writes_file(tmp_path, monkeypatch):
    import src.plots as plots_module

    monkeypatch.setattr(plots_module, "PLOTS_DIR", tmp_path)
    path = plot_residual_distribution(_fake_predictions(), filename="resid.png")
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_mlp_loss_curves_writes_file(tmp_path, monkeypatch):
    import src.plots as plots_module

    monkeypatch.setattr(plots_module, "PLOTS_DIR", tmp_path)
    histories = {
        "relu": {"train_loss_history": [0.5, 0.3, 0.2], "val_loss_history": [0.6, 0.4, 0.3]},
        "gelu": {"train_loss_history": [0.4, 0.25, 0.15], "val_loss_history": [0.5, 0.35, 0.25]},
    }
    path = plot_mlp_loss_curves(histories, filename="loss.png")
    assert path.exists()
    assert path.stat().st_size > 0


def test_new_real_data_plots_write_files(tmp_path, monkeypatch):
    import datetime as dt

    import src.plots as plots_module
    from src.plots import plot_forecasts_last_fold, plot_model_comparison, plot_price_and_vol

    monkeypatch.setattr(plots_module, "PLOTS_DIR", tmp_path)
    dates = np.array([np.datetime64(dt.date(2020, 1, 1)) + i for i in range(200)])
    rng = np.random.default_rng(1)
    vol = np.abs(rng.normal(20, 5, 200))
    paths = [
        plot_price_and_vol(dates, 3 + rng.normal(0, 0.1, 200).cumsum(), vol, filename="series.png"),
        plot_forecasts_last_fold(dates, vol, {"garch": vol * 0.9, "naive": vol * 1.1}, filename="fc.png"),
        plot_model_comparison(
            {
                "garch": {"rmse": [0.9, 0.95], "qlike": [0.8, 0.9]},
                "catboost": {"rmse": [1.0, 0.97], "qlike": [1.1, 0.9]},
                "naive": {"rmse": [1.0, 1.0], "qlike": [1.0, 1.0]},
            },
            filename="cmp.png",
        ),
    ]
    for path in paths:
        assert path.exists() and path.stat().st_size > 0
