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
