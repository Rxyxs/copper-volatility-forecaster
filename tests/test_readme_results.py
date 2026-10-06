"""Every number in the READMEs' results tables has to come from `reports/results.json`, the
file `main.py` writes. If the pipeline changes and the README is not updated, this fails."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "reports" / "results.json"

MODELS = {
    "en": {
        "garch": "GARCH(1,1)",
        "har_rv_macro": "HAR-X (HAR-RV + VIX, dollar)",
        "ewma": "EWMA (RiskMetrics)",
        "har_rv": "HAR-RV",
        "catboost": "CatBoost (Optuna, re-tuned per fold)",
        "catboost_no_macro": "CatBoost without VIX or dollar",
        "mlp_relu": "MLP (ReLU)",
        "mlp_gelu": "MLP (GELU)",
        "mlp_swish": "MLP (Swish)",
        "naive": "Persistence (last 5 days)",
    },
    "es": {
        "garch": "GARCH(1,1)",
        "har_rv_macro": "HAR-X (HAR-RV + VIX y dólar)",
        "ewma": "EWMA (RiskMetrics)",
        "har_rv": "HAR-RV",
        "catboost": "CatBoost (Optuna, ajustado en cada fold)",
        "catboost_no_macro": "CatBoost sin VIX ni dólar",
        "mlp_relu": "MLP (ReLU)",
        "mlp_gelu": "MLP (GELU)",
        "mlp_swish": "MLP (Swish)",
        "naive": "Persistencia (últimos 5 días)",
    },
}
COMPARISONS = {
    "en": {
        "HAR-X vs. GARCH(1,1)": "har_rv_macro_vs_garch",
        "CatBoost vs. GARCH(1,1)": "catboost_vs_garch",
        "HAR-RV vs. GARCH(1,1)": "har_rv_vs_garch",
        "EWMA vs. GARCH(1,1)": "ewma_vs_garch",
        "CatBoost vs. CatBoost without VIX or dollar": "catboost_vs_catboost_no_macro",
        "HAR-X vs. HAR-RV": "har_rv_macro_vs_har_rv",
        "MLP (ReLU) vs. persistence": "mlp_relu_vs_naive",
    },
    "es": {
        "HAR-X vs. GARCH(1,1)": "har_rv_macro_vs_garch",
        "CatBoost vs. GARCH(1,1)": "catboost_vs_garch",
        "HAR-RV vs. GARCH(1,1)": "har_rv_vs_garch",
        "EWMA vs. GARCH(1,1)": "ewma_vs_garch",
        "CatBoost vs. CatBoost sin VIX ni dólar": "catboost_vs_catboost_no_macro",
        "HAR-X vs. HAR-RV": "har_rv_macro_vs_har_rv",
        "MLP (ReLU) vs. persistencia": "mlp_relu_vs_naive",
    },
}
SHAP_GROUPS = {
    "en": {"return": "Return", "macro": "Macro (VIX, dollar index)", "calendar": "Calendar"},
    "es": {"return": "Retornos", "macro": "Macro (VIX, índice dólar)", "calendar": "Calendario"},
}
READMES = [("en", "README.md"), ("es", "README.es.md")]


def _fmt(value: float, decimals: int, lang: str) -> str:
    text = f"{value:.{decimals}f}"
    return text.replace(".", ",") if lang == "es" else text


def _p(value: float, lang: str) -> str:
    return ("< 0,001" if lang == "es" else "< 0.001") if value < 0.001 else _fmt(value, 3, lang)


@pytest.fixture(scope="module")
def results():
    if not RESULTS.exists():
        pytest.skip("reports/results.json not found: run `python main.py` first")
    return json.loads(RESULTS.read_text(encoding="utf-8"))


@pytest.mark.parametrize("lang, readme", READMES)
def test_results_table_matches_results_json(results, lang, readme):
    text = (ROOT / readme).read_text(encoding="utf-8")
    for model, label in MODELS[lang].items():
        p = results["pooled"][model]
        row = (
            f"| {label} | {_fmt(p['rmse_annual_pp'], 2, lang)} | {_fmt(p['qlike'], 3, lang)} | "
            f"{_fmt(p['rmse_vs_naive'], 2, lang)} | {_fmt(p['qlike_vs_naive'], 2, lang)} |"
        )
        assert row in text, f"{readme}: the row for {model} does not match results.json, expected {row}"


@pytest.mark.parametrize("lang, readme", READMES)
def test_diebold_mariano_table_matches_results_json(results, lang, readme):
    text = (ROOT / readme).read_text(encoding="utf-8")
    for label, key in COMPARISONS[lang].items():
        dm = results["diebold_mariano"][key]
        row = f"| {label} | {_p(dm['qlike']['p_value'], lang)} | {_p(dm['mse']['p_value'], lang)} |"
        assert row in text, f"{readme}: {key} does not match results.json, expected {row}"


@pytest.mark.parametrize("lang, readme", READMES)
def test_shap_table_matches_results_json(results, lang, readme):
    text = (ROOT / readme).read_text(encoding="utf-8")
    for group, label in SHAP_GROUPS[lang].items():
        row = f"| {label} | {_fmt(results['shap_groups_pct'][group], 2, lang)}% |"
        assert row in text, f"{readme}: SHAP group {group} does not match results.json, expected {row}"
