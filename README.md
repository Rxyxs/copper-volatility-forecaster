<div align="center">

# Copper Volatility Forecaster

**[English](README.md) | [Español](README.es.md)**

![Python](https://img.shields.io/badge/python-3.10-blue)
![Polars](https://img.shields.io/badge/polars-1.44-orange)
![CatBoost](https://img.shields.io/badge/catboost-1.2-yellow)
![Optuna](https://img.shields.io/badge/optuna-4.9-9cf)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

## Overview

Forecasts **forward realized volatility** of copper prices: given a history of
daily prices, the model predicts how volatile the price will be over the
**next 5 trading days**. Volatility forecasts like this feed into hedging,
options pricing, and risk-limit sizing for copper-exposed positions — relevant
to Chile as the world's largest copper producer.

The pipeline is built on **Polars** for feature engineering, **CatBoost** as
the regressor, and validated with **scikit-learn's `TimeSeriesSplit`** for
strict walk-forward cross-validation. **Optuna** is installed and wired into
the project for the hyperparameter-tuning phase that follows this initial
version.

## Data note

This initial version generates a **synthetic** daily copper price series
(GARCH(1,1) volatility clustering, so the series has genuine, recoverable
volatility regimes rather than i.i.d. noise). It is explicitly labeled as
synthetic in `main.py` and exists to build and validate the full pipeline
before real data is wired in. A natural next step is a real sourced series
(e.g. LME or COMEX copper futures).

## Why this matters: zero lookahead bias

Volatility-forecasting pipelines are especially prone to silent data leakage,
because a "future" value (realized volatility) is *derived from* the same
return series the features come from. Two rules are enforced end to end:

1. **Every feature is built from returns shifted by at least one day**
   (`.shift(1)` before any rolling window), so a feature at day *t* never sees
   the return realized on day *t* itself.
2. **Cross-validation is exclusively `TimeSeriesSplit`** — a walk-forward
   split where every validation fold is strictly later in time than its
   training fold. A shuffled random K-fold is never used, since it would let
   the model train on future volatility regimes and validate on the past.

## Architecture

```mermaid
flowchart LR
    A[Synthetic price series<br/>GARCH(1,1)] --> B[Log returns]
    B --> C["Lagged / rolling features<br/>(shift(1) before rolling)"]
    B --> D["Forward realized vol<br/>target (t+1..t+5)"]
    C --> E[TimeSeriesSplit<br/>5 walk-forward folds]
    D --> E
    E --> F[CatBoostRegressor<br/>per fold]
    F --> G[RMSE / MAE<br/>per fold + mean]
```

## Features

| Feature | Description |
|---|---|
| `realized_vol_{5,10,20,60}d` | Rolling std of *past* returns (shifted 1 day) |
| `mean_return_{5,10,20,60}d` | Rolling mean of *past* returns (shifted 1 day) |
| `lag_return_{1,2,3}` | Return 1/2/3 days ago |
| `day_of_week`, `month` | Calendar features (known in advance, no leakage) |

**Target**: `target_fwd_realized_vol` — std of returns over days *t+1* through
*t+5*, i.e. the volatility the model is asked to forecast.

## Results

Real output from running `main.py` (seed 42, 3000 simulated days, 2934 rows
after feature/target construction, 5-fold `TimeSeriesSplit`):

```
[fold 1/5] train=  489 val=  489 RMSE=0.002423 MAE=0.001915
[fold 2/5] train=  978 val=  489 RMSE=0.002900 MAE=0.002315
[fold 3/5] train= 1467 val=  489 RMSE=0.002657 MAE=0.002195
[fold 4/5] train= 1956 val=  489 RMSE=0.003262 MAE=0.002560
[fold 5/5] train= 2445 val=  489 RMSE=0.002471 MAE=0.002000
------------------------------------------------------------
CV mean RMSE: 0.002743 (+/- 0.000309)
CV mean MAE : 0.002197 (+/- 0.000230)
```

RMSE/MAE are in daily log-return units (the same scale as the volatility
target itself), so a mean RMSE of ~0.0027 means the model's forecast of
5-day forward realized volatility is off by about 0.27 percentage points of
daily return volatility on average — small relative to the ~1-3% daily moves
the underlying GARCH process itself produces.

## Getting started

```powershell
py -m venv venv
./venv/Scripts/pip install -r requirements.txt
./venv/Scripts/python main.py
```

## Roadmap

- Replace the synthetic price series with a real sourced copper series (LME/COMEX).
- Add an Optuna study to tune CatBoost hyperparameters per fold.
- Add SHAP feature-importance analysis.

## License

MIT — see [LICENSE](LICENSE).
