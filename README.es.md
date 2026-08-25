<div align="center">

# Copper Volatility Forecaster

**[English](README.md) | [Español](README.es.md)**

![Python](https://img.shields.io/badge/python-3.10-blue)
![Polars](https://img.shields.io/badge/polars-1.44-orange)
![CatBoost](https://img.shields.io/badge/catboost-1.2-yellow)
![Optuna](https://img.shields.io/badge/optuna-4.9-9cf)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

## Descripción general

Predice la **volatilidad realizada futura** del precio del cobre: dado un
historial de precios diarios, el modelo predice cuán volátil será el precio
durante los **próximos 5 días hábiles**. Este tipo de pronóstico alimenta
decisiones de cobertura (hedging), pricing de opciones y dimensionamiento de
límites de riesgo para posiciones expuestas al cobre — relevante para Chile
como el mayor productor mundial de cobre.

El pipeline está construido con **Polars** para la ingeniería de
características, **CatBoost** como regresor, y validado con
**`TimeSeriesSplit` de scikit-learn** para validación cruzada estricta
walk-forward. **Optuna** está instalado y conectado al proyecto para la fase
de ajuste de hiperparámetros que sigue a esta versión inicial.

## Nota sobre los datos

Esta versión inicial genera una serie **sintética** de precios diarios de
cobre (con agrupamiento de volatilidad tipo GARCH(1,1), de modo que la serie
tiene regímenes de volatilidad genuinos y recuperables, no ruido i.i.d.). Está
etiquetada explícitamente como sintética en `main.py` y existe para construir
y validar el pipeline completo antes de conectar datos reales. Un siguiente
paso natural es usar una serie real (por ejemplo, futuros de cobre de LME o
COMEX).

## Por qué esto importa: cero fuga de datos futuros (lookahead bias)

Los pipelines de pronóstico de volatilidad son especialmente propensos a fugas
silenciosas de datos, porque un valor "futuro" (la volatilidad realizada) se
*deriva* de la misma serie de retornos de donde salen las características. Se
imponen dos reglas de punta a punta:

1. **Cada característica se construye a partir de retornos desplazados al
   menos un día** (`.shift(1)` antes de cualquier ventana móvil), de modo que
   una característica del día *t* nunca ve el retorno realizado ese mismo
   día *t*.
2. **La validación cruzada usa exclusivamente `TimeSeriesSplit`** — una
   división walk-forward donde cada fold de validación es estrictamente
   posterior en el tiempo a su fold de entrenamiento. Nunca se usa un K-fold
   aleatorio con mezcla (shuffle), ya que permitiría que el modelo entrene con
   regímenes de volatilidad futuros y valide contra el pasado.

## Arquitectura

```mermaid
flowchart LR
    A[Serie sintética de precios<br/>GARCH(1,1)] --> B[Retornos logarítmicos]
    B --> C["Features rezagadas/móviles<br/>(shift(1) antes de rolling)"]
    B --> D["Target: volatilidad realizada<br/>futura (t+1..t+5)"]
    C --> E[TimeSeriesSplit<br/>5 folds walk-forward]
    D --> E
    E --> F[CatBoostRegressor<br/>por fold]
    F --> G[RMSE / MAE<br/>por fold y promedio]
```

## Características (features)

| Característica | Descripción |
|---|---|
| `realized_vol_{5,10,20,60}d` | Desviación estándar móvil de retornos *pasados* (desplazados 1 día) |
| `mean_return_{5,10,20,60}d` | Media móvil de retornos *pasados* (desplazados 1 día) |
| `lag_return_{1,2,3}` | Retorno de hace 1/2/3 días |
| `day_of_week`, `month` | Características de calendario (conocidas de antemano, sin fuga) |

**Target**: `target_fwd_realized_vol` — desviación estándar de los retornos
entre los días *t+1* y *t+5*, es decir, la volatilidad que el modelo debe
pronosticar.

## Resultados

Salida real de ejecutar `main.py` (semilla 42, 3000 días simulados, 2934 filas
tras construir features/target, `TimeSeriesSplit` de 5 folds):

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

RMSE/MAE están en unidades de retorno logarítmico diario (la misma escala que
el propio target de volatilidad), por lo que un RMSE promedio de ~0.0027
significa que el pronóstico del modelo sobre la volatilidad realizada futura
a 5 días se equivoca en promedio por unos 0.27 puntos porcentuales de
volatilidad de retorno diario — un margen pequeño frente a los movimientos
diarios de ~1-3% que produce el propio proceso GARCH subyacente.

## Cómo empezar

```powershell
py -m venv venv
./venv/Scripts/pip install -r requirements.txt
./venv/Scripts/python main.py
```

## Próximos pasos

- Reemplazar la serie sintética por una serie real de cobre (LME/COMEX).
- Agregar un estudio de Optuna para ajustar los hiperparámetros de CatBoost por fold.
- Agregar análisis de importancia de características con SHAP.

## Licencia

MIT — ver [LICENSE](LICENSE).
