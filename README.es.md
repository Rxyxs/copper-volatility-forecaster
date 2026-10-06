**[English](README.md) | [Español](README.es.md)**

# Copper Volatility Forecaster

[![CI](https://github.com/Rxyxs/copper-volatility-forecaster/actions/workflows/ci.yml/badge.svg)](https://github.com/Rxyxs/copper-volatility-forecaster/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue) ![Datos](https://img.shields.io/badge/datos-reales%20(LME%20%2B%20FRED)-2ea44f) ![Licencia](https://img.shields.io/badge/licencia-MIT-green)

Con 14 años de precios reales del cobre en la Bolsa de Metales de Londres, un GARCH(1,1) de 1986 pronostica la volatilidad de la semana siguiente tan bien como cualquier otro modelo que probé: un CatBoost ajustado con Optuna dentro de cada fold walk-forward no le gana, y el único que lo empata es un HAR-RV lineal al que se le suman el VIX y el índice dólar.

## Lo que encontré

| Hallazgo | Evidencia |
|---|---|
| **El modelo de 1986 sigue siendo el que hay que superar** | En 2.695 pronósticos diarios fuera de muestra (julio de 2015 a septiembre de 2026), GARCH(1,1) tiene el QLIKE más bajo, 0,380, y en esa métrica le gana a HAR-RV, a EWMA y a CatBoost (Diebold-Mariano, p ≤ 0,022). En RMSE los cuatro mejores modelos empatan estadísticamente, entre 8,59 y 8,86 puntos de volatilidad anualizada (p ≥ 0,22 contra GARCH). |
| **El gradient boosting no agrega nada acá** | CatBoost, ajustado con Optuna dentro de cada fold y solo con datos de entrenamiento, termina con QLIKE 0,455, peor que GARCH (p = 0,022). Su early stopping se queda con entre 10 y 95 árboles en cuatro de los cinco folds: una vez que se conoce la volatilidad reciente, queda poca estructura que aprender. |
| **El VIX y el dólar ayudan poco, y no de forma significativa** | SHAP le asigna al VIX y al dólar el 41% del peso del modelo CatBoost, pero sacarlos empeora su QLIKE en apenas 4% (p = 0,33). Sumar dos de ellos a HAR-RV lo mejora en 9% (p = 0,13) y lo deja como el único modelo a la par de GARCH (p = 0,80). SHAP mide cuánto se apoya un modelo en una variable, no cuánto vale esa variable. |
| **Todos le ganan a "la próxima semana será como la anterior"** | Frente a la persistencia, los modelos econométricos y CatBoost bajan el RMSE en cerca de 20% y el QLIKE en cerca de 60% (p < 0,001). La mejor red neuronal también le gana, pero solo después de corregir dos problemas de entrenamiento, y sigue siendo la más débil de los modelos serios (QLIKE 0,647). |
| **La mayor parte de la volatilidad de una semana no se puede pronosticar** | Ningún modelo explica ni el 10% de la variación de la volatilidad realizada de la semana siguiente (R² de Mincer-Zarnowitz menor a 0,10 en todos). Los modelos siguen el *nivel* del riesgo; ninguno anticipa un salto. |

## Por qué la volatilidad del cobre

Chile es el mayor productor de cobre del mundo, así que la volatilidad de su precio entra directo en el tamaño de las coberturas de mineras y tesorerías, en el precio de opciones y collares de cobre y en qué tan conservadoras se construyen las proyecciones de ingresos y fiscales. Un pronóstico de volatilidad a cinco días es el insumo estándar para las tres cosas. La pregunta acá es si el machine learning mejora los pronósticos econométricos clásicos cuando los datos son reales y la evaluación es honesta.

## Datos

Las tres series son públicas, gratuitas y las descarga el propio pipeline (`python main.py --download`). El análisis está congelado en `SAMPLE_END = 2026-10-02`, así que volver a correrlo reproduce este README.

| Serie | Fuente | Cobertura | Cómo se usa |
|---|---|---|---|
| Precio del cobre, dólares por libra | [mindicador.cl](https://mindicador.cl) (`libra_cobre`) | 3.435 días, del 2012-10-05 al 2026-10-02 | Retornos logarítmicos diarios, el objetivo y las features de retorno |
| VIX (CBOE) | [FRED](https://fred.stlouisfed.org/series/VIXCLS) `VIXCLS` | diaria | Nivel y cambios, solo cierres con fecha anterior al día del pronóstico |
| Índice dólar amplio (Reserva Federal) | [FRED](https://fred.stlouisfed.org/series/DTWEXBGS) `DTWEXBGS` | diaria, publicada cada semana | Cambios logarítmicos, solo una vez publicada la entrega semanal que los contiene |

La serie de cobre es el precio de Londres, no el de Nueva York: en julio de 2025, cuando el COMEX cotizaba sobre 5,5 dólares por libra por el temor a los aranceles, esta serie se mantuvo entre 4,4 y 4,6, unos 9.700 a 10.100 dólares por tonelada, el nivel de la LME. Antes de cualquier modelo hubo que resolver cuatro cosas en los datos, todas explícitas en `src/data.py` y con tests:

- **Sigue el calendario chileno.** En los feriados chilenos la LME opera pero la serie no tiene valor, así que el retorno siguiente cubre más de una sesión: el 95,3% de los retornos cubre un día hábil y el 4,6% cubre de dos a cuatro. Se dejan como están. Dividirlos por la raíz de los días hábiles transcurridos parece la corrección obvia, pero los deja *menos* volátiles que un día normal (0,94% contra 1,30% de desviación estándar diaria), porque muchos de esos feriados también lo son en Londres. Hacerlo bien requiere el calendario de la LME, que no está en los datos (ver Próximos pasos).
- **Dos huecos reales**: 7 días hábiles en la Navidad de 2014 y 18 en diciembre de 2017. Un retorno que cruza cinco o más días hábiles no es un retorno diario, así que se anula y nunca se usa.
- **Un día duplicado**, que se colapsa porque las dos copias son idénticas (un día con dos precios distintos detendría el pipeline).
- **El índice dólar llega con una semana de atraso.** La Reserva Federal lo publica una vez por semana (H.10, lunes en la tarde) con datos hasta el viernes anterior. Usar el valor diario sería usar información que nadie tenía todavía, así que cada fila solo ve valores ya publicados: de entre 4 y 10 días de antigüedad.

No existe una serie diaria gratuita de volumen para el cobre de la LME (mindicador no la tiene y stooq bloquea las descargas automatizadas), así que las features de volumen de la primera versión, simulada, de este proyecto ya no están.

![Precio del cobre y volatilidad realizada](reports/figures/copper_price_and_vol.png)

Catorce años del precio en la LME y su volatilidad realizada a 20 días, que promedia cerca de 21% al año y salta a 44% en abril de 2020 y a 51% en noviembre de 2021, después del estrangulamiento de octubre de 2021 en los contratos cercanos de la LME.

## Método

- **Objetivo:** la volatilidad realizada de los próximos cinco días hábiles, la raíz de la media de los retornos logarítmicos diarios al cuadrado (reportada anualizada, ×√245,5, los días observados por año).
- **Validación:** cinco folds walk-forward expansivos (`TimeSeriesSplit`) de 539 días cada uno. Todos los modelos usan los mismos folds y el mismo corte de información: los retornos hasta el día anterior al pronóstico.
- **Modelos:** persistencia (los últimos cinco días), EWMA (RiskMetrics, λ = 0,94), GARCH(1,1), HAR-RV (Corsi, 2009), HAR-X (HAR-RV más el nivel del VIX y la volatilidad a 20 días del dólar, elegidos antes de ver ningún resultado), CatBoost con 24 features, CatBoost sin las 11 features macro y una red MLP de PyTorch con tres funciones de activación.
- **Nada se ajusta con los datos que se evalúan:** CatBoost se vuelve a ajustar con 30 pruebas de Optuna dentro de cada fold, con el bloque de entrenamiento de ese fold; el último 20% del bloque es el holdout tanto de Optuna como del early stopping. La red usa el mismo holdout para detenerse. Un test reescribe los objetivos de validación de un fold y comprueba que sus pronósticos no cambian.
- **Métricas:** el RMSE en puntos de volatilidad anualizada, y **QLIKE** (Patton, 2011) como métrica para ordenar. Con una medida ruidosa como la volatilidad realizada a cinco días, QLIKE sigue ordenando los pronósticos como lo haría la varianza verdadera, cosa que el RMSE sobre la volatilidad no garantiza, y castiga más subestimar el riesgo que sobrestimarlo. Las diferencias se prueban con Diebold-Mariano sobre los 2.695 pronósticos juntos, con varianza de Newey-West porque los objetivos a cinco días de filas consecutivas se superponen.

## Resultados

| Modelo | RMSE (puntos de vol. anualizada) | QLIKE | RMSE vs. persistencia | QLIKE vs. persistencia |
|---|---:|---:|---:|---:|
| GARCH(1,1) | 8,79 | 0,380 | 0,80 | 0,36 |
| HAR-X (HAR-RV + VIX y dólar) | 8,59 | 0,385 | 0,78 | 0,36 |
| EWMA (RiskMetrics) | 9,21 | 0,417 | 0,83 | 0,39 |
| HAR-RV | 8,66 | 0,422 | 0,79 | 0,40 |
| CatBoost (Optuna, ajustado en cada fold) | 8,86 | 0,455 | 0,80 | 0,43 |
| CatBoost sin VIX ni dólar | 8,90 | 0,473 | 0,81 | 0,44 |
| MLP (ReLU) | 9,45 | 0,647 | 0,86 | 0,61 |
| MLP (GELU) | 9,64 | 0,686 | 0,87 | 0,64 |
| MLP (Swish) | 9,73 | 0,724 | 0,88 | 0,68 |
| Persistencia (últimos 5 días) | 11,03 | 1,066 | 1,00 | 1,00 |

Ordenados por QLIKE. Bajo 1 en las dos últimas columnas significa mejor que suponer que los próximos cinco días serán como los últimos cinco.

| Comparación (Diebold-Mariano) | p-valor, QLIKE | p-valor, RMSE | Lectura |
|---|---:|---:|---|
| HAR-X vs. GARCH(1,1) | 0,798 | 0,396 | Empate |
| CatBoost vs. GARCH(1,1) | 0,022 | 0,703 | GARCH mejor en QLIKE, empate en RMSE |
| HAR-RV vs. GARCH(1,1) | < 0,001 | 0,218 | GARCH mejor en QLIKE, empate en RMSE |
| EWMA vs. GARCH(1,1) | < 0,001 | < 0,001 | GARCH mejor |
| CatBoost vs. CatBoost sin VIX ni dólar | 0,334 | 0,735 | Sin ganancia detectable de las variables macro |
| HAR-X vs. HAR-RV | 0,127 | 0,763 | Sin ganancia detectable de las variables macro |
| MLP (ReLU) vs. persistencia | < 0,001 | < 0,001 | La MLP es mejor |

![Comparación de modelos contra la persistencia](reports/figures/model_comparison.png)

El RMSE y el QLIKE de cada modelo divididos por los de la persistencia, fold por fold y en escala logarítmica: GARCH, HAR-X, HAR-RV y EWMA quedan juntos, CatBoost les sigue el paso en RMSE pero se queda atrás en QLIKE, y las MLP quedan más cerca de la persistencia.

![Pronósticos en el último fold](reports/figures/forecasts_last_fold.png)

El último fold, de julio de 2024 a septiembre de 2026: los tres mejores modelos siguen el nivel de la volatilidad, pero ninguno ve venir un salto, el de abril de 2025 incluido; reaccionan después. Ese es el límite de lo que puede hacer un pronóstico de volatilidad a cinco días.

## Lo que agregan el VIX y el dólar

| Grupo de features | Participación en el peso SHAP |
|---|---:|
| Retornos | 57,94% |
| Macro (VIX, índice dólar) | 41,32% |
| Calendario | 0,74% |

En el modelo CatBoost ajustado sobre toda la serie, la volatilidad realizada a 60 días es la feature más importante, seguida de la volatilidad a 20 días de los cambios diarios del VIX. Leído solo, eso diría que las variables macro importan mucho. Las pruebas sin ellas dicen otra cosa: CatBoost sin el VIX ni el dólar es 4% peor en QLIKE y HAR-RV con dos de ellas es 9% mejor, y ninguna de las dos diferencias se distingue de cero en once años de pronósticos. Las series macro se mueven con la volatilidad del cobre, así que un modelo de árboles las usa con gusto, pero casi todo lo que traen ya está en la propia volatilidad reciente del cobre.

## Tercer enfoque: MLP de PyTorch (comparación de activaciones)

`src/deep_learning.py` entrena una red feed-forward pequeña (dos capas ocultas y salida Softplus para que los pronósticos sean positivos) con las mismas 24 features, folds y objetivo, con una pérdida Huber más un término de error relativo, y compara ReLU, GELU y Swish. Con datos reales aparecieron dos problemas que la primera versión, simulada, del proyecto arrastraba sin que se notaran:

- **La pérdida estaba mal escalada.** Con volatilidades diarias del orden de 0,01, el término Huber vale cerca de 0,00001 y el relativo cerca de 0,1, así que la red aprendía solo con el término relativo, que premia pronosticar de menos: pronosticaba cerca de la mitad de la volatilidad realizada (mediana 0,0058 contra 0,0112) y casi cero en el 12% de los días, lo que mandaba su QLIKE a los millones. Ahora el objetivo se divide por su media en el fold de entrenamiento, y los dos términos pesan lo que el diseño dice.
- **Sesenta épocas fijas sobreajustaban.** La pérdida de validación tocaba su mínimo cerca de la época 20 y subía después. Ahora el entrenamiento se detiene según el último 20% del bloque de entrenamiento, igual que CatBoost, y se queda con la mejor época: entre la 2 y la 16 según el fold y la activación.

Con las dos cosas corregidas, la red con ReLU le gana a la persistencia (QLIKE 0,647 contra 1,066, p < 0,001), pero sigue siendo la más débil de los modelos serios. Con 2.700 días de objetivos ruidosos, una red pequeña no tiene ventaja sobre un modelo de tres parámetros.

![Pronóstico vs. realizado](reports/figures/predicted_vs_actual.png)

Pronósticos contra volatilidad realizada en el último fold: todos los modelos comprimen sus pronósticos en una banda angosta, porque la mayor parte de la volatilidad de una semana es ruido que ningún pronóstico puede seguir.

![Distribución de los errores](reports/figures/residual_distribution.png)

Los errores son asimétricos: los modelos sobrestiman las semanas tranquilas por unos pocos puntos y subestiman las pocas semanas turbulentas por mucho más, que es justo lo que QLIKE castiga.

![Curvas de pérdida de la MLP por activación, animadas](reports/figures/mlp_loss_curves_animated.gif)

La pérdida de entrenamiento y de validación época por época, dibujada a medida que se entrenaba.

![Curvas de pérdida de la MLP por activación](reports/figures/mlp_loss_curves.png)

Las mismas curvas con la época elegida por el early stopping marcada: la pérdida de entrenamiento sigue bajando mientras la de validación se queda plana cerca de 0,30 desde las primeras épocas.

## Cómo se evita el sesgo de anticipación

1. **Las features del cobre solo usan retornos hasta el día anterior.** Toda ventana móvil corre sobre `log_return.shift(1)`; un test perturba el precio de un día y comprueba que las features de ese día no se mueven y las del día siguiente sí.
2. **Las features macro se unen por fecha de publicación.** El VIX entra con cierres hasta el día calendario anterior; el índice dólar, solo cuando ya salió su entrega semanal. Los dos se prueban perturbando un valor justo en el día en que debe, y en el que no debe, volverse visible.
3. **GARCH tiene el mismo corte de información que las features.** El pronóstico móvil de `arch` en el origen *o* actualiza su varianza con el retorno del propio día *o* (verificado empíricamente, no supuesto), así que el pronóstico de la fila *i* se toma desde el origen *i−1*, pidiendo un pronóstico a (horizonte+1) pasos y descartando el primero.
4. **Nada se ajusta ni se detiene con los datos con que se evalúa.** Optuna, el early stopping de CatBoost y el de la MLP usan el final del bloque de entrenamiento; los tests reescriben los objetivos de validación de un fold y comprueban que sus pronósticos no cambian.

## Qué cambió respecto de la primera versión

La primera versión de este proyecto corría sobre un mercado simulado GARCH-X, donde GARCH ganaba por construcción. Pasarlo a datos reales cambió más que las cifras:

- **Datos:** el precio real del cobre en la LME, el VIX y el índice dólar reemplazan a las series simuladas; el volumen salió porque no existe una serie real gratuita.
- **Fugas eliminadas:** CatBoost hacía early stopping con el mismo fold de validación con que se le evaluaba, y Optuna ajustaba una sola vez con el último 20% de la serie, que se superpone con el último fold. Ahora los dos usan solo el bloque de entrenamiento de cada fold.
- **La escala de la pérdida de la MLP y sus épocas fijas** se corrigieron, como se explica arriba.
- **Evaluación:** son nuevas las referencias de persistencia, EWMA y HAR-X, QLIKE, los tests de Diebold-Mariano y la prueba sin variables macro. El objetivo ahora es la raíz de la media de los cuadrados de los cinco retornos siguientes, lo que pronostica un GARCH de media cero, en vez de su desviación estándar muestral.

GARCH(1,1) sigue quedando arriba, y esta vez no porque los datos estuvieran hechos para él.

## Stack tecnológico

| Capa | Tecnología | Rol |
|---|---|---|
| Datos | **urllib**, **Polars** | Descarga, validación y features sin sesgo de anticipación |
| Econometría | **arch** (GARCH), **statsmodels** (HAR-RV, Newey-West) | Referencias y tests de Diebold-Mariano |
| Machine learning | **CatBoost**, **Optuna**, **scikit-learn** | Gradient boosting, ajuste por fold, particiones walk-forward |
| Deep learning | **PyTorch** | MLP con tres funciones de activación |
| Explicabilidad | **SHAP** | Atribución por feature y por grupo de features |
| Almacenamiento | **DuckDB** | Métricas y pronósticos fuera de muestra de cada corrida |

## Cómo correrlo

```powershell
py -m venv venv
./venv/Scripts/pip install -r requirements.txt
./venv/Scripts/python main.py --download   # una vez: 17 archivos de mindicador.cl y FRED en data/raw/
./venv/Scripts/python main.py              # unos 15 minutos en la CPU de un notebook
```

Escribe los gráficos en `reports/figures/`, el resumen detrás de cada cifra de este README en `reports/results.json` y los artefactos completos (valores SHAP, el modelo CatBoost, el historial de Optuna, un archivo DuckDB con cada pronóstico) en `outputs/`. El notebook [`02_CatBoost_Optuna_GARCH_Comparison.ipynb`](02_CatBoost_Optuna_GARCH_Comparison.ipynb) lee esos artefactos.

### Tests

```powershell
./venv/Scripts/pytest -v
```

57 tests, sin necesidad de red (los tests del pipeline corren sobre un mercado simulado pequeño con el mismo esquema que el real): validación de los archivos crudos (duplicados, fechas en fin de semana, unidades equivocadas, huecos), el corte de publicación del índice dólar, chequeos de anticipación en cada grupo de features, el corte de información de GARCH, un test de que un fold nunca ve sus propios objetivos de validación, QLIKE y Diebold-Mariano, el early stopping de la MLP, gráficos, la persistencia en DuckDB y un chequeo de que cada cifra de la tabla de resultados de los dos README coincide con `reports/results.json`.

## Próximos pasos

- Usar el calendario de feriados de la LME para escalar los retornos que cubren más de una sesión, en vez de tratarlos como un solo día.
- Precios intradiarios, para construir la varianza realizada con retornos de cinco minutos: el insumo estándar de HAR-RV, mucho menos ruidoso que la raíz de la media diaria.
- Un GARCH-X con el VIX en la ecuación de varianza, para probar las variables macro dentro del modelo que gana.
- Una historia más larga del cobre: mindicador parte en octubre de 2012.

## Licencia

MIT — ver [LICENSE](LICENSE).

## Autor

**Pablo Reyes** — [github.com/Rxyxs](https://github.com/Rxyxs)
