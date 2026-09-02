# Benchmark test-exp-local — EXP-015 / EXP-017 / EXP-019 / EXP-022

Ejecución real en `A1554-ubu` (RTX 4090) con el entorno Conda `rl`, el 2026-08-28.
Se usó el protocolo `biohub-legacy-seed42-holdout-common12-v1`: las mismas 12
muestras congeladas y etiquetadas para los cuatro modelos. Cada JSON contiene
las filas oficiales por dataset, no un score inferido desde logs.

## Resultado

| Orden board | Experimento | Arquitectura | Score test local | IC bootstrap 95% | CV histórico registrado | Runtime |
|---:|---|---|---:|---:|---:|---:|
| 1 | EXP-015 | Anisotropic U-Net 32ch | **0.8144** | [0.7551, 0.8703] | 0.8096 | 263.3 s |
| 2 | EXP-017 | Wide U-Net 64ch | **0.8099** | [0.7448, 0.8709] | 0.9010 | 415.3 s |
| 3 | EXP-022 | StarDist 3D 32ch | **0.7831** | [0.7222, 0.8396] | 0.8601 | 199.9 s |
| 4 | EXP-019 | Wide ResUNet 64ch | **0.7787** | [0.7037, 0.8510] | 0.8867 | 523.3 s |

Todos terminaron con `SUCCESS`, cobertura 12/12 y sin datasets omitidos.
El tablero completo está en `board/promotion_board.md` y `board/promotion_board.json`.

## Interpretación

La prueba confirma que `test-exp-local` calcula métricas oficiales por muestra,
reconstruye el score y produce intervalos/ranking reproducibles. No confirma
que el score sea todavía un predictor universal de Kaggle: el ranking test
(`015 > 017 > 022 > 019`) difiere del CV histórico (`017 > 019 > 022 > 015`).

Esto no demuestra que el evaluador esté mal. Los CV históricos se obtuvieron
con otros holdouts completos y pipelines de validación distintos; además,
EXP-022 conserva su inferencia histórica single-pass mientras los modelos U-Net
usan TTA `identity + flip_x`. Por ello este resultado debe tratarse como una
señal de generalización sobre el protocolo congelado, no como calibración LB.

## Recomendación operativa

Para experimentos futuros, usar el límite inferior bootstrap 95% como criterio
principal sólo después de mantener un protocolo oculto, un pipeline de
postprocesado comparable y al menos varios embriones/muestras. Registrar el
score de Kaggle después de cada submission y recalibrar la relación test→LB;
no ordenar candidatos únicamente por el CV histórico ni convertir este piloto
de 12 muestras en una garantía de leaderboard.
