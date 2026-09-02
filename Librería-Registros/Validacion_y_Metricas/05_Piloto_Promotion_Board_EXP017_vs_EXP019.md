---
tags: [biohub, validacion, promotion-board, exp-017, exp-019, piloto]
estado: finalizado
fecha: 2026-08-27
---

# Piloto Promotion Board — EXP-017 vs EXP-019

## Diseño ejecutado

- Nodo remoto: `A1558-UBU`, RTX 4090.
- Protocolo congelado: `biohub-legacy-seed42-holdout-common12-v1`.
- Datos: 12 pares `.zarr/.geff` pertenecientes al holdout histórico seed 42 de ambos modelos.
- Misma TTA: identidad + flip X.
- Mismo tracker: Hungarian, gap closing `12/16 µm`, mínimo 2 frames, sin mitosis.
- Matching oficial: `7 µm`.
- EXP-017: Wide Anisotropic U-Net 64ch, threshold histórico congelado `0.035`.
- EXP-019: Wide ResUNet3D 64ch, threshold histórico congelado `0.050`.
- Bootstrap pareado: `10.000` remuestreos, seed `20260827`.

## Resultados oficiales

| Métrica | EXP-017 | EXP-019 |
|---|---:|---:|
| Score oficial | **`0.8098`** | `0.7783` |
| IC bootstrap 95% | `[0.7431, 0.8727]` | `[0.7005, 0.8494]` |
| Adjusted Edge Jaccard | **`0.8098`** | `0.7783` |
| Edge Jaccard | **`0.7939`** | `0.7586` |
| Node Recall medio | **`0.8676`** | `0.8593` |
| Division Jaccard | `0.0000` | `0.0000` |
| Edge TP / FP / FN | `8492 / 808 / 1396` | `8150 / 856 / 1738` |
| Runtime total | **`421.0 s`** | `524.8 s` |
| Promotion Score (límite inferior) | **`0.7431`** | `0.7005` |

Comparación pareada:

- Delta EXP-017 − EXP-019: **`+0.0315`**.
- IC 95% del delta: **`[+0.0046, +0.0592]`**.
- Probabilidad bootstrap de que EXP-017 sea mejor: **`98.94 %`**.

## Robustez por dominio

| Grupo | EXP-017 | EXP-019 |
|---|---:|---:|
| `44b6` | **`0.4231`** | `0.3757` |
| `6bba` | **`0.8230`** | `0.7911` |

Peor muestra compartida: `44b6_0b24845f`. EXP-017 obtuvo `0.0413` y EXP-019 `0.1790`. Aunque EXP-017 gana en el agregado y en ambos grupos, EXP-019 fue menos malo en esta muestra extrema. Esto obliga a conservar resultados por dataset y no decidir sólo por el promedio.

## Concordancia externa

| Experimento | CV histórico reportado | Piloto comparable | Kaggle LB |
|---|---:|---:|---:|
| EXP-017 | `0.9010` | `0.8098` | **`0.6830`** |
| EXP-019 | `0.8867` | `0.7783` | `0.6430` |

El piloto reproduce correctamente el orden de Kaggle y su delta (`0.0315`) tiene el mismo sentido que el delta LB (`0.0400`). Con sólo dos puntos, cualquier correlación de ranking sería trivial y no constituye validación estadística de una métrica global.

## Decisión

**Adoptar el Promotion Board como infraestructura de decisión, pero no adoptar este holdout común de 12 muestras como métrica global definitiva.** El candidato promovido por este piloto es EXP-017.

Para convertirlo en protocolo global se requiere:

1. protocolo group-disjoint/OOF congelado para experimentos futuros;
2. thresholds seleccionados dentro de folds internos, no sobre el tablero final;
3. evaluación de al menos 5–10 arquitecturas con resultados Kaggle conocidos;
4. datasets con divisiones suficientes para validar el segundo término de la métrica;
5. reportar score, intervalo pareado, peor dataset, grupos biológicos, runtime y cobertura;
6. no reutilizar resultados del antiguo evaluador estimativo.

## Artefactos

- JSON crudos: `experiments/Promotion_Board_EXP017_EXP019_Pilot/remote_results/`.
- Tablero: `experiments/Promotion_Board_EXP017_EXP019_Pilot/board/promotion_board.json`.
- Reporte renderizado: `experiments/Promotion_Board_EXP017_EXP019_Pilot/board/promotion_board.md`.
