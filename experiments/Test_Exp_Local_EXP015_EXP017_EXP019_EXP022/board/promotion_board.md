# Experiment Promotion Board — biohub-legacy-seed42-holdout-common12-v1

Datasets comparables: `12`. Ranking: límite inferior bootstrap 95% del score oficial.

| Rank | Experimento | Score oficial | IC 95% | Peor muestra | Adj Edge | Div | Recall | Runtime | Decisión |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | EXP-015 | 0.8144 | [0.7551, 0.8703] | 0.1065 | 0.8144 | 0.0000 | 0.8913 | 263.3s | SUBMIT_CANDIDATE |
| 2 | EXP-017 | 0.8099 | [0.7448, 0.8709] | 0.0413 | 0.8099 | 0.0000 | 0.8676 | 415.3s | HOLD |
| 3 | EXP-022 | 0.7831 | [0.7222, 0.8396] | 0.2256 | 0.7831 | 0.0000 | 0.9090 | 199.9s | HOLD |
| 4 | EXP-019 | 0.7787 | [0.7037, 0.8510] | 0.1790 | 0.7787 | 0.0000 | 0.8593 | 523.3s | HOLD |

## Desglose por embrión

- `EXP-015`: 44b6=0.4611, 6bba=0.8261; peor dataset `44b6_0b24845f`=0.1065.
- `EXP-017`: 44b6=0.4219, 6bba=0.8232; peor dataset `44b6_0b24845f`=0.0413.
- `EXP-022`: 44b6=0.4605, 6bba=0.7939; peor dataset `44b6_0b24845f`=0.2256.
- `EXP-019`: 44b6=0.3757, 6bba=0.7915; peor dataset `44b6_0b24845f`=0.1790.

## Comparaciones pareadas

- `EXP-015 − EXP-017`: Δ `0.0045`; IC 95% `[-0.0059, 0.0138]`; P(Δ>0) `0.8080`.
- `EXP-015 − EXP-019`: Δ `0.0357`; IC 95% `[0.0087, 0.0634]`; P(Δ>0) `0.9960`.
- `EXP-015 − EXP-022`: Δ `0.0313`; IC 95% `[0.0206, 0.0449]`; P(Δ>0) `1.0000`.
- `EXP-017 − EXP-019`: Δ `0.0312`; IC 95% `[0.0052, 0.0579]`; P(Δ>0) `0.9912`.
- `EXP-017 − EXP-022`: Δ `0.0268`; IC 95% `[0.0107, 0.0449]`; P(Δ>0) `0.9998`.
- `EXP-019 − EXP-022`: Δ `-0.0044`; IC 95% `[-0.0253, 0.0155]`; P(Δ>0) `0.3416`.

## Limitaciones

- Confidence intervals quantify dataset-sampling variability, not split leakage or threshold-selection bias.
- A promotion board is globally comparable only when every experiment uses the same frozen, unseen protocol.
