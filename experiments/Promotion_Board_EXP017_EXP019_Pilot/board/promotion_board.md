# Experiment Promotion Board — biohub-legacy-seed42-holdout-common12-v1

Datasets comparables: `12`. Ranking: límite inferior bootstrap 95% del score oficial.

| Rank | Experimento | Score oficial | IC 95% | Peor muestra | Adj Edge | Div | Recall | Runtime | Decisión |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | EXP-017 | 0.8098 | [0.7431, 0.8727] | 0.0413 | 0.8098 | 0.0000 | 0.8676 | 421.0s | SUBMIT_CANDIDATE |
| 2 | EXP-019 | 0.7783 | [0.7005, 0.8494] | 0.1790 | 0.7783 | 0.0000 | 0.8593 | 524.8s | HOLD |

## Desglose por embrión

- `EXP-017`: 44b6=0.4231, 6bba=0.8230; peor dataset `44b6_0b24845f`=0.0413.
- `EXP-019`: 44b6=0.3757, 6bba=0.7911; peor dataset `44b6_0b24845f`=0.1790.

## Comparaciones pareadas

- `EXP-017 − EXP-019`: Δ `0.0315`; IC 95% `[0.0046, 0.0592]`; P(Δ>0) `0.9894`.

## Limitaciones

- Confidence intervals quantify dataset-sampling variability, not split leakage or threshold-selection bias.
- A promotion board is globally comparable only when every experiment uses the same frozen, unseen protocol.
