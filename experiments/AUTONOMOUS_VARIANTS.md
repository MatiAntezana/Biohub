# Variantes autónomas

> Vista generada desde `experiments/variant_proposals.jsonl`. Kaggle Public Score es la señal primaria; el test local permanece separado.

| Experimento | Carril | Padre | Score padre | Receta | Estado | Public Score |
|---|---|---|---:|---|---|---:|
| EXP-033 | exploitation | EXP-017 | 0.6830 | exploit_cosine_finetune | **SCORED** | 0.6510 |
| EXP-034 | exploitation | EXP-017 | 0.6830 | exploit_cosine_finetune | **SCORED** | 0.6360 |
| EXP-035 | adjacent_exploration | EXP-019 | 0.6430 | adjacent_groupnorm | **FAILED** | — |
| EXP-036 | exploitation | EXP-019 | 0.6430 | exploit_cosine_finetune | **KAGGLE_PENDING** | — |
| EXP-037 | exploitation | EXP-017 | 0.6830 | exploit_regularized | **SCORED** | 0.6510 |
| EXP-038 | adjacent_exploration | EXP-019 | 0.6430 | adjacent_attention | **KAGGLE_PENDING** | — |
| EXP-039 | exploitation | EXP-017 | 0.6830 | exploit_longer_lower_lr | **KAGGLE_PENDING** | — |
| EXP-040 | exploitation | EXP-017 | 0.6830 | exploit_cosine_finetune | **KAGGLE_PENDING** | — |
| EXP-041 | exploitation | EXP-017 | 0.6830 | exploit_regularized | **KAGGLE_PENDING** | — |
| EXP-042 | exploitation | EXP-019 | 0.6430 | exploit_cosine_finetune | **KAGGLE_PENDING** | — |
| EXP-043 | exploitation | EXP-017 | 0.6830 | exploit_regularized | **KAGGLE_PENDING** | — |
| EXP-044 | exploitation | EXP-019 | 0.6430 | exploit_regularized | **KAGGLE_PENDING** | — |
| EXP-045 | adjacent_exploration | EXP-033 | 0.6510 | adjacent_attention | **KAGGLE_PENDING** | — |
| EXP-046 | exploitation | EXP-017 | 0.6830 | exploit_regularized | **KAGGLE_PENDING** | — |

## Política

- 60% explotación de los mejores linajes Kaggle.
- 25% exploración adyacente con cambios arquitectónicos cercanos.
- 15% exploración novedosa; en cada lote completo hay al menos una variante novedosa.
- Ninguna variante puede promoverse por `train_loss`; solo sirve como diagnóstico de ejecución.
- Una carpeta materializada sigue con `kaggle.deployment_ready=false` hasta tener pesos exportados y notebook test-only validado.
