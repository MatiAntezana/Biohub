---
id: EXP-030B
estado: completado-cerrado
tipo: nueva-version-full-fit-de-exp-030
fecha: 2026-08-29
---

# EXP-030B — Dual-Seed Temporal Ensemble Full-Fit

## Estado real

**Completado y cerrado.** Entrenamiento full-fit dual-seed finalizado exitosamente en el nodo remoto `A1557` (GPU 0). Ambos seeds entrenaron sus 55 épocas completas sobre 199 pares de datos Biohub. Los checkpoints y recibos fueron sincronizados e incorporados al ledger canónico.

## Objetivo de la versión

Reentrenar los dos seeds de la estrategia temporal con todos los pares `.zarr/.geff` disponibles. No se retiene ningún embrión ni volumen para validación. Por lo tanto, esta ejecución no produce CV score local y sólo puede evaluarse en el gate posterior de inferencia/test/Kaggle.

## Configuración congelada

| Campo | Valor |
|---|---|
| Experimento padre | `EXP-030` |
| Seeds | `3407`, `9479` |
| Datos | `100 % train / 0 % validación` |
| Ventana temporal | `3` frames |
| UNet | canales `32,64,128`, salida `32` |
| Downsample | `(1,4,4)` |
| Epochs | `55` por seed |
| Iteraciones/epoch | `750` |
| Batch | `2` |
| Checkpoint desplegable | última época, sin selección por validación |
| Peso loss detección | `1.0` |
| Peso negativos | `0.02` |
| TTA futura | 8 vistas D4 XY |
| Fusión detección | `52.5 %` primaria / `47.5 %` secundaria |
| Guard | retención mínima `0.90` |

## Archivos canónicos

- Launcher: `experiments/EXP-030B_DualSeed_Temporal_Ensemble_FullFit/train.py`.
- Contrato: `experiments/EXP-030B_DualSeed_Temporal_Ensemble_FullFit/experiment.json`.
- Entrenador compartido: `official_starter_kit/scripts/train_unet_transformer.py`.
- Runtime y split full-fit: `src/strategies/experiment_runtime.py`.
- Ensemble/TTA: `src/strategies/ensemble_temporal.py`.
- Estrategia madre: [[../Estrategias/Estrategia_Experimental_DualSeed_TTA_ILP|Dual-Seed + TTA + ILP]].
- Carpeta propia: `experiments/EXP-030B_DualSeed_Temporal_Ensemble_FullFit/`.

## Lanzamiento remoto

```bash
python infra/experimentctl.py run EXP-030B --node A1556 --dry-run
python infra/experimentctl.py run EXP-030B --node A1556  # sólo cuando esté READY y se ordene explícitamente
```

## Monitoreo verificable

- Split y recibos: `experiments/EXP-030B_DualSeed_Temporal_Ensemble_FullFit/artifacts/<run_id>/`.
- Checkpoints: `experiments/EXP-030B_DualSeed_Temporal_Ensemble_FullFit/checkpoints/<run_id>/<role>/training_checkpoint.pth`.
- Reporte factual: `experiments/EXP-030B_DualSeed_Temporal_Ensemble_FullFit/validation/<run_id>/REPORT.md`.
- Estado global: `.state/STATE.md` y `experiments/registry.jsonl`.

## Registro de resultado pendiente

Al terminar se deben anotar: nodo/GPU, entorno, timestamps, cantidad real de datasets, hashes de ambos checkpoints, última loss por seed, errores/warnings y decisión posterior. El campo CV debe quedar como `N/A — full-fit sin validación`; nunca convertir la loss de train en score de validación.
