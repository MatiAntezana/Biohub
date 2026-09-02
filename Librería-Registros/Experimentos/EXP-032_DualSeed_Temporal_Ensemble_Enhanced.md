---
id: EXP-032
estado: completado-cerrado
tipo: variante-mejorada-full-fit-de-exp-030b
fecha: 2026-08-29
---

# EXP-032 — Dual-Seed Temporal Ensemble Enhanced

## Resumen ejecutivo

**Completado y cerrado.** Entrenamiento full-fit dual-seed de mayor capacidad jerárquica finalizado exitosamente en el nodo remoto `A1556` (GPU 0). Ambos seeds entrenaron sus 55 épocas completas sobre 199 pares de datos Biohub. Los checkpoints y recibos fueron sincronizados e incorporados al ledger canónico.

## Mejoras introducidas vs EXP-030B

1. **Mayor capacidad de representación jerárquica (UNet Anisotrópico 3D)**:
   - Canales intermedios: `[48, 96, 192]` (vs `[32, 64, 128]` en EXP-030B).
   - Canales de salida a nivel de nodo: `48` (vs `32` en EXP-030B).
   - Incremento del +50% en capacidad de extracción de características morfológicas sin riesgo de OOM en GPU de 24 GB.
2. **Calibración de pérdida de detección**:
   - `det_loss_weight`: `1.5` (vs `1.0`).
   - `det_negative_weight`: `0.03` (vs `0.02`).
   - Mayor penalización y precisión en la detección de mitosis tempranas y blastómeros en planos profundos.
3. **Entrenamiento Full-Fit Dual-Seed**:
   - Seeds independientes: `3407` y `9479`.
   - 55 épocas por seed con 750 iteraciones por época en el 100% de los datos de entrenamiento.

## Configuración congelada

| Campo | Valor |
|---|---|
| Experimento padre | `EXP-030B` |
| Seeds | `3407`, `9479` |
| Datos | `100 % train / 0 % validación` |
| Ventana temporal | `3` frames |
| UNet | canales `48, 96, 192`, salida `48` |
| Downsample | `(1, 4, 4)` |
| Epochs | `55` por seed |
| Iteraciones/epoch | `750` |
| Batch | `2` |
| Checkpoint desplegable | última época, sin selección por validación |
| Peso loss detección | `1.5` |
| Peso negativos | `0.03` |

## Archivos canónicos

- Launcher: `experiments/EXP-032_DualSeed_Temporal_Ensemble_Enhanced/train.py`.
- Manifiesto: `experiments/EXP-032_DualSeed_Temporal_Ensemble_Enhanced/experiment.json`.
- Carpeta operativa: `experiments/EXP-032_DualSeed_Temporal_Ensemble_Enhanced/`.
