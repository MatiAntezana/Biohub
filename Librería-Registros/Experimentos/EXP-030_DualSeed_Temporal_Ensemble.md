---
id: EXP-030
estado: codigo-listo-esperando-gpu
tipo: estrategia-experimental-independiente
fecha: 2026-08-26
---

# EXP-030 — Dual-Seed TemporalUNet + Transformer Ensemble

## Estado real

**Código listo; entrenamiento no iniciado.** Debe esperarse a que una computadora remota con RTX 4090 esté libre. No ejecutar en la Mac local.

## Hipótesis

Dos modelos temporales entrenados desde cero con seeds independientes, combinados mediante calibración, TTA y controles de degradación, deberían reducir la varianza de detección y resolver asociaciones ambiguas mejor que el pipeline frame-a-frame.

## Configuración congelada

| Campo | Valor |
|---|---|
| Seeds | `3407`, `9479` |
| Ventana temporal | `3` frames |
| UNet | canales `32,64,128`, salida `32` |
| Downsample | `(1,4,4)` |
| Split | leave-one-embryo-prefix-out, fold `0` |
| Epochs | `40` por seed |
| Iteraciones/epoch | `600` |
| Batch | `2` |
| Peso loss detección | `1.0` |
| Peso negativos | `0.02` |
| TTA futura | 8 vistas D4 XY |
| Fusión detección | `52.5 %` primaria / `47.5 %` secundaria |
| Guard | retención mínima `0.90` |
| Reverse support | `30 %` |

## Lanzamiento remoto

El controlador valida, congela y sincroniza el código. Preparar primero sin lanzar:

```bash
python infra/experimentctl.py run EXP-030 --node A1556 --dry-run
```

Para lanzar, repetir de forma explícita sin `--dry-run`:

```bash
python infra/experimentctl.py queue EXP-030
python infra/experimentctl.py run EXP-030 --node A1556
```

## Monitoreo verificable

- Recibos y métricas: `experiments/EXP-030_DualSeed_Temporal_Ensemble/artifacts/<run_id>/`.
- Logs: `experiments/EXP-030_DualSeed_Temporal_Ensemble/logs/`.
- Checkpoints: `experiments/EXP-030_DualSeed_Temporal_Ensemble/checkpoints/<run_id>/<role>/training_checkpoint.pth`.
- Estado global: `.state/STATE.md` y `experiments/registry.jsonl`.

## Gate posterior

No subir automáticamente a Kaggle al terminar. Primero ejecutar la evaluación ensemble oficial, comprobar hashes y estimar runtime del test privado. Luego usar el handoff preparado en [[../Bitacora_Estrategica/Mensaje_Handoff_Antigravity_EXP-030|Mensaje para Antigravity]].
