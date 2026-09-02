---
id: EXP-031
estado: codigo-listo-esperando-gpu
tipo: estrategia-experimental-independiente
fecha: 2026-08-26
---

# EXP-031 — Detector-First + ILP preservado

## Estado real

**Revisión full-fit lista para relanzar.** La ejecución embryo-disjoint anterior fue detenida por instrucción del usuario. Esta revisión reemplaza su configuración de entrenamiento por `100 % train / 0 % validación` y debe ejecutarse en una computadora remota. La Mac local sólo se usa para editar y orquestar.

## Hipótesis

Una loss que concentre el aprendizaje negativo en falsos positivos difíciles reducirá candidatos basura. Mantener el ILP como topología base y reparar sólo nodos libres debería evitar la regresión observada cuando motion matching reemplaza todo el grafo.

## Configuración congelada

| Campo | Valor |
|---|---|
| Seed | `7319` |
| Ventana temporal | `3` frames |
| Split | full-fit, todos los pares detectados para train, validación vacía |
| Epochs | `65` |
| Iteraciones/epoch | `850` |
| Batch | `2` |
| Checkpoint desplegable | última época, sin selección por validación |
| Loss detección | focal positives + top-k hard negatives |
| Ratio hard negatives | `48 × N_positivos` |
| Zona ignorada ZYX | kernel `(3,5,5)` |
| Peso detección | `2.0` |
| Peso negativo | `0.25` |
| ILP | preservado, umbral `0.48` |
| Motion | sólo unmatched, gates `6/10 µm` |

## Lanzamiento remoto

```bash
python infra/experimentctl.py run EXP-031 --node A1556
```

Para preflight sin cómputo:

```bash
python infra/experimentctl.py run EXP-031 --node A1556 --dry-run
```

## Monitoreo verificable

- Split, recibos y métricas: `experiments/EXP-031_Detector_First_ILP_Preservado/artifacts/<run_id>/`.
- Log: `experiments/EXP-031_Detector_First_ILP_Preservado/logs/<run_id>.log`.
- Checkpoint: `experiments/EXP-031_Detector_First_ILP_Preservado/checkpoints/<run_id>/model/training_checkpoint.pth`.
- Estado global: `.state/STATE.md` y `experiments/registry.jsonl`.

## Criterio de éxito

Esta revisión no tiene embrión retenido ni CV local. La loss sólo sirve para auditar estabilidad. El resultado debe registrarse como `N/A — full-fit sin validación` hasta medir Node Recall, precisión/conteo, adjusted edge Jaccard y score oficial en el gate posterior de inferencia/test/Kaggle. El barrido de threshold se realiza después del entrenamiento sin cambiar el linker.
