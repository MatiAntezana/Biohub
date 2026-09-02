---
tags: [biohub, estrategia, experimental, temporal-unet, transformer, ensemble, tta, ilp]
estado: codigo-listo
experimento: EXP-030
---

# Estrategia experimental Dual-Seed + TTA + ILP

## Propósito

Esta es una línea **aparte del pipeline campeón de heatmaps 3D**. Su objetivo es comprobar si una arquitectura temporal entrenada de punta a punta puede mejorar simultáneamente detección y asociación, y si la combinación de dos inicializaciones independientes reduce errores sistemáticos.

La propuesta nace de [[../../analisis-estrategia-extra|analisis-estrategia-extra.md]] y fue contrastada con [[../../analisis_exhaustivo_biohub_track_your_cells|analisis_exhaustivo_biohub_track_your_cells.md]]. Ambos textos estudian la misma familia de notebook; este experimento implementa la receta positiva y congela sus constantes para poder medirla como rama independiente.

## Pipeline implementado

1. `TemporalUNet3D` sobre ventanas de tres frames.
2. `SimpleNodeTransformer` para puntuar aristas entre frames consecutivos.
3. Entrenamiento desde cero de dos modelos con seeds `3407` y `9479`.
4. Split leave-one-embryo-prefix-out; por defecto el fold 0 mantiene un embrión completo fuera del entrenamiento.
5. TTA D4 de ocho vistas en XY.
6. Alineación de media/desvío del seed secundario y fusión `0.525 / 0.475`.
7. Guard de retención: fallback por frame si la fusión conserva menos de `90 %` de los candidatos primarios.
8. Fusión de aristas secundaria sólo con bajo margen y consenso de mejor padre.
9. Evidencia forward/reverse mediante media armónica ponderada `70/30`.
10. ILP para obtener un grafo globalmente consistente.

## Código

- Entrenamiento remoto: `experiments/EXP-030_DualSeed_Temporal_Ensemble/train.py`, orquestado por `infra/experimentctl.py`.
- Primitivas de ensemble/TTA: `src/strategies/ensemble_temporal.py`.
- Splits, seeds, hashes y sentinels: `src/strategies/experiment_runtime.py`.
- Configuración del experimento: [[../Experimentos/EXP-030_DualSeed_Temporal_Ensemble|EXP-030]].
- Versión full-fit con todos los datos: [[../Experimentos/EXP-030B_DualSeed_Temporal_Ensemble_FullFit|EXP-030B]].

## Diferencias respecto de los experimentos anteriores

- No es otro U-Net de heatmap por frame: aprende representación temporal y enlaces de forma conjunta.
- No depende de un único checkpoint.
- La topología final se delega a ILP, no sólo a Hungarian por distancia.
- Las transformaciones y fusiones están implementadas como funciones versionadas, sin parchear strings de un notebook.

## Riesgos controlados

- Dos seeds duplican el costo de entrenamiento.
- Un ensemble puede reducir recall: se mitiga con el guard por frame.
- La validación proxy del notebook analizado no era oficial: EXP-030 no se promueve a Kaggle sin evaluación oficial del starter kit.
- El notebook original superponía múltiples heurísticas. Aquí se registra por separado la fase de entrenamiento y la futura fase de inferencia.

## Criterio de promoción

La estrategia sólo pasa a empaquetado Kaggle si:

1. existen los dos checkpoints y sus SHA-256;
2. la ejecución emitió `experiments/EXP-030_DualSeed_Temporal_Ensemble/artifacts/<run_id>/TRAINING_COMPLETE.json` y fue reconciliada en el ledger;
3. no hay solapamiento de prefijos de embrión entre train y validación;
4. la evaluación oficial mejora o complementa al champion actual;
5. runtime estimado de inferencia entra dentro del límite de la competencia.

## Variante full-fit EXP-030B

EXP-030B conserva arquitectura, seeds y política de ensemble, pero usa todos los datos para train y ninguna validación. Aumenta el presupuesto de cada seed de `40 × 600` a `55 × 750`. Su checkpoint desplegable es el de la última época y su CV es obligatoriamente `N/A`; EXP-030 original permanece como referencia histórica con split por embrión.
