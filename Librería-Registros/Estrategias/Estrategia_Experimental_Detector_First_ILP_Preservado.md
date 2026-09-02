---
tags: [biohub, estrategia, experimental, detector-first, hard-negative, ilp, ablation]
estado: codigo-listo
experimento: EXP-031
---

# Estrategia experimental Detector-First con ILP preservado

## Propósito

Esta segunda línea implementa las correcciones sugeridas por la auditoría exhaustiva y por [[../../problematica_hipotesis_y_conclusion|problematica_hipotesis_y_conclusion.md]]. La hipótesis es que el cuello de botella no es necesariamente el backbone, sino la cantidad de falsos positivos que contamina el conjunto de candidatos del linker.

## Hipótesis verificable

Si se reduce la masa de picos falsos de alta confianza sin degradar excesivamente el Node Recall, entonces:

$$
Edge\ Recall \approx Node\ Recall^2 \times Conditional\ Linking\ Accuracy
$$

debería mejorar porque el Transformer y el ILP recibirán un espacio de candidatos menos ruidoso.

## Cambios implementados

1. Backbone temporal y Transformer se mantienen para no mezclar cambio de arquitectura con cambio de objetivo.
2. Nueva loss de detección:
   - focal loss para todos los centros anotados;
   - zona anisotrópica de exclusión alrededor de positivos;
   - sólo los `48 × N_positivos` negativos más difíciles contribuyen al gradiente;
   - mayor peso negativo (`0.25`) que en la rama base.
3. Revisión de despliegue full-fit: todos los pares se usan para train y no se crea loader de validación.
4. Reparación de movimiento únicamente en fuentes y destinos que quedaron libres tras ILP.
5. Preservación explícita de aristas ILP con confianza mínima.
6. Divisiones conservadoras con mutual-nearest-neighbor real, continuidad hasta `t+2`, crecimiento de separación y ángulo real entre ramas.
7. Reportes de candidatos, aceptados y rechazos; los contadores no pueden volverse negativos.

## Código

- Entrenamiento remoto: `experiments/EXP-031_Detector_First_ILP_Preservado/train.py`, orquestado por `infra/experimentctl.py`.
- Loss detector-first: `src/strategies/detector_precision.py`.
- Reparación auditable: `src/strategies/ilp_preserving_repair.py`.
- Hook explícito de loss: parámetro opcional `detection_loss_fn` en `official_starter_kit/scripts/train_unet_transformer.py`.
- Configuración del experimento: [[../Experimentos/EXP-031_Detector_First_ILP_Preservado|EXP-031]].

## Por qué es distinta de EXP-030

- EXP-030 pregunta si el ensemble temporal completo aporta robustez.
- EXP-031 pregunta si limpiar el detector y preservar ILP mejora el resultado sin acumular heurísticas.
- EXP-031 usa un solo seed inicialmente: la atribución de cualquier delta permanece clara.

## Ablaciones posteriores obligatorias

1. Loss original vs. hard-negative focal.
2. ILP solo vs. ILP + motion unmatched.
3. Con y sin reparaciones de división.
4. Barrido de threshold de detección manteniendo linker fijo.
5. Reportar node precision/recall, adjusted edge Jaccard y Division Jaccard oficial por embrión.

## Revisión full-fit vigente

La primera ejecución embryo-disjoint fue detenida. La configuración vigente de EXP-031 usa `65` épocas de `850` iteraciones, `100 %` de los datos para entrenamiento y `0 %` para validación. El checkpoint se selecciona por última época; no existe CV local válido para esta revisión.
