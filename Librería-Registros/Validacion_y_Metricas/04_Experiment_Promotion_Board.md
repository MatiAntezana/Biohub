---
tags: [biohub, validacion, promotion-board, metrica-oficial, bootstrap]
estado: implementado-piloto
fecha: 2026-08-27
---

# Experiment Promotion Board

## Objetivo

Decidir qué experimento merece una ejecución Kaggle usando evidencia comparable. El tablero no estima scores desde losses, logits ni autoconsistencia: consume exclusivamente conteos por dataset producidos por la implementación oficial sobre predicciones y ground truth GEFF.

## Regla de comparabilidad

Dos experimentos sólo pueden compartir tablero cuando tienen:

1. el mismo `protocol_id`;
2. exactamente los mismos datasets no vistos;
3. la misma métrica y distancia de matching;
4. configuración de inferencia congelada antes de la comparación;
5. cobertura completa y cero datasets omitidos.

## Ranking

La métrica base es la oficial:

$$
Score = AdjustedEdgeJaccard + 0.1 \times DivisionJaccard
$$

El valor de promoción es el límite inferior del intervalo bootstrap del 95% del score oficial. Esto favorece experimentos buenos y estables entre datasets. Runtime, cobertura, errores y validez del artefacto funcionan como gates: no se mezclan con la calidad mediante pesos arbitrarios.

## Decisiones

- `SUBMIT_CANDIDATE`: primer candidato elegible del ranking.
- `HOLD`: candidato válido, pero superado en el protocolo actual.
- `REJECT`: falló cobertura, runtime, score finito o integridad.
- La diversidad para ensemble debe medirse con predicciones OOF; no se infiere del nombre de la arquitectura.

## Flujo futuro obligatorio

Cada receta nueva debería generar primero una variante `EXP-XXX-CV` sobre un protocolo congelado group-disjoint. Sólo las recetas promovidas pasan a `EXP-XXX-FF` con todos los datos. El full-fit se envía, pero la evidencia de selección proviene de su réplica CV.

## Código

- Inferencia legacy real: `test-exp-local/evaluator.py`.
- Constructor del tablero: `test-exp-local/build_board.py`.
- Agregación y bootstrap: `test-exp-local/promotion_board.py`.
- Validador de artefactos: `test-exp-local/validate_evaluation.py`.
- Protocolo congelado: `test-exp-local/protocols/legacy_exp017_exp019_pilot.json`.
- Tests: `test-exp-local/tests/test_promotion_board.py`.

## Corrección de integridad

La primera versión histórica del evaluador no evaluaba checkpoints: derivaba números desde logs, fijaba `Division Jaccard = 0.15` y usaba valores por defecto. Ese comportamiento fue eliminado. Ningún resultado generado por esa versión puede usarse para decisiones. La implementación vigente y única está en `test-exp-local/`.

## Piloto EXP-017 vs EXP-019

Protocolo: `biohub-legacy-seed42-holdout-common12-v1`. Contiene 12 de los 20 volúmenes del holdout seed 42 original que están completos en A1558. Sirve para comprobar ejecución pareada y concordancia de ranking; no alcanza por sí solo para certificar una métrica global porque el split histórico fue aleatorio por volumen y los thresholds se ajustaron sobre la misma familia de holdout.

Los resultados se registran en `experiments/Promotion_Board_EXP017_EXP019_Pilot/` y en [[05_Piloto_Promotion_Board_EXP017_vs_EXP019|el reporte del piloto]].

### Resultado resumido

| Experimento | Score oficial | IC bootstrap 95% | Recall | Runtime | Decisión |
|---|---:|---:|---:|---:|---|
| EXP-017 | `0.8098` | `[0.7431, 0.8727]` | `0.8676` | `421.0 s` | `SUBMIT_CANDIDATE` |
| EXP-019 | `0.7783` | `[0.7005, 0.8494]` | `0.8593` | `524.8 s` | `HOLD` |

Comparación pareada `EXP-017 − EXP-019`: delta `+0.0315`, IC 95% `[+0.0046, +0.0592]`, `P(Δ>0)=98.94 %`. El orden coincide con el leaderboard (`0.683 > 0.643`), pero dos modelos no bastan para certificar validez global.
