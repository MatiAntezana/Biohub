# Mitosis-005: resultado nulo y siguiente diagnóstico

Historial: `tests-006` y `diagnose-006` completaron. El paso activo es
`docs/LEARNING_MITOSIS_007.md`. No relanzar los comandos `006` de este documento.

## Conclusión del diseñador

La preparación ya está operativa. El brazo `mitosis_add` no mejoró: produjo el
mismo CSV que el control. No se autoriza repetirlo con otro umbral ni enviarlo.
Antes de modificar el modelo, se necesita separar cobertura de candidatos,
capacidad del head y restricciones del brazo aditivo.

Se implementó `diagnose-mitosis` para obtener esa evidencia sin reentrenar el
detector ni los heads, regenerar caches o evaluar confirmación. No es un nuevo
brazo ni produce una candidata publicable.

## Verificado en los artefactos recuperados

- `tests-005`: receipt completado; el operador reportó 53 pruebas sin omisiones.
- `prepare-005`: receipt completado y `ready_for_experiments=true`. Caches 8/8
  desarrollo, 8/8 confirmación y 4/4 visible. La política v2 resolvió ambos casos
  de colisión reportados sin cambiar la paridad histórica.
- SHA de replay verificado nuevamente sobre el archivo real:
  `46b4008a8021f6882eaf4661ac814d70762f7d5190cbf86747e2630d33d5af37`.
- SHA de `mitosis-005/development/candidate.csv` y del control verificados iguales:
  `aa3caaddd53391f94b5802953e9a719f554f888712e431e64cc873f6d0ce0aba`.
- Score de desarrollo de ambos: **0.9012742110**, delta **0**. Jaccard de aristas
  0.8814367451; de divisiones 0.1304347826, con TP3/FP8/FN12 según la métrica.
  El rechazo es correcto y no habilita confirmación ni publicación.

Los 591 forks del grafo completo no son 591 divisiones anotadas evaluables: no
interpretar TP3/591 como precisión del detector. Las anotaciones son parciales.
El recall de nodos 0.970535 tampoco equivale a score Kaggle 0.970.

## Hallazgo 1: había parejas para puntuar

Codex reconstruyó en CPU únicamente las propuestas, desde los NPZ ya calculados,
sin ejecutar el head ni imágenes. Se usaron los filtros y orden originales:
prior conocido >=0.05, radio 12 µm, máximo seis hijas. Los grafos corresponden al
CSV idéntico al control. El diagnóstico remoto verificará la coincidencia exacta
entre esta enumeración y las parejas efectivamente puntuadas por el código original.

| Muestra | Parejas evaluables | Añadibles antes del head y de resolver competencia |
|---|---:|---:|
| 44b6_c50204e0 | 2399 | 597 |
| 44b6_d5e7d891 | 3739 | 733 |
| 44b6_ddf577ad | 4293 | 673 |
| 44b6_e57ff5c6 | 416 | 108 |
| 6bba_07e24132 | 3978 | 316 |
| 6bba_09961292 | 3868 | 421 |
| 6bba_48816121 | 1190 | 165 |
| 6bba_5a4d9360 | 73 | 14 |
| Total | **19956** | **3027** |

"Añadible" significa que incluye la hija existente y otra sin padre; no que sea
una mitosis real ni que todas puedan añadirse simultáneamente. `approved_pairs=0`
no se explica por ausencia total de propuestas. Falta conocer sus scores.

Además, el matching geométrico de transferencia de priors solo cubre entre
36.6% y 65.8% de los nodos según muestra. Los demás no reciben un prior inventado.
No sabemos todavía cuántas divisiones reales quedan fuera por esa restricción.
Subir globalmente la tolerancia de 0.9 µm arriesgaría mezclar identidades.

## Hallazgo 2: entrenamiento que requiere diagnóstico

| Head | Positivos/negativos | BCE ponderada época1 → época12 | Batches sin positivos en época12 |
|---|---:|---:|---:|
| 44b6 | 26 / 2329 | 2.1421 → 3.8036 | 128 / 148 |
| 6bba | 125 / 4936 | 2.3233 → 2.2212 | 231 / 317 |

La BCE de 6bba termina ligeramente por debajo de la primera época, pero no muestra
un descenso sostenido. La de 44b6 termina por encima. Los valores se acumulan
durante actualizaciones: NO son una evaluación final de un checkpoint fijo.

Con los pesos de clase actuales, un predictor constante 0.5 tendría BCE ponderada
1.3710 y 1.3521 en esos conjuntos. Es una referencia algebraica
`2 * N_neg / N * log(2)`, no un modelo evaluado ni prueba concluyente de colapso.
Los minibatches se agrupan por muestra y predominan pasos solo negativos, con
pesos positivos 89.5769 y 39.488 y clipping global 1.0. Una optimización dominada
por esos pasos es una hipótesis plausible que debe comprobarse con scores reales.

## Diagnóstico implementado y límites

`core/mitosis_diagnostic.py`:

1. Verifica el manifiesto de la release fuente `005`, paridad y hashes de plan,
   control, candidatos y heads. Solo permite reutilización diagnóstica si el
   código científico, configuraciones, vendor y notebooks permanecen idénticos.
2. Ejecuta el scorer original en desarrollo y captura sus scores, sin cambiar
   logits, umbral, candidatos, features o restricciones. Verifica orden y cantidad
   contra la enumeración y exige que siga sin cambiar el grafo de referencia.
3. Solo después de predecir abre GT de la muestra de desarrollo. Usa el matching
   completo de la métrica archivada y conserva el mapeo correcto de IDs.
4. Separa forks exactos por: extremo no emparejado, división ya presente, madre
   incompatible, hija previa incorrecta, segunda hija ocupada, pareja no puntuada
   o score bajo. Los no puntuados se desglosan por prior, radio o top-k.
5. Compara scores de parejas anotadas positivas, negativas confirmadas y
   desconocidas. Mide también el head contrario sobre las coordenadas GT de
   divisiones de desarrollo para distinguir problemas del modelo y de localización.
6. Reconstruye la composición de batches del entrenamiento existente, sin entrenar.

Los forks exactos consecutivos son una atribución de cobertura, NO sustituyen
el Jaccard oficial con ventanas temporales. No etiquetar la ausencia de anotación
como negativo. No barrer umbrales, evaluar confirmación o usar estas salidas como
una candidatura. Si el replay del brazo deja de ser idéntico, detenerse y reportar.

La excepción de compatibilidad entre hashes existe SOLO para `diagnose-mitosis`.
Los gates de `prepare`, `experiment` y `package` siguen exigiendo la misma release.
Un receipt diagnóstico no es un source válido para entrenar ni publicar.

## Decisiones posteriores, todavía no autorizadas al operador

- Si el head puntúa bajo incluso las divisiones GT: diseñar minibatches equilibrados
  y una pérdida coherente con ese muestreo, sin duplicar el balance por clase;
  mantener separación de prefijos y épocas fijas. No resolverlo bajando el umbral.
- Si el head reconoce GT pero no los puntos predichos: estudiar el desajuste entre
  coordenadas/features de entrenamiento e inferencia y entrenamiento con errores
  de localización físicamente consistentes, solo en el grupo de entrenamiento.
- Si domina `not_scored`: mejorar la procedencia de IDs para transferir priors antes
  del suavizado, o diseñar propuestas con evidencia de imagen; no inventar priors.
- Si dominan madres/hijas ya asignadas incorrectamente: el brazo aditivo no puede
  repararlas. Priorizar una asociación global con contexto temporal, como la fase
  HOCT prevista, tras revisar su procedencia, compatibilidad y admisibilidad.

Estos son criterios para elegir el siguiente cambio, no cuatro variantes que
Antigravity deba ejecutar. La decisión final vuelve a Codex con el diagnóstico.
No hay evidencia de un nuevo score Kaggle ni garantía de alcanzar 0.970.

## Comandos para Antigravity

Release diagnóstica:
`601d2b57979319f2b3f315e0da099ceb47e0d8100784357a61690475ed5341fb`.
56 pruebas locales aprobadas. Suite completa: 63, con una prueba nueva de matching
real incluida únicamente en la suite remota. No se instalaron dependencias ni
se ejecutaron trabajos remotos en esta revisión.

Conservar todos los runs `005`. No iniciar `prepare-006`, reentrenamiento o Kaggle.
Si alguno de los IDs siguientes existe, consultar primero; no sobrescribirlo.
Usar exclusivamente A1688 y el acceso UCBV existente. Ante GPU ocupada, esperar
sin detener ningún proceso. MLflow continúa prohibido.

```bash
cd /Users/matias/Proyects/Biohub
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/verify.py
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py doctor --node A1688
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1688 --task tests --run-id b0970-tests-006
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1688 --run-id b0970-tests-006
```

Esperar `completed`, `tests_passed=true`, 63 pruebas sin omisiones y hash exacto.
Solo entonces:

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1688 --task diagnose-mitosis --source-run b0970-prepare-005 --tests-run b0970-tests-006 --run-id b0970-diagnose-006
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1688 --run-id b0970-diagnose-006
```

Esperar `completed` y `disposition=diagnostic_only_no_submission`. Recuperar:

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py fetch --node A1688 --run-id b0970-tests-006
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py fetch --node A1688 --run-id b0970-diagnose-006
```

`fetch` exige destino local nuevo. Si ya se recuperó, no repetirlo. Entregar
receipts, `diagnostic.json`, `training-audit.json`, los ocho `*.diagnostic.json`
y `*.scores.json`, y el log. Informar máximo/mediana de scores por head, scores
de positivos conocidos, cobertura de forks y sus motivos. Ante cualquier fallo,
preservar artefactos y entregar traceback; no editar código ni instalar sustitutos.
