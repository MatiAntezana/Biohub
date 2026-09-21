# 014 — cobertura y novedad de supervisión antes de volver a entrenar

Diseñador: Codex. Operador: Antigravity. Fecha: 2026-09-13.
Estado: implementado y verificado localmente; ejecución real014 pendiente.
Actualización de responsabilidad: el usuario continuará solo con Antigravity.
La revisión posterior a014 la realiza ahora Antigravity según
`AUTONOMOUS_CONTINUATION.md`; no requiere regresar a Codex. El alcance y código
de014 siguen intactos y no habilitan por sí mismos entrenamiento/publicación.
Release de contenido (no commit Git):
`becd1f5dcac099c2227ef7d46ecef7a33b04fb2c044bed2a92c158d5268a7121`.

## 1. Evidencia que motiva014

013 pasó247/247 pruebas en A1555, pero terminó
`insufficient_supervision_no_submission` antes del ajuste. En44b6 había3
comparaciones de eventos procedentes de solo2 madres/2 volúmenes, por debajo
del mínimo de3 madres. En6bba había22 comparaciones de16 madres/11 volúmenes.
La auditoría local reproducible confirmó que las25 comparaciones eran idénticas
a preferencias012 orientadas por su signo, dentro de la misma muestra/madre.
No se bajará el mínimo de013 ni se contará duplicación como nueva supervisión.

Fuentes013 fijadas por SHA256 en `configs/coverage_audit.json`. Resultado:
`549398e62b79a9e8ceffe34b9727936f4eb4c61944a17a95492a32b8f40a8e5b`.
Todos los archivos científicos001–013 se mantienen byte a byte. Solo cambian
los drivers operativos para añadir014; no cambian modelos, parámetros o notebooks.

## 2. Preguntas y alcance congelado

1. ¿Cuántos forks no se pueden representar por falta de correspondencia de
   extremos, y qué detecciones cercanas existen antes/después del postprocesado?
2. ¿Cuántos pares reales quedan fuera del radio12 µm o del top‑6 congelados?
3. ¿Cuántos forks cubiertos carecen de comparaciones completamente anotadas?
4. ¿Las tres muestras adicionales aportan nuevos eventos y preferencias únicas?

La expansión autorizada usa metadatos005, no scores ni selección de umbrales:

| Muestra44b6 | Divisiones anotadas | Uso autorizado |
|---|---:|---|
| 44b6_12dfb391 | 1 | Supervisión previamente expuesta |
| 44b6_a21120c2 | 2 | Supervisión previamente expuesta |
| 44b6_e28840c6 | 2 | Supervisión previamente expuesta |

Estas muestras están fuera de TODOS los paneles reservados y del corpus011.
Son5 divisiones potenciales, no5 ejemplos utilizables ya comprobados. No son
muestras nuevas para la historia del proyecto ni un holdout independiente.
Se rechaza cualquier superposición con FIT/calibración011 o paneles005.
No incorporar otras muestras44b6 o6bba sin una nueva decisión del diseñador.

Se reutilizan12 FIT44b6 +23 FIT6bba:35 muestras y59 forks anotados históricos.
El audit014 abre GT solo de esas35 +3 adicionales; no abre GT ni imágenes de
calibración, desarrollo, confirmación o visible. Verificar hashes de los53 NPZ
de evidencia011 es una comprobación de integridad, no evaluación de esos paneles.
Detector full-fit y exposición previa declarados; no afirmar independencia por
embrión ni confundir esta partición con una validación nueva.

## 3. Dos tareas separadas

`prepare-coverage` (GPU exclusiva):

- Fuente exacta `b0970-events-013`; valida cadena005–013 y paridad histórica.
- Ejecuta champion congelado dual-seed/DeepCenter/ILP/guardia de colisiones solo
  en las3 imágenes aprobadas. No abre GT ni entrena.
- Guarda candidatos crudos, referencia CSV de las3 muestras y evidencia011
  medida en coordenadas predichas. El CSV es un cache, NO una submission:
  no debe cumplir ni simular el presupuesto de filas del test completo.
- Congela `corpus.json`, receta, partición y hashes. No reescribe el corpus011.

`audit-coverage` (CPU limitada):

- Fuente exacta `b0970-coverage-corpus-014`, de la misma release y nodo.
- Extrae preferencias directas de las3 muestras con la regla012 existente:
  solo cambios íntegramente confirmados; unknown/mixtos excluidos. No optimizer.
- Compara cada fork GT con el matching oficial congelado a7 µm, el grafo final,
  los candidatos crudos guardados antes del ILP global/postprocesado y el pool011.
- Conserva clases separadas: `unmatched_endpoint`, `outside_frozen_pool`,
  `covered_without_confirmed_contrast`, `usable_event`. Registra distancias
  físicas, vecino y empates, ocupación del vecino en el matching, rank y límites.
- Para extremos sin matching distingue `near_anchor_unmatched`,
  `near_raw_but_no_near_anchor`, `no_near_detection_in_saved_candidates`.
  Proximidad NO acredita identidad. No hay atribución causal a una etapa exacta
  del ILP/postprocesado; tampoco se declara que el detector nunca vio la célula.
- Explica unknown por padre/hija detectados sin matching o progenitor GT no
  anotado. Cancela cambios compartidos antes de clasificar comparaciones.
- Reproduce los conteos y vectores013 sobre35 FIT; cualquier divergencia aborta.
- Deduplica vectores firmados exactos dentro de una misma muestra/madre.
  Separa duplicados de012, duplicados de las nuevas preferencias directas,
  repeticiones dentro del lote y vectores nulos. No fusiona diferentes madres
  ni vectores cercanos por tolerancia. Novedad numérica no prueba independencia.
- Guarda `new-supervision-44b6.npz`: diferencias orientadas hacia la preferencia
  confirmada, claves y origen direct/event. No son pesos entrenados ni grafos.
  Informa aparte madres de preferencias y forks biológicos utilizables:
  muchas comparaciones de una división siguen siendo una sola división.
- Verifica hashes antes/después; nunca aplica transacciones o modifica matching.

No hay entrenamiento, scoring del ranker, selección de margen, búsqueda,
evaluación oficial de tracking, reparación de grafos ni publicación en014.
Resultado correcto: `designer_review_no_submission`, incluso si hay0 ejemplos
nuevos. No existe un gate014 que habilite Kaggle.

## 4. Recursos y contratos

Nodo único A1555, alias a1555, conda train-55. Python local exclusivo:
`/Users/matias/miniforge3/envs/AIEnv/bin/python`. Usar `ops/biohub.py`, que emplea
`core.remote_executor.execute_remotely` y los aliases seguros UCBV. No SSH ad-hoc,
Python genérico, cambios de proxies/host keys/sockets/servicios, otros nodos,
instalación de paquetes o MLflow. No detener VLLM ni procesos ajenos.

Tests y auditoría: `cpu_evaluation_v1`, CUDA oculta, hasta2 CPU lógicas,
nice>=10, límite por proceso8 GiB, guardias de CPU/RAM/disco existentes.
Preparación: `gpu_exclusive_v1`, GPU sin procesos ajenos, VRAM libre>=12 GiB,
disco>=24 GiB. Una GPU ajena ocupada bloquea la preparación, pero no una auditoría
CPU admitida. Lock común Biohub en ambas; no lanzarlas en paralelo.
`deferred_resources` + `run_created=false` no consume el ID: esperar.
Un receipt creado consume el ID. No relanzar un job fallido/incierto ni borrar locks.

## 5. Secuencia exacta para Antigravity

Ejecutar por etapas; no pegar todos los launches en un bloque sin esperar.
Los3 IDs siguientes están predeclarados. Si alguno existe, consultar estado
y devolver evidencia; no inventar otro ID ni sobrescribir.

Verificación local (sin GT, inferencia o fitting):

```bash
cd /Users/matias/Proyects/Biohub
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/verify.py
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/audit_coverage_inputs.py
```

Exigir269/269 pruebas locales y auditoría `passed=true`, hash014 arriba,
53 evidence_npz,35 historical_raw_npz, expansión exacta3/5,
historical_novelty:3+22 duplicadas,0 nuevas. `gt_opened=false`,
`images_opened=false`. `historical005_chain_verified=false` local es deliberado;
la cadena005 completa se valida en el worker remoto antes de inferir.

Primero tests remotos, CPU:

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py doctor --node A1555 --task tests
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1555 --task tests --run-id b0970-tests-014
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1555 --run-id b0970-tests-014
```

Esperar `completed`, `tests_passed=true`,285/285 full_discovery, cero
skips/fallos/errores, hash014, aislamiento CPU. No usar tests013.
Después preparación GPU de3 muestras únicamente:

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py doctor --node A1555 --task prepare-coverage
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1555 --task prepare-coverage --source-run b0970-events-013 --tests-run b0970-tests-014 --run-id b0970-coverage-corpus-014
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1555 --run-id b0970-coverage-corpus-014
```

Esperar `completed`, `ready_for_coverage_audit=true`, datasets=3,
`gt_opened=false`, `source_unchanged=true`. Si falla o espera recursos, no continuar.
Después auditoría CPU:

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py doctor --node A1555 --task audit-coverage
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1555 --task audit-coverage --source-run b0970-coverage-corpus-014 --tests-run b0970-tests-014 --run-id b0970-coverage-014
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1555 --run-id b0970-coverage-014
```

Consultar `status` cada60–120s, nunca duplicar launches por silencio. Al completar
o fallar cada job, descargar COMPLETO con `fetch`, solo IDs realmente creados:

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py fetch --node A1555 --run-id b0970-tests-014
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py fetch --node A1555 --run-id b0970-coverage-corpus-014
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py fetch --node A1555 --run-id b0970-coverage-014
```

No excluir NPZ/cache. `fetch --resume` solo completa una descarga parcial cuyo
receipt terminal coincide; preserva archivos existentes. Si difieren hashes,
reportar rutas; no borrar para forzar reemplazo ni regenerar fuentes.

## 6. Entrega obligatoria y punto de parada

Devolver comandos,3 receipts, hash014, logs, tests.json, admisión/aislamiento;
corpus.json, split.json, input-audit,3 evidencias y cache completo; coverage.json
y su SHA256,38 reportes por muestra, new-direct JSON/NPZ y new-supervision NPZ.

Resumir por prefijo y por histórico/adicional: total de forks, etapas de pérdida,
disponibilidad raw/anchor, causas unknown, forks utilizables, comparaciones
confirmadas, duplicadas y únicas. Enumerar los5 forks de las3 muestras adicionales
y por qué sirven/no sirven. No presentar madres negativas de preferencias como
divisiones nuevas. Distinguir proximidad, matching y falta de anotación.

Exigir `source_unchanged=true`, dataset_count=38; `model_trained=false`,
`threshold_selected=false`, `threshold_selection_attempted=false`,
`calibration_evaluated=false`, `development_evaluated=false`,
`tracking_evaluated=false`, `confirmation_evaluated=false`,
`visible_evaluated=false`, `uses_gt_for_edits=false`, `graph_edits_applied=false`,
`graph_exported=false` en el audit, `kaggle_submission_allowed=false`, `promoted=false`.

Cerrar014 y realizar la revisión escrita de Antigravity conforme a
`AUTONOMOUS_CONTINUATION.md`, sin esperar a Codex. No ejecutar learn-events,
rank-repairs, experiment, package, HOCT, entrenamiento, notebook, upload, push
o submission a partir de un receipt014. Una campaña posterior exige nueva
predeclaración, implementación, tests y gates; no hay promoción automática.
014 identifica dónde intervenir después; no demuestra una mejora de Kaggle
ni que cinco divisiones adicionales basten para alcanzar0.970.
