# 011 — selector aprendido sobre detecciones predichas

Diseñador: Codex. Operador: Antigravity. Fecha: 2026-09-09.
Estado histórico: ejecutado en A1555 y rechazado; desarrollo sin cambios.
Los comandos de este documento NO son órdenes activas. Seguir `REPAIR_RANKING_012.md`.
No es un notebook aprobado ni una mejora medida de Kaggle.
Release: `64f6f7de375a1d861d6a72727d06a5e50d20e5529348e788bdd18e0777547ead`.

## 1. Decisión a partir de010

010-CPU terminó con135/135 tests y fuentes intactas. Se verificó el SHA de
`counterfactual.json`: `e48cde4dc6ef988295d92174b163ebb9934062b9ef73c23837d20e12fbb19bfb`.
Referencia pública:0.9012742110. Nueve reparaciones elegidas CON GT dieron
0.9370949901; diez dieron0.9418622937. No son resultados de un selector.
La diferencia de0.035821 NO se suma al0.933 de Kaggle.

011 implementa la hipótesis pendiente: aprender a reconocer enlaces/divisiones
en las coordenadas realmente producidas por V19, con evidencia específica de
cada célula, y comparar también la evidencia que se destruye al reparar.
El algoritmo nunca recibe GT al elegir aristas. No lee ediciones/IDs del oracle010.
El único acceso a su informe es verificar el hash de la evidencia del diseño.

## 2. Qué queda congelado

- Detector, ambos seeds, DeepCenter, V19, cuantización/procedencia y pesos originales.
- Referencia de desarrollo de005 y cadena científica005–010 por hashes.
- Detecciones, coordenadas, tiempos y cardinalidad de cada grafo.
- Radio12µm y seis vecinos para proponer divisiones; no se amplía por GT76.
- Evaluador oficial y gates originales de desarrollo.
- Confirmación y visible permanecen cerrados. Ninguna publicación está autorizada.
- No se entrena una CNN, no se descarga ningún modelo ni se instala dependencia.

Los forks/aristas ya presentes se conservan en el universo de puntuación aunque
estén fuera del radio; conservarlos no introduce nuevas propuestas de largo alcance.

## 3. Datos nuevos y separación

`split_corpus` usa exclusivamente `plan.json` archivado en005. Excluye todos los
paneles desarrollo/confirmación/visible y `previously_exposed`. Nunca elige por
scores de predicción. Por prefijo selecciona hasta24 muestras con divisiones
anotadas, ordenadas por cantidad y hash estable, más cuatro sin divisiones anotadas.
Tres muestras positivas y dos de las otras se reservan para calibración mediante
otro orden determinístico del mismo hash. El resto es entrenamiento.

La selección determinística sobre el plan existente da:

| Prefijo fuente | Muestras de entrenamiento | Forks GT anotados allí | Muestras de calibración | Forks GT anotados allí |
|---|---:|---:|---:|---:|
|44b6|12|10|5|3|
|6bba|23|49|5|7|

Son59 forks de entrenamiento ANTES de matching/cobertura; no59 positivos
garantizados para el modelo. Si faltan clases o quedan menos de3 positivos o10
negativos confirmados en un ajuste, aborta. No convertir desconocidos en negativos.

Total45 muestras nuevas para detector+evidencia y ocho de desarrollo con detecciones
reutilizadas:53 archivos de evidencia. No repetir `prepare-005`, ni inferir sus
paneles de confirmación/visible. `split.json` registra todos los nombres antes de
calcular las detecciones nuevas. Antigravity no modifica esa selección.

Para evaluar44b6 se usan exclusivamente ajuste Y calibración de6bba, y viceversa.
La normalización de features solo usa entrenamiento. La calibración interna es
por muestra, no se afirma independencia biológica entre muestras del mismo prefijo.
El detector original es full-fit: tampoco se llama holdout integral al desarrollo.

## 4. Representación y aprendizaje

Cada detección aporta un recorte medido en SU posición y SU frame: campo físico
±6µm, rejilla5×9×9, interpolación trilineal. Se registran contraste y fracción del
recorte dentro de la imagen. La correlación ignora la amplitud mediante centrado y
normalización; regiones planas tienen bandera explícita de contraste ausente.
El padding no se cuenta como intensidad medida. No se inventan células o imágenes.

Dos modelos lineales logísticos L2 por prefijo, cuatro ajustes en total:

- Enlaces:14 features de distancia física, residuales de movimiento anterior y
  posterior, disponibilidad de contexto, correlación madre-hija, contraste,
  cobertura de imagen y pertenencia del enlace a la referencia.
- Divisiones:18 features simétricas entre hijas de geometría, movimiento del
  centro del par, continuidad de las hijas, correlaciones específicas de ambas
  hijas y madre, contraste conjunto/asimetría y disponibilidad de evidencia.

El movimiento procede de la referencia predicha, no de GT; puede contener errores.
No es un encoder de imagen aprendido. La hipótesis de que estos descriptores
transfieran mejor sigue sin demostrar. Mantener el modelo pequeño responde a la
escasez de positivos y al sobreajuste observado en007.

Para crear etiquetas de entrenamiento se hace matching público de7µm como en009.
Un enlace solo es negativo si el padre de la hija está explícitamente anotado y
es otro; un par exige evidencia anotada de ambas hijas. Sin evidencia:unknown,
excluido del ajuste, no del universo de inferencia. Labels y features siempre
corresponden a detecciones predichas; no se insertan coordenadas GT faltantes.

Optimización convexa predeclarada: BCE media50% por clase +0.5λ||w||², λ=1,
intercepto libre, L-BFGS-B, máximo500 iteraciones y gradiente infinito<=1e-5.
No búsqueda de épocas, arquitectura o regularización. Los logits NO se presentan
como probabilidades calibradas de la población sin anotar.

## 5. Decisor sin GT y abstención

Se puntúan todos los enlaces/forks propuestos y todos los de referencia con
features congeladas. Para cada madre se comparan:

1. Conservar su estado (ganancia0).
2. Proponer dos hijas, incluso si otra madre ya ocupa alguna.
3. Si actualmente tiene dos hijas, conservar solo una para revisar una bifurcación.

No se hace relinking general de continuaciones ni se permite borrar arbitrariamente
todo el linaje. Las nuevas continuaciones solo retiran una rama de una bifurcación.
Un padre desplazado puede quedar sin hija: esto se contabiliza explícitamente.

Ganancia experimental de la transacción:

`sum(logits de aristas añadidas) - sum(logits de aristas retiradas)`
`+ cambio de logits de forks en TODOS los padres afectados`
`- 2 × cambio en el número de padres afectados sin descendiente`.

Así no se ignora el enlace previo de una hija ni una división que se destruye.
No es una probabilidad conjunta ni un ILP nuevo. Por madre se conserva la mejor
alternativa de ganancia positiva; empates abstienen. Después se seleccionan en
orden determinístico de ganancia las transacciones con escrituras no solapadas
(salidas de padres y entradas de hijas). Se revalida el grafo final y el estado
solicitado. No se encadenan reparaciones recalculando features sobre sus propias salidas.

La calibración fuente evalúa EXACTAMENTE cuatro márgenes nuevos de energía:
`1,2,4,8`, más la opción abstenerse. No modifica el antiguo umbral0.80.
Las propuestas de TODAS las muestras de calibración se congelan antes de leer
su GT. Solo es elegible un margen con:

- score de calibración al menos+0.001;
- edgeJ y divisionJ sin caída;
- ninguna muestra con caída de adjusted edgeJ mayor que0.015;
- ningún rechazo por presupuesto de cambios.

Entre los elegibles se elige el mejor score de calibración; empate exacto favorece
el margen mayor. Si ninguno pasa, `margin=null`: abstención, no fallback inventado.
Esta es una selección predeclarada sobre el prefijo fuente; no una búsqueda en
desarrollo, confirmación o Kaggle. No añadir valores tras mirar resultados.

Límite de seguridad: transacciones seleccionadas <=max(1,floor(0.002×nodos)).
Si excede el límite, se rechaza ESA política en ESA muestra y se conserva toda la
referencia; no se recortan los top-N cambios ni se oculta el rechazo en los gates.
El límite no fuerza un número mínimo de cambios y no ajusta cardinalidad del CSV.

## 6. Desarrollo y aislamiento de resultados

Los cuatro modelos y los dos márgenes se congelan antes de predecir desarrollo.
Se guardan las OCHO predicciones completas antes de abrir GT de desarrollo.
Cada predicción usa el modelo del otro prefijo. Después se reproduce la referencia
005 (conteos exactos y floats con atol1e-10) y se evalúa la candidata mediante el
evaluador oficial, en grafos nuevos porque el matching muta atributos internos.

Se aplican gates originales: delta score>=0.003, edgeJ/divisionJ no inferiores,
ningún prefijo retrocede y caída adjusted edgeJ por muestra no mayor que0.015.
Una política que desborde el presupuesto de ediciones queda rechazada aunque
otras muestras compensen el promedio. Una métrica requerida indefinida aborta.

Se permite exportar NPZ de predicciones diagnósticas sin GT en la decisión; no
confundirlos con los grafos oracle010, que siguen prohibidos como training/input.
No se genera `submission.csv` de candidata, bundle, entrenamiento final ni notebook.
Incluso si pasa desarrollo, la única siguiente acción es revisión del diseñador:
el receipt011 nunca habilita `experiment`, `package`, confirmación o Kaggle.

Flags esperadas en evaluación:
`model_trained=true`, `threshold_selected=true` (solo calibración fuente),
`tracking_evaluated=true`, `deployable_policy_evaluated=true` (algoritmo sin GT,
NO aprobación para despliegue), `uses_gt_for_edits=false`, `graph_exported=true`
(NPZ diagnósticos), `confirmation_evaluated=false`, `visible_evaluated=false`,
`promoted=false`, `kaggle_submission_allowed=false`.

## 7. Recursos y comandos autorizados

Usar A1688 mediante `ops/biohub.py` y los aliases UCBV seguros. No servicios nuevos,
instalaciones, cambios en VLLM, claves, proxy o procesos ajenos.

- `tests`: CPU limitado como010.
- `prepare-selector`: GPU exclusiva, >=12GiB VRAM libre y ningún proceso GPU
  ajeno. Además exige24GiB de disco libre al preparar. Si VLLM ocupa GPU, esperar.
- `evaluate-selector`: excepción CPU explícita nueva, CUDA oculta, dos CPU lógicas,
  nice>=10, RLIMIT_AS8GiB por proceso, pools limitados y admisión CPU/RAM/disco.
  Ajusta modelos convexos pequeños y evalúa; no ejecuta detector ni CNN. Puede
  coexistir con VLLM solo si las guardias pasan. Mantiene el lock común Biohub.

No se promete tiempo de ejecución ni impacto cero sobre servicios que comparten
CPU/RAM/IO. No usar otro nodo sin indicación del usuario. No reanudar daemons viejos.

Desde la Mac:

```bash
cd /Users/matias/Proyects/Biohub
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/verify.py
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py doctor --node A1688 --task tests
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1688 --task tests --run-id b0970-tests-011
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1688 --run-id b0970-tests-011
```

Exigir153/153 CPU locales y168/168 remotos completos, cero skips/fallos/errores.
Luego, solo con tests completos de ESTA release:

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py doctor --node A1688 --task prepare-selector
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1688 --task prepare-selector --source-run b0970-prepare-005 --tests-run b0970-tests-011 --run-id b0970-selector-corpus-011
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1688 --run-id b0970-selector-corpus-011
```

Exigir `completed`, `ready_for_selector=true`,53 datasets, `gt_opened=false` y
la misma identidad de código. Solo entonces:

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py doctor --node A1688 --task evaluate-selector
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1688 --task evaluate-selector --source-run b0970-selector-corpus-011 --tests-run b0970-tests-011 --run-id b0970-selector-011
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1688 --run-id b0970-selector-011
```

Consultar estado cada60–120s; un proceso vivo sin salida nueva no requiere relaunch.
`deferred_resources` con `run_created=false` no consume ID: esperar y volver a
consultar recursos. Si existe receipt, consultar estado; nunca sobrescribirlo.
Una falla posterior al lanzamiento consume el ID. Recuperar y reportar, no parchear.
No repetir preparación005 ni ensayos006–010. No prolongar la grilla.

Al terminar cada corrida, o al fallar, traer su carpeta una sola vez:

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py fetch --node A1688 --run-id b0970-tests-011
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py fetch --node A1688 --run-id b0970-selector-corpus-011
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py fetch --node A1688 --run-id b0970-selector-011
```

No ejecutar fetch de IDs que no se lanzaron. Las carpetas nuevas pueden ser
grandes: contienen evidencia medida y los45 grafos de referencia nuevos. No borrar
fuentes ni runs anteriores para ahorrar disco; reportar falta de espacio.

## 8. Informe requerido y detención

Devolver:

1. IDs, code hash, receipts, tests.json, logs, admisión y pruebas de aislamiento CPU.
2. `split.json`, `corpus.json`:53 evidencias con hashes,45 muestras nuevas y8 desarrollo.
3. `fit-44b6.json`/`fit-6bba.json`: positivos/negativos/unknown por muestra,
   convergencia, métricas de entrenamiento y procedencia GT. No confundir éstas con generalización.
4. `calibration-*.json`: los cuatro márgenes completos por prefijo, fallos de
   presupuesto, métrica oficial y margen final o abstención. No ocultar ensayos perdedores.
5. `models.json`, `selector-*.json`, manifiestos de predicción y `*.edits.json`.
6. `selector.json`: score y delta global y por prefijo, TP/FP/FN de enlaces/divisiones,
   cambios, regresiones por muestra, abstenciones, presupuesto y gate desarrollo.

Detenerse al finalizar o ante fallo. `next_action=designer_review_only` NO autoriza
confirmación ni publicación. `rejected_no_kaggle_submission` es un resultado,
no una orden de bajar márgenes. El objetivo0.970 sigue sin demostrarse.
