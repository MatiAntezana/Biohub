# Ensayo 009: geometría sobre propuestas reales y cobertura de forks

Entrega de diseño: 2026-09-09. **Cerrado: ejecutado; no repetir comandos009.**
Resultados y correcciones: `docs/COUNTERFACTUAL_MITOSIS_010.md`. El punto activo
es tests010/audit010. Este documento conserva el protocolo histórico009.

## Decisión basada en probe-008

Codex verificó receipt completed, los seis modelos y `probe.json` de008. Su SHA
es `ab7466b9358d74c05865ec818a78963c5385c65ad92f4c4394091dc825b6c576`.
Ese hash corresponde a **probe.json**, no al receipt: este último tiene SHA
`afffd4fed4bd81e2409df07045eb144cd8368de589ff44782ffe8d6b0f724bf8`.
Tests008: el operador reportó 89/89 aprobados. No hubo confirmación ni tracking.

| Datos evaluados | Modelo | AUROC | AP |
|---|---|---:|---:|
| 44b6 | Histórico | 0.9444 | 0.6183 |
| 44b6 | Geometría | 0.9484 | 0.7647 |
| 6bba | Histórico | 0.7954 | 0.2631 |
| 6bba | Geometría | 0.9795 | 0.4267 |

La geometría mejora ambos rankings frente al histórico en ambos grupos. Es
evidencia para investigarla sobre detecciones, no para publicar un notebook.
El **AUROC 0.9795 no es un score Kaggle 0.9795**. Solo se evaluaron 4/11 positivos
en coordenadas GT y negativos seleccionados. A0.80 el recall fue 0%/9.09%:
la baja FPR de ese punto no demuestra un sistema operativo de alta precisión.

Dos matices del reporte del operador:

- Imagen sola no mostró transferencia fiable en estas pruebas lineales. No se
  demuestra que toda información de imagen sea inútil ni que falle cualquier encoder.
- Combinado sí supera AP de geometría en 6bba: 0.4942 vs0.4267, aunque pierde AUROC.
  En44b6 pierde ambas métricas claramente. No hay mejora consistente de añadir
  esa imagen. No se afirma que la geometría domine todas las métricas y grupos.

Se elige por decisión del diseñador la geometría como baseline de este ensayo,
no como modelo final. Se mantienen EXACTOS los dos JSON de008: no hay nueva
lambda, entrenamiento, calibración, ajuste de umbral ni encoder. Esta decisión
usa desarrollo ya observado;009 sigue siendo desarrollo adaptativo, no holdout.

## Hipótesis siguiente

El cuello pendiente es doble: pasar de coordenadas GT a coordenadas predichas,
y permitir proponer forks que el brazo aditivo ni siquiera podía considerar.
En006 ningún fork GT exacto estaba entre sus19956 parejas puntuadas. Proponer
solo hijas libres o exigir siempre priors transferidos puede impedir descubrir
divisiones aunque su geometría tenga señal. La cobertura nueva debe medirse.

009 enumera y puntúa, pero **no selecciona ni aplica enlaces**. No elimina la
necesidad de evidencia para distinguir mitosis de dos células vecinas. Si el
ranking sobre propuestas falla, no se implementa un solver confiando en él.

## Algoritmo implementado

1. Leer exclusivamente la referencia de desarrollo de `prepare-005`, fijada por
   hash. Son nodos V19 postprocesados: incluyen los sintéticos históricos que
   sobrevivieron a la guardia. No son todos observaciones crudas.009 no añade,
   borra, mueve ni vuelve a redondear ningún nodo.
2. Para cada madre en t, buscar hijas en t+1 a distancia física <=12µm; conservar
   las seis más cercanas, desempate por ID local. Enumerar todos sus pares, como
   máximo15 por madre. Radio y K únicos; no hacer barridos de sensibilidad.
3. No filtrar por padre ocupado, hija existente ni presencia de prior neuronal.
   Esto permite proponer reasignaciones, no autorizarlas. Distancia/prior ausente
   no se convierte en probabilidad inventada. Se limita a1M pares por muestra:
   si se excede, falla sin truncar ni podar para ocultarlo.
4. Calcular las mismas once features de008 y aplicar el JSON geométrico entrenado
   en el otro prefijo, con su normalización intacta. Los cuatro flags temporales
   usan la longitud real de la imagen, no el máximo frame detectado. Se leen
   metadatos Zarr (ejes/unidades/shape), no píxeles, CNN ni detector.
5. Reconstruir el conjunto legado con `build_candidates` e `inventory` originales
   y caches dual-seed fijados. Es un comparador de cobertura del enumerador,
   no una nueva ejecución del head histórico ni una candidata alternativa.
6. Guardar `.proposals.npz` y su hash para TODAS las ocho muestras antes de abrir
   GEFF de GT para evaluación. Recién entonces ejecutar el matching oficial
   archivado, con las mismas reglas de006, y guardar etiquetas por separado.

Los IDs en triples/footprints son índices locales del `Lineage` cargado desde
reference.csv, no IDs originales del CSV ni índices del grafo crudo. Sus
coordenadas/orden siguen el loader histórico intacto. No trasladar estos IDs a
otro archivo para aplicar aristas por cuenta propia.

## Evidencia que debe devolver el ensayo

- Cobertura de todos los forks GT de dos hijas, con razones explícitas:
  `unmatched_endpoint`, `nonconsecutive_gt_fork`, `outside_radius`,
  `outside_nearest_children` o `scored`. No insertar triples GT en el pool.
- Para cada fork cubierto: score, posición entre TODOS los pares de su madre
  (incluidos desconocidos), mejor/peor rango en empates y número de competidores.
  `unique_top1_gt_forks` exige rango peor1; no romper empates usando GT.
- Comparación con cobertura del conjunto legado. Distinguir forks ya presentes
  de forks nuevos cubiertos: reencontrar una división existente no es ganancia.
- `edit_footprint`: enlaces que sería necesario añadir/retirar y tipo de operación
  (`already_present`, `additive_second_child`, `new_two_child_fork`,
  `requires_reassignment`). Es una descripción local, no prueba de que convenga
  aplicarla ni una solución conjunta compatible entre madres vecinas.
- AP/AUROC y métricas a0.80 únicamente para pares con etiqueta confirmada. Los
  extremos no emparejados o padres GT desconocidos se etiquetan -1; nunca0.
  Se informa por separado cuántos pares desconocidos superan0.80. Esa cantidad
  no es un conteo de FP ni puede omitirse para inflar la precisión.

Las AP de008 y009 tienen universos de candidatos distintos; no interpretar su
diferencia directamente como mejora/regresión del modelo.009 no calcula score
oficial de tracking, no evalúa confirmación y no construye submission.csv.
Los fingerprints de GT certifican lo leído y su invariancia durante matching;
no afirman que se dispusiera de ese fingerprint en la preparación005.

## Decisión posterior: no automática

Si aparecen forks verdaderos nuevos bien rankeados en ambos grupos, se podrá
diseñar un experimento pareado con reasignación conjunta y costes de retirar
enlaces existentes. Deberá calibrarse/validarse sin mirar confirmación ni usar
scores geométricos como probabilidades ya calibradas. Un radio no prueba mitosis.

Si se recupera cobertura pero no ranking, hará falta evidencia adicional de
imagen condicionada a las hijas o continuidad temporal. Si no hay cobertura,
el problema será de propuestas/detección/matching y no se resolverá bajando0.80.
009 no autoriza ninguno de estos pasos posteriores, ni elegir otro radio/K/lambda.
No se sostiene todavía una mejora sobre0.933 ni una receta garantizada de0.970.

## Integridad y alcance operativo

Módulo nuevo `core/mitosis_proposals.py`, configuración y tests propios. Solo se
modifican los dos drivers; todo código científico008 anterior permanece exacto.
Se fija su hash agregado y se mantiene la cadena de verificaciones005–008.
Los13 archivos fuente heredados, referencia/caches de desarrollo y JSON008 usados
se verifican antes/después. No se cargan modelos balanceados007 ni se reentrena.
Un receipt de `propose-mitosis` no puede habilitar prepare, experiment o package.

Release009:
`c6ca21f53e51615126b99db8f351226a3db3d4cee5c26023bba64ea3a62baf43`.
94 tests locales aprobados. Suite completa106, pendiente en A1688, incluye un
nuevo fixture de matching oficial con hija ocupada. Codex no ejecutó inferencia
de datos reales, entrenamiento, conexiones remotas ni Kaggle en esta entrega.

## Comandos exactos para Antigravity

Leer AGENTS.md y este documento. A1688 fue autorizado explícitamente; respetar
UCBV, aliases seguros y GPU ajena. No modificar dependencias, config ni hashes.
No MLflow. Conservar todos los runs anteriores; no relanzar005–008.

```bash
cd /Users/matias/Proyects/Biohub
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/verify.py
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py doctor --node A1688
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1688 --task tests --run-id b0970-tests-009
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1688 --run-id b0970-tests-009
```

Esperar completed, tests_passed=true, **106/106 sin omisiones** y hash exacto.
Solo entonces:

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1688 --task propose-mitosis --source-run b0970-prepare-005 --tests-run b0970-tests-009 --run-id b0970-proposals-009
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1688 --run-id b0970-proposals-009
```

El JSON de configuración resuelve como fuente secundaria inmutable `probe-008`;
no agregar flags ni copiar modelos a005. Consultar estado cada60–120s; lanzamiento
asíncrono, no duplicar si termina SSH o no hay salida. Si el ID ya existe, consultar
su estado. GPU ocupada: esperar sin detenerla. Cualquier fallo exige entregar el
receipt/traceback/log al diseñador; no forzar gates ni modificar el presupuesto.

Final esperado: completed, `proposal_assay_only_no_submission`, source_unchanged=true;
confirmation_evaluated, tracking_evaluated, graph_modified y threshold_selected=false.
Recuperar una sola vez por destino:

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py fetch --node A1688 --run-id b0970-tests-009
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py fetch --node A1688 --run-id b0970-proposals-009
```

Devolver receipts/logs, `proposals.json`, `prediction-manifest.json`, ocho
`.prediction.json`, `.proposals.npz`, `.labels.npz` y `.diagnostic.json`.
Tabla por grupo de cobertura legado/nueva/no-presente, unique-top1, AP/AUROC,
TP/FP/FN/TN conocidos y desconocidos a0.80. Desglosar los forks GT individuales,
sus rangos y footprints. No interpretar null como0 ni `covered` como recuperación
aplicada. Detenerse y devolver a Codex: sin confirmación, nuevos entrenamientos,
HOCT, rewiring, package o Kaggle cualquiera sea el resultado.
