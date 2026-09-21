# Archivo histórico — RUNBOOK de campaña005

Documento preservado solo como antecedente. NO EJECUTAR sus comandos.
Sus referencias a A1688 y revisiones de Codex NO son autorización vigente.
Los únicos nodos actuales son A1555/A1557, con las condiciones de AGENTS.md.
Para trabajar usar RUNBOOK.md y AUTONOMOUS_CONTINUATION.md.

Todos los comandos se lanzan desde `/Users/matias/Proyects/Biohub` en la Mac.
No ejecutar el proyecto viejo. No editar el código para que un gate pase.
Los IDs siguientes corresponden a la campaña con procedencia capturada por etapa,
tras `b0970-prepare-004`. Leer primero `docs/FIX_PREPARE_004.md`.
Conservar runs/releases `001`–`004`; sus tests no habilitan otra release.
Si un ID `005` ya existe, consultar su estado antes de actuar. Un ID nuevo solo
corresponde a un lanzamiento explícitamente nuevo.

## 1. Verificación y entorno

```bash
cd /Users/matias/Proyects/Biohub
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/verify.py
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py doctor --node A1688
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1688 --task tests --run-id b0970-tests-005
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1688 --run-id b0970-tests-005
```

`launch` devuelve rápidamente y la tarea continúa independientemente de SSH.
Consultar `status` a intervalos de 60–120 segundos; no relanzar porque no haya
salida nueva. El estado necesario es `completed` y `result.tests_passed=true`.
La suite remota incluye SCIP real e imports en frío de tracksdata/GEFF/Zarr,
y no admite pruebas omitidas. Si falla, recoger
el log y devolverlo al diseñador. No instalar PySCIPOpt en la Mac para entrenar.

Rutas remotas declaradas:

- Código: `/home/alumno1/Documents/Biohub-0970/releases/<code_hash>`.
- Corridas: `/home/alumno1/Documents/Biohub-0970/runs/<run_id>`.
- Assets: `/home/alumno1/Documents/Biohub-0970/assets`.
- Imágenes/GT públicos existentes: `/home/alumno1/Documents/Biohub/data/train`.
- Pesos/outputs históricos de solo lectura:
  `/home/alumno1/Documents/Biohub-mejorado/experiments/SCORE-0970`.

El rename local a `Biohub-viejo` NO implica que la ruta remota haya cambiado.
Si doctor muestra una GPU ocupada, esperar; no matar ni detener esa carga.
Si falta un paquete/version, informar el nombre exacto. El entorno existente
no se actualiza a `latest` como reparación improvisada.

## 2. Preparar base, folds y caches una sola vez

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1688 --task prepare --source-run b0970-tests-005 --run-id b0970-prepare-005
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1688 --run-id b0970-prepare-005
```

Este paso exige el receipt de tests del mismo hash de código. Copia únicamente
los tres modelos de identidades conocidas a `assets/champion`; verifica configs
y hashes; reproduce V19 sobre sus grafos archivados; exige paridad byte a byte;
fija los paneles; prepara ejemplos públicos; entrena ambos folds mitóticos y
calcula caches dense dual-seed para desarrollo, confirmación y visible.

Archivos relevantes: `replay/parity.json`, `plan.json`, `models.json`,
`train-44b6/training.json`, `train-6bba/training.json`, `cache-*/cache.json`.
Los parches se calculan una vez por entrenamiento. No se reutiliza como modelo
de evaluación el checkpoint antiguo `ng_a1688_mitosis_balanced_20260907`.

Un `FileNotFoundError` de los assets históricos se resuelve identificando el
archivo correcto y verificando su SHA; no con otro seed. No recuperar caches sin
receipt asignándoles hashes de los modelos actuales: eso inventaría procedencia.
Un fallo de paridad detiene la campaña y se devuelve con `replay/parity.json`.

## 3. Primera hipótesis: mitosis conservadora

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1688 --task experiment --source-run b0970-prepare-005 --arm mitosis_add --run-id b0970-mitosis-005
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1688 --run-id b0970-mitosis-005
```

El control se evalúa junto a la candidata sobre idénticos datos. Si falla
desarrollo, no se consulta el score de confirmación de ese brazo. Si falla
confirmación, `result.disposition=rejected_no_kaggle_submission`. Es un resultado
científico terminado que debe reportarse; no corresponde publicar ni retocar
el umbral de 0.80. Reportar los conteos TP/FP/FN y la cantidad de divisiones añadidas.

Solo `eligible_for_final_training_and_visible_runtime_check` habilita el paso 5.

## 4. Fase HOCT (hipótesis completa)

La primera hipótesis funciona sin HOCT. Para habilitar esta fase completar
**antes de tests/prepare de su campaña** los campos de procedencia en
`configs/hoct.json`, leyendo la fuente oficial y las reglas oficiales de Kaggle:

- `source_revision`: commit exacto, 40 hex, del código auditado compatible.
- `api_sha256`: SHA-256 real de `_api.py` de esa revisión.
- `review_evidence_url`: página oficial que sustenta admisibilidad/uso de pesos.
- `admissibility_reviewed`: true solo después de esa revisión.

No usar otro modelo ni inventar una fecha de publicación. Los pesos `general_v1`
se verifican contra el SHA preparado. Si reglas, ABI o licencia no permiten el uso,
esta fase se bloquea y se informa el punto exacto. La fase mitótica permanece útil.

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1688 --task hoct-assets --run-id b0970-hoct-assets-005
```

El script fija la revisión, construye un wheel sin dependencias, lo instala en
`assets/hoct-site` aislado, verifica API y descarga/verifica el checkpoint.
No cambia el conda compartido. Las dependencias faltantes se reportan, no se
instalan versiones adivinadas. Como cambió la configuración, ejecutar de nuevo
tests y prepare con IDs nuevos; mezclar hashes de campañas está bloqueado.

Sobre ese `prepare` congelado ejecutar en este orden, con IDs nuevos y un brazo
por vez: `hoct`, `hoct_mitosis`, `hoct_rescue`, `full`, sustituyendo solo el valor
de `--arm`. Sus parámetros ya están definidos; no crear grillas adicionales.
Cada brazo produce sus gates y puede ser rechazado. No publicar todos los que
terminen; solo un brazo elegible, priorizando el mejor score de desarrollo entre
los elegibles. La confirmación sirve como aprobación/rechazo, no como objetivo de
optimización. No cambiar thresholds después de ver confirmación.

## 5. Dependencias offline y modelo final

Para una campaña elegible:

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1688 --task wheels --run-id b0970-wheels-005
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1688 --run-id b0970-wheels-005
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1688 --task package --source-run b0970-mitosis-005 --run-id b0970-package-005
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1688 --run-id b0970-package-005
```

`wheels` descarga versiones exactas, sin modificar conda. Si tracksdata registra
su commit original en `direct_url.json`, construye el wheel de ese commit exacto.
Un paquete de desarrollo sin procedencia/wheel publicable produce un fallo explícito. Entregar ese fallo al diseñador
para incorporar su fuente inmutable/wheel correcto; no reemplazarlo por una
versión cercana. Tampoco borrar un wheelhouse parcial para ocultar el fallo.
Si Kaggle usa otra versión menor de Python, la plantilla aborta; se requiere
una preparación compatible, no saltarse esa comprobación.

`package` entrena el head final con ambos grupos y las mismas épocas fijas,
ejecuta el brazo elegido sobre las muestras visibles y verifica rango de filas,
cambio real respecto de la base y presupuesto de tiempo. No usa el score visible
del head final como confirmación. Genera `bundle/`, `dataset-upload/`, `kaggle/`
y `publish-ready.json`. El notebook incorpora hashes concretos, no placeholders.

## 6. Traer paquete y ejecutar el notebook en Kaggle

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py fetch --node A1688 --run-id b0970-package-005
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/kaggle.py upload --package artifacts/A1688/b0970-package-005
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/kaggle.py push --package artifacts/A1688/b0970-package-005
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/kaggle.py status --package artifacts/A1688/b0970-package-005
```

`fetch` crea una carpeta local nueva; no sobrescribe una descarga anterior.
`upload` crea un dataset **privado** con un ZIP íntegro; `push` crea el kernel
**privado nuevo** `matiasantezana/biohub-0970-b0970-package-005`, con GPU activada e internet
desactivado. Nunca usa el kernel compartido EXP-056. Los códigos source se
verifican mediante hash; el notebook procesa las muestras test de esa ejecución.
No necesita un CSV visible precalculado para inferir el test oculto.

Cuando Kaggle confirme que el notebook completó:

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/kaggle.py download --package artifacts/A1688/b0970-package-005
```

El script descarga únicamente CSV/receipt, valida identidad del bundle, nombres
de muestras, filas, grafo, duración y paridad visible con la remota. No relanzar
`download` antes de terminar ni borrar su carpeta para intentar otra versión.
Si hay diferencia numérica, el diseñador decide el siguiente diagnóstico; el
operador no publica una salida diferente bajo el mismo resultado local.

## 7. Submission y score exactos

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/kaggle.py submit --package artifacts/A1688/b0970-package-005
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/kaggle.py score --package artifacts/A1688/b0970-package-005
```

`submit` usa `competition_submit_code` con la versión exacta registrada en push.
Respeta el máximo propio de un envío por día UTC, coteja duplicados y persiste
la intención antes de la llamada de red. Un timeout de red no autoriza repetir
la mutación: reconciliar por run/kernel/ref real en Kaggle y entregar evidencia.
`score` solo registra el score de la referencia exacta; PENDING no es un fracaso.

Un score <0.970 mantiene la misión abierta. No volver a enviar la misma CSV ni
rebautizar la candidata como champion. Para una nueva hipótesis, devolver a Codex
los informes; Antigravity no modifica código o reglas científicas por su cuenta.

## Entregable del operador

Un resumen con run IDs, code hash, resultados desarrollo/confirmación por grupo,
TP/FP/FN de aristas/divisiones, filas/nodos/aristas, tiempos y hashes. Si publicó:
dataset, kernel, versión, submission ref, estado y score obtenidos del API.
Adjuntar o indicar la ruta de receipts/logs/CSV correspondientes. Las credenciales
de Kaggle/SSH permanecen en su ubicación habitual y nunca entran al paquete.
