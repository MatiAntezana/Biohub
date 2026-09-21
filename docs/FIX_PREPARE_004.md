# Gap-2 sin etiqueta: procedencia capturada — 2026-09-08

Resuelto en `prepare-005`: paridad correcta, 20/20 muestras cacheadas y ambas
colisiones reparadas. `mitosis-005` luego fue rechazado por delta cero. El paso
actual está en `docs/DIAGNOSIS_MITOSIS_005.md`; no relanzar los comandos históricos.

## Evidencia y alcance

`b0970-prepare-004` pertenece a la release
`fb40805f0295e0c08d44e5215a0df61ab1e2b8b275fcb42238caa1969bf45a30`.
El operador reportó 45 pruebas remotas aprobadas en `b0970-tests-004`.
Los artefactos recuperados están en
`/Users/matias/Proyects/Biohub/artifacts/A1688/b0970-prepare-004/`.

La preparación completó:

- Paridad V19 byte a byte. SHA del CSV verificado localmente:
  `46b4008a8021f6882eaf4661ac814d70762f7d5190cbf86747e2630d33d5af37`.
- Ambos heads de 12 épocas, con los mismos hashes que `003`:
  44b6 `0900030a4977baec3deedc3226f1339401e682656eaeb1aed2ece414558b8e6f`;
  6bba `3321db1f4352463543f61dd8b872b10c560cb48a75fb29dd766b178734ff73cb`.
- Desarrollo 8/8 en 372.45 segundos. SHA de `reference.csv` verificado contra
  `cache.json`: `aa3caaddd53391f94b5802953e9a719f554f888712e431e64cc873f6d0ce0aba`.
  La colisión anterior `6bba_48816121` quedó reparada según la política v1.
- Confirmación 7/8; fallo en `6bba_df673a83`. No hay preparación aprobada ni
  evaluación del brazo nuevo que permita atribuir una ganancia de tracking.

## Causa y decisión de implementación

El nodo observado 3146 y el nodo interpolado 24135 redondean al vóxel
`(12,45,113,31)`. La función congelada se llama `recover_strict_gap2`: crea nodos
sin `gap_synthetic`, mientras que `close_single_frame_gaps` sí añade esa etiqueta.
Por eso la guardia v1 clasificó erróneamente 24135 como observado.

No se usa `gap2_recovered` en aristas para decidir qué nodo es sintético: esas
aristas también tocan detecciones observadas. Tampoco se utiliza `ID > máximo`
como heurística de procedencia, ni se modifica el notebook histórico.

`core/node_provenance.py` captura la creación efectiva:

1. Antes del filtro histórico, conserva el conjunto de IDs de entrada observados.
2. Envuelve temporalmente `close_single_frame_gaps` y `recover_strict_gap2` en el
   espacio de nombres utilizado por el filtro. En cada llamada toma los IDs
   ANTES de la mutación y registra los nuevos IDs devueltos y su etapa creadora.
3. Devuelve exactamente el resultado original. No añade atributos ni modifica
   coordenadas, enlaces o decisiones del filtro congelado. Restaura ambas
   funciones incluso si hay una excepción; cada muestra tiene su propio registro.
4. Pasa esta procedencia explícita a `core/quantization.py`. Un nodo sin
   procedencia conocida no se convierte automáticamente en observado. Reutilizar
   una identidad observada/generada o aportar procedencia contradictoria aborta.

La guardia `observed_over_gap_synthetic_exact_voxel_v2` mantiene la regla
conservadora: exactamente un observado y todos los demás sintéticos conocidos
en el mismo vóxel entero permiten retirar solo esos sintéticos y sus aristas
incidentes. No se transfieren enlaces, mueven observados, cambian NMS/umbrales,
rellenan filas ni se podan otros nodos en cascada. Varios observados, solo
sintéticos o una colisión de procedencia desconocida siguen rechazándose.
La reparación es transaccional: una colisión no resuelta impide aplicar las otras.

La instrumentación no cambia el filtro histórico; la guardia SÍ cambia el grafo
cuando retira una interpolación en conflicto. Puede afectar continuidad y no
se presenta como mejora de score medida. Replay, inferencia remota y Kaggle
comparten `Production.filtered_graph`, también para control y candidata.

## Auditoría esperada

Cada `*.rounding-collisions.json` incluye:

- `policy=observed_over_gap_synthetic_exact_voxel_v2` y
  `provenance_policy=input_snapshot_and_captured_creation_stage_v1`.
- Conteos `input_observed_nodes`, `input_synthetic_nodes_by_stage` y lista
  `unknown_provenance_ids`.
- Por colisión: `node_origins`, `observed_ids`, `gap_synthetic_ids` (ambas etapas),
  `gap2_synthetic_ids`, `unknown_ids`, vóxel y coordenadas.
- IDs retirados y extremos de sus enlaces, cardinalidad y estado final.

Sobre la fixture del incidente, 3146 queda como `observed_input` y 24135 como
`recover_strict_gap2`: se retira 24135 y sus dos enlaces, preservando sus vecinos.
La nueva ejecución debe informar sus resultados reales; no forzar esos IDs o
conteos si cambia la inferencia. Exigir cero colisiones sin resolver y validación
estructural completa, además de todos los gates existentes.

## Verificación y release nueva

Hash del código corregido:
`6db6f0671fa7358bf961a5c2b7005438b03f05b16fd424ff086ee62e1fb2b607`.

47 pruebas locales aprobadas y parseo estático correcto. La suite completa
recoge 53 pruebas; se deben ejecutar en A1688 sin omisiones, incluyendo SCIP e
imports tracksdata/GEFF/Zarr. AIEnv no tiene esas dependencias completas y no se
instalaron. No se ejecutaron GPU, entrenamiento remoto ni submissions en esta
corrección.

Las ocho pruebas nuevas incluyen creación in-place, aislamiento entre muestras,
restauración ante errores, conflictos de identidad, etiquetas engañosas y la
colisión reportada. Una ejecuta la función Gap-2 histórica real y sus helpers
sobre un caso pequeño, sustituyendo solo refinamiento de imagen por identidad:
el resultado original coincide exactamente con el instrumentado. Otra atraviesa
la integración de producción. Esto NO prueba paridad completa sobre los datos
reales: `prepare-005` debe volver a reproducir el mismo SHA histórico sin excepción.

Conservar `001`–`004`, incluidos heads, grafos crudos, caches, logs y receipts.
No editar sus releases ni transformar un estado fallido en completado. Este
parche no implementa reanudación de etapas entre hashes: la preparación nueva
es completa, sin copiar manualmente heads/caches parciales de `004` a `005`.

## Secuencia exacta para Antigravity

Leer antes `AGENTS.md`, este documento y `docs/RUNBOOK.md`. Usar A1688 y el
acceso configurado por UCBV. No cambiar de nodo ni interferir con otras cargas.
Si un ID ya existe, consultar su estado; nunca sobrescribirlo ni relanzar por
silencio de SSH. Si la GPU está ocupada, esperar sin detener sus procesos.

```bash
cd /Users/matias/Proyects/Biohub
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/verify.py
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py doctor --node A1688
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1688 --task tests --run-id b0970-tests-005
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1688 --run-id b0970-tests-005
```

Esperar `status=completed`, `result.tests_passed=true`, las 53 pruebas sin
omisiones y `code_hash` coincidente con el anterior. Un receipt `004` no habilita
esta release. Si hay discrepancia de hash o un test falla, detenerse y reportar.

Solo entonces:

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1688 --task prepare --source-run b0970-tests-005 --run-id b0970-prepare-005
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1688 --run-id b0970-prepare-005
```

Esperar preparación `completed`, paridad `passed=true`, ambos heads y caches
completos de desarrollo, confirmación y visible. Revisar las auditorías de
`6bba_48816121` y `6bba_df673a83`. Después continuar con `b0970-mitosis-005`,
usando `--source-run b0970-prepare-005`, según RUNBOOK. No publicar directamente
por haber superado la preparación: siguen pendientes los gates del experimento,
modelo final, filas/tiempo y validación real de Kaggle.

Ante fallo, preservar y recuperar receipt/log y auditoría de la muestra, entregar
comando, run ID, code hash y traceback a Codex. No agregar etiquetas a mano,
alterar el hash de paridad, desactivar verificaciones ni improvisar un parche.
El siguiente objetivo medible es completar esta preparación y evaluar la
hipótesis; 0.970 solo se declara con el score de la submission exacta en Kaggle.
