# Corrección de imports y reanudación — 2026-09-08

Documento histórico: la corrección de imports pasó los tests remotos `002`.
La reanudación vigente está en `docs/FIX_PREPARE_002.md`, con IDs `003`.
No volver a ejecutar los comandos históricos siguientes como campaña nueva.

## Diagnóstico confirmado

Run fallido: `b0970-prepare-001`, A1688, release
`0dc710f3cb93fad5a09c1db55ad9a2254538cc3c62542d5eb4290809b5de751d`.
Evidencia preservada: `artifacts/A1688/b0970-prepare-001/receipt.json` y `worker.log`.

Al ejecutar `python .../ops/biohub.py`, Python deja `ops/` en su ruta de imports.
Insertar la raíz del proyecto no lo quitaba. El archivo `ops/packaging.py`
ocultaba el paquete externo `packaging`; la cadena tracksdata → GEFF → Zarr
fallaba al importar `packaging.version`. Se reprodujo el mismo error localmente.
No demuestra una falla científica ni de pesos, GPU o tracking. Replay no terminó
y aún no se puede afirmar paridad V19 ni mejora de score.

Se identificó además la misma colisión potencial entre `ops/kaggle.py` y el SDK
Kaggle. Este segundo caso se comprobó inspeccionando la resolución de imports,
sin autenticar ni hacer consultas/publicaciones a Kaggle.

## Cambios del diseñador

- Renombrado el constructor a `ops/bundle_build.py`, con imports actualizados.
- `core/cli_bootstrap.py` elimina `ops/` como ruta de módulos de primer nivel;
  conserva la raíz del proyecto y los directorios externos de dependencias.
- Aplicado al controlador, CLI Kaggle, runtime del notebook y verificador local.
- Nuevas regresiones en intérpretes frescos; los modos script y `-m` se conservan.
- Smoke obligatorio tracksdata/GEFF/Zarr en la suite remota completa.

Release corregida:
`dde0926a43196615b5d1be6433d9fc550acd9353b6d4acea67cef74456f7d660`.
28 pruebas locales pasaron. AIEnv no tiene Zarr ni PySCIPOpt; no se instalaron
paquetes. El smoke completo y SCIP siguen pendientes de la ejecución remota.
No se modificaron checkpoints, umbrales, épocas, paneles, gates ni algoritmo.
Los artefactos fallidos y la release anterior se conservan para auditoría.

## Instrucción para Antigravity

No volver a ejecutar la release `0dc710...`. No parchear su carpeta inmutable,
reinstalar `packaging`, eliminar caches/locks ni sobrescribir IDs `001`.
El dispatcher sincronizará el código nuevo a otra carpeta identificada por hash.

Desde la Mac, verificar y consultar disponibilidad; luego lanzar tests nuevos:

```bash
cd /Users/matias/Proyects/Biohub
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/verify.py
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py doctor --node A1688
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1688 --task tests --run-id b0970-tests-002
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1688 --run-id b0970-tests-002
```

Si la GPU está ocupada, esperar sin detener cargas. Si `tests-002` ya existe,
consultar su estado y no duplicarlo. Verificar `code_hash` de la release corregida,
`status=completed` y `result.tests_passed=true`. La suite incluye 34 pruebas,
entre ellas SCIP y `RuntimeImportTests`; un error de imports no se omite.

Solo después de tests completados, lanzar:

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1688 --task prepare --source-run b0970-tests-002 --run-id b0970-prepare-002
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1688 --run-id b0970-prepare-002
```

No usar `--source-run b0970-tests-001`: pertenece a otro hash. `prepare-002` vuelve
a verificar assets y paridad V19 antes de entrenar. No hereda un supuesto éxito
de replay del run fallido. Los assets ya presentes se verifican sin borrarlos.

Si prepare completa, continuar con el paso 3 del RUNBOOK actualizado: experimento
`b0970-mitosis-002` y source `b0970-prepare-002`. Los pasos restantes también usan
IDs `002`. Un rechazo de gates sigue bloqueando Kaggle; esta reparación no
autoriza publicar ni afirma que 0.970 esté alcanzado.

Si falla cualquier etapa, conservarla y devolver comando, run ID, hash, receipt,
traceback y último log. No cambiar dependencias ni algoritmos por cuenta propia.
