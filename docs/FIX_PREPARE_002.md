# Corrección de IDs GEFF y reanudación — 2026-09-08

Documento histórico: `prepare-003` ya confirmó la paridad aquí exigida.
La reanudación vigente está en `docs/FIX_PREPARE_003.md`, con IDs `004`.
No repetir los comandos históricos siguientes como campaña nueva.

## Evidencia comprobada

El operador reportó `b0970-tests-002` completed con 34 tests aprobados en A1688.
El posterior `b0970-prepare-002` falló en el gate de paridad, no en entrenamiento.
Su release era `dde0926a43196615b5d1be6433d9fc550acd9353b6d4acea67cef74456f7d660`.

Recibos y log preservados en `artifacts/A1688/b0970-prepare-002/`.
Codex comparó la totalidad de `replay/replayed.csv` con el histórico de solo
lectura `Biohub-mejorado/experiments/SCORE-0970/inputs/v19/submission.csv`:

- 235679 filas en ambos, con 119881 nodos y 115798 aristas.
- Cero diferencias en coordenadas `t,z,y,x`.
- 119878 IDs de nodo distintos por renumeración.
- Cero diferencias en aristas, incluido su orden, tras traducir los IDs.
- Hash esperado: `46b4008a8021f6882eaf4661ac814d70762f7d5190cbf86747e2630d33d5af37`.
- Hash del replay fallido: `016c718817bcfcca3da13d5b491a75f6e9f663e32261afbcb3c757c7bcee0cf4`.

El log muestra la conversión IndexedRXGraph → RustWorkXGraph con renumeración.
`InMemoryGraph.from_geff` no conserva IDs dispersos como 0, 1, 3. El notebook
original sí los conserva mediante `IndexedRXGraph.from_geff`.

## Corrección y alcance

`ops/workflow.py::replay` invoca directamente `graph_from_geff` del namespace del
notebook congelado. No duplica ni redefine ese lector y no cambia los loaders
de anotaciones, que usan su propia correspondencia interna de IDs.

El cambio no modifica detectores, pesos, coordenadas, enlaces, configuraciones,
gates, notebook histórico ni hash de referencia. No se relaja la comprobación
para aceptar solamente cardinalidad o equivalencia de grafo.

Release nueva:
`3ccd9e51d913b664064d3a40c286488ad51a35fd2291d3b4ea45ca3cb189909c`.

32 pruebas locales aprobadas. Se agregaron cuatro regresiones: backend original
IndexedRXGraph con retorno simple/tupla; conservación de IDs dispersos y orden
de aristas; rechazo de un CSV renumerado con idéntica geometría; bloqueo de
entrenamiento cuando falla paridad. Estas fixtures locales no ejecutan GPU ni
sustituyen la prueba real sobre GEFF de la remota.

El operador reportó que un diagnóstico manual con IndexedRXGraph obtuvo el
hash esperado. Es evidencia favorable, pero no un receipt completed de prepare
en esta release. `prepare-003` debe certificarlo de nuevo. No editar receipts
anteriores ni afirmar que ya hubo entrenamiento o mejora en Kaggle.

## Comandos para Antigravity

Seguir AGENTS.md y el RUNBOOK actualizado. Conservar las releases y corridas
`001`/`002`. No reinstalar paquetes, cambiar el hash esperado ni parchear la
release remota vieja. El dispatcher sincroniza la nueva automáticamente.

```bash
cd /Users/matias/Proyects/Biohub
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/verify.py
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py doctor --node A1688
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1688 --task tests --run-id b0970-tests-003
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1688 --run-id b0970-tests-003
```

Esperar `completed` y `result.tests_passed=true`; verificar hash nuevo y 38 tests
completos sin omisiones. Ante GPU ocupada, esperar sin detenerla. Si el ID existe,
consultarlo; no duplicarlo. Solo después de tests correctos:

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1688 --task prepare --source-run b0970-tests-003 --run-id b0970-prepare-003
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1688 --run-id b0970-prepare-003
```

El gate exige `replay/parity.json: passed=true` y ambos hashes exactamente iguales
al esperado. Solo entonces prepare continúa con paneles, folds y caches. Esperar
a que TODO prepare complete antes del experimento `b0970-mitosis-003`, con source
`b0970-prepare-003`, según paso 3 del RUNBOOK. No usar tests/prepare `002` como source.

Ante otro error, conservar y devolver comando, run ID, hash, receipt, traceback
y log. No forzar gates ni hacer nuevas reparaciones experimentales del código;
devolver el diagnóstico al diseñador. Esta corrección no autoriza saltar a Kaggle.
