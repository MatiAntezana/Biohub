# Colisión de un gap sintético al redondear — 2026-09-08

Nota histórica: `prepare-004` reparó esta colisión y completó desarrollo, pero
encontró otra en confirmación causada por un Gap-2 sin etiqueta. La corrección
vigente y el reinicio con IDs `005` están en `docs/FIX_PREPARE_004.md`.
Los comandos `004` de este documento se conservan como historial, no para relanzar.

## Avance real y fallo

`b0970-tests-003`: 38 pruebas aprobadas según el reporte del operador.
`b0970-prepare-003` sí completó paridad V19 y entrenamiento de ambos heads.
Codex verificó el SHA del CSV de replay recuperado:
`46b4008a8021f6882eaf4661ac814d70762f7d5190cbf86747e2630d33d5af37`.
No contiene vóxeles duplicados. Se verificaron también los checkpoints:

- 44b6, 12 épocas, SHA `0900030a4977baec3deedc3226f1339401e682656eaeb1aed2ece414558b8e6f`.
- 6bba, 12 épocas, SHA `3321db1f4352463543f61dd8b872b10c560cb48a75fb29dd766b178734ff73cb`.

Cada head se entrenó solo con su prefijo y sin selección de épocas en otro grupo.
Esto no demuestra mejora de tracking: no llegaron a evaluarse las candidatas.
El cache de desarrollo completó seis muestras antes de `ValueError: Duplicate detections`.
El panel quedó incompleto; sus artefactos son evidencia, no un source aprobado.

El operador aisló el caso `6bba_48816121`: el observado 14434 en
`(48,24.64,181.92,64.0)` y el nodo `gap_synthetic=1` 26981 en
`(48,24.9036,182.397,64.126)` redondean ambos a `(48,25,182,64)`.
Se usaron esos valores para la regresión local. El nuevo tratamiento aún debe
comprobarse sobre la muestra real en la remota, no solo sobre esa fixture.

## Decisión del diseñador

La política `observed_over_gap_synthetic_exact_voxel_v1` actúa sobre la salida
del filtro histórico antes de construir/exportar el linaje entero:

1. Agrupa por el vóxel exacto resultante del redondeo histórico, incluyendo tiempo.
2. Solo si hay UN observado y los demás tienen `gap_synthetic=1`, descarta esos
   sintéticos y las aristas que los tocan. No redirige ninguna arista al observado.
3. Mantiene todos los demás nodos, coordenadas, IDs y orden de aristas; no aplica
   poda en cascada. No mueve células, agrega jitter, cambia NMS ni rellena filas.
4. Si hay varios observados, solo sintéticos o procedencia ambigua, aborta con
   evidencia; nunca escoge un ganador arbitrario ni desactiva `Lineage.validate`.
5. Escribe `*.rounding-collisions.json`: voxel, IDs, coordenadas, nodos/enlaces
   retirados y cardinalidad antes/después. También registra colisiones sin solución.

La implementación está en `core/quantization.py` y `Production.filtered_graph`.
La comparten replay, inferencia remota y notebook Kaggle. El control y todos los
brazos parten de la misma referencia corregida. **Sí cambia el grafo cuando se
activa**, y puede perder continuidad: su efecto no se presenta como mejora medida.
Es corrección de colisión de una interpolación, no ajuste para cumplir cardinalidad.

Se mantienen pesos, hiperparámetros, épocas, notebooks congelados, gates y hash
histórico esperado. La guardia no debe alterar el CSV visible sin colisiones;
el replay nuevo vuelve a exigir paridad exacta, no una tolerancia.

## Verificación y release

Release: `fb40805f0295e0c08d44e5215a0df61ab1e2b8b275fcb42238caa1969bf45a30`.
39 pruebas locales aprobadas; 45 pruebas completas se exigen en A1688, incluyendo
SCIP e imports de GEFF/Zarr. No se instalaron dependencias ni se ejecutó GPU local.

Conservar toda la corrida `003`, sus heads, sus grafos crudos y receipts. No editar
su release ni hacer pasar su estado failed a completed. La nueva preparación es
completa; este parche no implementa migración de etapas entre hashes. No copiar
los heads o caches parciales a `004` ni saltar etapas manualmente.

## Comandos exactos para Antigravity

```bash
cd /Users/matias/Proyects/Biohub
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/verify.py
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py doctor --node A1688
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1688 --task tests --run-id b0970-tests-004
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1688 --run-id b0970-tests-004
```

Esperar completed, tests_passed=true, 45 pruebas sin omisiones y hash correcto.
No detener cargas si doctor muestra GPU ocupada. Si el ID ya existe, consultar
su estado antes de actuar. Solo con tests correctos:

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1688 --task prepare --source-run b0970-tests-004 --run-id b0970-prepare-004
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1688 --run-id b0970-prepare-004
```

Comprobar `replay/parity.json: passed=true` y las auditorías de colisión. En la
muestra afectada, reportar los IDs/cuentas reales del nuevo run; no exigir a
mano una cifra de borrados para que coincida con la fixture. Si surge una colisión
sin resolución, conservar `*.rounding-collisions.json` y devolverla a Codex.
No modificar la política para avanzar.

Después de TODO prepare completed, seguir paso 3 del RUNBOOK: `b0970-mitosis-004`
con source `b0970-prepare-004`. Mantener todos los gates antes de publicar. No hay
submissions nuevas ni evidencia de score >=0.970 durante esta reparación.
