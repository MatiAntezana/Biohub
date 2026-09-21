# Corrección operativa010-CPU — no esperar una GPU para evaluación CPU

Documento histórico:010-CPU ya completó. No repetir sus comandos. La entrega
activa es `docs/SELECTOR_MITOSIS_011.md`, que añade evaluate-selector al allowlist
CPU; la fase prepare-selector continúa siendo GPU exclusiva.

Diseñador: Codex. Operador: Antigravity. Nodo: A1688, sin cambio de servidor.
Esta revisión **no cambia la hipótesis, métricas, matching, fuentes ni parámetros
del ensayo010**. Reemplaza sus comandos de lanzamiento anteriores.

## Motivo y decisión

El operador informó A1688 con VLLM activo, GPU100%,21264 MiB ocupados y2708 MiB
libres. No lanzó tests010 ni el contrafactual. Hizo lo correcto bajo el contrato
anterior: no detener procesos ajenos ni eludir la guardia.

La guardia GPU estaba aplicada a todas las tareas. La auditoría del código muestra
que `audit-mitosis` evalúa grafos en CPU y la suite utiliza fixtures CPU, SCIP e
imports de dependencias; no necesita inferencia real en CUDA. La excepción ahora
está implementada por el diseñador, no es una edición improvisada del operador.

Solo `tests` y `audit-mitosis` tienen perfil `cpu_evaluation_v1`. Todas las demás
tareas conservan `gpu_exclusive_v1`, procesos GPU vacíos y>=12 GiB VRAM libre.
No existe `--skip-gpu`, `--force`, selector libre de perfil ni fallback de entrenamiento.
El notebook, los heads y la lógica de `core/mitosis_counterfactual.py` permanecen
intactos. Las fuentes005–009 siguen verificándose mediante la misma cadena de hashes.

## Aislamiento y admisión

- El proceso nuevo recibe `CUDA_VISIBLE_DEVICES=-1` **antes** de importar PyTorch.
  También sus subprocesos. Antes/después se exige cero dispositivos CUDA visibles
  y `cuda_initialized=false`. Es el mecanismo documentado por
  [PyTorch](https://docs.pytorch.org/docs/2.14/cuda_environment_variables.html).
- Afinidad a un máximo de dos CPU lógicas permitidas y prioridad `nice>=10`.
  Solo se limita el proceso Biohub recién creado, nunca VLLM ni sus threads.
  [Semántica de afinidad de Python](https://docs.python.org/3/library/os.html#os.sched_setaffinity).
- Pools de cálculo limitados a dos hilos, interop de PyTorch a uno. SCIP ya tenía
  su límite de un hilo. La suite en su propio subproceso reaplica los límites.
- `RLIMIT_AS` máximo8 GiB **de espacio de direcciones por proceso**, heredado por
  los hijos; no eleva límites heredados menores. No es un límite agregado RSS ni
  una reserva de RAM. Si una biblioteca no puede cargarse dentro de ese límite,
  informar el error y detenerse, sin quitarlo.
  [Semántica de RLIMIT_AS](https://docs.python.org/3/library/resource.html#resource.RLIMIT_AS).
- Antes del lanzamiento y de la ejecución: disco libre>=12 GiB;
  `MemAvailable >= max(16 GiB, 25% de RAM total)`;
  carga1min<=0.75×CPU lógicas del host y afinidad disponible.
- El lock común `gpu.lock` se conserva como exclusión entre trabajos Biohub,
  incluidos los de CPU. No borrarlo ni modificarlo. No se inicia otra tarea Biohub
  simultánea. La única coexistencia nueva permitida aquí es con cargas ajenas
  que usan la GPU, sin solicitar CUDA para Biohub.

Estos controles limitan la competencia por recursos; **no garantizan impacto cero
sobre un servicio que comparte RAM, disco y CPU**. El doctor es una foto de estado,
no una reserva. Si no hay margen, esperar o informar al diseñador. No matar,
pausar, renicear, reconfigurar ni invocar el endpoint del servicio VLLM.

Si la admisión falla antes de lanzar, devuelve `status=deferred_resources`,
`run_created=false` y motivos: **no crea carpeta/receipt ni consume el ID**.
No ejecutar `status`/`fetch` como si hubiera un job. Un nuevo intento con el mismo
ID solo es seguro después de una respuesta explícita de ese tipo. Una respuesta
incierta exige consultar estado primero; nunca repetir una mutación a ciegas.
Si el recurso cambia entre admisión y worker, o el lock está tomado, el worker
puede fallar y preserva su receipt: ese ID ya está consumido.

## Release y pruebas

Release anterior, no ejecutada según el operador:
`4a3db15a64615e0e266fdcc9906966710be828f639017e9bcfff625e79086bd5`.

Nueva release operativa:
`cf085ecf611e15034fce58c6fa3a8ee41f458fb019a0feabe272102ba4f2feba`.

121/121 pruebas locales pasaron. La suite remota completa tiene135 pruebas:
las119 anteriores más15 pruebas operativas y una prueba remota de aislamiento
real. No se omite ninguna prueba SCIP/GEFF/Zarr/metric ni la fixture de paridad.
`ops/verify.py --full` usa discovery completo y genera `tests.json`; cualquier
skip, fallo, descubrimiento vacío o ejecución parcial rechaza el resultado.
La Mac no ejecutó las135 completas ni un ensayo real.

## Orden autorizado de ejecución

No continuar esperando para lanzar los IDs antiguos cuando se libere la GPU.
Usar los nuevos IDs con sufijo `-cpu`. La receta sigue siendo010, no una011.

```bash
cd /Users/matias/Proyects/Biohub
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/verify.py
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py doctor --node A1688 --task tests
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1688 --task tests --run-id b0970-tests-010-cpu
```

Si la respuesta fue `launched`, seguir el job sin duplicarlo:

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1688 --run-id b0970-tests-010-cpu
```

Exigir `completed`, mismo hash, `tests_passed=true`,135/135 sin skips y
`resource_profile=cpu_evaluation_v1`. Revisar `tests.json`, su `cpu_runtime`
y `cpu-isolation.json`. Solo entonces:

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py doctor --node A1688 --task audit-mitosis
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1688 --task audit-mitosis --source-run b0970-prepare-005 --tests-run b0970-tests-010-cpu --run-id b0970-counterfactual-010-cpu
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1688 --run-id b0970-counterfactual-010-cpu
```

El último `status` solo corresponde si hubo `launched`. Consultar cada60–120s;
no lanzar el ensayo si los tests fallan. Usar únicamente el CLI con los aliases
seguros UCBV y el intérprete AIEnv local. No usar `--worker` manualmente ni instalar
paquetes/servicios para evitar un fallo. No cambiar de nodo.

Al terminar o fallar, recuperar solo los runs que sí se crearon, una vez cada uno:

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py fetch --node A1688 --run-id b0970-tests-010-cpu
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py fetch --node A1688 --run-id b0970-counterfactual-010-cpu
```

Entregar receipts/logs, `admission.json`, `environment.json`, `cpu-isolation.json`,
`cpu-runtime-before.json`, `cpu-runtime-after.json`, `tests.json` y todas las
salidas científicas previstas en `COUNTERFACTUAL_MITOSIS_010.md`. Conservar
`uses_gt_for_edits=true`, `tracking_evaluated=true`, `graph_exported=false`,
`confirmation_evaluated=false`, `promoted=false`.

No entrenar, evaluar confirmación/visible ni publicar. Detenerse y devolver
evidencia incluso si el score contrafactual supera0.970: no es score de un modelo.
