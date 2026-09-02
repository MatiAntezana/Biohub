# Protocolo obligatorio de experimentación

## Inicio de sesión

1. Activar `AIEnv` con `source ~/miniforge3/etc/profile.d/conda.sh && conda activate AIEnv`.
2. Leer `AGENTS.md`, estas reglas y las vistas `.state/STATE.md` y `.state/QUEUE.md`.
3. Ejecutar `python infra/experimentctl.py doctor` y `python infra/experimentctl.py nodes --probe`.
4. Ejecutar `python infra/experimentctl.py validate EXP-XXX` antes de encolar o lanzar.

## Flujo

1. `queue EXP-XXX` registra el trabajo. Sin dependencias pendientes queda `READY`; de lo contrario, `QUEUED`.
2. `run EXP-XXX --node auto` consulta VRAM libre y lanza en la mejor GPU permitida. Un nodo explícito también selecciona su GPU local con más memoria libre.
3. `experiment_scheduler.py` solo considera trabajos `READY` cuyo manifiesto tenga `auto_launch=true`; los demás nunca se autolanzan.
4. El worker escribe log, heartbeat y recibo terminal dentro de la carpeta del experimento.
5. `reconcile` recupera recibos y repara vistas. Un éxito genera reporte factual y estado `DOC_PENDING`.
6. `close RUN_ID --conclusion-file ... --decision ... --doc-link ...` aplica el gate documental.

## Flujo Kaggle obligatorio

1. Marcar `kaggle.deployment_ready=true` solo cuando exista el paquete completo y sus pesos de inferencia.
2. Ejecutar `python infra/kaggle_pipeline.py preflight EXP-XXX`.
3. Ejecutar `python infra/kaggle_pipeline.py push EXP-XXX`; esto construye, valida, sube con T4 y registra la versión.
4. Mantener `python infra/kaggle_pipeline.py watch JOB_ID`; descarga logs, impide train, valida `submission.csv`, realiza la submission y registra el score.
5. Ante `NEEDS_REPAIR`, corregir código/pesos y crear una versión nueva. No repetir indefinidamente un error determinístico.
6. Al recibir score, el pipeline debe anexarlo a `experiments/kaggle_scores.jsonl` y regenerar `experiments/KAGGLE_LEADERBOARD.md`.
7. Para decidir por Kaggle, usar exclusivamente esa tabla. Para decidir por test local, usar exclusivamente el Promotion Board correspondiente.

### Publicación automática después del entrenamiento

1. Después de cada `reconcile`, buscar todas las corridas `SUCCEEDED`, `DOC_PENDING` o `CLOSED`, no solamente propuestas del planificador de variantes.
2. Si el manifiesto declara `kaggle.auto_publish_after_train=true`, `auto_push=true` y `deployment_ready=true`, ejecutar automáticamente preflight y push cuando exista cupo Kaggle.
3. Registrar la identidad con `experiment_id + run_id + package_sha256` para impedir pushes duplicados y permitir una versión nueva después de un reentrenamiento real.
4. Si el cupo está ocupado, mantener el candidato pendiente y volver a considerarlo en ciclos posteriores. Los fallos transitorios usan backoff y un presupuesto acotado; los determinísticos quedan bloqueados para reparación.
5. Tras el push, el mismo ciclo de `watch` debe validar que solo se usó test, descargar y validar el CSV, hacer la submission y esperar el Public Score.
6. Esta ruta es obligatoria durante `Modo automatico`; la orden `Subir mejores versiones a kaggle` es un mecanismo manual adicional y no debe ser necesaria para completar el flujo normal.

## Control del modo automático

- `python infra/agent_commands.py "Modo automatico"` inicia o adjunta los ciclos desacoplados y mantiene un supervisor foreground que emite progreso compacto. El agente debe seguir la sesión hasta una orden de corte.
- `python infra/agent_commands.py "Cortar automatico"` detiene esos ciclos y devuelve un resumen con una pregunta de continuidad.
- `python infra/agent_commands.py "Cortar train remoto"` corta hijos registrados y recupera su checkpoint más reciente.
- `python infra/agent_commands.py "Subir mejores versiones a kaggle"` promueve hasta tres candidatos con métricas comparables y package test-only listo.

### Bucle de comunicación del agente

1. Ejecutar `Modo automatico` mediante un comando que pueda permanecer activo.
2. Si la herramienta entrega un `session_id`, hacer polling en intervalos no mayores a 60 segundos.
3. Convertir `AUTOMATIC_UPDATE` en una actualización inmediata para el usuario.
4. Convertir `AUTOMATIC_HEARTBEAT` en una línea breve de continuidad; no repetir logs ni explicaciones sin cambios.
5. Continuar aunque un entrenamiento tarde horas. No emitir respuesta final por el mero hecho de que el daemon quedó desacoplado.
6. Ante `Cortar automatico`, detener el daemon, esperar `AUTOMATIC_DETACHED`, resumir corridas, artefactos, Kaggle, scores, errores y preguntar cómo proseguir.
7. En cada snapshot incluir `remote_gpu_activity`: experimentos detectados, nodo, GPU, PID, VRAM y si la corrida está registrada. Notificar inmediatamente altas y bajas de procesos CUDA.

### Secuencia científica automática

1. Reconciliar y ejecutar primero EXP-031.
2. Su éxito habilita EXP-030B; ejecutarlo a continuación. `READY` por sí solo no salta esta prioridad.
3. Mientras alguno siga sin entrenamiento remoto exitoso, `variant_planner.py` devuelve `blocked_by_bootstrap` y no crea carpetas.
4. Después, mantener hasta cinco variantes activas con la proporción 60/25/15 definida en `.agents/autonomous-exploration.json`.
5. Materializar solo recetas listadas, registrar linaje/hash y encolar. El scheduler limita a dos corridas remotas simultáneas y elige GPU por VRAM libre.
6. Consultar `experiments/AUTONOMOUS_VARIANTS.md` para estado de búsqueda, `KAGGLE_LEADERBOARD.md` para el resultado oficial y el Promotion Board para el test local independiente.
7. Para una variante `TRAINED`, exportar y publicar su dataset de pesos mediante `variant_kaggle.py`; solo después habilitar `deployment_ready`, hacer push test-only y alimentar el siguiente ciclo con su Public Score.
8. Aplicar el mismo autopublicador a experimentos base y bootstrap ya empaquetados; nunca esperar una orden manual una vez satisfechos los gates.

## Criterios de parada

- Parar antes de lanzar si falla el esquema, el entorno remoto, CUDA, el dataset, una dependencia o el aislamiento de rutas.
- Marcar `FAILED` cuando el proceso termina con código distinto de cero.
- Marcar `ORPHANED` cuando no existe proceso verificable y el heartbeat está vencido.
- No promover a Kaggle sin checkpoint íntegro, hashes, validación del cargado de pesos y contrato CSV correcto.
- Parar el notebook antes de inferencia si el dataset resuelto no está bajo `test/` o contiene `train/`.
