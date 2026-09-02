# Reglas persistentes de Biohub

## Invariantes

- Toda ejecución de control en la Mac usa obligatoriamente `AIEnv`. Los scripts locales se reejecutan automáticamente mediante `conda run -n AIEnv` si hace falta.
- Los workers remotos usan el entorno GPU declarado para cada nodo: `train-52` en A1552, `train-55` en A1555, `train-56` en A1556, `train-57` en A1557, `train-58` en A1558, `train-59` en A1559, `train-60` en A1560, `train-63` en A1563 y `train-88` en A1688. `AIEnv` es exclusivamente local y no debe fingirse su activación remota.
- Todas las conexiones SSH y rsync a computadoras remotas utilizan exclusivamente los alias de host configurados en `~/.ssh/config` a través del gateway UCBV (`ssh a1552`, `ssh a1555`, `ssh a1557`, etc.).
- A1555 está habilitada como nodo de entrenamiento permitido (`ONLINE`, `training_ready=true`). A1556 está en estado `RESERVED`.
- Todo entrenamiento o evaluación pesada se ejecuta en una GPU remota autorizada. La Mac local solo valida, prepara, sincroniza y monitorea.
- Kaggle se usa para inferencia y submission, nunca para entrenamiento. Los kernels GPU deben declarar `NvidiaTeslaT4`; P100 está prohibida.
- Cada experimento posee su carpeta bajo `experiments/`. Sus logs, artefactos, checkpoints, snapshots y validaciones no pueden mezclarse con los de otro experimento.
- Un experimento usa un checkpoint canónico resumible por rol/seed. Los pesos de inferencia exportados para Kaggle son artefactos de despliegue, no checkpoints activos.
- La fuente de verdad operativa es `experiments/registry.jsonl`. `STATE.md`, `QUEUE.md` y `RESULTS.md` se regeneran desde datos estructurados.
- Una dependencia satisfecha solo cambia el trabajo a `READY`. El scheduler puede iniciarlo únicamente si el manifiesto opta explícitamente por `execution_policy.auto_launch=true`; el valor por defecto y los experimentos existentes es `false`.
- La selección automática consulta todas las GPUs de los nodos permitidos y elige la GPU individual con mayor VRAM libre que cumpla `min_free_vram_mib`. El índice elegido se fija con `CUDA_VISIBLE_DEVICES` y se registra en el recibo.
- Una corrida exitosa pasa a `DOC_PENDING`. Solo llega a `CLOSED` con conclusión, decisión y enlace a documentación narrativa existentes.
- Los recibos y manifiestos se escriben atómicamente. Un `RUNNING` sin proceso o heartbeat vigente se reconcilia como `ORPHANED`, nunca como éxito supuesto.
- El CSV de competencia tiene exactamente: `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id`.
- Todo notebook Kaggle debe emitir `BIOHUB_DATASET_SCOPE` y `BIOHUB_DATASET_PATH`, rechazar cualquier ruta `train/`, fallar si no existe `test/` y generar un CSV real no vacío. Nunca se aceptan placeholders.
- La subida y submission automática a Kaggle requieren `deployment_ready=true`, preflight estático limpio, scope runtime `test`, corrida `COMPLETE` y CSV validado. Los errores transitorios se reintentan de forma acotada; los determinísticos quedan `NEEDS_REPAIR`.
- Una corrida reconciliada como `SUCCEEDED`, `DOC_PENDING` o `CLOSED` se publica automáticamente durante `Modo automatico` cuando el manifiesto autoriza `kaggle.auto_publish_after_train=true` y `auto_push=true`. Esto aplica a experimentos base, bootstrap y variantes; no requiere una orden posterior del usuario.
- `deployment_ready=false` nunca se salta: el ciclo mantiene el experimento como pendiente de empaquetado, y solo después de exportar pesos, versionar su dataset y aprobar el preflight puede hacer push. Un experimento futuro con Kaggle habilitado debe incluir desde su creación el scaffold test-only y `auto_publish_after_train=true`.
- `experiments/kaggle_scores.jsonl` es el ledger append-only exclusivo de scores oficiales. `experiments/KAGGLE_LEADERBOARD.md` se genera desde ese ledger y ordena por el mejor Public Score de Kaggle de cada experimento.
- El ranking Kaggle y el Promotion Board local nunca se mezclan: un experimento puede liderar uno y quedar detrás en el otro. Ninguno sobrescribe al otro.
- Los checkpoints pesados (`.pt`, `.pth`, `.ckpt`, `.safetensors`) se respaldan obligatoriamente en el repositorio privado de Hugging Face (`Matycaba2016/biohub-checkpoints`) mediante `infra/hf_sync.py`. Los binarios pesados locales se purgan tras la subida exitosa, conservando el marcador `.hf_remote` y los metadatos/logs en el repositorio. Está terminantemente prohibido commitear checkpoints pesados a Git.

## Seguridad operativa

- No eliminar ni sobrescribir procesos o artefactos remotos legacy durante una reconciliación.
- No usar rutas compartidas legacy como `experiment_artifacts/`, `eval_tmp/` o `official_starter_kit/weights/` para corridas normalizadas.
- No reportar loss de entrenamiento como evidencia de generalización. Full-fit sin holdout registra validación `N/A`.
- Conservar logs completos en disco y emitir al agente únicamente resúmenes JSON compactos.
- La CLI pública de Kaggle no ofrece cancelación segura de una sesión sin borrar el notebook. Por eso el corte test-only ocurre dentro del notebook mediante excepción inmediata; el monitor externo corta retries/submission y conserva evidencia.

## Semántica de órdenes operativas

- `Cortar train remoto` está autorizado a enviar `SIGTERM` solamente al `child_pid` registrado de cada corrida `RUNNING`. Después debe sincronizar el último checkpoint, artefactos y log; nunca buscar ni matar procesos por nombre global.
- `Subir mejores versiones a kaggle` usa score público de Kaggle como primera fuente, luego métricas globales del evaluador. Está prohibido rankear por loss de entrenamiento. Solo promueve paquetes `deployment_ready` que pasen el gate test-only.
- `Modo automatico` automatiza cola opt-in, selección de GPU, reconciliación, promoción declarada, monitoreo, validación y submission. Primero debe completar `EXP-031 → EXP-030B`. Solo después puede proponer variantes mediante las recetas acotadas de `.agents/variant-recipes.json`; no puede editar código arbitrariamente ni cerrar conclusiones científicas `DOC_PENDING`.
- `Modo automatico` no puede quedar esperando la orden `Subir mejores versiones a kaggle` cuando ya existe un paquete post-train elegible. En cada ciclo debe vigilar jobs existentes, respetar el cupo Kaggle, publicar el siguiente candidato, validar su CSV, enviarlo a la competencia y registrar su Public Score.
- `Cortar automatico` detiene el daemon operativo y resume sus eventos. Los procesos remotos ya lanzados continúan hasta finalizar o hasta recibir además `Cortar train remoto`.

## Supervisión visible y persistente

- El daemon desacoplado garantiza continuidad del trabajo, pero no sustituye la presencia del agente. La orden `Modo automatico` debe usar `automatic_mode.py supervise`, permanecer en foreground y mantener vivo el turno del agente.
- Mientras el modo esté activo, está prohibido responder con un mensaje final que dé por terminada la tarea. El agente debe seguir esperando la salida del supervisor y publicar actualizaciones por `commentary`.
- `.agents/monitoring.json` define la cadencia y los eventos notificables. Se reportan inicios, progreso de epochs, estados terminales, errores, transiciones Kaggle, submissions, scores y variantes. Un heartbeat sin cambios se emite cada 60 segundos.
- La salida al agente debe ser compacta. Los logs completos permanecen en disco; solo se exponen el experimento, nodo, estado, último epoch/métricas, transición Kaggle, score o error relevante.
- Si llega un nuevo mensaje del usuario durante la supervisión, atenderlo sin perder el daemon. `Cortar automatico` detiene el daemon y permite cerrar el turno con un resumen; `Cortar train remoto` es una orden separada.
- Existe un heartbeat de Codex cada cinco minutos con id `biohub-automatic-live-monitor` para recuperar visibilidad si el turno foreground se interrumpe. Al procesar `Cortar automatico`, el agente debe desactivarlo mediante la herramienta de automatizaciones después de detener el daemon.
- El modo automático escanea todos los nodos habilitados (`ONLINE` y `training_ready=true`) mediante `infra/remote_gpu_monitor.py`. Debe notificar procesos CUDA iniciados/detenidos e identificar el experimento desde su comando cuando contenga `EXP-XXX`.
- La actividad detectada se guarda en `.state/remote-gpu-activity.json`. Una asociación `registered` permite unirla con su run y métricas; una asociación `command` o `unknown` es solo observacional y nunca amplía autoridad operativa.

## Política de búsqueda autónoma

- El objetivo primario es el Public Score de Kaggle registrado en `experiments/kaggle_scores.jsonl`; se excluyen baselines y submissions sin score.
- La asignación de cada lote completo es 60% explotación, 25% exploración adyacente y 15% exploración novedosa, con al menos dos propuestas exploratorias y una novedosa por lote de cinco.
- La explotación parte de los mejores linajes Kaggle reproducibles. La exploración tiene presupuesto limitado y no puede desplazar indefinidamente a los linajes ganadores.
- Se prohíben configuraciones duplicadas, más de dos variantes consecutivas del mismo padre/familia y la selección por loss de entrenamiento.
- `experiments/variant_proposals.jsonl` es append-only y `experiments/AUTONOMOUS_VARIANTS.md` es su vista generada.
- Una variante materializada queda `kaggle.deployment_ready=false` hasta exportar pesos, publicar su dataset de pesos, construir un notebook test-only y pasar el preflight. `variant_kaggle.py` realiza ese cierre de forma determinística después del train. Nunca usar pesos del padre como sustituto silencioso.
- `.agents/kaggle-automation.json` gobierna concurrencia, backoff y presupuesto de reintentos. La identidad de despliegue incluye experimento, `run_id` y hash del paquete: una corrida no se duplica, pero un reentrenamiento posterior sí genera una nueva versión aunque conserve el mismo notebook.
