# Biohub — reglas operativas para agentes

Antes de modificar, ejecutar o desplegar un experimento:

1. Leer `.agents/rules.md` y `.agents/experiment-protocol.md` completos.
2. Leer `.state/STATE.md` y `.state/QUEUE.md`; son vistas generadas y no se editan a mano.
3. Usar `python infra/experimentctl.py ...` para validar, encolar, ejecutar, reconciliar y cerrar corridas. `run --node auto` selecciona la GPU remota con más VRAM libre.
4. Toda operación local debe ejecutarse con `conda activate AIEnv`; los entrypoints de infraestructura lo fuerzan automáticamente.
5. No ejecutar entrenamiento ni evaluación GPU pesada en la Mac local.
6. No escribir checkpoints, logs ni artefactos fuera de la carpeta propia del experimento.
7. No considerar cerrada una corrida mientras permanezca en `DOC_PENDING`.
8. El autolanzamiento remoto solo está autorizado cuando el manifiesto declara `execution_policy.auto_launch=true`.
9. Kaggle es exclusivamente test/inferencia. Usar `infra/kaggle_pipeline.py`: cualquier señal de train bloquea retries y submission; un notebook sin test debe fallar por su guardia interna.
10. En `Modo automatico`, toda corrida reconciliada como `SUCCEEDED`, `DOC_PENDING` o `CLOSED` cuyo manifiesto declare `kaggle.auto_publish_after_train=true`, `auto_push=true` y `deployment_ready=true` debe entrar automáticamente al preflight, push, monitor, submission y consulta de Public Score. No esperar la orden manual `Subir mejores versiones a kaggle`.
11. Almacenamiento desacoplado: Todo checkpoint pesado (`.pt`, `.pth`, `.ckpt`, `.safetensors`) se respalda obligatoriamente en Hugging Face Hub (`infra/hf_sync.py`) y se purga del disco local al completar la reconciliación. Queda estrictamente prohibido commitear binarios pesados a Git.

## Órdenes exactas del usuario

- `Cortar train remoto`: ejecutar `python infra/agent_commands.py "Cortar train remoto"`. Señala únicamente los procesos registrados como RUNNING y trae checkpoints, recibos y logs.
- `Subir mejores versiones a kaggle`: ejecutar `python infra/agent_commands.py "Subir mejores versiones a kaggle"`. Selecciona como máximo tres paquetes elegibles según `.agents/ranking.json`, los sube y crea jobs monitoreables.
- `Modo automatico`: ejecutar `python infra/agent_commands.py "Modo automatico"`. El comando inicia o adjunta el daemon y queda en foreground supervisándolo. El agente no debe cerrar el turno ni responder con un mensaje final mientras continúe activo.
- `Cortar automatico`: ejecutar `python infra/agent_commands.py "Cortar automatico"`. Detiene el daemon, muestra el resumen y formula la pregunta de continuidad. No corta entrenamientos remotos por sí solo. También pausar o eliminar mediante la herramienta de automatizaciones el heartbeat `biohub-automatic-live-monitor` indicado en `.agents/monitoring.json`.

### Persistencia obligatoria de la sesión en modo automático

- Después de invocar `Modo automatico`, mantener abierta la llamada de supervisión y leer su salida incremental. Si la herramienta devuelve un identificador de sesión/proceso, seguirla con polls de hasta 60 segundos; no abandonarla después del mensaje inicial.
- Comunicar en `commentary` cada transición real y los cambios de epoch/métricas. Si no cambia nada, enviar un heartbeat breve como máximo cada 60 segundos indicando qué sigue corriendo, nodo y último progreso conocido.
- Nunca volcar logs completos al chat. Traducir los eventos JSON compactos `AUTOMATIC_UPDATE` y `AUTOMATIC_HEARTBEAT` a resúmenes claros.
- Solo terminar el turno de supervisión cuando el usuario escriba `Cortar automatico`, cuando una política quede pausada requiriendo una decisión humana o cuando exista un bloqueo irrecuperable que deba informarse.
- Si el daemon muere accidentalmente pero su estado seguía `RUNNING`, el supervisor debe reiniciarlo y avisar. Un estado explícito `STOPPED` no se reinicia.
- En cada ciclo consultar todas las GPUs de nodos `ONLINE` con `training_ready=true`, no solo las corridas presentes en la cola. Informar qué `EXP-XXX` está usando cada nodo/GPU, su PID y VRAM cuando exista un proceso CUDA.
- Un proceso GPU descubierto por comando pero no registrado se notifica como externo/no administrado. Monitorearlo es read-only y no autoriza detenerlo, adoptarlo ni modificar sus artefactos.

`experiments/registry.jsonl` es la fuente canónica append-only. Los documentos narrativos de
`Librería-Registros/` explican decisiones científicas, pero no sustituyen el ledger.

Los resultados locales y Kaggle son rankings distintos: consultar `experiments/RESULTS.md` para experimentación local y `experiments/KAGGLE_LEADERBOARD.md` para el ranking exclusivo por Public Score oficial.

## Orden y variantes autónomas

- Al iniciar `Modo automatico`, la secuencia bloqueante es `EXP-031 → EXP-030B` (EXP-030B es la versión full-fit vigente de EXP-030). No generar variantes nuevas hasta que ambos entrenamientos remotos hayan terminado.
- Después se aplica `.agents/autonomous-exploration.json`: 60% de las propuestas mejoran los mejores linajes por Public Score de Kaggle, 25% exploran cambios cercanos y 15% reservan caminos nuevos.
- Solo se permiten recetas de `.agents/variant-recipes.json`. Cada variante debe registrar padre, score del padre, carril, receta, seed y hash de configuración en `experiments/variant_proposals.jsonl`.
- La planificación nunca ejecuta entrenamiento directamente: materializa una carpeta aislada, la encola y el scheduler común decide el lanzamiento remoto según VRAM y presupuesto de concurrencia.
- Al terminar una variante, `variant_kaggle.py` recupera el checkpoint ya sincronizado, exporta pesos-only, crea/versiona un dataset de pesos propio, habilita el notebook test-only y deja que el pipeline haga push, monitor, submission y score.
- La publicación post-train no es exclusiva de variantes: EXP-030B, EXP-031 y cualquier experimento futuro elegible siguen la política `.agents/kaggle-automation.json`. `Subir mejores versiones a kaggle` permanece como orden manual extraordinaria, no como requisito del flujo automático.
