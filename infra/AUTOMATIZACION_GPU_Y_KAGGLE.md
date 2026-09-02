# Automatización de GPU remota y Kaggle

## Qué queda automático

- `experimentctl run EXP-XXX --node auto` consulta en vivo todas las GPUs de los nodos permitidos, exige el mínimo de VRAM del manifiesto y fija la GPU elegida con `CUDA_VISIBLE_DEVICES`.
- `experiment_scheduler.py` puede consumir la cola continuamente, pero solo lanza manifiestos con `execution_policy.auto_launch=true`.
- `kaggle_pipeline.py` construye el notebook, hace preflight test-only, sube una versión T4, persiste logs, valida el CSV, realiza la submission y consulta el score.
- Los logs completos viven en `.state/kaggle/JOB_ID/`; la salida normal es JSON compacto.
- Al entrar en modo automático se respeta primero `EXP-031 → EXP-030B`. EXP-030B es la nueva versión full-fit de EXP-030 y depende del éxito remoto de EXP-031.
- Tras completar ambos, `variant_planner.py` mantiene una búsqueda Kaggle-guided con recetas reproducibles y carpetas aisladas.

## Comandos

```bash
conda activate AIEnv

# Inventario en vivo, sin lanzar trabajos
python infra/experimentctl.py nodes --probe

# Selección automática; dry-run no entrena
python infra/experimentctl.py run EXP-031 --node auto --dry-run

# Scheduler opt-in
python infra/experiment_scheduler.py --once --dry-run
python infra/experiment_scheduler.py --interval 60

# Vista previa determinista: propone pero no persiste, encola ni entrena
python infra/variant_planner.py plan --count 5 --seed 20260828

# Kaggle: comprobar sin red/mutaciones externas más allá de construir el notebook local
python infra/kaggle_pipeline.py preflight EXP-030
python infra/kaggle_pipeline.py push EXP-030 --dry-run

# Flujo externo real, autorizado solo con deployment_ready=true
python infra/kaggle_pipeline.py push EXP-030
python infra/kaggle_pipeline.py watch JOB_ID --interval 60
```

## Política test-only

El notebook tiene una guardia interna: solo acepta rutas con un componente `test`, rechaza cualquier componente `train`, falla si no encuentra `.zarr` de test y no fabrica submissions dummy. Emite dos eventos estructurados en sus logs:

```text
BIOHUB_DATASET_SCOPE={"scope":"test",...}
BIOHUB_DATASET_PATH=/kaggle/input/.../test/...
```

El monitor exige esos eventos al finalizar. Si ve train, un scope distinto, un CSV vacío o columnas incorrectas, pasa el job a `POLICY_BLOCKED` y no reintenta ni envía la submission.

Kaggle no publica una operación segura para cancelar solamente la sesión activa desde su CLI. El corte efectivo se realiza con la excepción de la guardia dentro del notebook; desde afuera se detienen el workflow, los reintentos y la submission. Nunca se usa `kernels delete`, porque borraría el notebook.

## Reintentos

Solo se reintentan automáticamente errores de infraestructura transitorios y hasta `max_retries`. Tracebacks de código, CUDA OOM, archivos o módulos ausentes quedan `NEEDS_REPAIR`, preservando el log para que un agente haga una corrección informada antes de subir una versión nueva.

## Dos rankings independientes

- `experiments/RESULTS.md` y los Promotion Boards contienen evaluaciones locales.
- `experiments/KAGGLE_LEADERBOARD.md` contiene exclusivamente el mejor Public Score oficial de Kaggle por experimento.
- El ledger factual de Kaggle es `experiments/kaggle_scores.jsonl`.

El pipeline actualiza automáticamente el ledger y la tabla cuando una submission obtiene score. No se comparan directamente valores locales con valores Kaggle ni se supone que ambos rankings deben coincidir.

## Estado de los paquetes actuales

- EXP-015, EXP-017 y EXP-022: paquetes legacy normalizados con `deployment.json`, T4, pesos obligatorios y guard test-only. Son el top 3 actual según el Promotion Board común de 12 muestras.
- EXP-030: paquete Kaggle completo y `deployment_ready=true`, todavía sin score global estructurado para entrar al top 3.
- EXP-030B y EXP-031: política preparada, pero `deployment_ready=false` hasta que existan sus pesos exportados, `inference.py`, notebook y metadata propios. El controlador rechazará una subida prematura.

## Órdenes de alto nivel

Las frases exactas están respaldadas por `infra/agent_commands.py`:

```bash
python infra/agent_commands.py "Cortar train remoto"
python infra/agent_commands.py "Subir mejores versiones a kaggle"
python infra/agent_commands.py "Modo automatico"
python infra/agent_commands.py "Cortar automatico"
```

`Modo automatico` no retorna inmediatamente: inicia o se adjunta al daemon y mantiene un supervisor foreground. Emite `AUTOMATIC_UPDATE` cuando cambia una corrida, epoch, job Kaggle o score, y `AUTOMATIC_HEARTBEAT` cada 60 segundos sin cambios. Esto permite que el agente conserve el turno abierto y muestre la evolución en lugar de desaparecer después de iniciar el daemon.

Cuando una corrida termina y queda reconciliada como `SUCCEEDED`, `DOC_PENDING` o `CLOSED`, el daemon consulta `.agents/kaggle-automation.json`. Si el manifiesto declara `auto_publish_after_train=true`, `auto_push=true` y `deployment_ready=true`, el experimento entra automáticamente a preflight, push, monitoreo, submission y consulta de Public Score. Esto cubre experimentos base, bootstrap y variantes, y no requiere invocar manualmente `Subir mejores versiones a kaggle`.

El despliegue se identifica por `experiment_id`, `run_id` y hash del paquete. Así se evita duplicar una misma corrida y se permite publicar otra versión después de un reentrenamiento real. Si Kaggle alcanzó su cupo, el candidato queda pendiente; los errores transitorios tienen backoff y reintentos acotados, mientras los determinísticos quedan para reparación.

Además, cada ciclo ejecuta un inventario read-only de procesos CUDA en todos los nodos habilitados. `.state/remote-gpu-activity.json` registra nodo, GPU, PID, VRAM, experimento inferido y asociación con la cola. Los procesos externos se muestran, pero no se adoptan ni se detienen automáticamente.

La selección de las tres mejores usa `.agents/ranking.json`: public score primero y luego métricas globales registradas; jamás loss de train. `Cortar train remoto` solo usa PID registrados y sincroniza checkpoints/artefactos/logs. `Cortar automatico` detiene el daemon, no los entrenamientos ya lanzados.

## Generación autónoma de variantes

La política canónica está en `.agents/autonomous-exploration.json` y el catálogo permitido en `.agents/variant-recipes.json`. Cada lote completo contiene tres variantes de explotación, una exploración adyacente y una novedosa. El padre se toma de los mejores Public Scores reproducibles; EXP-000 y entradas sin score quedan excluidos.

El planner registra cada propuesta en `experiments/variant_proposals.jsonl`, genera `experiments/AUTONOMOUS_VARIANTS.md` y materializa `train.py`, `variant.json`, `experiment.json`, documentación narrativa y subcarpetas propias. El trainer usa todos los datos, CUDA remota, checkpoint resumible y logs compactos. `train_loss` queda marcado solo como diagnóstico.

Las nuevas carpetas no nacen listas para Kaggle: `deployment_ready=false` impide usar accidentalmente pesos del padre. Al finalizar el train, `variant_kaggle.py` recupera el checkpoint sincronizado, exporta pesos-only, crea o versiona un dataset de pesos exclusivo, valida el notebook test-only y habilita la promoción. El pipeline existente se encarga luego del push, monitor, submission y score.
