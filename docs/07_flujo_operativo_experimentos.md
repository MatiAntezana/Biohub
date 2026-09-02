# Flujo operativo determinístico de experimentos

Este es el único flujo soportado para EXP030, EXP030B, EXP031 y experimentos futuros.

## Reinicio seguro de contexto

Después de `/clear` o al comenzar otra sesión, leer en este orden:

1. `AGENTS.md`;
2. `.agents/rules.md` y `.agents/experiment-protocol.md`;
3. `.state/STATE.md` y `.state/QUEUE.md`.

Las vistas Markdown son regenerables. La fuente canónica es `experiments/registry.jsonl`; no se corrige manualmente una vista divergente.

## Preparar y lanzar

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate AIEnv
python infra/experimentctl.py doctor
python infra/experimentctl.py nodes
python infra/experimentctl.py validate EXP-031
python infra/experimentctl.py queue EXP-031
python infra/experimentctl.py run EXP-031 --node A1556 --dry-run
python infra/experimentctl.py run EXP-031 --node A1556
```

Los entrypoints locales verifican `AIEnv` y se reejecutan allí automáticamente si se invocan desde otro Python. Esta regla cubre controlador, watcher, monitor Kaggle, validaciones y exportación de pesos. Los workers remotos utilizan sus entornos GPU: A1552=`train-52`, A1556=`train-56`, A1557=`train-57` y A1560=`train-60`. A1560 permanece registrada como `OFFLINE` y el controlador rechazará lanzamientos hasta que se la vuelva a verificar.

El último comando es el único que inicia cómputo. Genera un `run_id`, congela código/configuración, sincroniza con `rsync`, activa el entorno Conda del nodo y deja un worker desacoplado. La salida del controlador es un JSON compacto; el detalle queda bajo `experiments/EXP-XXX/logs/`.

`READY` no autoriza un lanzamiento automático. Cuando termina EXP031, EXP030B cambia de `QUEUED` a `READY`, pero permanece detenido hasta otro `experimentctl run` explícito.

El nodo se elige con `--node A1552`, `--node A1556`, `--node A1557` o `--node A1560`. Antes de crear una corrida real, el controlador comprueba por SSH el entorno Conda, CUDA y las dependencias Biohub. Un nodo `OFFLINE`, como actualmente A1560, es rechazado antes de escribir un evento `RUNNING`.

## Monitorear y recuperar

El lanzamiento crea un watcher local desacoplado. Si la Mac duerme o el watcher termina, el worker remoto continúa y sus recibos siguen siendo la fuente de verdad.

```bash
python infra/experimentctl.py status
python infra/experimentctl.py reconcile
```

`reconcile` descarga recibos, verifica hashes y regenera vistas. Un heartbeat vencido con procesos inexistentes se registra como `ORPHANED`; no se supone éxito ni se borra material remoto.

## Cerrar la documentación

Una corrida exitosa termina en `DOC_PENDING` y genera `validation/RUN_ID/REPORT.md`. Para cerrarla:

```bash
python infra/experimentctl.py close RUN_ID \
  --conclusion-file experiments/EXP-XXX/validation/RUN_ID/CONCLUSION.md \
  --decision "Promover, iterar o descartar con justificación" \
  --doc-link "Librería-Registros/Experimentos/EXP-XXX.md"
```

La conclusión y el documento narrativo deben existir. Al cerrar, una dependencia puede pasar a `READY`, pero no se lanza ningún proceso.

## Checkpoint y Kaggle

Los experimentos normalizados guardan `training_checkpoint.pth` con modelo, optimizador, próxima época, RNG y configuración. Para generar pesos de inferencia sin crear otro checkpoint activo:

```bash
python infra/export_inference_checkpoint.py \
  --checkpoint experiments/EXP-XXX/checkpoints/RUN_ID/model/training_checkpoint.pth \
  --output experiments/EXP-XXX/kaggle/artifacts/RUN_ID/model_state_dict.pth
```

Kaggle solo puede ejecutar inferencia con T4. El submission debe contener exactamente `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id`.
