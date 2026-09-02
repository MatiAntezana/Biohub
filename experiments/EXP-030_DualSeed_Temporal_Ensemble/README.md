# EXP-030 — carpeta operativa normalizada

La ejecución vigente se define en `experiment.json` y usa `train.py`. Cada corrida posee:

- recibos y métricas en `artifacts/<run_id>/`;
- logs en `logs/`;
- checkpoints resumibles en `checkpoints/<run_id>/<role>/training_checkpoint.pth`;
- snapshot con hashes en `code_snapshot/<run_id>/`;
- reporte factual en `validation/<run_id>/REPORT.md`.

Validar con `python infra/experimentctl.py run EXP-030 --node A1556 --dry-run`. Ninguna corrida se inicia sin repetir el comando sin `--dry-run` de forma explícita.

Los recibos legacy bajo la antigua ruta compartida se registran como evidencia histórica y no se borran ni se consideran automáticamente exitosos.

