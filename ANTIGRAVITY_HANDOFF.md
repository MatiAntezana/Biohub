# Mensaje para Antigravity — campaña014

Este es el mensaje de la etapa inmediata. El mensaje permanente está en
ANTIGRAVITY_CONTINUITY_PROMPT.md y su protocolo en docs/AUTONOMOUS_CONTINUATION.md.
El usuario continuará solo con Antigravity: ya no requiere revisión de Codex.

Trabajá en /Users/matias/Proyects/Biohub. Leé completos AGENTS.md,
docs/AUTONOMOUS_CONTINUATION.md, docs/RUNBOOK.md y docs/COVERAGE_SUPERVISION_014.md.
Primero completar014. No editar SU código, recetas, límites, umbrales o notebooks;
no repetir005–013. La continuidad posterior usa nuevas releases y predeclaraciones.

Release de contenido autorizada (NO commit Git):
becd1f5dcac099c2227ef7d46ecef7a33b04fb2c044bed2a92c158d5268a7121.
Python local exclusivo: /Users/matias/miniforge3/envs/AIEnv/bin/python.
Nodo de014: A1555, alias a1555, conda train-55. Pasar --node A1555 en esta campaña.
Solo A1555 y A1557 están autorizados. A1557 requiere habilitación verificada en
una release posterior conforme al protocolo; no sirve cambiar el argumento de014.
Toda ejecución remota por ops/biohub.py / core.remote_executor.execute_remotely.
No SSH ad-hoc, python3 genérico, proxies manuales, cambios de host keys/claves,
sockets/servicios/procesos, instalaciones o MLflow. No intervenir en VLLM.

013 pasó247 tests pero NO entrenó:44b6 tenía2 forks utilizables; las25
comparaciones del estrato event duplicaban preferencias012. No bajar mínimos.
014 audita cobertura y supervisión nueva, SIN fitting ni score de tracking.

Seguí por etapas la sección5 de docs/COVERAGE_SUPERVISION_014.md:

1. Ejecutar ops/verify.py: exigir269/269 pruebas locales.
2. Ejecutar ops/audit_coverage_inputs.py: passed=true, hash014 autorizado,
   evidence_npz=53, historical_raw_npz=35,3 muestras adicionales/5 divisiones.
   historical_novelty:3 duplicadas en44b6 y22 en6bba,0 nuevas en ambos.
   La auditoría local no abre GT/imágenes ni entrena.
3. Doctor tests → launch b0970-tests-014 → status.
   Exigir completed, tests_passed=true,285/285 full_discovery, sin skips/fallos/
   errores, hash014 y aislamiento CPU. Tests013 no sustituyen los de014.
4. Doctor prepare-coverage → launch b0970-coverage-corpus-014 con
   --node A1555 --task prepare-coverage --source-run b0970-events-013
   --tests-run b0970-tests-014. Esperar completed y ready_for_coverage_audit=true.
   Esta tarea usa GPU EXCLUSIVA: no lanzar si hay procesos GPU ajenos; esperar.
   Solo infiere44b6_12dfb391,44b6_a21120c2,44b6_e28840c6 con el champion congelado.
   Son muestras previamente expuestas autorizadas para supervisión, NO holdout.
   No abre GT ni entrena; guarda referencias/candidatos/evidencia de3 imágenes.
5. Doctor audit-coverage → launch b0970-coverage-014 con
   --node A1555 --task audit-coverage --source-run b0970-coverage-corpus-014
   --tests-run b0970-tests-014. Esperar completed y designer_review_no_submission.
   Usa CPU limitada. Reutiliza35 FIT011 y las3 adicionales; nunca GT de
   calibración/desarrollo/confirmación/visible. No aplica reparaciones.
6. Fetch COMPLETO de cada ID creado, incluso si falló. Entregar evidencia
   y cerrar014 sin entrenar/publicar con ese receipt. Después hacer la revisión
   de diseño y predeclarar la próxima campaña según AUTONOMOUS_CONTINUATION.md;
   no esperar otra decisión de Codex ni saltar los gates de una futura candidata.

No pegar todos los launches juntos. Consultar status cada60–120s. CPU tests/audit
pueden coexistir con GPU ajena si pasan las guardias; preparación GPU no.
Lock común, sin jobs Biohub paralelos. deferred_resources/run_created=false no
consume ID: esperar. Un receipt creado consume el ID: no relanzar ni inventar
otro. Si falla, conservar traceback/receipt/log. No corregir código ni parámetros
de esa corrida. Un bug reproducible se trata con fixture y nueva release de
mantenimiento según el protocolo, no con un parche de emergencia al job activo.

No usar filtros que excluyan NPZ/cache al descargar. fetch --resume solo completa
una descarga parcial con receipt terminal coincidente; no sobrescribe existentes.
Si faltan fuentes o no coinciden hashes, reportar rutas exactas, preservar todo
y detenerse. No regenerar corpus011 ni las preferencias012/013.

Entregar receipts/hashes/logs de3 runs, tests.json y auditorías de recursos;
corpus.json/split/input-audit,3 evidence NPZ y cache completo; coverage.json y
SHA256,38 reportes individuales, new-direct JSON/NPZ y new-supervision-44b6.npz.
Resumir pérdidas de extremos/radio/top-K/anotación, por prefijo y muestra.
Enumerar los5 forks adicionales y su utilidad. Separar comparaciones, madres de
preferencias y divisiones biológicas. Reportar duplicados contra012, contra
preferencias directas nuevas y dentro del lote. Unknown NO es FP ni negativo.
Proximidad raw/anchor no prueba identidad ni identifica la etapa exacta de poda.

Flags del audit: source_unchanged=true, dataset_count=38, model_trained=false,
threshold_selected=false, threshold_selection_attempted=false, graph_exported=false,
graph_edits_applied=false, uses_gt_for_edits=false, tracking_evaluated=false,
calibration_evaluated=false, development_evaluated=false, confirmation_evaluated=false,
visible_evaluated=false, kaggle_submission_allowed=false, promoted=false.
El CSV de preparación es un cache de3 referencias, NO una submission.

No entrenamiento, calibración, HOCT, experiment, package, learn-events,
rank-repairs, notebook, upload, push o submission DESDE014. Una campaña futura
debe implementar su camino científico y cumplir sus propios requisitos antes
de avanzar. No cambiar el control, presupuesto de filas ni guardias.
0.970 sigue siendo un objetivo no demostrado.
