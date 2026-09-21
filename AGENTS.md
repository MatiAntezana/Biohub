# Contrato de operación y continuidad de Biohub

## Autoridad vigente — 2026-09-13

El usuario continuará SOLO con Antigravity. Desde ahora Antigravity asume diseño
incremental, implementación, ejecución e interpretación dentro de
`docs/AUTONOMOUS_CONTINUATION.md`. No debe esperar otra revisión de Codex para
continuar después de un diagnóstico. Esta autorización sustituye el rol antiguo
de “operador que no puede diseñar”; NO permite eludir gates o alterar evidencias.

Leer completos este contrato y AUTONOMOUS_CONTINUATION.md, luego STATUS,
RUNBOOK y el handoff de la campaña activa. Las órdenes antiguas de “volver a
Codex/detenerse para revisión” se satisfacen ahora con una revisión escrita de
Antigravity y la predeclaración de la siguiente campaña. Un resultado fallido
no se transforma en aprobatorio y un receipt diagnóstico no habilita publicación.

## Invariantes

1. Trabajar en /Users/matias/Proyects/Biohub. Biohub-viejo y Biohub-mejorado son
   antecedentes de solo lectura. Preservar fuentes, pesos, receipts y releases
   históricas. No reactivar dispatchers/watchers antiguos.
2. MLflow está prohibido: no instalar, importar, configurar o inicializar,
   ni crear mlruns/. Usar manifiestos/receipts JSON.
3. Python local exclusivo /Users/matias/miniforge3/envs/AIEnv/bin/python.
   La Mac controla, inspecciona y ejecuta tests estáticos/toy; nunca entrenamiento
   ni inferencia real. El cómputo científico real va a las remotas autorizadas.
4. ÚNICOS nodos autorizados: A1555/a1555/train-55 y A1557/a1557/train-57.
   Pasar --node explícitamente. El default A1688 y otros nodos del JSON
   histórico NO son autorización vigente. No consultar ni ejecutar otras compus.
   Usar core.remote_executor.execute_remotely mediante los drivers verificados
   y aliases seguros UCBV. No SSH ad-hoc, IP directa, host-key bypass, cambios
   de proxies, sockets, claves, redes, servicios o procesos ajenos.
5. La campaña014 sigue fijada a A1555. No cambiarla a A1557 por argumento o falsificando
   fuentes. Habilitar A1557 después mediante una nueva release operativa, suite
   completa en destino, inventario y traslado íntegro de fuentes cuando haga
   falta. Distinguir nodo de origen de un receipt y nodo actual de ejecución.
   El procedimiento obligatorio está en AUTONOMOUS_CONTINUATION.md, sección3.
6. Se permite implementar correcciones e hipótesis nuevas dentro del protocolo,
   con predeclaración, tests, manifest/hash y run ID nuevos. No modificar código
   o parámetros de una corrida activa ni receipts para evitar una falla.
   No cambiar la métrica, los gates, las reglas de seguridad ni las pruebas
   para ocultar errores. No bajar mínimos a posteriori para forzar fitting.
7. Guardar y comprobar procedencia real de dependencias/datos/modelos. No
   inventar hashes, revisiones, licencias o admisibilidad. No instalar “latest”
   en conda compartido como reparación rápida. Nuevas dependencias verificadas
   solo en destino aislado y en una nueva release; nunca MLflow.
8. El champion de control conserva ambos checkpoints fijados y DeepCenter.
   Un seed ausente es fallo, no fallback. No sustituir pesos históricos:
   cualquier modelo candidato nuevo tiene identidad/artefactos propios.
9. No inventar células, enlaces o coordenadas para optimizar artificialmente
   score/cardinalidad. Rescate nuevo exige evidencia real de imagen y tiempo.
   CSV completo: estrictamente 234000 < filas < 242000, excluyendo cabecera.
   No padding/poda para cumplir la cota; no exigirla a caches parciales.
10. GT solo para entrenamiento/calibración/evaluación explícitos. Inferencia
    nunca recibe GT, listas de divisiones verdaderas ni grafos oracle010.
    Unknown no es negativo/FP. El notebook procesa todos los test de Kaggle;
    no sustituirlos por cuatro nombres visibles o un CSV precalculado.
11. Detector full-fit y paneles ya expuestos no son holdout integral. Mantener
    separación fuente/prefijo evaluado, registrar exposición y fijar receta
    antes de evaluar. Confirmación no se usa para seleccionar hiperparámetros.
12. Jobs fallidos/rechazados: conservar, recuperar, interpretar y registrar.
    Una corrección justificada produce otra release/run. No relanzar una
    mutación incierta, borrar locks/receipts o sobrescribir artefactos.
13. Una tarea Biohub por nodo bajo el lock común. GPU ajena ocupada: esperar,
    nunca detenerla. Tareas CPU aprobadas mantienen cpu_evaluation_v1, CUDA
    oculta y límites de CPU/RAM/disco. GPU mantiene gpu_exclusive_v1 y sus
    guardias. No reclasificar tareas CUDA como CPU para evitar admisión.
    Paralelismo entre A1555/A1557 solo tras validar ambos entornos, con tareas
    independientes e IDs distintos. Una publicación/cuota centralizada.
14. Publicar solo candidatas con tests completos, paridad de referencia V19,
    gates pareados de desarrollo/confirmación, modelo final compatible, contrato
    visible/tiempo, bundle íntegro y smoke real Kaggle. Inferencia compartida
    entre remota/notebook. El builder antiguo NO integra por sí solo ranking014.
    Implementar/probar esa integración antes de empaquetar una nueva familia.
15. Publicación solo mediante ops/kaggle.py, kernel privado nuevo por run.
    Máximo interno una submission por día UTC y por artefacto único, aunque
    Kaggle conceda más. No modificar el kernel histórico. Escritura incierta:
    reconciliar el resultado real antes de otra acción; no borrar *_attempted.
16. Alcanzar0.970 solo se declara con score finito >=0.970 de la referencia
    exacta COMPLETE en Kaggle. No usar un score local, recall de nodos, AUROC
    o la submission más reciente de otro proceso como prueba.
17. No contratar servicios, gastar dinero/créditos, ampliar a otras máquinas,
    acceder a datos privados nuevos o modificar invariantes sin el usuario.
    Consultarlo solo por una decisión/permiso realmente faltante o bloqueo
    irresoluble; no derivar cada resultado rutinario de nuevo a Codex.

## Campaña inmediata014 y transición

014 conserva su código/receta/hash y gates; no entrenar ni publicar desde014.
Seguir docs/COVERAGE_SUPERVISION_014.md y ANTIGRAVITY_HANDOFF.md:
verificación local → audit_coverage_inputs → tests014 → prepare-coverage014 →
audit-coverage014 → fetch completo → revisión escrita por Antigravity.

Se reutilizan35 muestras FIT011, preferencias012 y evidencia013 con hashes.
Única expansión014:44b6_12dfb391,44b6_a21120c2,44b6_e28840c6, previamente
expuestas y autorizadas solo como supervisión, nunca holdout. Prepare-coverage
usa GPU exclusiva y24 GiB libres; no abre GT. Audit-coverage usa CPU limitada;
GT solo de35 FIT +3 adicionales. Nunca GT de calibración/desarrollo/confirmación/
visible en014. No modificar matching, radio/K, detector o colisiones de014.

Verificar hashes de todos los NPZ011 no equivale a evaluar los paneles reservados.
No contar comparaciones repetidas como nuevas divisiones. Proximidad a una
detección no demuestra identidad celular ni causa exacta de pérdida.

Tras014, no esperar a Codex: aplicar el árbol de AUTONOMOUS_CONTINUATION.md.
Predeclarar la siguiente hipótesis con NEXT_EXPERIMENT_TEMPLATE.md, implementar
lo necesario en una nueva release y repetir pruebas. Una futura campaña de
tracking tiene que superar sus propios gates:014 no autoriza saltarlos.

Status/fetch son las vías de seguimiento; no Python genérico/SSH improvisado.
Fetch completo, sin excluir NPZ/cache. --resume solo para descarga parcial con
receipt terminal coincidente, preservando existentes. Ante corrupción, reportar
y conservar; no borrar para forzar reemplazo. Deferred_resources/run_created=false
no consume ID. Un receipt existente sí lo consume.

Actualizar docs/STATUS.md y fichas por campaña con hash/nodo/run, resultados,
decisión, exposición y próximo comando. Al reanudar, leer ese estado y retomar
sin repetir corridas. Para contradicciones con textos históricos, prevalecen
las instrucciones vigentes del usuario y este contrato.
