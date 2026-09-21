# Ejecución vigente por Antigravity

## Responsabilidad y nodos

El usuario trabajará solo con Antigravity. Leer completos AGENTS.md y
[el protocolo de continuidad](AUTONOMOUS_CONTINUATION.md): Antigravity asume
diseño incremental, implementación, ejecución e interpretación, sin volver a
Codex después de cada campaña. No puede modificar evidencias o saltar gates.

Únicos nodos autorizados: A1555 y A1557. Usar --node explícito, aliases seguros
y core.remote_executor.execute_remotely. El default histórico A1688 no está
autorizado.014 permanece en A1555; A1557 requiere habilitación en una release
operativa nueva, fuentes verificadas y pruebas completas en destino.

## Etapa inmediata014

Seguir [COVERAGE_SUPERVISION_014.md](COVERAGE_SUPERVISION_014.md), sección5,
y [ANTIGRAVITY_HANDOFF.md](../ANTIGRAVITY_HANDOFF.md). Ahí están los comandos
exactos, hash y cantidades de pruebas de esta release.

Orden obligatorio:

1. Reconciliar los IDs existentes; no duplicar launches.
2. Verificación local y auditoría de hashes/novedad.
3. b0970-tests-014: CPU, suite completa del mismo hash.
4. b0970-coverage-corpus-014: GPU exclusiva, solo las3 muestras autorizadas.
5. b0970-coverage-014: CPU, solo35 FIT históricos +3 adicionales.
6. Fetch completo, interpretar y registrar.014 no entrena ni publica.

Si la GPU está ocupada, esperar sin detener procesos. Las tareas CPU solo
avanzan si pasan sus guardias de CPU/RAM/disco; se conserva el lock común.
No repetir fuentes005–013, bajar mínimos o alterar modelos de control.

## Después de014

No esperar a Codex: aplicar el árbol de
[AUTONOMOUS_CONTINUATION.md](AUTONOMOUS_CONTINUATION.md). Predeclarar una nueva
hipótesis con [NEXT_EXPERIMENT_TEMPLATE.md](NEXT_EXPERIMENT_TEMPLATE.md),
implementar los módulos/tareas que falten, congelar otra release y verificar.

No hay comandos futuros prefabricados: comprobar que cada tarea esté
implementada y probada. El experiment/package antiguo no incorpora
automáticamente el ranking nuevo. La continuidad permite implementar esa
integración, no fingir que un receipt014 ya es una candidata publicable.

Tests → control/candidata pareados → gates de desarrollo/confirmación → modelo
final compatible → contrato visible/tiempo → paridad remota/Kaggle → bundle/smoke
→ publicación controlada. Mantener los requisitos numéricos de protocol.json.
Una submission diaria interna como máximo; score de referencia exacta COMPLETE.

Persistir hash/nodo/run, artefactos, resultado, decisión y próximo comando en
STATUS y fichas de campaña. Al retomar, usar receipts para continuar sin repetir.

## Antecedentes de solo lectura

Los comandos005 antes presentes aquí quedaron en
[RUNBOOK_005_HISTORICAL.md](RUNBOOK_005_HISTORICAL.md). No son órdenes activas,
y sus nodos/reglas de revisión antiguos no prevalecen sobre AGENTS.md.
Los diagnósticos previos siguen disponibles en sus documentos históricos.
