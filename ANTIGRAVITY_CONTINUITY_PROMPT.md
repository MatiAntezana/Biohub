# Mensaje permanente para Antigravity — continuidad de Biohub

Quiero que, de acá en adelante, te hagas cargo de la continuidad del proyecto
en `/Users/matias/Proyects/Biohub`: interpretar resultados, decidir la siguiente
hipótesis, implementar el código que falte, verificarlo, ejecutar experimentos
y publicar únicamente candidatas que superen todos los controles. Ya no voy a
volver a Codex entre campañas. No te limites a ejecutar una orden y esperar que
Codex decida: realizá vos esa revisión, dejala escrita y continuá dentro del
protocolo. No confundas autonomía con cambiar gates o fabricar resultados.

Leé completos `AGENTS.md`, `docs/AUTONOMOUS_CONTINUATION.md`, `docs/STATUS.md`,
`docs/RUNBOOK.md`, `ANTIGRAVITY_HANDOFF.md` y el documento de la campaña activa.
El protocolo de continuidad sustituye las frases históricas que exigían volver
a Codex; no modifica las prohibiciones científicas de cada corrida ni sus flags.

Solo están autorizadas A1555 y A1557. A1555 es preferente y014 debe completarse
allí, porque su release verifica ese nodo y sus fuentes. A1557 puede utilizarse
después de habilitarlo con inventario, hashes, transferencia segura si falta
algo, pruebas completas del nuevo hash en destino y una nueva release compatible.
No cambiar receipts de origen para hacerlos pasar por ejecuciones en otro nodo.
No usar A1688 ni ninguna otra computadora aunque figure en un JSON histórico.
Siempre --node explícito, executor/aliases seguros UCBV y Python local exclusivo
`/Users/matias/miniforge3/envs/AIEnv/bin/python`. Nada de GPU/entrenamiento real
en la Mac, MLflow, procesos ajenos, cambios de proxies/servicios o gastos nuevos.

Primero reconciliá el estado real de014. Si está pendiente, seguí su handoff;
si está ejecutándose, consultá status; si terminó, verificá y descargá todo,
sin repetirlo.014 no entrena ni publica. Después usá su diagnóstico para elegir
UNA siguiente hipótesis: supervisión deduplicada, cobertura de candidatos,
procedencia/localización, rescate con imagen o asociaciones temporales.
El árbol de decisión, límites de variantes y criterios están en el protocolo.
Si no hay suficiente señal mitótica, cambiá al cuello de botella demostrado:
no repitas heads/umbrales ni cuentes duplicados como nueva supervisión.

Antes de cada experimento, completá `docs/NEXT_EXPERIMENT_TEMPLATE.md` en una
ficha nueva: evidencia, hipótesis falsable, control, cambios acotados, variantes,
particiones, recursos y gates. Implementá los comandos que todavía no existan;
no inventes que un diagnóstico ya se integra con experiment/package/notebook.
Congelá código/config/datos/modelos por hashes, ejecutá tests locales y la suite
completa remota del mismo hash, y usá run IDs nuevos sin sobrescribir históricos.

Conservá el champion dual-seed como control. GT nunca entra en inferencia;
unknown no es negativo. Selección/calibración solo en fuente, desarrollo pareado
y confirmación de la receta congelada. No modificar la métrica o los gates.
Una mejora local no demuestra Kaggle. Antes de publicar, verificar integración
remota/notebook, paridad, pesos activos, bundle, filas estrictamente entre234000
y242000, tiempo y smoke. Solo ops/kaggle.py, identidad exacta y como máximo una
submission por día UTC. No publicar variantes rechazadas ni barridos de umbrales.

Ante un bug: conservar evidencia, crear fixture, corregir en otra release y
volver a verificar. Ante un gate rechazado: registrar el rechazo y continuar
con la siguiente hipótesis justificada; no debilitarlo. GPU ocupada: esperar o
usar una tarea/nodo ya habilitados y admitidos, nunca detener trabajos ajenos.

Después de cada campaña, actualizá STATUS y la bitácora con hashes, nodos,
comandos, artefactos, métricas por grupo, interpretación y próximo paso exacto.
Cuando te diga “continuá”, retomá desde esos receipts sin reiniciar la búsqueda.
Preguntame solo por permisos/credenciales/datos/recursos faltantes, un cambio
de restricciones o un bloqueo real que no puedas resolver dentro del alcance.

La meta sigue siendo score Kaggle>=0.970, pero no lo prometas: declará éxito
solo con el score real de la referencia exacta COMPLETE. Si no se alcanza,
informá honestamente lo aprendido y qué impide avanzar.
