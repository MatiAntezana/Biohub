# Biohub — laboratorio reproducible hacia 0.970

Objetivo: superar el mejor score Kaggle confirmado, **0.933**, y alcanzar **0.970**.
Este repositorio implementa hipótesis y controles de ejecución. No contiene una
receta cuyo 0.970 haya sido medido ni pesos finales ya aprobados.

Continuidad vigente: el usuario trabajará solo con Antigravity, que asume las
siguientes decisiones/implementaciones bajo el [protocolo de continuidad](docs/AUTONOMOUS_CONTINUATION.md).
Únicos nodos autorizados: A1555/A1557.014 sigue fijado a A1555; A1557 necesita
habilitación y verificación en una nueva release antes de campañas compatibles.
[Mensaje permanente para Antigravity](ANTIGRAVITY_CONTINUITY_PROMPT.md).

Estado actual:013 pasó247/247 tests remotos pero se detuvo antes de entrenar:
44b6 solo aportaba2 eventos utilizables; las25 comparaciones adicionales de ambos
prefijos duplicaban preferencias012. No se baja el mínimo para repetir el ajuste.
Siguiente paso: [cobertura y novedad de supervisión014](docs/COVERAGE_SUPERVISION_014.md).
Reutiliza35 muestras FIT de011 y prepara únicamente3 muestras44b6 previamente
expuestas, con5 divisiones anotadas y utilidad aún no medida. Separa pérdidas de
matching, disponibilidad de detecciones y radio/top-K; deduplica comparaciones.
014 no entrena ni aplica enlaces. La preparación GPU guarda referencias para las
3 imágenes; el análisis CPU no exporta grafos/CSV ni evalúa score. No es holdout.
Calibración, desarrollo, confirmación y Kaggle permanecen cerrados en014.

**Diseño:** ensamble dual-seed congelado → referencia V19 restaurada → contexto
temporal HOCT / mitosis aprendida / rescate con imagen → grafo válido → Kaggle.

La primera prueba histórica fue `mitosis_add` (rechazada en005): conserva todos los nodos y enlaces de la
referencia y permite añadir una segunda hija únicamente con evidencia neuronal
y de imagen. Aísla el efecto del nuevo clasificador. Los brazos `hoct`,
`hoct_mitosis`, `hoct_rescue` y `full` continúan la hipótesis de asociación temporal
y rescate, después de resolver sus dependencias externas.

La referencia aplica una guardia estructural explícita: rechaza un nodo de gap
sintético que colisione al redondear con un único nodo observado, sin mover el
observado ni transferir enlaces. Replay, remota y Kaggle comparten esa regla.
Registra la creación efectiva de sintéticos, incluso Gap-2 sin etiqueta; no
relaja el contrato de unicidad ni los gates. Ver `docs/FIX_PREPARE_004.md`.

- [Diseño y criterios de decisión](docs/DESIGN.md)
- [Comandos exactos para la ejecución](docs/RUNBOOK.md)
- [Mensaje completo para Antigravity](ANTIGRAVITY_HANDOFF.md)
- [Reglas del operador](AGENTS.md)
- [Estado comprobado y pendientes](docs/STATUS.md)

La Mac prepara, verifica, consulta y publica. El entrenamiento y la inferencia
remota de la campaña activa se ejecutan en **A1555**, mediante
`core.remote_executor.execute_remotely`. Pasar `--node A1555` explícitamente:
el JSON conserva el default histórico A1688 para preservar la cadena de hashes.
La remota usa `/home/alumno1/Documents/Biohub-0970`; nunca se renombra su carpeta
de datos `Biohub` por el cambio de nombre realizado en la Mac.

Chequeo local, sin entrenamiento ni conexión remota:

```bash
cd /Users/matias/Proyects/Biohub
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/verify.py
```

La ejecución completa se hace siguiendo RUNBOOK. Cada lanzamiento crea un ID
nuevo y una copia de código identificada por hash. Los recibos, logs, modelos,
caches y paquetes quedan separados del código. Las guardias impiden publicar
una variante que falle los criterios congelados.

`notebooks/submission.ipynb` es la plantilla ejecutable del notebook. El paso
`package` produce su versión concreta con hashes y pesos reales. No editar
manualmente la plantilla para saltarse ese paso.
