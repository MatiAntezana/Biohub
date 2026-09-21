# Continuidad autónoma de Biohub — Antigravity

Instrucción vigente del usuario, 2026-09-13: a partir de esta entrega trabajará
solo con Antigravity. Los únicos nodos remotos autorizados son A1555 y A1557.
Este documento sustituye las órdenes antiguas de esperar una decisión de Codex
después de cada diagnóstico. No sustituye resultados, hashes, gates o recibos.

## 1. Responsabilidad y límites de la autonomía

Antigravity asume diseño incremental, implementación, tests, ejecución remota,
interpretación, empaquetado y eventual publicación de candidatas verificadas.
No necesita volver a Codex para una corrección técnica o la siguiente hipótesis
dentro de este protocolo. El usuario sigue siendo quien decide ampliaciones de
recursos, gastos, credenciales, datos no autorizados o cambios de restricciones.

Objetivo: superar el mejor score Kaggle histórico documentado,0.933, y alcanzar
un public score >=0.970 para una submission exacta en estado COMPLETE. No hay
una fórmula demostrada que garantice ese resultado. Un resultado negativo exige
interpretación y una nueva decisión, no promesas ni repetición indefinida.

Autonomía significa poder preparar una NUEVA release y un NUEVO experimento,
no editar una corrida en marcha o hacer pasar un gate fallido. Las flags
`designer_review_no_submission`, `rejected_no_kaggle_submission` o
`kaggle_submission_allowed=false` de una corrida histórica siguen siendo ciertas.
La revisión de diseño la realiza ahora Antigravity y queda registrada; NO
convierte ese mismo recibo en permiso para entrenar, empaquetar o publicar.

## 2. Arranque y estado que debe reconstruirse

Trabajar en `/Users/matias/Proyects/Biohub`. Leer completos, en este orden:

1. `AGENTS.md` y este documento.
2. `docs/STATUS.md`, `docs/RUNBOOK.md` y `ANTIGRAVITY_HANDOFF.md`.
3. `docs/COVERAGE_SUPERVISION_014.md` y los recibos de la campaña activa.
4. Para interpretar resultados anteriores, sus documentos específicos: no
   convertir comandos de001–013 en nuevas órdenes de ejecución.

No suponer que014 sigue pendiente: comprobar los IDs y artefactos existentes.
Al momento de escribir este documento no se encontró un receipt014 descargado
en A1555 local; esto NO demuestra que no exista una ejecución en la remota.
Consultar estado antes de lanzar cuando haya duda. No duplicar corridas.

014 conserva la release de contenido, no commit Git:
`becd1f5dcac099c2227ef7d46ecef7a33b04fb2c044bed2a92c158d5268a7121`.
Sus269 tests locales y285 tests completos previstos corresponden a ESA release.
Completar sus etapas en A1555 conforme al handoff. No añadir archivos de código
o configuración antes de completar014: eso cambia el hash y bloquea su cadena.
Modificar documentación Markdown no cambia el hash científico actual.

Si014 ya terminó: validar recibos, hashes y38 reportes antes de interpretar.
No repetir detector ni aprender otra vez con los mismos inputs por conveniencia.
Si014 tiene un bug: conservar la release/corrida, reproducir con un fixture,
registrar la corrección y crear una release de mantenimiento con IDs nuevos;
nunca cambiar su resultado histórico ni eliminar una guardia para que complete.

## 3. Nodos autorizados y continuidad en A1557

| Nodo | Alias | Conda | Intérprete remoto esperado |
|---|---|---|---|
| A1555 | a1555 | train-55 | /home/alumno1/miniconda3/envs/train-55/bin/python |
| A1557 | a1557 | train-57 | /home/alumno1/miniconda3/envs/train-57/bin/python |

Verificar identidad e intérprete reales con el executor antes de ejecutar.
La autorización de uso NO acredita GPU libre, dependencias completas, datos
replicados o suficiente espacio. A1555 es preferente por contener la cadena
histórica. No ejecutar ni consultar otras computadoras, incluidos A1688,
A1552,A1558,A1560. Una carpeta local `artifacts/A1688` es un archivo histórico,
no autorización para conectarse a ese nodo.

El JSON cluster histórico aún enumera otros nodos/default A1688 por trazabilidad.
NO usar ese default: siempre pasar `--node` explícito.014 exige A1555 por código
y procedencia. No basta cambiar un comando a A1557: debe rechazarse.

Después de014, habilitar A1557 en una NUEVA release operativa, antes de usarlo
para una tarea científica que actualmente lo rechaza:

1. Declarar política viva que permita exclusivamente A1555/A1557 y default seguro
   A1555. Probar rechazo de cualquier otro nodo en CLI, worker y dispatcher.
   Preservar las configuraciones históricas al verificar fuentes antiguas.
2. Inspeccionar por `core.remote_executor.execute_remotely`: identidad, conda,
   paquetes/versiones, CPU/RAM/disco/GPU, rutas de datos y assets. Si doctor no
   expone una comprobación necesaria, implementar una consulta acotada de solo
   lectura con tests; no usar SSH ad-hoc o comandos inventados.
3. Reutilizar archivos que coincidan exactamente con hashes. Si faltan, crear
   una transferencia explícita de las fuentes necesarias mediante aliases
   seguros, carpeta de staging nueva, manifiesto completo y verificación de
   destino. No sobreescribir existentes ni copiar credenciales o proyectos ajenos.
   El executor actual sincroniza código, NO provisiona por sí solo datos/assets.
4. Los receipts de origen continúan diciendo A1555/A1688 según su procedencia
   real. Registrar por separado `source_node`, `execution_node` y hashes de
   transferencia. No reescribir `node` ni fabricar receipts para superar checks.
5. Adaptar los verificadores de la NUEVA release para aceptar esas fuentes
   históricas exactas y el nodo de ejecución permitido, con tests de corrupción,
   fuente faltante, manifiesto falso y aislamiento. No desactivar la paridad.
6. Ejecutar la suite COMPLETA del nuevo hash en cada nodo que vaya a ejecutar
   trabajo científico. Tests en A1555 no prueban el entorno de A1557.

Hasta completar lo anterior, continuar en A1555. Si su GPU está ocupada, la
preparación espera; los diagnósticos CPU admitidos pueden avanzar. No migrar014
silenciosamente ni enviar su misma tarea a dos hosts.

Después de habilitar ambos, como máximo una tarea Biohub por nodo bajo su lock.
Se permite paralelizar dos tareas independientes con recursos admitidos y IDs
globalmente distintos; no entrenar/evaluar dependencias antes de que sus inputs
estén congelados. Centralizar publicación/cuota Kaggle en un único controlador.

## 4. Invariantes que no se pueden negociar

- Nada de MLflow: no instalar, importar, configurar ni crear `mlruns/`.
- Python local exclusivo `/Users/matias/miniforge3/envs/AIEnv/bin/python`.
  La Mac solo controla, inspecciona artefactos y ejecuta tests estáticos/toy.
  Entrenamiento e inferencia real se ejecutan en las remotas autorizadas.
- Usar el executor y aliases seguros UCBV. No tocar procesos ajenos, proxies,
  sockets, servicios, claves, verificación de host ni redes. No limpiar locks
  para evitar una espera. Un problema de infraestructura no se arregla matando VLLM.
- `Biohub-viejo`, `Biohub-mejorado`, fuentes/recibos/checkpoints y releases
  históricas permanecen de solo lectura. No reactivar dispatchers antiguos.
- El champion de control conserva dual-seed, DeepCenter y hashes congelados.
  No sustituirlo por single-seed ni cambiar sus pesos/parámetros. Un modelo nuevo
  de una hipótesis futura tiene otro nombre/manifest; no reemplaza el control.
- GT solo en entrenamiento/calibración/evaluación explícitos. Inferencia pura
  no recibe GT, IDs de forks verdaderos ni grafos reparados por oracle010.
  Ausencia de anotación es UNKNOWN, no negativo ni falso positivo demostrado.
- No inventar detecciones/enlaces para subir la métrica o completar filas.
  Recuperación nueva exige evidencia medida en imagen y coherencia temporal.
- CSV completo: estrictamente `234000 < filas < 242000`, sin contar cabecera.
  No padding, duplicación o poda para cumplirlo. No aplicar esa cota a caches
  parciales de entrenamiento, como las3 referencias014.
- Mantener guardias de CPU/GPU/RAM/disco/tiempo y validez de linajes. No ampliar
  la lista de tareas CPU sin demostrar que no usan CUDA y añadir pruebas.
- Mantener gates numéricos de `configs/protocol.json`; no modificarlos mirando
  resultados. Confirmación y Kaggle no son optimizadores de hiperparámetros.
- Máximo interno: una submission por día UTC y por artefacto único. No gastos,
  créditos, servicios pagos, datos privados nuevos o ejecución fuera de estos
  nodos sin permiso del usuario. Verificar reglas vigentes antes de activos externos.

## 5. Bucle de trabajo que sustituye la revisión de Codex

Para cada ciclo, seguir esta secuencia; no saltar del diagnóstico a Kaggle:

1. **Reconciliar estado.** Leer recibos reales. Clasificar la última corrida como
   completada, fallida, diferida o mutación de resultado incierto. Recuperar todo.
2. **Interpretar evidencia.** Escribir qué cambió, qué no cambió, causas apoyadas
   y limitaciones. Separar score oficial, AUROC/AP, recall, cobertura y conteos.
3. **Elegir UNA hipótesis.** Priorizar el cuello de botella medido; predeclarar
   mecanismo, inputs, controles, cambios, máximo de variantes, recursos y gates.
   Usar `docs/NEXT_EXPERIMENT_TEMPLATE.md`; guardar ficha antes de calcular
   outputs que se usarán para elegir variantes. No retocar la ficha retrospectivamente.
4. **Implementar lo que falte.** Crear módulos/configs/tareas versionados y tests.
   No simular que existen comandos de campañas futuras ni adaptar un receipt
   diagnóstico para que lo acepte una tarea antigua de publicación.
5. **Congelar.** Identificar código por `code_manifest`/`code_hash`, dependencias,
   datos, política y modelos. Guardar versión inmutable. Al verificar una fuente
   vieja, usar su manifiesto archivado validado, no un manifiesto actual recortado.
   Probar que cada cambio permitido está listado y que el resto permanece idéntico.
6. **Verificar y ejecutar.** Tests locales/toy; luego full suite remota del hash
   actual, sin omisiones. Doctor/admisión, launch único, status y fetch completo.
7. **Decidir.** Si falla un requisito, esa candidata no se publica. Documentar
   rechazo e ir a la siguiente hipótesis válida de la hoja de ruta. Si supera
   los criterios de esa fase, avanzar a la siguiente fase compatible y probada.
8. **Persistir continuidad.** Actualizar STATUS y bitácora con último comando,
   IDs/hash/nodo, artefactos, conclusión y próximo comando exacto. Tras una
   interrupción, retomar por receipts; no empezar el proyecto desde cero.

Por ciclo: como máximo3 hipótesis científicas, y como máximo2 variantes
predeclaradas por hipótesis más su control. No grillas abiertas ni búsquedas
indefinidas. Cada ciclo termina en una revisión escrita de evidencia por
Antigravity. Para abrir otro debe haber un cuello de botella nuevo/no ensayado
o nueva información causal: no renombrar el mismo barrido rechazado.
No es un límite para inflar seis notebooks diarios: la cuota de publicación
sigue siendo una diaria y solo para candidatas que superen TODOS los gates.

Una GPU ocupada implica espera/replanificación permitida, no fracaso científico.
Una excepción de código implica fixture + corrección versionada, no cambio de
hipótesis. Una mejora nula válida implica rechazar la hipótesis, no depurar el
evaluador hasta obtener otro score. No prometer trabajo futuro si el entorno
de ejecución no queda realmente activo; dejar el siguiente paso persistido.

## 6. Decisión concreta después de014

Leer los38 reportes, no solo el resumen. Desglosar por prefijo y por muestras
históricas/adicionales. Las3 adicionales aportan5 forks anotados POTENCIALES.
Contar por separado madres de preferencias negativas, divisiones verdaderas,
comparaciones, vectores únicos y volúmenes. No sumar las25 repeticiones013.

| Evidencia014 | Siguiente hipótesis permitida | Condición para no seguir |
|---|---|---|
| Nuevos forks utilizables y preferencias confirmadas suficientes | A. Aprendizaje de ranking con supervisión deduplicada | Duplicados, una sola clase, eventos escasos o mala discriminación entre prefijos |
| Pares verdaderos excluidos por top‑K/radio | B. Propuestas más completas, limitadas y sin GT | No mejora cobertura nueva, excede recursos o solo expande desconocidos |
| Hay detecciones cercanas pero no correspondencia, o se pierde el prior al suavizar | C. Auditoría de identidad/procedencia y localización | Solo proximidad: no justifica reasignar por GT ni ensanchar tolerancias a ciegas |
| Extremos ausentes en el grafo final y evidencia cruda/imagen disponible | D. Recuperación conservadora apoyada en imagen | Sin máximo real, ambigüedad o incremento de ruido/coste |
| Datos de mitosis siguen insuficientes y la principal pérdida está en enlaces | E. Asociación temporal de continuaciones y tejido denso | No hay labels válidos o integración neuronal/temporal verificable |

Orden de elección: resolver primero problemas de integridad/procedencia
confirmados; luego A si tiene supervisión suficiente; si no, B/C/D según la
pérdida medida. Si no quedan eventos para A, pasar a E en lugar de repetir heads
mitóticos con otros umbrales. No ejecutar todas las ramas automáticamente.

### A. Ranking supervisado, no incremento artificial de activaciones

Reutilizar preferencias directas012 y las nuevas014, deduplicadas y con
proveniencia de cada fila. Evitar un tercer estrato que simplemente vuelva a
pesar las mismas comparaciones. Conservar la función de potencial conjunta y
un ajuste lineal regularizado por prefijo como primera hipótesis.

Antes de ajustar, informar dimensionalidad efectiva, filas/madres/volúmenes por
tipo, positivos/negativos directos y forks GT utilizables. Como mínimo conservar
los controles científicos013:3 madres beneficiosas,10 perjudiciales y3 forks
utilizables en2 volúmenes por prefijo. Contar después de deduplicar; no bajar
mínimos. Son controles de degeneración, no garantía de suficiencia estadística.

Balancear por evento, no por número de alternativas de la misma madre. Una
receta fuente congelada, sin buscar épocas con el prefijo evaluado. Los scores
no se llaman probabilidades calibradas salvo demostración apropiada. Antes de
editar enlaces, comprobar ranking/AUROC/AP confirmed-only en ambos prefijos,
forks verdaderos frente a TODOS los competidores (incluidos unknown), precisión
y abstención. No promover una simple subida de logits como ocurrió en007.

### B. Cobertura del generador

Si el problema dominante medido es top‑6, un primer ensayo acotado puede comparar
K=8 y K=12 contra K=6, manteniendo radio12 µm y el resto de la receta. Predeclarar
ambas variantes antes del diagnóstico. No aumentar simultáneamente radio,
umbral del detector y smoothing. Si predominan pérdidas por radio, justificar
otra hipótesis de movimiento/incertidumbre física, no sumar cambios al ensayo K.

Enumerar candidatos exclusivamente con detecciones/imágenes/contexto. Nunca
añadir el par GT por conocerlo. No cambiar silenciosamente los límites de pares,
memoria o tiempo si una variante excede el presupuesto: rechazarla o diseñar
otro generador limitado en un ciclo nuevo. Medir cobertura de forks nuevos,
expansión de unknowns, coste y capacidad del ranker para ordenarlos. Cobertura
mayor por sí sola no autoriza aplicar ni publicar enlaces.

### C. Correspondencias y procedencia

Un punto cercano sin matching puede ser otra célula, competencia de asignación
o mala localización; no es una detección ausente demostrada. El matching oficial
es evaluación y permanece intacto. Para investigar pérdida de prior neuronal,
rastrear IDs/procedencia desde las detecciones crudas al postprocesado; no tomar
GT como mapa de transferencia ni dar probabilidad artificial a un prior ausente.
Una corrección de procedencia requiere fixture con ambigüedad, paridad del
control y comprobación de que no altera filas/enlaces sin evidencia científica.

### D. Recuperación de detecciones

No usar coordenadas GT ni interpolación geométrica aislada como nuevas células.
Primera hipótesis limitada: huecos de un frame con máximos medidos, contraste,
coherencia con parches vecinos y exclusión espacial. El módulo existente de
rescate ofrece controles iniciales de SNR>=4, NCC>=0.5 y máximo
min(200,0.005*N) propuestas por muestra. No bajar esos controles para llenar CSV.

La hipótesis debe integrar opcionalmente el nodo y sus dos enlaces, rechazar
ambigüedad y respetar colisiones/linajes. El camino actual de rescue pasa por
HOCT: no asumir que es ejecutable si esa dependencia sigue sin verificar.
Una integración alternativa sin HOCT exige módulo, tests y release propios.

### E. Continuaciones y contexto temporal

Si solo hay pocas mitosis supervisadas, investigar continuaciones, fragmentación
y cruces de identidad con evidencia temporal del mismo volumen. Empezar por un
modelo regularizado con rasgos medidos de movimiento, parches y prior neuronal
con procedencia válida; comparar transacciones completas, no solo una arista
atractiva ignorando el padre desplazado. Limitar la zona y el presupuesto de
cambios mediante una política sin GT, fijada antes de validación.

HOCT es una opción, NO un requisito ni un modelo listo. `configs/hoct.json`
tiene revisión/API/admisibilidad sin completar. Antes de usarlo, verificar
fuentes oficiales, revisión inmutable, licencia/reglas Kaggle, checkpoint, ABI
real, preprocesado y memoria. No instalar una dependencia mutable ni aceptar
features/máscaras ficticias. Si no puede verificarse, avanzar con una hipótesis
sin esa dependencia, no desbloquearla con `admissibility_reviewed=true` inventado.

## 7. Validación y controles contra sobreajuste

Usar la métrica oficial congelada y `estimated_number_of_nodes`, no el antiguo
proxy de100 frames ni cantidad de nodos anotados. La mejor validación pequeña
no garantiza el mejor Kaggle. No truncar/optimizar artificialmente la métrica.

Registrar exposiciones por muestra: FIT, calibración, desarrollo, confirmación,
visible e histórico. Detector full-fit y dos prefijos son limitaciones conocidas.
No llamar a confirmación un holdout virgen: ya existe historial de campañas.
No inventar independencia biológica por nombre de archivo.

Para cada candidata futura que sí aplique tracking:

1. Modelo y calibración solo en prefijo fuente; evaluar el otro. Incorporar las
   muestras expuestas014 SOLO en supervisión44b6, nunca como validación nueva.
2. Congelar todos los scores/predicciones de una fase antes de abrir sus GT.
   Selección de margen únicamente en calibración fuente con regla predeclarada.
3. Control y candidata en las mismas muestras, con el mismo detector y métrica.
   Registrar cambios reales de nodos/enlaces, no solo floats redondeados.
4. Desarrollo: delta score>=0.003; sin regresión de Jaccard de aristas/divisiones
   agregados ni score de cada prefijo; ninguna muestra pierde más de0.015 de
   Jaccard ajustado. Métrica requerida indefinida => candidata rechazada.
5. Elegir como máximo UNA receta por hipótesis para confirmación, por criterio
   predeclarado en desarrollo. Congelar pesos/config antes. Confirmación exige
   delta score>=0.002 y los mismos controles de regresión. No probar la segunda
   variante en confirmación como reacción al fallo de la primera.
6. Si falla confirmación, rechazar esa candidata; no ajustar con sus errores
   por muestra ni moverla a entrenamiento conservando el nombre de holdout.
   Registrar cada consulta y su exposición. La siguiente hipótesis debe surgir
   de entrenamiento/desarrollo, no de un barrido sobre confirmación.

La tendencia por prefijo, errores y cantidad de eventos acompañan al promedio.
Una muestra escasa o unknowns masivos impiden afirmar precisión poblacional.
No descartar el gate porque la referencia histórica0.933 tuviera score local
distinto: se compara contra el control de la misma ejecución, no contra números
de benchmarks incompatibles. La incertidumbre de transferencia sigue explícita.

## 8. Integración real con notebook y publicación

ADVERTENCIA:013/014 solo son diagnósticos/preparación. El `experiment` y
`package` actuales usan los brazos antiguos y el head antiguo; no consumen
automáticamente el nuevo ranking o `new-supervision-44b6.npz`. Antes de publicar
una futura mejora, implementar y probar su integración end-to-end:

- Un único módulo de inferencia para remota y Kaggle; el notebook solo carga el
  bundle y lo llama. Sin lógica científica duplicada ni CSV privado precalculado.
- En pruebas de ablación, demostrar que el nuevo camino realmente carga los
  pesos/config correctos y cambia exactamente lo esperado. Probar también el
  control/no-op, checkpoint faltante, input inesperado y prohibición de leer GT.
- Extender, si hace falta, el workflow de experimentos y el builder con un schema
  explícito del nuevo modelo. No etiquetar un ranker como si fuera el viejo head.
  Los receipts de tests, evaluación, configuración, pesos y paquete deben estar
  enlazados; discrepancia => abortar, nunca adaptar metadatos a posteriori.
- Solo después de los gates, entrenar modelo final con receta ya congelada.
  El posterior chequeo visible mide contrato/tiempo y paridad, no generalización.
- Bundle offline con dependencias/revisiones/hashes reales. Nuevas dependencias
  solo verificadas en destino aislado; no actualizar el conda compartido como
  arreglo rápido. No MLflow bajo ninguna circunstancia.
- Mantener paridad byte a byte de la referencia V19 sobre sus inputs archivados,
  y paridad de la CANDIDATA entre ejecución remota y GPU Kaggle visible. El hash
  del control y el de una candidata mejorada son cosas distintas. Si difieren
  remota/Kaggle, investigar implementación/numerismo; no relajar la comparación.
- Procesar TODOS los test montados por Kaggle; no fijar los cuatro nombres
  visibles. Linajes válidos, CSV dentro de cota y presupuesto end-to-end de7200s
  interno (no una afirmación del timeout oficial privado), con tiempo medido.

Publicar exclusivamente mediante `ops/kaggle.py`, después de `publish-ready`
real y smoke validado. Leer su ayuda y el receipt del paquete antes de comandos.
El flujo existente es upload → push → status → download/smoke → submit → score;
cada mutación una vez, identidad exacta de run/kernel/version/bundle/ref.
El kernel debe ser privado y nuevo para ese run; no editar el histórico.
No inferir que la última submission de otra tarea es la propia.

Antes de datos/modelos externos y de publicar, comprobar reglas vigentes y
disponibilidad de la competencia con fuentes oficiales. No eludir restricciones.
Si la cuota interna diaria está agotada, esperar; no crear otras cuentas o
procesos. Una escritura incierta se reconcilia en modo lectura: nunca borrar
`*_attempted`, receipts o locks para repetir upload/push/submit.

Después de score COMPLETE: registrar referencia exacta, score, artefacto y
receta. Si no mejora, conservar el champion y volver al diagnóstico: el score
oculto no señala por sí solo qué umbral cambiar. Si mejora pero es<0.970,
preservar la nueva campeona verificada y continuar con una hipótesis ortogonal.
No convertir el leaderboard en un barrido diario de parámetros.

## 9. Cuándo continuar y cuándo pedir intervención al usuario

Continuar sin pedir una revisión de Codex ante resultados normales, gates
rechazados, necesidad de implementar el siguiente módulo o bugs reproducibles
con solución dentro del alcance. Cada modificación crea otra release/test/run.

Pedir al usuario únicamente si faltan credenciales, permisos, datos inaccesibles,
licencia/admisibilidad no resoluble, recursos/gastos adicionales, reparación de
servicios ajenos, modificación de una invariante, o si se agotó la hoja de ruta
sin evidencia que justifique otro experimento distinto. Entregar el bloqueo y
la mínima decisión necesaria, no volver a ejecutar lo mismo para aparentar avance.

Si ambas remotas están ocupadas, esperar y avanzar solo con trabajo seguro que
sí pueda hacerse; no reportarlo como fracaso algorítmico ni usar otro nodo.
Si existe un run activo, consultarlo antes de iniciar otro con la misma finalidad.

Entrega breve tras cada campaña: hash/node/run, tests, fuentes, cambio propuesto,
delta pareado por fase/prefijo, errores, decisión y próximo paso. Actualizar
`docs/STATUS.md` y una ficha/bitácora persistente. El usuario puede decir
“continuá” y la tarea retoma desde ese estado, sin necesitar conversación con Codex.

Éxito solo con evidencia real de score>=0.970 en la referencia exacta COMPLETE.
Si no se alcanza, decirlo con claridad. Este protocolo organiza una búsqueda
rigurosa; no garantiza que los datos, modelos y recursos alcancen la meta.
