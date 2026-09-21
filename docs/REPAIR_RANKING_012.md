# 012 — aprender a comparar reparar contra conservar

Diseñador: Codex. Operador: Antigravity. Fecha: 2026-09-10.
Estado histórico: ejecutado en A1555,202/202 pruebas, rechazado sin ediciones ni
mejora en calibración/desarrollo. No hay nuevo score Kaggle. Sus comandos ya NO
son órdenes activas; continuar únicamente con `docs/EVENT_RANKING_013.md`.
Release: `610edd946b5085b211db2c61c42da0ad3d8db9f414de88ac8ae7b882a9f60fdb`.

## 1. Decisión y evidencia011

011 terminó en A1555 con168/168 tests. Los hashes del código, corpus, informe,
dos modelos y diez propuestas de calibración fueron verificados localmente.
El hash de código es una identidad de contenido, NO una revisión Git.

En calibración, margen1 seleccionó59 transacciones en44b6 (delta score0) y32
en6bba (delta−0.0027311164). De esas91,90 añadían una rama, una retiraba una rama
y ninguna reasignaba una hija. En6bba_74686d6a: edgeTP+1, edgeFP+2, edgeFN−1,
divisionFP+3, divisionTP+0. Ninguna política desbordó el presupuesto. Margen2
cambió7/1 transacciones sin mejorar; márgenes4/8 no editaron. Ambos prefijos
seleccionaron abstención, desarrollo quedó0.9012742110 y no se abrió confirmación.

Los AUROC de enlaces0.9932/0.9923 eran de entrenamiento. Los clasificadores de
forks solo tenían5/29 positivos confirmados, no millones de divisiones supervisadas.
Los millones de propuestas unknown siguen sin ser negativos ni aciertos medidos.

Limitación del diseño011: entrenaba clasificadores de factores independientes y
sumaba sus logits con una penalización fija. No aprendía directamente a preferir
una transacción completa frente a conservar.012 prueba esa hipótesis; no asegura
alcanzar0.970 y sus deltas públicos no se suman al0.933 histórico de Kaggle.

## 2. Congelación y recursos

- Usar A1555 (`a1555`, `train-55`), confirmado por el usuario al continuar esta etapa.
  Siempre pasar `--node A1555`; no cambiar nodo ni usar el default histórico A1688.
  `configs/cluster.json` permanece intacto para preservar la procedencia005–011.
- Reutilizar `b0970-selector-corpus-011` y `b0970-selector-011`, NO regenerarlos.
  Los hashes concretos están fijados en `configs/repair_ranker.json`.
- Se conservan las detecciones, coordenadas, tiempos, dos seeds, V19, cuantización,
  radio12µm/K6, patches medidos y referencias de desarrollo005.
- No inferencia GPU, nueva CNN, descarga de modelos, instalación ni HOCT.
- `tests` y `rank-repairs` usan `cpu_evaluation_v1`: CUDA oculta, hasta2 CPU
  lógicas, nice>=10, pools acotados, RLIMIT_AS8GiB por proceso, lock común Biohub.
  Exigen RAM disponible>=max(16GiB,25% RAM), load1m<=0.75×CPU y disco>=12GiB.
- GPU ajena ocupada no impide esta tarea CPU si pasan las guardias; no se toca
  VLLM ni ningún servicio/proceso ajeno. No se promete impacto cero de CPU/IO.
- Todas las ejecuciones remotas pasan por `ops/biohub.py` y
  `core.remote_executor.execute_remotely`, con aliases UCBV/host-key checking.

## 3. Aprendizaje de una diferencia de potencial

Se conservan las alternativas del universo011: dos hijas, incluso ocupadas, y
retiro de una rama cuando la madre ya tiene dos. Conservar es siempre la referencia.
No se hace relinking general ni se introduce una hipótesis de borrar todo un linaje.

Cada transacción incluye TODOS los padres afectados. Se calcula
`Phi(grafo reparado) − Phi(referencia)` con34 variables:

- suma de13 descriptores de enlaces añadidos menos retirados;
- suma de18 descriptores de forks creados menos destruidos;
- cambios en cantidad de aristas, forks y padres afectados sin descendencia.

Los descriptores siguen usando los patches propios de cada célula, velocidades
previas/siguientes, contraste y geometría en unidades físicas. Incluyen factores
del padre desplazado y continuación de hijas. No hay encoder temporal nuevo ni
relectura de imágenes: usa evidencia011. El contexto permanece congelado en la
referencia; no se recalcula a partir de sus propias decisiones.

Se omite `reference_edge`: ser parte de la solución previa no es evidencia de
imagen. El coste de crear/retirar una arista, fork o huérfano se aprende junto con
el resto. No se usan los logits/weights011 ni su penalización fija de2.

### Preferencias con anotaciones parciales

Solo se leen GT de las35 muestras de ajuste ya fijadas en011. Se usa el matching
de7µm existente; la validez de estas etiquetas depende de ese matching público.

- Beneficial: cada arista añadida es correcta Y cada retirada es incorrecta.
- Harmful: cada añadida es incorrecta Y cada retirada es correcta.
- Unknown: alguna modificación no tiene sus extremos/padre anotados; se excluye.
- Mixed: mezcla mejoras y daños conocidos; se excluye, no se arbitra un ganador.

Es una preferencia estructural estricta, no una etiqueta del score oficial.
Una alternativa incorrecta frente a una referencia correcta también enseña una
preferencia. Su orientación se invierte en la pérdida, SIN inventar un nuevo
evento biológico ni añadir una célula GT al grafo. `beneficial` y `harmful` se
reportan separadamente; una preferencia invertida no cuenta como otro positivo.

El filtro de endpoints anotados solo reduce el trabajo de construir ejemplos de
TRAINING; la inferencia enumera todas las alternativas, incluidos unknowns.
No se leen IDs/ediciones del oracle010 para generar ejemplos ni tomar decisiones.

Guardia por prefijo: al menos10 preferencias,10 madres distintas identificadas
por muestra+fila, y3 forks verdaderos cubiertos por los candidatos de entrenamiento.
Esto NO garantiza independencia biológica ni abundancia de reparaciones positivas.
Si falla: `insufficient_supervision_no_submission`, sin ajustar ni abrir calibración.

### Ajuste fijo y alternativas difíciles

Una sola función lineal conjunta por prefijo. Potencial de conservar exactamente0:
sin intercepto, sin centrar diferencias. Escala RMS calculada solo en entrenamiento;
columnas sin señal quedan inactivas. Pérdida logística de preferencias
`softplus(−signo × diferencia_puntuada)` + L2 conλ1.

Dos rondas predeclaradas, ambas sobre el mismo entrenamiento:

1. Ajuste con igual peso total por madre para evitar que madres con muchos
   candidatos dominen el conjunto.
2. Seleccionar la preferencia peor satisfecha por madre usando el primer ajuste;
   reajustar con50% del peso original y50% de esas preferencias difíciles.

No se minan ejemplos de calibración/desarrollo ni unknowns. No buscar epochs,
regularización o rondas nuevas. L-BFGS-B, máximo500 iteraciones por ajuste,
ftol1e−12, gtol1e−7; exigir convergencia y gradiente infinito<=1e−5.
Los índices de la segunda ronda se refieren a `preferences-<prefijo>.npz`.
Guardar y recuperar ese tensor y su JSON/hash, exclusivamente como evidencia de
entrenamiento; nunca usarlos como entrada del predictor.

## 4. Calibración, inferencia y evaluación

Ambos modelos se ajustan antes de predecir calibración. Se congelan las diez
listas de propuestas antes de abrir las anotaciones de calibración.

Se prueba SOLO margen1.0 de la NUEVA puntuación, contra abstenerse. No es el
umbral probabilístico histórico0.80. La calibración fuente puede habilitarlo si
delta score>=0.001 y pasan las mismas guardias011 de enlaces, divisiones,
regresiones por muestra y presupuesto. Si no, `margin=null`.

Se reutilizan las10 muestras de calibración011, ya observadas: no son un holdout
nuevo. Los prefijos son proxies de dominio y el detector sigue full-fit. No se
afirma independencia biológica ni evaluación integral fuera de entrenamiento.

Se guarda el modelo con su decisión antes de predecir desarrollo. Para44b6 se
usa ajuste Y calibración6bba, y viceversa. Las ocho predicciones completas se
congelan antes de abrir GT de desarrollo. Primero se reproduce la referencia005
con conteos exactos/atol1e−10 y luego se aplica el evaluador oficial sin cambios.

Por madre se conserva la mejor alternativa de ganancia positiva; empate abstiene.
Se aplican solo transacciones sin escrituras solapadas, con ganancia>1.0 si la
política fue habilitada. El límite es el original0.2% de nodos en transacciones:
si lo excede se rechaza la política entera de esa muestra, no se recorta un topN.
Nodos y coordenadas no cambian. No hay cuotas mínimas de ediciones.

Desarrollo exige delta>=0.003, sin retroceso de edgeJ/divisionJ ni de prefijos;
se mantienen las tolerancias por muestra originales. Incluso un resultado
`designer_review_only` no habilita confirmación, visible, package o Kaggle.

## 5. Comandos exactos para Antigravity

### A. Verificar código y completar la descarga011

```bash
cd /Users/matias/Proyects/Biohub
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/verify.py
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py fetch --node A1555 --run-id b0970-selector-corpus-011 --resume
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py fetch --node A1555 --run-id b0970-selector-011 --resume
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/audit_selector_inputs.py
```

Exigir186/186 tests CPU locales. `--resume` es solo para los dos directorios
parciales existentes con receipt terminal coincidente. Agrega archivos ausentes
sin sobrescribir los presentes; no usar exclusiones ni rsync con proxy manual.
El CLI ya comprueba hashes de lo que existía antes de transferir. Una corrupción
existente NO se arregla borrando: informar a Codex. Una falla SSH/transporte se
reporta sin relanzar entrenamiento ni cambiar la red.

La auditoría exige53 NPZ de evidencia y8 NPZ de predicción, hashes fijados,
modelos y propuestas; también igualdad de arrays de la abstención011. No lee
imágenes/GT ni entrena. Su `historical005_chain_verified=false` es deliberado:
la disponibilidad de la cadena005 se verifica en la remota antes del ensayo.
El corpus `cache-training` también se trae con fetch completo; no es necesario
copiar las imágenes `.zarr` originales a la Mac. No borrar datos por falta de disco.

### B. Suite remota de ESTA release

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py doctor --node A1555 --task tests
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1555 --task tests --run-id b0970-tests-012
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1555 --run-id b0970-tests-012
```

Esperar `completed`,202/202, tests_passed=true, cero skips/errores/fallos, hash012
y aislamiento CPU. Suite descubierta localmente; aún NO ejecutada en la remota
por Codex. No reutilizar tests011 como aprobación de012.

### C. Ranking012 sobre el corpus existente

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py doctor --node A1555 --task rank-repairs
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1555 --task rank-repairs --source-run b0970-selector-corpus-011 --tests-run b0970-tests-012 --run-id b0970-ranking-012
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1555 --run-id b0970-ranking-012
```

El worker audita entradas011 y toda la cadena005–011 por hashes. No se falsifica
un receipt011 con el hash012 para permitir reutilización. Si faltan fuentes
históricas en A1555, devolver rutas faltantes: no repetir005 ni cambiar de nodo.

Consultar estado cada60–120s. `deferred_resources` con run_created=false no consume
ID: esperar recursos sin tocar servicios. Cualquier receipt creado consume el ID;
no relanzar ni sobrescribir un job incierto o fallido. No fijar ETA sin evidencia.

Al terminar o fallar, recuperar los IDs que realmente se hayan lanzado:

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py fetch --node A1555 --run-id b0970-tests-012
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py fetch --node A1555 --run-id b0970-ranking-012
```

## 6. Entrega y detención obligatoria

Devolver code hash, comandos, receipts/logs, tests.json y auditorías de CPU;
salida de audit_selector_inputs; preferences JSON/NPZ/hashes, conteos beneficial/
harmful/mixed/unknown por muestra, madres/forks supervisados; dos rondas y sus
preferencias difíciles; modelos, calibración completa y decisión1.0/null;
ocho predicciones/ediciones y las listas development-proposals con hashes
(se conservan incluso si la política se abstuvo); ranking.json con métricas/gates y deltas globales,
por prefijo y por muestra. Reportar cuántas transacciones añaden, podan o reasignan.

Métricas de ajuste no son generalización. Unknown no equivale a FP, ni a éxito.
Flags: confirmation_evaluated=false, visible_evaluated=false,
kaggle_submission_allowed=false, uses_gt_for_edits=false, source_unchanged=true.
tracking_evaluated/model_trained=false si faltó supervisión; true si el ensayo
completo terminó. No ocultar abstenciones, insuficiencia, errores ni regresiones.

DETENERSE incluso si pasa desarrollo o un score local supera0.970. Ningún receipt012
autoriza publicación. Codex revisa antes de abrir confirmación o construir una
candidata Kaggle. No modificar el notebook histórico, instalar MLflow ni otros
paquetes, bajar umbrales, abrir HOCT o lanzar otro ensayo por cuenta propia.
