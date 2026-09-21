# Ensayo 007: recuperar señal antes de reasignar enlaces

**Cerrado el 2026-09-08:** tests007 73/73 y learn007 completed. El balanceo empeoró
AUROC entre grupos y produjo 15/90 FP a 0.80. Checkpoints rechazados para tracking.
El resto del documento conserva la receta histórica, no ordena repetirla.
Siguiente instrucción activa: `docs/REPRESENTATION_MITOSIS_008.md`.

## Qué demostró diagnose-006

Codex verificó el receipt y SHA del diagnóstico recuperado:
`83f418ad9e48be2e16fd28fd6c1aa29bc2036f017730ec4873bda47c41e7ab4c`.
`tests-006` y `diagnose-006` terminaron; el operador reportó 63 tests aprobados.
Confirmación no fue evaluada. No hubo candidata nueva ni submission.

Hay dos problemas distintos:

1. Los scores del head son bajos incluso sobre coordenadas GT. El máximo sobre
   las 15 divisiones GT fue 0.1485067, lejos del umbral fijado de 0.80.
2. Ningún fork GT exacto se encuentra entre las 19956 parejas realmente puntuadas.
   Por tanto, bajar el umbral del mismo scorer no recuperaría esos forks exactos.

De los 15 forks: cinco tienen segunda hija ocupada, dos una hija existente
incorrecta, uno una madre incompatible con el brazo aditivo, tres tienen algún
extremo sin matching, dos están presentes y dos no llegan al head por ausencia
de priors. En estos últimos faltan ambos priors y las distancias físicas están
dentro de 12 µm. Esto NO prueba por sí solo que la tolerancia 0.9 µm sea la única
causa: también pueden faltar aristas crudas o un matching no ambiguo.

"Extremo no emparejado" tampoco demuestra necesariamente una hija no detectada:
puede ser madre o hija, o un problema de localización/asignación del matching.
Estas categorías son de forks consecutivos exactos, no equivalen al conteo de
divisiones con ventanas de la métrica oficial.

Corrección de nombres en el reporte: los cuatro GT de las muestras 44b6 fueron
puntuados por el head ENTRENADO EN 6bba (SHA `3321db1f...`); los once GT de 6bba
por el entrenado en 44b6 (`0900030a...`). Se respeta el entrenamiento cruzado.
Los nuevos informes nombran explícitamente ambos grupos.

## Hipótesis priorizada

El predominio de minibatches sin positivos y el peso positivo muy alto, junto
con clipping global, puede perjudicar la optimización. No se afirma que sea una
causa única ya demostrada. La prueba cambia la forma de presentar las clases al
optimizador manteniendo exactamente la red, features y datos.

Cada batch tendrá 8 positivos y 8 negativos. Dentro de cada clase se recorren
ciclos aleatoriamente permutados; los positivos se reutilizan, sin generar nuevos
ejemplos ni etiquetas. El número de pasos por época sigue siendo `ceil(N/16)`:
148 para 44b6 y 317 para 6bba. Cada época visita una parte de los negativos y
repite positivos; no se afirma equivalencia muestra por muestra al entrenamiento
antiguo. Se registra exposición por clase y cantidad de ejemplos únicos vistos.

Se usa BCEWithLogitsLoss sin ponderación adicional: `pos_weight` multiplica el
término positivo, de modo que mantener 89.58/39.49 después de balancear los
batches introduciría otro desequilibrio deliberado. La elección 1.0 es parte
de este ensayo, no una calibración de probabilidades. [Definición oficial de PyTorch](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BCEWithLogitsLoss.html).

Se mantienen seed42, arquitectura MitosisHead, parches y once features originales,
AdamW lr0.0003, weight_decay0.01, clipping1.0, doce épocas y checkpoint final.
No se inicializa desde los heads fallidos ni se elige una época mirando desarrollo.
No hay grilla, early stopping, jitter nuevo ni cambio del umbral 0.80.

## Implementación y alcance

- Código nuevo: `core/mitosis_learning.py`.
- Receta y hashes de los 13 archivos fuente: `configs/mitosis_learning.json`.
- Tarea nueva: `ops/biohub.py --task learn-mitosis`.
- Todos los archivos de entrenamiento/inferencia históricos permanecen intactos;
  el trainer nuevo vive fuera de `core/nextgen/mitosis.py`.

Se reutilizan los tensores de `prepare-005`, abiertos con mmap de solo lectura.
Los hashes fueron calculados sobre los artefactos recuperados reales; no son
placeholders ni se recomputan para aceptar un archivo distinto. Se verifican
antes y después del ensayo, junto con receipt, plan, ejemplos y heads antiguos.
No se regeneran caches de imágenes, no se ejecuta el detector ni se copia encima
de un checkpoint anterior. Cada modelo nuevo se guarda dentro de `learn-007`.

La compatibilidad entre releases es específica de este ensayo: el código de la
red/producción/features/protocolo original debe permanecer idéntico. Solo los
archivos nuevos del ensayo y los drivers revisados están permitidos. `prepare`,
`experiment` y `package` mantienen sus controles de identidad y tarea fuente.
Un receipt `learn-mitosis` no permite pasar a ninguna de esas tareas.

## Qué se medirá, sin confundirlo con tracking

Ambos heads terminan sus doce épocas ANTES de abrir la comparación de desarrollo.
Se compara head histórico y balanceado sobre los mismos tensores anotados:

| Grupo de datos evaluado | Head entrenado en | Ejemplos | Positivos | Negativos confirmados |
|---|---|---:|---:|---:|
| 44b6, solo desarrollo | 6bba | 314 | 4 | 310 |
| 6bba, solo desarrollo | 44b6 | 574 | 11 | 563 |

Se guarda por grupo/muestra: AUROC, BCE balanceada de evaluación, BCE ordinaria,
scores positivos/negativos, TP/FP/FN/TN, recall y tasa de falsos positivos a 0.80.
Los logits por ejemplo quedan disponibles para auditoría, no para que el operador
elija un umbral. La BCE balanceada de evaluación usa el mismo cálculo para ambos
checkpoints; no se comparan directamente pérdidas de entrenamiento con pesos
diferentes. AUROC usa ranking de logits para evitar empates por saturación numérica.

Los logs de entrenamiento registran batches, normas de gradiente antes del
clipping, fracción de pasos recortados y exposición de clases. La evaluación final
sobre el propio entrenamiento se etiqueta como tal, nunca como validación.

No hay evaluación del panel de confirmación, score de tracking ni CSV candidato.
El scope usa coordenadas GT de desarrollo: reconocerlas no demuestra reconocer
parejas de detecciones ni resolver el grafo. Solo hay 4/11 positivos por grupo y
los negativos se muestrearon al construir ejemplos; sus métricas no son precisión
de despliegue ni validación integral independiente del detector full-fit.

Resultado único permitido: `learning_assay_only_no_submission`. Incluso si mejora
el AUROC o el recall, el operador se detiene y devuelve evidencia al diseñador.

## Qué haría después según la evidencia

Si la separación positivo/negativo mejora de forma consistente entre grupos,
el siguiente diseño deberá corregir la topología: propuestas de forks con
evidencia de imagen y reasignación conjunta de sus enlaces, no solo añadir una
hija libre. Debe comparar el beneficio de una división con el coste de retirar
asignaciones previas y evaluar los efectos sobre sus vecinos, bajo restricciones
de un padre, hasta dos hijas y pares mitóticos explícitos. Los nodos se conservarían.

Antes de ese solver hay que recuperar cobertura de candidatos: auditar IDs
observados antes del suavizado para transferir priors sin ampliar ciegamente el
radio, y no hacer que un prior inexistente se convierta en probabilidad inventada.
HOCT sigue siendo una alternativa pendiente de revisión de procedencia/ABI/reglas,
no se instala ni se lanza como efecto secundario de este ensayo.

Si los scores solo suben pero también crecen falsos positivos, o la separación
no generaliza al grupo contrario, este head no autoriza reasignar enlaces.
Corresponde revisar representación/señal de imagen y generalización antes de
confiar en sus decisiones. No bajar 0.80 ni elegir variantes ad hoc.

Estos pasos posteriores requieren diseño y pruebas nuevos. NO están implementados
ni autorizados para ejecución en esta entrega. El ensayo 007 no promete 0.970:
atiende un requisito de aprendizaje y deja visible el problema topológico pendiente.

## Comandos exactos para Antigravity

Release nueva:
`32edb4bd6b2cfcdf78b8e0c86bac9bf4a37d74e787d4e9a12adfa20678161ddc`.
65 pruebas locales aprobadas; suite completa de 73 pendiente en A1688. Una prueba
nueva de optimizer/serialización con datos de juguete se ejecuta solo en esa suite
remota. Codex no entrenó modelos ni ejecutó trabajos remotos en esta revisión.

Leer `AGENTS.md` y este documento. Conservar los runs anteriores. No relanzar
prepare/diagnose, no modificar pesos/parámetros ni ejecutar Kaggle. Usar AIEnv
en la Mac y A1688 mediante el wrapper UCBV. No instalar MLflow ni dependencias
alternativas. GPU ocupada: esperar, sin detener cargas ajenas.

```bash
cd /Users/matias/Proyects/Biohub
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/verify.py
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py doctor --node A1688
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1688 --task tests --run-id b0970-tests-007
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1688 --run-id b0970-tests-007
```

Esperar `completed`, `tests_passed=true`, 73 tests sin omisiones y code hash exacto.
Solo entonces:

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1688 --task learn-mitosis --source-run b0970-prepare-005 --tests-run b0970-tests-007 --run-id b0970-learn-007
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1688 --run-id b0970-learn-007
```

Esperar `completed`, `learning_assay_only_no_submission` y `source_unchanged=true`.
`confirmation_evaluated` y `tracking_evaluated` deben ser false. Recuperar:

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py fetch --node A1688 --run-id b0970-tests-007
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py fetch --node A1688 --run-id b0970-learn-007
```

Si un run ID o destino local ya existe, consultar antes; nunca sobrescribir o
duplicar el trabajo por falta de salida SSH. Entregar receipts, `learning.json`,
`models.json`, los dos `*.scores.json`, ambos `train-*/training.json`, checkpoints
y log. Incluir tabla histórico/balanceado por grupo con AUROC, recall, FPR y
scores de positivos/negativos. Ante fallo, conservar y devolver traceback; no
modificar hashes para aceptar archivos diferentes ni improvisar otra receta.
