# Ensayo 008: separar señal geométrica, imagen y capacidad del clasificador

**Cerrado el 2026-09-08:** tests00889/89 y probe008 completed. Geometría mejoró
AUROC/AP frente al histórico en ambos grupos; combinado no mejoró consistentemente.
No se midió tracking. El resto conserva la receta histórica, no ordena repetirla.
Instrucción activa: `docs/PROPOSALS_MITOSIS_009.md`.

## Decisión después de learn-007

`tests-007` y `learn-007` están completed. El operador reportó 73/73 pruebas.
Codex verificó los checkpoints recuperados y `learning.json`, SHA:
`a9f7924a3b5082950cb7939e1d53d76fc8a005a2b81e79a6c4c4cb43cd93d052`.
Los heads balanceados quedan rechazados para tracking y reasignación de enlaces.

| Datos de desarrollo | Head entrenado en | AUROC histórico → balanceado | FP a 0.80 histórico → balanceado |
|---|---|---:|---:|
| 44b6 (4 positivos / 310 negativos) | 6bba | 0.9444 → 0.8081 | 0 → 15 |
| 6bba (11 positivos / 563 negativos) | 44b6 | 0.7954 → 0.6987 | 0 → 90 |

El ajuste sobre su propio entrenamiento fue casi perfecto, sin transferirse al
otro grupo. Los scores históricos eran finitos y su ranking era mejor: estar
debajo de 0.80 NO demuestra un fallo numérico. El balanceo cambió la escala de
scores y también el modelo; no demostró mejorar la identificación de mitosis.
No corresponde rescatar `007` bajando un umbral ni hacer otra grilla de épocas.

También hay una limitación de representación comprobable en
`core/nextgen/mitosis.py:example_tensors`: para una misma madre, las cuatro
imágenes de entrada son las mismas independientemente del par de hijas. Solo
las siete variables geométricas cambian. La rama de imagen puede reconocer el
contexto de una madre, pero sola no puede elegir entre esos pares de hijas.
La prueba de imagen sola se interpreta con esta limitación, no como un detector
capaz de certificar una tripleta exacta.

## Hipótesis y prueba predeclarada

Evaluar si un clasificador de capacidad limitada conserva señal transferible,
sin volver a ajustar el encoder de imagen con apenas 26/125 positivos. Se usa
exclusivamente el encoder HISTÓRICO de `prepare-005`, nunca los pesos balanceados
de `learn-007`. El encoder queda en eval, sin gradientes ni actualización alguna.

Se ajustan exactamente estas tres variantes en cada grupo:

| Variante | Entradas | Parámetros ajustados, incluido intercepto |
|---|---|---:|
| `geometry` | 7 variables geométricas + 4 flags temporales originales | 12 |
| `frozen_image` | 64 features del encoder histórico congelado | 65 |
| `geometry_frozen_image` | Ambas anteriores, concatenadas | 76 |

Es una ablación de representación con lectura lineal regularizada; no una
demostración causal aislada de que toda la caída de `007` vino de la capacidad.
La dimensión de las entradas y el objetivo también difieren del head completo.

Receta única, `configs/mitosis_probe.json`:

- Regresión logística full-batch, inicio en cero y `lambda=1.0` fija.
- Objetivo: `0.5*mean(BCE positivos) + 0.5*mean(BCE negativos)
  + 0.5*lambda*sum(w**2)`. El intercepto no se penaliza.
- No hay minibatches, sobremuestreo ni un segundo `pos_weight`.
- Media/desviación ajustadas SOLO sobre el grupo de entrenamiento del modelo.
  Las columnas con desviación menor de `1e-6` se anulan también en evaluación;
  no se estima una normalización nueva mirando el grupo contrario.
- SciPy L-BFGS-B con gradiente analítico: maxiter500, ftol1e-12, gtol1e-7.
  Se exige `success=true` y norma infinito del gradiente <=1e-5; no converger
  produce un fallo, no un checkpoint aceptado. Las condiciones de terminación
  siguen la [documentación de SciPy](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html).
- Sin búsqueda de hiperparámetros, selección de modelo, calibración ni cambio de
  umbral. La extracción usa batches16; no se instala ninguna dependencia nueva.

Se completan los seis ajustes antes de puntuar desarrollo. Para datos 44b6, tanto
encoder como clasificador se entrenaron en 6bba, y viceversa. El source incluye
los ejemplos de entrenamiento por prefijo originales; no se redefine ese split.
La evaluación cruzada selecciona solo las ocho muestras de desarrollo fijadas,
314/574 ejemplos respectivamente. No se evalúan los paneles de confirmación,
visible ni la métrica de tracking. No se ejecuta el detector.

## Métricas y lectura del resultado

Se compara cada variante contra el head histórico completo en los mismos
ejemplos. Se guardan logits, AUROC, average precision (AP), BCE balanceada,
prevalencia, TP/FP/FN/TN, recall/FPR/precisión y cuantiles por grupo y muestra.
AP se calcula sin interpolación y agrupando empates, según su
[definición de referencia](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
Es un resumen de ranking, no una búsqueda o selección de umbral de operación.

0.80 se mantiene únicamente como punto descriptivo comparable con `007`. Un
modelo fuertemente regularizado puede no superar 0.80 y conservar buen ranking;
esa ausencia, por sí sola, no se clasifica como fracaso de aprendizaje. Tampoco
se confunden scores del entrenamiento balanceado con probabilidades calibradas.

El informe conserva TODAS las variantes, sin elegir un ganador automáticamente.
El diseñador revisará AP y AUROC conjuntamente en ambos grupos y los contrastes
combinado vs geometría e imagen, no solo la mejor media:

1. Si combinado mejora el ranking respecto del histórico y geometría en ambos
   grupos, habrá evidencia para investigar una lectura regularizada del encoder.
   Aún faltará comprobar propuestas reales, calibración sin fuga y topología.
2. Si imagen no aporta frente a geometría, la siguiente arquitectura deberá
   condicionar explícitamente la imagen en madre y ambas hijas. No se concluye
   que las imágenes carezcan de señal: esta prueba es lineal y su parche actual
   no identifica por separado las hijas.
3. Si ninguna variante mejora consistentemente, se cierra esta familia de
   reajustes sobre los mismos tensores. No se hace `008b` con otra lambda: se
   requiere otro diseño de representación/datos, no otro umbral en Kaggle.

Con 4/11 positivos, negativos seleccionados y detector full-fit, esto no es un
holdout integral ni una estimación de precisión de despliegue. El AUROC/AP aquí
no es el score Kaggle. Sigue pendiente el bloqueo de `006`: entre las propuestas
realmente puntuadas no había forks GT exactos. Ningún cambio de clasificador
arregla por sí solo los extremos sin matching, priors ausentes e hijas ocupadas.
No hay garantía ni medición nueva de 0.970.

## Aislamiento e integridad

Implementación nueva: `core/mitosis_probe.py`, configuración y tests propios.
Solo cambian los dos drivers para exponer `probe-mitosis` y verificar la suite.
Código científico original, notebook, pesos, datos y artefactos previos intactos.
Los 13 archivos fuente `005` se verifican antes y después; tensores mmap solo
lectura. Se fija además el hash agregado de TODOS los archivos científicos de
la release007: no se permiten cambios ni incorporaciones no revisadas.

Los probes se guardan como JSON con esquema `mitosis_linear_probe_v1`, no como
un checkpoint MitosisHead válido. Su scope es exclusivamente
`representation_assay_only_no_submission`. La tarea exige tests de la release
actual y su receipt no sirve de fuente para prepare, experiment ni package.
No lee los checkpoints rechazados007 ni crea un CSV/notebook candidato.

## Comandos exactos para Antigravity

Release008:
`0d6db10113d4201f32c6ae6686a96c2ace3deadb3d73db3847a86423154c8de5`.
78 pruebas locales aprobadas. Suite completa de **89** pruebas, incluidas tres
pruebas nuevas de runtime/optimizador con datos de juguete, pendiente en A1688.
Codex no entrenó modelos, ejecutó inferencia real ni inició conexiones remotas.

Leer AGENTS.md y este documento; no relanzar007/prepare/diagnose. A1688 es la
excepción de nodo explícitamente autorizada por el usuario; mantener aliases y
reglas de `/Users/matias/.local/UCBV-Access/docs`. No modificar servicios, acceso,
paquetes, recetas, hashes ni cargas ajenas. No MLflow. GPU ocupada: esperar.

```bash
cd /Users/matias/Proyects/Biohub
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/verify.py
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py doctor --node A1688
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1688 --task tests --run-id b0970-tests-008
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1688 --run-id b0970-tests-008
```

Esperar completed, tests_passed=true, **89/89 sin omisiones** y hash exacto.
No reemplazar los tests con los receipts007. Solo entonces:

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1688 --task probe-mitosis --source-run b0970-prepare-005 --tests-run b0970-tests-008 --run-id b0970-probe-008
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1688 --run-id b0970-probe-008
```

Consultar estado con intervalos razonables (60–120 s). `launch` es asíncrono:
una conexión terminada no significa fallo del worker. No relanzar ni sobrescribir
un ID existente. Ante respuesta incierta, consultar el mismo ID. Ante fallo,
conservar receipt, traceback y log y devolver al diseñador, sin reparación ad hoc.

Final necesario: completed, disposition `representation_assay_only_no_submission`,
source_unchanged=true; confirmation_evaluated, tracking_evaluated,
threshold_selected y model_selected todos false. Recuperar una vez:

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py fetch --node A1688 --run-id b0970-tests-008
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py fetch --node A1688 --run-id b0970-probe-008
```

Entregar receipts/logs, `probe.json`, `models.json`, ambos `*.scores.json` y los
seis `train-*/*.json`. Tabla con las cuatro variantes (histórico más tres probes)
por grupo: AUROC, AP, BCE balanceada, FP/TP/FN/TN y métricas a 0.80; añadir métricas
propias de entrenamiento, convergencia y deltas. No convertir un null en cero.
Detenerse y devolver evidencia a Codex cualquiera sea el resultado. Ninguna
salida autoriza confirmación, rewiring, HOCT, package o Kaggle en esta campaña.
