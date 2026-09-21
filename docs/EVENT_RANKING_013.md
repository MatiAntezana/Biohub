# 013 — aprendizaje equilibrado por evento, antes de editar el tracking

Diseñador: Codex. Operador: Antigravity. Fecha: 2026-09-10.
Estado histórico: completed, insufficient_supervision_no_submission; no se entrenó.
247/247 pruebas remotas aprobadas. Release013:
`a693c90ed4a8c23e4207f2c9dfdc8fe74bba5b3a48505b7fd8651fbe163024d8`.
La auditoría posterior confirmó25/25 comparaciones duplicadas de012. No repetir
estos comandos. Campaña activa: `COVERAGE_SUPERVISION_014.md` y handoff actual.

## 1. Motivo del cambio y alcance

012 terminó correctamente en A1555, con202/202 pruebas remotas, pero no editó
ningún grafo en calibración ni desarrollo. Desarrollo siguió0.9012742110283266.
Su ranking.json tiene SHA256
`a8cce44372df645366d509e66ac87c32d49fc6bedbf2824889335cc4124b592d`.

Dos limitaciones del diseño012 quedaron verificadas:

- Las reparaciones beneficiosas eran3/518 y12/928. Tras la ronda difícil solo
  recibían0.4990% y1.3201% del peso. El modelo44b6 puntuaba negativamente sus
  tres reparaciones beneficiosas;6bba favorecía6/12. La precisión global98–99%
  no demostraba discriminación útil.
- Ninguna propuesta de calibración superó1.0: máximos0.978367 y0.692738. No
  basta bajar ese margen: primero debe mejorar el ordenamiento de alternativas.

013 es un ensayo de aprendizaje y diagnóstico, NO un experimento de tracking.
No materializa grafos reparados, CSV ni notebook; no llama apply_transactions.
No evalúa el score oficial de una nueva candidata. Una hipótesis individual
beneficiosa no garantiza mejorar la métrica mitótica contextual oficial.

El recall de nodos0.970535 de012 NO es el score Kaggle objetivo0.970.

## 2. Entradas inmutables, particiones y recursos

Reutilizar `b0970-ranking-012` y los53 tensores medidos de
`b0970-selector-corpus-011`, sin volver a prepararlos. Configuración nueva:
`configs/event_ranker.json`; pesos/modelos012, preferencias JSON/NPZ y recibos
fijados por hashes. Los datos y ciencia001–012 permanecen intactos.

Los modelos nuevos se ajustan en las mismas35 muestras:12 del prefijo44b6 y23
del6bba. Solo el prefijo fuente interviene en ajuste y calibración. Se reutilizan
las10 muestras de calibración011; las8 de desarrollo reciben el modelo del
otro prefijo. Se usan descriptores medidos y detecciones de un detector full-fit:
esto NO es un holdout integral ni demuestra independencia biológica por embrión.
Un prefijo es un proxy de dominio; una madre por volumen es una unidad de peso,
no una afirmación de independencia estadística entre todas las muestras.

Nodo único A1555, alias seguro a1555, conda train-55. Pasar --node A1555 siempre.
CPU limitada mediante el worker existente: CUDA oculta, hasta2 CPU lógicas,
nice>=10, pools acotados, RLIMIT_AS8GiB. Guardias CPU/RAM/disco y lock común
sin cambios. No nueva inferencia GPU, CNN, instalación, modelos externos ni HOCT.
No intervenir en VLLM, servicios, procesos ajenos, claves, proxies o sockets.
Todas las llamadas remotas pasan por ops/biohub.py y execute_remotely.

## 3. Supervisión y función aprendida

Se mantiene la función lineal conjunta de34 diferencias de potencial012:
factores de aristas, forks y contadores de TODOS los padres afectados. No se
introducen logits independientes ni penalizaciones manuales. Conservar tiene
potencial exactamente0; sin intercepto ni centrado. RMS de entrenamiento,
L2=1, un único ajuste L-BFGS-B por prefijo, máximo500 iteraciones, ftol1e−12,
gtol1e−7 y gradiente infinito<=1e−5. Sin segunda ronda ni búsqueda de variantes.

Tres estratos con masas predeclaradas:

| Estrato | Masa de la pérdida | Fuente |
|---|---:|---|
| Reparación beneficiosa frente a conservar | 25% | Preferencias confirmadas012 |
| Reparación perjudicial frente a conservar | 25% | Preferencias confirmadas012 |
| Fork correcto frente a alternativas confirmadas | 50% | GT de ajuste + hipótesis011 |

Dentro de cada estrato cada madre tiene igual peso total, repartido entre sus
comparaciones. Más alternativas de una madre no equivalen a más eventos.
Los eventos pueden aparecer en varios estratos: se informa el número de madres
únicas entre todos ellos, sin sumar duplicados como nuevas divisiones.

Para el tercer estrato se busca cada fork anotado cuyos tres extremos fueron
emparejados con detecciones y cuyo par existe en el universo congelado011.
Se compara su transacción completa con las alternativas de ESA madre y con
conservar, incluyendo forks que la referencia YA reconstruye correctamente.
También se considera retirar una rama solo donde el universo011 ya lo permite.
No se añaden nodos GT, pares fuera del radio/K ni escenarios oracle010.

La etiqueta se obtiene de la diferencia simétrica de aristas entre las dos
hipótesis: se cancelan primero los cambios compartidos. Se acepta la comparación
solo si TODA diferencia conocida favorece al fork correcto. Diferencias unknown
o mixtas se excluyen. Un cambio desconocido compartido que se cancela NO se está
declarando correcto. Ningún grafo contrafactual se exporta o aplica.

Guardias por prefijo: al menos3 madres de reparaciones beneficiosas,10 de
perjudiciales y3 forks con comparaciones utilizables en al menos2 volúmenes.
Esto limita degeneración, NO garantiza potencia estadística. Si falta evidencia:
`insufficient_supervision_no_submission`, sin fitting, calibración ni desarrollo.

Se registran pesos efectivos y métricas por estrato para013 y el modelo012
guardado. El012 se usa solo como comparador, nunca como inicialización. No se
presenta precisión de entrenamiento como generalización ni se cuentan las
comparaciones invertidas como nuevos eventos.

## 4. Predicción, calibración y evaluación diagnóstica

Ambos ajustes terminan antes de las predicciones de calibración. Se guardan TODAS
las alternativas con scores013 y012: positivas, negativas y forks intactos.
Las10 tablas y su manifiesto se congelan antes de abrir GT de calibración.
La función de predicción no admite GT, anotaciones ni selección por etiquetas.

Regla única de calibración, solo con prefijo fuente:

1. `trial_margin = max(0, máxima ganancia de alternativas perjudiciales confirmadas)`.
2. Selección diagnóstica: mejor alternativa positiva única por madre y ganancia
   estrictamente mayor al margen. Empate => abstención. No hay topN o cuota.
3. Se requiere al menos10 alternativas perjudiciales conocidas en calibración,
   al menos1 propuesta beneficiosa seleccionada, recall de madres con alternativa
   beneficiosa>=0.5 y ninguna propuesta perjudicial/mixta seleccionada.
4. Si falla, `diagnostic_margin=null`. No ampliar una grilla, reducir el margen
   manualmente ni relabelar unknown para forzar una decisión.

Cero perjuicios confirmados por esta construcción en calibración NO es una
garantía de precisión poblacional. Las muestras ya fueron observadas y muchas
propuestas siguen siendo desconocidas. Se informa su fracción seleccionada.
El margen solo sirve para el diagnóstico; no autoriza editar ningún enlace.

Modelos y decisiones se guardan antes de predecir desarrollo. Las8 tablas del
prefijo opuesto se congelan antes de abrir GT de desarrollo, incluso si hubo
abstención fuente. No se selecciona ningún parámetro con el desarrollo.

Informar por prefijo:

- AUROC y TP/FP/FN/TN sobre alternativas estrictamente confirmadas; excluye
  unknown, mixtas y no-ops. No equivale a precisión del pool completo.
- Ganancias, ranking de forks GT frente a TODAS las alternativas, incluidos
  unknown y conservar. Separar forks ya presentes de nuevas oportunidades.
- Recall de madres con reparación beneficiosa y tipo de propuestas best únicas
  a margen0, al margen fuente013 y al margen real archivado012 (null).
- Conteos de unknown seleccionados, abstenciones y eventos fuera del pool/no
  emparejados. No interpretar unknown como FP ni como acierto.

Gate de discriminación (no es el gate de tracking): no retroceso de AUROC
confirmed-only ni de nuevos forks unique-top1 en ningún prefijo; al menos un
nuevo unique-top1 adicional frente a012 y una propuesta beneficiosa al margen
fuente; ninguna perjudicial/mixta best seleccionada en desarrollo. Incluso si
pasa: `designer_review_only`, NO habilita tracking, confirmación o Kaggle.

## 5. Secuencia exacta para Antigravity

Desde `/Users/matias/Proyects/Biohub` en la Mac:

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/verify.py
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/audit_event_inputs.py
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py doctor --node A1555 --task tests
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1555 --task tests --run-id b0970-tests-013
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1555 --run-id b0970-tests-013
```

Exigir tests completed/tests_passed=true, suite full_discovery sin omisiones,
errores o fallos, del hash013 y en A1555. Comparar conteos exactos con el handoff.
La auditoría local exige53 evidence_npz y2 preference_npz intactos. No lee GT,
entrena o verifica la disponibilidad local de toda la cadena005; esa cadena
se comprueba en remota usando sus manifiestos históricos verificados, no recibos
reescritos con el hash actual. No ejecutar audit_selector_inputs.py012 bajo013.

Solo después de tests aprobados:

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py doctor --node A1555 --task learn-events
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1555 --task learn-events --source-run b0970-ranking-012 --tests-run b0970-tests-013 --run-id b0970-events-013
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1555 --run-id b0970-events-013
```

Consultar status cada60–120s, no relanzar ante silencio. deferred_resources con
run_created=false no consume ID: esperar sin tocar servicios. Un receipt creado
consume el ID; si ya existe, inspeccionarlo, no sobrescribir ni elegir otro ID.
Toda falla de datos, hashes, entorno o transporte se reporta; no improvisar fixes.
No usar python3 genérico, SSH directo o comandos ad-hoc para explorar archivos
remotos: status/fetch y el intérprete local fijado bastan para obtener el reporte.

Al finalizar o fallar, recuperar únicamente los IDs que realmente se lanzaron:

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py fetch --node A1555 --run-id b0970-tests-013
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py fetch --node A1555 --run-id b0970-events-013
```

Si una descarga queda parcial, --resume solo sobre la misma carpeta con receipt
terminal coincidente; no excluye NPZ ni sobrescribe archivos. Si falta una fuente
histórica, reportar su ruta exacta, sin regenerar005–012 ni cambiar de máquina.

## 6. Evidencia de salida y detención

Entregar receipts, code hash, tests.json, worker.log y auditorías CPU; training
JSON/NPZ con conteos por estrato, masas, eventos y GT hashes; ambos fits y modelos;
10 scores de calibración y8 de desarrollo, sus manifiestos,18 labels NPZ y
diagnostic JSON; calibración por prefijo; models.json y event-ranking.json/hash.

Las tablas scores.npz tienen `ids` (madre,hija1,hija2; hija2=-1 indica poda),
`scores` (columnas event013,ranker012) y `changed`. No son grafos predichos.
Labels separados:1 beneficiosa,−1 perjudicial,0 mixta,2 desconocida,3 no-op.
Un sentinel de comparación [madre,−1,−1] en training JSON significa conservar;
jamás un nodo sintético. No adivinar nombres de campos ausentes en los JSON.

Flags obligatorias: tracking_evaluated=false, graph_exported=false,
uses_gt_for_edits=false, confirmation_evaluated=false, visible_evaluated=false,
kaggle_submission_allowed=false, promoted=false, source_unchanged=true.
threshold_selection_attempted distingue intento de threshold_selected (algún
margen no-null). Falta de supervisión => model_trained=false.

DETENERSE incluso si pasa el gate. Sin notebook, confirmación, visible, package,
HOCT, API Kaggle ni nuevas variantes. Codex revisa antes de decidir un ensayo
de tracking. Ninguna métrica de013 demuestra haber alcanzado0.970 en Kaggle.
