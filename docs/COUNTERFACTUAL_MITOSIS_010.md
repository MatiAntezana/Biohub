# 010 — efecto de reparar forks, medido con la métrica oficial

Fecha: 2026-09-09. Diseñador: Codex. Operador: Antigravity.
Estado: completado como010-CPU; histórico, no volver a ejecutar. No es un nuevo modelo.
El punto activo es `docs/SELECTOR_MITOSIS_011.md`. Resultados comprobados en
`docs/STATUS.md`: nueve reparaciones con GT0.937095, diez0.941862; no son scores
de un modelo desplegable ni de Kaggle.

**Corrección operativa activa: `docs/CPU_WORKER_010.md`.** El operador informó
GPU ocupada antes de lanzar010. Solo cambió la asignación de recursos de tests y
evaluación; la receta científica de este documento está intacta. Usar los IDs
`b0970-tests-010-cpu` y `b0970-counterfactual-010-cpu`, con121 tests locales y135
remotos completos. La sección5 siguiente es el protocolo anterior, no una orden
de esperar a VLLM ni lanzar los IDs sin sufijo.

## 1. Qué cambió en nuestra interpretación de 009

Se verificaron receipt y `proposals.json`, SHA
`6909d585e718e98323ed19af7d23216031150de4a70e7413ad3fd4b30e3d9e8f`.
Tests009: 106/106 según el operador. La release fue
`c6ca21f53e51615126b99db8f351226a3db3d4cee5c26023bba64ea3a62baf43`.

- 11/15 forks cubiertos, frente a cero en el universo del scorer anterior.
  Pero **dos ya están en el grafo de referencia**: hay nueve oportunidades nuevas.
- Cuatro GT unique-top1 incluyen esos dos ya presentes: solo dos oportunidades
  nuevas son unique-top1. El mejor score GT, 0.73397, corresponde a uno ya presente.
- 1 645 258 propuestas, 764 etiquetas conocidas: **0.04644%** del universo.
  Los 2903 desconocidos sobre 0.80 no son FP confirmados, ni están certificados.
- AUROC confirmed-only 0.7908/0.8340; **AP 0.0741/0.0938**. El ranking no permite
  aprobar un reasignador. La prevalencia ~1–2% corresponde al subconjunto conocido,
  no a la población completa. No atribuimos toda la caída de AP a la prevalencia.

El modelo geométrico008 fue ajustado en coordenadas GT. En009 recibe coordenadas
predichas, suavizadas y redondeadas. Puede haber un desajuste de representación;
distancias residuales de matching no prueban por sí solas su causa ni identidad
biológica correcta. La auditoría010 medirá esas distancias con matching nuevo.

## 2. Pregunta y alcance

¿Corregir los forks identificables sobre las detecciones existentes aporta una
ganancia real en el evaluador oficial, después de retirar aristas incompatibles?

La cobertura009 usa forks exactos bajo matching completo. El evaluador archivado
usa ventanas de división con contexto parental y descendiente. **No son métricas
equivalentes**: agregar el par correcto no asegura recuperar una división oficial.

010 es un **contrafactual con GT**, exclusivamente dentro de evaluación pública:
usa las anotaciones para elegir reparaciones y medir consecuencias en copias en
memoria. No es inferencia, no estima precisión de un selector real y no demuestra
que pueda encontrarse ese conjunto sin GT. Tampoco es un máximo matemático:
no busca todos los subconjuntos ni todos los cambios posibles del grafo.

No entrena, no infiere imágenes, no mueve/agrega/elimina nodos, no vuelve a puntuar
propuestas, no cambia radio/K/umbral. No toca confirmación ni visible. **No exporta
grafos contrafactuales, CSV, GEFF, NPZ de linajes, checkpoints ni paquetes**.
Se exportan solo métricas, conteos y auditorías JSON, marcados como no publicables.

## 3. Tres intervenciones congeladas, más efectos individuales

Siempre partir de la referencia intacta005; nunca encadenar escenarios.

| Escenario de evaluación con GT | Forks nuevos esperados | Pregunta |
|---|---:|---|
| `covered_nonpresent` | 9 | Efecto de reparar todos los GT cubiertos que faltan |
| `covered_unique_top1_nonpresent` | 2 | Efecto del subconjunto GT unique-top1; **no** aplica los demás top1 desconocidos |
| `matched_nonpresent` | 10 | Añade el GT con extremos emparejados fuera del radio009; separa limitación de cobertura |

Además, evaluar cada uno de los diez forks del último conjunto **por separado**.
No sumar esos deltas como si fueran el efecto conjunto: la métrica no es lineal.
El escenario de diez no amplía radios en producción ni inserta las tres detecciones
GT sin matching. Los dos forks ya presentes no se contabilizan como recuperaciones.

Cada intervención pone exactamente las dos hijas requeridas para las madres
seleccionadas. Quita solamente otros enlaces salientes de esas madres y padres
incompatibles de esas hijas; mantiene todo el resto del grafo. Rechaza solicitudes
contradictorias, sin elegir un ganador ni podar adicionalmente para pasar validación.

## 4. Controles

1. Requiere tests de la release actual completados, sin skips.
2. Reutilización exclusivamente diagnóstica: ciencia009 intacta salvo drivers y
   tres archivos010 nuevos. Config010 fija hashes reales de los artefactos009 y
   resultado005; verificadores anteriores conservan toda la cadena005–009.
3. Verifica los ocho NPZ de predicción009 contra el manifiesto ya congelado.
4. Antes de cualquier intervención, reproduce los conteos por muestra de la
   referencia005 exactamente y sus métricas float con tolerancia absoluta1e-10.
   Referencia agregada histórica: score0.9012742110283266, edgeJ0.8814367450731088,
   divisionJ0.13043478260869565, divisiones TP3/FP8/FN12.
5. Abre únicamente los ocho nombres de desarrollo. El campo `split=confirmation`
   del resultado005 era una etiqueta interna para exigir carga cross-prefix;
   **no** identifica el panel de confirmación.010 registra `split=development`.
6. Exige GT idéntico al fingerprint009 y matching completo idéntico; compara
   footprints. No inventa un nuevo matching tolerante cuando no coincide.
7. El evaluador se importa del vendor congelado; cada evaluación recibe grafos
   nuevos porque el matching modifica atributos internamente.
8. Conserva los nodos, cardinalidad y referencia en disco/memoria. Revalida hashes
   de fuentes y GT al terminar. Ningún receipt010 habilita experiment/package.

Las flags cambian honestamente respecto de009:
`uses_gt_for_edits=true`, `tracking_evaluated=true`,
`counterfactual_graphs_modified_in_memory=true`; en cambio,
`source_graph_modified=false`, `graph_exported=false`,
`deployable_policy_evaluated=false`, `confirmation_evaluated=false`,
`promoted=false`, `kaggle_submission_allowed=false`.
No resumir010 como «todas las flags negativas».

## 5. Comandos originales — sustituidos por CPU_WORKER_010.md

Leer primero AGENTS.md, este documento y las normas UCBV. No modificar servicios,
aliases, entorno ni host-key checking. A1688 sigue siendo la excepción explícita
del usuario a la distribución general de GPUs. Conservar las corridas001–009.

```bash
cd /Users/matias/Proyects/Biohub
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/verify.py
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py doctor --node A1688
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1688 --task tests --run-id b0970-tests-010
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1688 --run-id b0970-tests-010
```

La verificación local tiene106 pruebas CPU; **no confundir con las106 de la suite
remota009**. La suite completa010 tiene119 pruebas e incluye paridad del adaptador
con el evaluador de producción y recuperación de una división en fixture sintética.
Solo después de `completed`, `tests_passed=true`,119/119 sin skips:

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --node A1688 --task audit-mitosis --source-run b0970-prepare-005 --tests-run b0970-tests-010 --run-id b0970-counterfactual-010
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --node A1688 --run-id b0970-counterfactual-010
```

Histórico: consultar estado cada60–120s, sin duplicar launches. Si ya existe un ID, consultar
su estado; no borrar ni sobrescribir. La tarea es de evaluación CPU, pero conserva
el worker remoto y lock exclusivo existentes. La corrección CPU activa elimina
solo la exigencia de GPU libre para tests/evaluación, con límites nuevos de CPU/RAM.
No iniciar trabajos Biohub paralelos.
`arm=mitosis_add` puede aparecer en el receipt como metadato CLI predeterminado;
no significa que se haya ejecutado ese experimento.

Al finalizar (o fallar), recuperar cada destino nuevo una sola vez:

```bash
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py fetch --node A1688 --run-id b0970-tests-010
/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py fetch --node A1688 --run-id b0970-counterfactual-010
```

## 6. Entrega y siguiente decisión exclusiva de Codex

Devolver receipts, code hash, logs, `baseline-parity.json`, `counterfactual.json`,
ocho `*.baseline.json` y ocho `*.counterfactual.json`. Informar por escenario/grupo
score y deltas, TP/FP/FN de aristas/divisiones y número de ediciones; por fork,
delta de panel, rango009, residuales y padres desplazados. Mantener el signo de
los deltas; no omitir reparaciones perjudiciales ni seleccionar solo las exitosas.

- Si las nueve reparaciones aportan señal sin daño relevante de enlaces, diseñar
  luego un selector **entrenado en detecciones predichas**, con evidencia específica
  de ambas hijas y continuidad temporal, y competencia explícita entre padres.
- Si los forks exactos no recuperan divisiones oficiales, localizar en las
  auditorías el contexto temporal/matching que falta antes de entrenar otro head.
- Si ni reparaciones perfectas de este conjunto ofrecen margen útil, relegar este
  brazo y revisar detecciones/asociaciones más amplias. Esto no descarta toda mitosis.

Son criterios de interpretación, **no órdenes automáticas de nuevos experimentos**.
Sea cual sea el score contrafactual, detenerse y devolver evidencia. Incluso un
score local>=0.970 en este ensayo **no es una solución ni mejora medida de Kaggle**.
El objetivo0.970 sigue sin demostrarse.
