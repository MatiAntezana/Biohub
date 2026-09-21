# Hipótesis y experimentos congelados

Documento del diseño histórico inicial; sus referencias a un “paso actual” no
son órdenes vigentes. Para el estado real y las siguientes decisiones, leer
STATUS.md, AGENTS.md y AUTONOMOUS_CONTINUATION.md. La continuidad posterior
a014 queda a cargo de Antigravity, no de otra revisión de Codex.

## Qué problema intentamos corregir

La evidencia archivada en `Biohub-mejorado/experiments/SCORE-0970/AUDIT_FINDINGS.md`
descarta interpretar los cambios de score local como una ganancia garantizada en
Kaggle. A01 mejoró el diagnóstico ampliado, pero obtuvo **0.930**. El calibrador
geométrico mejoró desarrollo y retrocedió **−0.0006208** en confirmación. No se
reintroducen esas variantes como resultados positivos.

En el panel de ocho muestras, el presupuesto de errores del matching final fue
5578 TP / 333 FP / 329 FN de aristas. Entre los FN: 129 involucraban un extremo
GT sin nodo predicho emparejado; 181 carecían de candidato neuronal entre los nodos
finalmente emparejados; 15 eran candidatos descartados y 4 se perdían después.
Es una atribución descriptiva, no prueba de causalidad aislada.

La nueva hipótesis es que contexto temporal aprendido y evidencia de imagen
pueden discriminar asociaciones ambiguas, divisiones reales y titileo mejor que
reglas geométricas globales. Para pasar de 0.933 a 0.970 se necesita una ganancia
real de 0.037; no existe evidencia que permita prometer ese salto con estos
parámetros. Las pruebas decidirán qué parte aporta señal.

## Base y restauración

La arquitectura efectiva auditada es la del paquete público de 32 canales,
ventana temporal 2 y downsample `(1,4,4)`, con los dos seeds identificados en
`configs/models.json`. No es el modelo de 48 canales del antiguo fast loop.
Pesos de mezcla: detección secundaria 0.80, aristas secundarias 0.15; TTA activo.

La referencia seleccionada es V19, ref Kaggle `56063258`, score histórico 0.933.
Su postprocesado se reconstruyó estáticamente desde el snapshot V20 y el script
original `prepare_v19_axial_psf.py`: escala axial efectiva /1.25 en sus tres
funciones de distancia/posición y configuración V19. La consistencia `cost.T`
de V20 queda desactivada. Procedencia y hashes: `legacy/provenance.json`.

`prepare` exige reproducir byte a byte el CSV V19 sobre los cuatro grafos crudos archivados:
`46b4008a8021f6882eaf4661ac814d70762f7d5190cbf86747e2630d33d5af37`.
Si no coincide, se detiene antes de entrenar. `prepare-003` ya logró esa paridad
con IndexedRXGraph; su CSV y hash fueron verificados localmente. Esto certifica
posprocesado sobre inputs congelados; no demuestra ganancia del nuevo head.
La reproducción GPU completa se comprueba después en Kaggle.

### Guardia estructural de colisiones al redondear

La nueva referencia es V19 con `observed_over_gap_synthetic_exact_voxel_v2`.
La procedencia se obtiene del conjunto de IDs de entrada y de los IDs nuevos
capturados alrededor de `close_single_frame_gaps` y `recover_strict_gap2`.
No se deduce de la magnitud del ID ni de atributos en nodos/aristas: la segunda
rutina histórica omite la etiqueta del nodo y sus aristas también tocan observados.
La instrumentación no modifica las salidas del notebook congelado.

Después del filtro histórico, si exactamente un observado y uno o más sintéticos
con procedencia capturada caen en el mismo `(t,z,y,x)` entero, se rechazan esos
sintéticos y únicamente sus aristas incidentes. No se modifica la detección
observada, no se transfieren enlaces, no hay jitter ni poda en cascada.
Dos observados, solo sintéticos o procedencia ambigua siguen produciendo error.

Es una modificación estructural explícita de la referencia donde aparecen
colisiones; puede afectar continuidad y no se presume que mejore el score.
La guardia se comparte entre replay, inferencia remota y Kaggle, y entre control
y candidata antes de evaluar el head. No se usa para ajustar el número de filas.
El CSV histórico visible verificado no contiene vóxeles duplicados; la nueva
release debe volver a pasar el mismo hash sin ninguna excepción.
Auditoría por muestra: `*.rounding-collisions.json`, incluyendo origen por nodo,
conteos por etapa y procedencias desconocidas. Detalle: `docs/FIX_PREPARE_004.md`.

## Brazos y parámetros

Resultado observado: `mitosis-005` fue rechazado por delta cero; `diagnose-006`
confirmó scores bajos en GT y ausencia de forks GT exactos entre las propuestas
puntuadas. La hipótesis aditiva no se promueve. `learn-007` subió scores pero
empeoró AUROC cruzado y FP, por lo que tampoco habilita tracking. `probe-008`
mostró mejor AUROC/AP con geometría respecto al histórico en ambos grupos GT.
009 midió ese modelo intacto sobre nodos predichos: cubrió once forks, dos ya
presentes; AP confirmed-only0.0741/0.0938, sin aprobar reasignaciones.
El paso actual es `docs/COUNTERFACTUAL_MITOSIS_010.md`: medir reparaciones elegidas
con GT sobre copias en memoria y localizar el contexto temporal que limita el
score oficial. Es un ensayo de evaluación, no inferencia, no máximo matemático
ni modelo publicable. No modifica los brazos siguientes ni habilita candidata.

| Brazo | Cambios respecto de los nodos/enlaces de referencia |
|---|---|
| control | Ninguno; referencia pareada calculada en la misma ejecución |
| mitosis_add | Conserva todos los nodos/enlaces; añade segunda hija libre solo cuando la pareja exacta recibe evidencia del head |
| hoct | Reestima asociaciones con contexto de hasta cinco frames; mitosis limitadas a parejas ya presentes en la referencia |
| hoct_mitosis | HOCT más parejas mitóticas aprobadas por el head y variables de tripleta en SCIP |
| hoct_rescue | HOCT más recuperación localizada de un frame con máximos reales de imagen |
| full | Asociación, mitosis aprendida y rescate conjuntamente |

Configuración inicial completa: `configs/nextgen.json`. No se elige mirando Kaggle.
HOCT aporta 0.65 de la mezcla cuando existe prior dual-seed; sin prior se exige
evidencia HOCT. Se conservan candidatos neuronales >=aprox.0.05 y vecinos a 12 µm.
Un prior ausente nunca se convierte en una probabilidad geométrica inventada.

El head mitótico usa cuatro parches 3D (`t−1,t,t+1,t+2`) y once rasgos de geometría
y disponibilidad temporal, invariantes al orden de las hijas. El umbral inicial
es 0.80. Se entrena con BCE ponderada por desequilibrio y doce épocas fijas,
seed42, AdamW lr0.0003. Se cachean los parches reales una vez. No se usa jitter
inconsistente entre geometría e imagen. Los scores del head no se presentan como
probabilidades calibradas.

Un negativo requiere que ambas hijas tengan padres anotados y que la pareja no
corresponda a la madre. Ausencia de anotación significa desconocido. En
`mitosis_add` no se roba una hija asignada a otra madre ni se reemplaza un fork.
En los brazos globales el ILP admite la segunda hija únicamente mediante la
tripleta exacta, no por un descuento general de coste mitótico.

El rescate exige extremo previo/posterior aislado, máximo de intensidad medido,
SNR>=4, correlación de parches>=0.5 y separación de detecciones presentes.
Máximo `min(200, 0.005*N)` nodos por muestra. El ILP exporta un rescate solo si
elige sus dos enlaces. No se usa para completar filas. SCIP: 60 segundos por
solución; un brazo con rescate puede resolver dos veces por muestra.

## Validación y límites de independencia

Se entrenan dos heads con épocas fijas: uno con 44b6 y otro con 6bba. Cada muestra
de desarrollo/confirmación se predice con el head del prefijo contrario. Ningún
grupo evaluado se usa para seleccionar época, calibrar o entrenar su head.
Los prefijos son una aproximación de agrupación biológica, no identidad biológica
independiente demostrada. Con dos grupos no se inventa un tercer holdout.

Desarrollo mantiene las ocho muestras archivadas. Confirmación selecciona ocho
nuevas por divisiones anotadas y densidad, excluyendo los paneles ya expuestos;
el algoritmo está congelado antes de inferencia. Sus nombres quedan en plan.json.
Los veinte ejemplos visibles/de desarrollo/confirmación se cachean desde el
mismo detector. El promedio global nunca oculta una regresión de grupo/muestra.

La métrica es la copia oficial archivada, commit
`075fc5f5a52d11077f9dc2b074644618f26939e2`; no el proxy de `val_cache.json`.
Se usan los `estimated_number_of_nodes` del GEFF, nunca el número escaso de
anotaciones como cardinalidad total. Un score oficial diagnóstico >1 puede
aparecer por su ajuste firmado; no se recorta ni se utiliza para explotar nodos
falsos. Todos los linajes se validan estructural y espacialmente.

## Decisión automática

- Delta de score mínimo: desarrollo +0.003; confirmación +0.002.
- Sin regresión del Jaccard de aristas ni de divisiones agregado.
- Sin regresión del score agregado de ninguno de los dos grupos.
- Ninguna muestra pierde más de 0.015 de Jaccard ajustado.
- Toda métrica necesaria indefinida bloquea la promoción.
- Luego se entrena un head final con ambos grupos, con el mismo número fijo de
  épocas. Ese head final solo pasa chequeo visible de formato/cambio/tiempo;
  sus muestras visibles no vuelven a presentarse como validación independiente.
- Filas visibles y de submission: `234000 < N < 242000`. Inferencia end-to-end
  remota y smoke Kaggle deben completar dentro del presupuesto de 7200 segundos.
  Es un presupuesto nuestro, no una afirmación sobre el límite oficial privado.
- Antes de submission, el CSV visible generado por Kaggle debe coincidir con el
  de la remota. Si difiere por hardware o empaquetado se devuelve al diseñador;
  el operador no ajusta tolerancias.

El score local permite decidir qué hipótesis merece una prueba; el API de Kaggle
es la única evidencia del score Kaggle. Máximo un envío diario y por paquete.

## HOCT: dependencia experimental explícita

El puente punto→imagen usa 19 características medidas; no es una segmentación
preentrenada validada. La función upstream `create_graph_from_points` inspeccionada
estaba sin implementar, por lo que se invoca directamente el ABI TorchScript.
Las máscaras, normalización, memoria, ABI y rendimiento requieren pruebas reales.
Un fallo aborta el brazo y se informa; no produce máscaras/features ficticios.

La fase HOCT requiere registrar revisión Git de 40 hex, hash real de `_api.py`,
checkpoint verificado y evidencia de admisibilidad en las reglas oficiales.
`configs/hoct.json` contiene huecos explícitos porque esa revisión no fue completada
para este proyecto. El operador puede completar hechos verificables, no elegir
otra arquitectura ni modificar el puente. `hoct-assets` instala solo en un target
aislado y deja intacto el conda compartido. Una incompatibilidad vuelve al diseñador.

Fuentes técnicas del diseño previo: [API HOCT](https://github.com/royerlab/hoct/blob/main/src/hoct/_api.py),
[batching](https://github.com/royerlab/hoct/blob/main/src/hoct/data/_batching.py),
[métrica oficial congelada](https://github.com/royerlab/kaggle-cell-tracking-competition/tree/075fc5f5a52d11077f9dc2b074644618f26939e2).
Los enlaces móviles son orientación para el operador; el runtime exige hashes
y revisión inmutables, no descarga `main` como dependencia de ejecución.
