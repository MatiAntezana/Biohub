# Plantilla de predeclaración — siguiente experimento Biohub

Plantilla, NO experimento aprobado ni comandos ejecutables. Copiar a una ficha
Markdown nueva por campaña, completar ANTES de generar resultados de selección
y congelar su SHA256 en el manifiesto/receipt de la release correspondiente.
No editar una ficha congelada: cualquier cambio exige otra versión/run.

## Identidad y punto de partida

- Campaña/fecha/responsable:
- Hipótesis A/B/C/D/E u otra dentro del alcance, con evidencia nueva explícita:
- Run fuente, nodo de origen, hashes de receipts y artefactos:
- Nodo de ejecución (A1555 o A1557), tests y admisión requeridos:
- Release/control archivados y comparación con la campeona vigente:
- Exposiciones previas de cada panel y limitaciones de independencia:

## Hipótesis falsable

- Cuello de botella medido (conteos, denominadores y archivos):
- Mecanismo propuesto y por qué estos datos lo apoyan:
- Qué observación concreta lo rechazaría:
- Qué incertidumbre NO resuelve el ensayo:

## Cambio acotado

- Módulos/configs nuevos y drivers modificados:
- Archivos/modelos/parámetros protegidos que no cambian:
- Control:
- Variante1:
- Variante2 opcional (máximo2; declarar ahora o escribir “no habrá variante2”):
- Features, reglas de candidatos, parámetros, seeds, épocas/optimizer:
- Conteo de supervisión única por madre/fork/volumen; unknown/mixtos:
- Criterios de suficiencia antes de fitting y de abstención:
- Regla de calibración fuente; criterio fijo de selección en desarrollo:

## Datos y ejecución

- Listas exactas de FIT, calibración, desarrollo y confirmación; hashes:
- Prueba de ausencia de GT en inferencia y de exclusión del prefijo evaluado:
- Necesita inferencia GPU nueva o reutiliza cache verificado:
- Dependencias/revisiones/hashes/admisibilidad (no asumir modelos listos):
- CPU/GPU/RAM/disco/tiempo máximos y condición de rechazo por recursos:
- IDs nuevos de tests/preparación/ensayo; dependencias entre tareas:
- Comandos exactos IMPLEMENTADOS y comprobados por ayuda/tests:
- Transferencias autorizadas y verificación destino si usa A1557:

## Verificación y decisión predeclarada

- Fixtures unitarios/regresión, pruebas end-to-end y suite completa por nodo:
- Comprobación del código realmente ejecutado y de pesos/config activos:
- Métrica oficial y control pareado idénticos:
- Desarrollo: delta>=0.003 y gates restantes de protocol.json:
- Confirmación: solo receta elegida, delta>=0.002 y gates restantes:
- Outputs guardados antes de abrir GT de cada evaluación:
- Integración necesaria en inferencia/experiment/package/notebook:
- Paridad referencia, paridad candidata remota/Kaggle, filas y tiempo:
- Si falla: qué se conserva y qué siguiente decisión está justificada:
- Si pasa: siguiente fase exacta; nunca promover un receipt diagnóstico:

## Resultado (sección nueva posterior; no reescribir la predeclaración)

- Receipt/hash, estado real y rutas de logs/artefactos:
- Cantidades esperadas/procesadas, métricas por prefijo/muestra, deltas:
- Diferencias reales del grafo y política de abstención:
- Limitaciones, desviaciones o fallos de integridad:
- Aceptada/rechazada/bloqueada y evidencia de esa decisión:
- Kaggle: no enviado / ref y score COMPLETE exactos; nunca inferido:
- Próximo paso/comando y actualización de STATUS:
