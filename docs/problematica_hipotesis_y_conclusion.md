# Análisis Técnico: Problemática y Conclusiones en Seguimiento Celular 3D

## 1. La Problemática

### 1.1 Contexto y Síntomas
* **Estancamiento del modelo:** Al utilizar una arquitectura base pública (3D UNet / Node-Transformer) sin *fine-tuning*, el rendimiento de la predicción alcanza un techo del cual no logra avanzar.
* **Discordancia Local vs. Leaderboard (LD):** Las técnicas tradicionales de posprocesamiento logran incrementar la puntuación en la validación local, pero provocan una caída en la métrica del Leaderboard global.

### 1.2 Causa Raíz Identificada
El problema fundamental se localiza en el módulo de **detección**:
* El detector identifica con éxito las células verdaderas (*Ground Truth*), pero simultáneamente genera una masa desproporcionada de falsos positivos ("candidatos basura").
* Esta piscina sobrepoblada de candidatos contamina la fase de enlazado (*linking*), obligando al algoritmo a elegir entre demasiadas opciones incorrectas, lo que resulta en enlaces erróneos y degradación del grafo de seguimiento.

### 1.3 Impacto Matemático en la Métrica
La métrica del dominio se compone de:
$$\text{Score} = \text{Adjusted Edge Jaccard} + 0.1 \times \text{Division Jaccard}$$

El comportamiento del *recall* de aristas (*edges*) se aproxima mediante:
$$\text{Edge Recall} \approx (\text{Node Recall})^2 \times \text{Conditional Linking Accuracy}$$

* **Efecto cuadrático:** Una pequeña pérdida en el *recall* de nodos impacta exponencialmente la recuperación de aristas, dado que un enlace solo es válido si ambos nodos de extremo son detectados y conectados correctamente.
* **Complexidad de divisiones:** Para eventos de división celular, la exigencia es aún mayor: la célula madre y ambas células hijas deben detectarse previamente para que cualquier algoritmo de ranking o enlazado pueda registrar la división.

---

## 2. Metodología de Diagnóstico

Para aislar las fuentes de error sin interferencia entre componentes, se establece una evaluación descompuesta en cuatro ejes:

1. **Evaluación de Detección:**
   * Medición del *Node Recall* en un radio de coincidencia de $7\,\mu\text{m}$.
   * Comparación del conteo de nodos predichos vs. el conteo estimado provisto.

2. **Evaluación de Enlazado (*Linking*):**
   * Medición de la exactitud de enlace (*linking accuracy*) aplicada **únicamente** sobre nodos verdaderos (*Ground Truth*) detectados de manera exitosa.

3. **Techo de Detección (*Detection Ceiling / Oracle*):**
   * Determinación de la máxima puntuación de aristas alcanzable con las detecciones actuales mediante un *linker* ideal (GT-aware).

4. **Evaluación de Divisiones:**
   * Análisis de la disponibilidad de detección de células madre/hijas, *recall@K* de candidatos, calidad del ranking y balance final de verdaderos/falsos positivos.

---

## 3. Conclusión y Recomendaciones

1. **No cambiar la arquitectura base prematuramente:**
   * La combinación de *3D UNet + Node-Transformer* sigue siendo un *backbone* extremadamente competitivo en este dominio. El estancamiento no proviene de una limitación intrínseca de la arquitectura.

2. **Aislar el control de experimentos:**
   * No se deben alterar el detector y el *linker* de forma simultánea. Al evaluar diferentes *linkers*, el detector debe mantenerse fijo; al evaluar detectores, el *linker* debe ser el mismo. De lo contrario, los cambios en el grafo y la piscina de candidatos distorsionan la validación local.

3. **Priorizar la calibración y precisión del detector:**
   * El objetivo primario debe ser elevar la precisión del detector para **reducir el ruido de candidatos basura** antes de enviar las detecciones al *linker*. Si la precisión condicional de enlace ya es alta, invertir esfuerzo en un nuevo *linker* no aportará mejoras significativas sin antes limpiar la entrada del detector.

4. **Reentrenamiento integral en lugar de posprocesamiento:**
   * El uso de posprocesamiento sobre *checkpoints* públicos preentrenados ha llegado a su límite práctico. La estrategia efectiva consiste en reentrenar la canalización (*pipeline*) completa desde cero, ajustando las fases de preprocesamiento de datos y refinando el entrenamiento de visión por computadora.
