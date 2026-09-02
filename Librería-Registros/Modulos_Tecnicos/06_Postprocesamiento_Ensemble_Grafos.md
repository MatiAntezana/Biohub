---
tags: [modulo, postprocesamiento, ensemble]
---

# 🧩 Módulo 06: Postprocesamiento y Ensembling de Grafos

## 💡 Estrategias de Refinamiento
1. **Poda de Nodos Aislados:** Eliminar nodos que no posean aristas entrantes ni salientes (falsos positivos de detección).
2. **Eliminación de Trayectorias Huérfanas Cortas:** Suprimir cadenas de células con duración $< L_{\min}$ frames (ej: $< 3$ frames) que suelen ser artefactos de ruido.
3. **Interpolación de Gaps (Gap Closing):** Si una célula desaparece en $t$ pero reaparece en $t+2$ en la misma posición esperada, reconectar con un nodo interpolado.
4. **Ensemble de Detectores (WBF 3D / NMS):** Fusionar centroides de U-Net 3D, StarDist y Cellpose promediando coordenadas ponderadas por score de confianza.
5. **Verificación de Formato de Submission:** Asegurar que todo `source_id` y `target_id` referencie un `node_id` existente y que todos los datasets de test estén presentes.
