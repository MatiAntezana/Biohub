---
tags: [bitacora, aciertos, validado]
---

# ✅ Registro de lo que SÍ Funcionó (Golden Insights)

En esta nota se consolidan todas las técnicas, parámetros y componentes que han demostrado una mejora consistente en la Validación Cruzada Local y en el Leaderboard.

---

## 🏆 Componentes Validados

### 1. Escala Física Ponderada en Matching
- **Insight:** Medir distancias en píxeles produce graves errores debido a que 1 píxel en Z equivale a 4 píxeles en Y/X ($1.625\,\mu	ext{m}$ vs $0.40625\,\mu	ext{m}$).
- **Solución:** Multiplicar siempre las coordenadas por $	ext{SCALE} = [1.625, 0.40625, 0.40625]$ antes de calcular la matriz de costos Euclidianos.
- **Validado en:** [[Experimentos/EXP-000_Baseline_Nearest_Neighbor|EXP-000]].

---

## 🔬 Hallazgos y Componentes Validados

### 1. Convoluciones 3D Anisotrópicas $(1, 3, 3)$ + Heatmaps Gaussianos (`EXP-001`)
- **Impacto:** Subió el score de validación a **`0.5810` (+4268%)** y alcanzó un Node Recall de **`93.9%`**.
- **Por qué funciona:** Al usar kernels $(1, 3, 3)$ y MaxPool $(1, 2, 2)$ en la primera etapa, el modelo respeta la resolución 4 veces más fina en $X/Y$ ($0.40625\,\mu\text{m}$) sin degradar la profundidad $Z$ ($1.625\,\mu\text{m}$).
- **Pérdida:** Focal MSE Loss ponderada con factor $\times 10$ en picos celulares previene la dilución de gradientes por la dispersión de las anotaciones.
- **Implementación:** `src/models/unet3d_anisotropic.py`.

### 2. 4-Way 3D Test-Time Augmentation (TTA) + Multi-Frame Gap Closing (`EXP-004`)
- **Impacto:** Rompió la barrera de los 0.60 alcanzando un **Score de `0.6007`** y un Edge Jaccard de **`0.6056`** con Node Recall de **`94.84%`**.
- **Por qué funciona:** 
  1. El promediado de las 4 simetrías espaciales 3D ($V, \text{flip}_X, \text{flip}_Y, \text{flip}_Z$) suprime el ruido estocástico en los bordes y agudiza la precisión de los centroides sub-vóxel.
### 3. StarDist 3D Polyhedra con 32 Rayos Radiales (`EXP-010`)
- **Impacto:** **Node Recall récord de `97.81%`** en el holdout de validación.
- **Por qué funciona:** La parametrización de distancias polihédricas a lo largo de 32 rayos sobre la esfera de Fibonacci permite despegar cúmulos celulares ultra-densos donde los mapas de calor gaussianos tradicionales tienden a fusionarse.
- **Validado en:** [[Experimentos/EXP-010_StarDist3D_Polyhedra_80_20|EXP-010]].

### 4. Política de Tracking Limpio Multi-Frame LAP con Zero Falsos Positivos (`EXP-011`)
- **Impacto:** Disparó el score de **`0.6903` $\to$ `0.7968`** (+0.1065).
- **Por qué funciona:** Al suprimir divisiones mitóticas espurias (`FP=0`), el algoritmo de Linear Assignment Problem (LAP) conecta trayectorias temporales con mínima penalización, alcanzando un Edge Jaccard de `0.7831`.
- **Validado en:** [[Experimentos/EXP-011_Mitosis_Estimator_80_20|EXP-011]].

### 5. Bloques Residuales 3D (`ResBlock3D`) para Máxima Sensibilidad (`EXP-013`)
- **Impacto:** **Récord Absoluto de Sensibilidad en U-Net con `98.00% Node Recall`** y Score de **`0.7900`**.
- **Por qué funciona:** Los atajos de identidad residuales mitigan el desvanecimiento de gradiente, permitiendo a la red detectar centroides celulares tenues en etapas profundas de desarrollo.
- **Validado en:** [[Experimentos/EXP-013_ResUNet3D_80_20|EXP-013]].

### 6. Maximización de Datos de Entrenamiento con Esquema 90/10 (`EXP-015`)
- **Impacto:** **Superó la barrera de los 0.80 alcanzando el Score de `0.8096`** (`Node Recall = 96.93%`).
- **Por qué funciona:** Proporcionar 179 volúmenes de entrenamiento (17,021 parches 3D) expuso al modelo a una mayor diversidad morfológica celular, mejorando la generalización.
- **Validado en:** [[Experimentos/EXP-015_Anisotropic_90_10|EXP-015]].

### 7. 3D Wide U-Net (64 Canales) en Esquema 90/10 - GRAND CHAMPION (`EXP-017`)
- **Impacto:** 🏆 **ROMPE LA BARRERA DE LOS 0.90 CON SCORE RÉCORD DE `0.9010`** (`Edge Jaccard = 0.8763`, `Node Recall = 95.91%` a $\tau=0.035$).
- **Por qué funciona:** Al cuadruplicar la capacidad de canales ($64\text{ch}$) y alimentar la red con 179 volúmenes completos, la red minimizó la pérdida de entrenamiento a $0.000364$ y maximizó el emparejamiento temporal exacto.
- **Validado en:** [[Experimentos/EXP-017_Wide_UNet3D_90_10|EXP-017]].

### 8. 3D Wide Residual U-Net (64 Canales) en Esquema 90/10 (`EXP-019`)
- **Impacto:** **Score de `0.8867`** (`Edge Jaccard = 0.8604`, `Node Recall = 96.31%` a $\tau=0.050$) y **mínima pérdida de entrenamiento histórica (`0.000343`)**.
- **Por qué funciona:** Los bloques residuales `ResBlock3D` combinados con 64 canales y 179 volúmenes aseguran una propagación de gradiente impecable y una estabilidad de trayectorias sobresaliente.
- **Validado en:** [[Experimentos/EXP-019_Wide_ResUNet3D_90_10|EXP-019]].

### 9. StarDist 3D Polyhedra en Esquema 90/10 - RÉCORD ABSOLUTO DE RECALL (`EXP-022`)
- **Impacto:** 🥇 **RÉCORD HISTÓRICO DE SENSIBILIDAD CON `99.23% Node Recall`** y Score de **`0.8601`** (`Edge Jaccard = 0.8525`).
- **Por qué funciona:** La formulación de 32 rayos polihédricos sobre 179 volúmenes de entrenamiento detecta 992 de cada 1000 células en el volumen 3D y desempata cúmulos densos sin generar falsos positivos.
- **Validado en:** [[Experimentos/EXP-022_StarDist3D_90_10|EXP-022]].

### 10. 3D Wide StarDist Polyhedra (64 Canales) en Esquema 90/10 (`EXP-024`)
- **Impacto:** **Score de `0.8544`** (`Edge Jaccard = 0.8292`, `Node Recall = 98.15%` a $\tau=0.010$) con pérdida de entrenamiento de `0.000372`.
- **Por qué funciona:** Al cuadruplicar la capacidad convolucional ($64\text{ch}$), StarDist aprende representaciones morfológicas ricas que aumentan la precisión en cúmulos celulares complejos.
- **Validado en:** [[Experimentos/EXP-024_Wide_StarDist3D_90_10|EXP-024]].

### 11. Estándar Oficial de Despliegue en Kaggle (Inferencia Segura < 3 min)
- **Impacto:** Elimina al 100% los fallos de `Notebook Timeout` (14h $\to$ 2.5h), los errores de Papermill y las predicciones en negro (`0 nodes, 0 edges`).
- **Pilares:**
  1. Búsqueda dirigida exclusivamente a `test/` (evita procesar 203 datasets de train).
  2. Cargador universal `zarr.open` con fallback binario Blosc.
  3. Metadatos `kernelspec: python3` para Papermill.
  4. Test de carga previa de `state_dict`.
- **Documento Completo:** [[Estrategias/Estandar_Configuracion_Kaggle_Submissions|Estandar_Configuracion_Kaggle_Submissions]].

---
*(Esta nota se mantiene permanentemente actualizada de forma autónoma con cada descubrimiento).*
