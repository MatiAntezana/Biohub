# 🧬 Contexto de la Competencia: Biohub - Cell Tracking During Development

> **Documento de Contexto Maestro (`folder-context.md`)**  
> *Diseñado para que cualquier agente o modelo de Inteligencia Artificial entienda de inmediato el problema biológico, la estructura de datos, las métricas de evaluación, el ecosistema de código, los modelos campeones y las reglas operativas del proyecto.*

---

## 📌 1. Resumen Ejecutivo y Objetivo Biológico

* **Nombre de la Competencia:** *Biohub - Cell Tracking During Development* (Kaggle Code Competition).
* **Organizador:** Chan Zuckerberg Biohub San Francisco.
* **Problema Científico:** Reconstruir el linaje celular completo en 3D+Tiempo (árboles genealógicos y grafos acíclicos dirigidos - DAGs) a partir de grabaciones de microscopía de fluorescencia 4D durante el desarrollo embrionario del pez cebra (*Danio rerio*).
* **Objetivo del Algoritmo:**
  1. **Detección Espacial (Nodos):** Predecir las coordenadas 3D $(t, z, y, x)$ de cada centroide celular activo en cada fotograma temporal.
  2. **Asociación Temporal (Enlaces / Edges):** Conectar los centroides entre fotogramas consecutivos $(t \to t+1)$ o con salto de fotograma (*gap closing*, $t \to t+2$).
  3. **Divisiones Celulares (Mitosis):** Identificar cuándo una célula madre da origen a dos células hijas ($1 \to 2$).

---

## 📦 2. Estructura y Física de los Datos

### A. Anisotropía Espacial (CRÍTICO)
Los vóxeles de microscopía no son cúbicos; el eje axial ($Z$) tiene 4 veces menor resolución que los ejes laterales ($Y, X$):
$$\text{SCALE} = [1.625,\, 0.40625,\, 0.40625]\,\mu\text{m/vóxel}$$
> [!IMPORTANT]
> Toda distancia euclidiana o cálculo de costo de emparejamiento temporal DEBE multiplicar las coordenadas por este vector de escala física antes de calcular distancias.

### B. Formato de Archivos
* **Imágenes 4D (`.zarr`):** Volúmenes $(T, Z, Y, X)$ en formato Zarr v3 con compresión Blosc, típicamente de forma $(100, 64, 256, 256)$ en uint16.
* **Etiquetas de Entrenamiento (`.geff`):** Grafos biológicos (Graph Exchange Format) con propiedades de nodos $(t, z, y, x)$ y enlaces de linaje.
* **Directorio en Kaggle:**
  ```text
  biohub-cell-tracking-during-development/
  ├── train/                  # 199 videos (Imágenes + Etiquetas .geff) -> SOLO PARA ENTRENAR
  ├── test/                   # Videos de Test (4 en modo edición / ~199 privados en Submit)
  └── sample_submission.csv   # Formato requerido (Solo contiene IDs de test/)
  ```

---

## 📐 3. Métrica Oficial y Reglas de Evaluación

* **Métrica:** Promedio ponderado de **Edge Jaccard** (asociaciones temporales correctas con penalización por densidad) y **Division Jaccard** (detección exacta de mitosis).
* **Restricciones de Kaggle:**
  * Competencia de código pura (*Code Competition*).
  * Sin acceso a internet durante la inferencia.
  * Límite estricto de tiempo: **9 a 12 horas máximo**.
  * Hardware permitido: **GPU Tesla T4 (Dual T4 x2)** o **CPU (4 vCPUs)**. *Las GPUs P100 (sm_60) están prohibidas por reglas y generan error de CUDA.*

---

## 🏆 4. Modelos Campeones y Hallazgos Consolidados

| Experimento | Arquitectura | Esquema | Score CV | Hallazgo Principal |
|---|---|:---:|:---:|---|
| **`EXP-017`** | **3D Wide Anisotropic U-Net (64ch)** | `90/10` | **`0.9010`** 🏆 | **Grand Champion:** Cuadruplicar canales a 64 y entrenar en 179 volúmenes rompió la barrera de 0.90. |
| **`EXP-019`** | **3D Wide ResUNet (64ch)** | `90/10` | **`0.8867`** | Bloques residuales `ResBlock3D` con menor pérdida de entrenamiento histórica (`0.000343`). |
| **`EXP-022`** | **StarDist 3D Polyhedra (32 Rayos)** | `90/10` | **`0.8601`** | **Récord Histórico de Recall (99.23%)**: Despega cúmulos densos sin mapas gaussianos continuos. |
| **`EXP-028`** | **3D Wide Attention U-Net (64ch)** | `90/10` | *En Progreso* | *Spatial Attention Gates* 3D para filtrar el 98% de líquido de fondo. |
| **`EXP-029`** | **3D Full-Volume GroupNorm U-Net** | `90/10` | *En Progreso* | `GroupNorm3d` (8 grupos) para estabilidad numérica total con batch size 1 en volúmenes completos. |

---

## 🚀 5. Protocolo Estricto de Despliegue en Kaggle (Kaggle Submissions)

Cualquier cuaderno generado para Kaggle DEBE seguir estas 5 reglas obligatorias:
1. **Búsqueda Exclusiva en `test/`:**  
   Nunca incluir `train/` en la búsqueda de inferencia; de lo contrario, Kaggle procesará 203 videos en lugar de 4 durante el *Save Version* y tardará 8 horas en lugar de 2 minutos.
2. **Cargador Universal Zarr (`zarr.open`):**  
   Cargar usando `zarr.open(path)["0"][t]` con fallback Blosc2, evitando nombres fijos como `{t}.0.0.0` para prevenir imágenes vacías de ceros (`0 nodes, 0 edges`).
3. **Metadatos `kernelspec` en el `.ipynb`:**  
   Incluir `"kernelspec": {"name": "python3", "display_name": "Python 3"}` para evitar errores de Papermill.
4. **Validación Previa de `state_dict`:**  
   Probar localmente `model.load_state_dict(...)` antes de subir para evitar desajustes de capas.
5. **Configuración de Acelerador:**  
   * GPU: `"enable_gpu": "true"`, `"machine_shape": "NvidiaTeslaT4"`.
   * CPU: `"enable_gpu": "false"`, con `torch.set_num_threads(4)` para inferencia rápida AVX-512.

---

## 🖥️ 6. Infraestructura de Hardware y Reglas Operativas

1. **Máquina Local del Usuario (Mac):**  
   **CERO cómputo pesado.** Solo se utiliza para edición de código, control de versiones y sincronización SSH/Kaggle CLI.
2. **Cluster Físico Remoto (4x Servidores con GPU NVIDIA RTX 4090 24GB):**  
   * `A1557-UBU` (`100.87.128.33:2222`)
   * `A1556-UBU` (`100.107.113.22:2222`)
   * `A1560-UBU` (`100.84.253.117:2222`)
   * `A1552-UBU` (`100.112.138.117:2222`)
3. **Bóveda de Conocimiento Obsidian:**  
   Ubicada en `Librería-Registros/`. Todo nuevo experimento o hallazgo se documenta automáticamente en [01_Registro_Experimentos.md](file:///Users/matias/Proyects/Biohub/Librería-Registros/01_Registro_Experimentos.md).

---

## 🗂️ 7. Mapa de Directorios del Repositorio

```text
/Users/matias/Proyects/Biohub/
├── folder-context.md               # Este archivo (Contexto maestro para IAs)
├── src/                            # Código fuente modular
│   ├── dataset.py                  # Dataset por parches 3D con augmentations
│   ├── dataset_full_volume.py      # Dataset de volumen completo (64x256x256)
│   ├── models/
│   │   ├── unet3d_anisotropic.py   # AnisotropicUNet3D (EXP-017 Champion)
│   │   ├── resnet_unet3d.py        # ResUNet3D (EXP-019)
│   │   ├── stardist3d.py           # StarDist3D Polyhedra (EXP-022)
│   │   ├── attention_unet3d.py     # AttentionUNet3D (EXP-028)
│   │   └── unet3d_groupnorm.py     # GroupNormUNet3D (EXP-029)
│   └── tracking/
│       └── gap_closing_tracker.py  # Algoritmo de tracking LAP + Gap Closing
├── champion_weights/               # Pesos .pt de los modelos campeones
├── kaggle_submission_exp017/       # Paquete de envío Kaggle EXP-017 (Wide UNet T4)
├── kaggle_submission_exp019/       # Paquete de envío Kaggle EXP-019 (ResUNet T4)
├── kaggle_submission_exp022/       # Paquete de envío Kaggle EXP-022 (StarDist CPU)
└── Librería-Registros/             # Bóveda Obsidian de Bitácoras y Experimentos
```

---

## 🧪 8. Ramas experimentales independientes preparadas

Además de la familia de detectores 3D por heatmaps, existen dos estrategias experimentales y una nueva variante full-fit:

* **EXP-030:** versión original con dos seeds de `TemporalUNet3D + Node Transformer`, TTA de ocho vistas, fusión calibrada, retention guard e ILP. Conservar su historial y checkpoints.
* **EXP-030B:** nueva versión de EXP-030; entrena ambos seeds con `100 %` de los datos, sin validación, durante `55 × 750` por seed.
* **EXP-031:** revisión detector-first full-fit con hard-negative mining, `100 %` train / `0 %` validación, `65 × 850`, ILP preservado y motion/divisiones sólo sobre nodos no asociados.

EXP-030B y la revisión de EXP-031 deben entrenarse únicamente en una GPU remota. Su estado no debe cambiarse a “En progreso” hasta observar el sentinel `RUNNING.json` correspondiente. Como no tienen holdout, su CV debe registrarse como `N/A — full-fit sin validación`, nunca como la loss de train.

---

## 🚦 9. Experiment Promotion Board

El evaluador y selector reusable están concentrados en `test-exp-local/`: inferencia real, validación, agregación, CLI, protocolo congelado y tests. Sólo acepta artefactos con conteos oficiales por dataset, mismo `protocol_id` y exactamente las mismas muestras. Ordena por el límite inferior bootstrap 95% del score oficial y usa integridad, cobertura y runtime como gates.

El piloto remoto EXP-017 vs EXP-019 está en `experiments/Promotion_Board_EXP017_EXP019_Pilot/`: EXP-017 `0.8098`, EXP-019 `0.7783`, delta pareado `+0.0315` con IC 95% `[+0.0046, +0.0592]`. El framework queda adoptado; el protocolo legacy de 12 muestras sigue siendo piloto y no una métrica global definitiva.
