# 🧬 Biohub-mejorado: Autoresearch V2 Framework

Un marco de investigación autónoma de doble velocidad (**Dual-Speed Autoresearch**) diseñado para superar las limitaciones de AutoML tradicional y escalar el rendimiento en reconstrucción de linajes celulares 3D+t (**Chan Zuckerberg Biohub Cell Tracking Competition**).

---

## ⚡ La Filosofía: Autoresearch de Doble Velocidad (Dual-Speed)

El autoresearch clásico (estilo Karpathy) fuerza una única escala temporal y espacial (un archivo `train.py`, 5 minutos fijos de GPU, una métrica escalar ciega). En Biohub, entrenar una red volumétrica 3D durante 50 épocas toma **2 a 3 horas en una RTX 4090 de 24 GB**, haciendo inviable la exploración autónoma ágil.

**Autoresearch V2 desacopla la investigación en dos bucles complementarios:**

```
                               ┌─────────────────────────────┐
                               │   INTELLIGENCE INGESTION    │
                               │  Kaggle Forums & Literature │
                               └──────────────┬──────────────┘
                                              ▼
                               ┌─────────────────────────────┐
                               │     SCIENTIFIC AGENT        │
                               │   Hypothesis & Memory       │
                               └──────┬───────────────┬──────┘
                                      │               │
                 Fast-Loop (CPU)      │               │ Slow-Loop (GPU)
                 10-15 segundos       │               │ 2-3 horas
                                      ▼               ▼
                       ┌────────────────────┐   ┌────────────────────┐
                       │  graph_builder.py  │   │      train.py      │
                       │  - Node Budgeting  │   │  - Wide UNet 3D    │
                       │  - Mitosis 1 -> 2  │   │  - Attention T=5   │
                       │  - Gap-2 Closing   │   │  - Mitosis Head    │
                       └─────────┬──────────┘   └─────────┬──────────┘
                                 │                        │
                                 ▼                        ▼
                       ┌────────────────────┐   ┌────────────────────┐
                       │  DIAGNOSTIC ENGINE │   │   HF CHECKPOINTS   │
                       │  Sub-metrics JSON  │   │   & CACHE UPDATE   │
                       └────────────────────┘   └────────────────────┘
```

1. **Fast-Loop (CPU / Memoria — 0.5 a 2 segundos por corrida)**:
   - **Espacio de búsqueda**: [`core/graph_builder.py`](file:///Users/matias/Proyects/Biohub-mejorado/core/graph_builder.py).
   - **Objetivo**: Optimizar la construcción del grafo sobre representaciones neuronales cacheadas.
   - **Palancas clave**:
     - *Node Budgeting*: Supresión de la penalización de nodos ($T_{\text{pred}} \le T_{\text{true}}$).
     - *Bifurcación Mitótica*: Enlace 1 a 2 células hijas (`SAFE_DIV`) para disparar el `Division Jaccard`.
     - *Cierre de Gaps Temporales*: Re-conexión $t \to t+2$ para mitigar parpadeos ópticos.
     - *Poda de Ruido*: Supresión de componentes huérfanas de corta duración.
   - **Rendimiento**: Más de 100 experimentos por hora sin tocar GPUs ni cuota de Kaggle.

2. **Slow-Loop (GPU Clúster — 2 a 3 horas)**:
   - **Espacio de búsqueda**: Backbones convolucionales 3D, cabezales de atención temporal ($T=5, T=6$), funciones de pérdida focal.
   - **Orquestación**: Clúster de 4x NVIDIA RTX 4090 (`A1552`, `A1556`, `A1557`, `A1560`) conectadas por Tailscale vía SSH puerto 2222.
   - **Activación**: Se dispara únicamente cuando el Fast-Loop detecta saturación en la capacidad de las representaciones actuales.

---

## 📊 Diagnóstico en Profundidad (Beyond Scalar Blindness)

A diferencia de sistemas que solo reportan un número escalar (`score: 0.933`), el evaluador diagnóstico ([`core/diagnostic_evaluator.py`](file:///Users/matias/Proyects/Biohub-mejorado/core/diagnostic_evaluator.py)) entrega un desglose exacto de las causas de pérdida de puntos:

- `node_penalty_multiplier`: Impacto de sobre-predicción de centroides.
- `node_recall_7um`: Porcentaje de células reales capturadas.
- `division_jaccard`: Desempeño exacto en mitosis (TP, FP, FN).
- `edges_recovered` vs `edges_fragmented` vs `edges_lost_to_detection` vs `wrong_association`.
- `z_axis_displacement_error`: Desviación focal por anisotropía óptica.

## 🖥️ Arquitectura Remote-First (Ejecución Estricta en Clúster Remoto)

Siguiendo las directivas operativas del proyecto:
- **La Mac local actúa estrictamente como Controladora / Supervisora**: aloja el código, versionado Git, redacción de hipótesis y registro de bitácora.
- **TODO el cómputo, benchmark, evaluación y entrenamiento se ejecuta en las máquinas remotas autorizadas** (`A1556-UBU`, `A1552-UBU`, `A1560-UBU`, `A1557-UBU`) conectadas vía **Tailscale (puerto SSH 2222, usuario `alumno1`)**.
- [`core/remote_executor.py`](file:///Users/matias/Proyects/Biohub-mejorado/core/remote_executor.py) sincroniza automáticamente el código vía `rsync` hacia `~/Documents/Biohub-mejorado` en la máquina remota y ejecuta sobre el entorno Conda correspondiente (`train-56` / `train-52` / `train-60`), devolviendo la telemetría a la Mac.
- Por defecto, `fast_loop/evaluate_fast.py` y `fast_loop/orchestrator.py` corren **en modo remoto**.

---

## 📁 Estructura del Repositorio

```
Biohub-mejorado/
├── program.md                     # Directivas del sistema y protocolo para el agente
├── results.tsv                    # Ledger canónico append-only de experimentos
├── MEMORY_FAILURES.md             # Memoria acumulativa de hipótesis fallidas
├── LATEST_COMMUNITY_DISCOVERIES.md# Feed sintetizado de hallazgos externos
├── core/
│   ├── data_types.py              # Definiciones tipadas (Detections, Edges, Telemetry)
│   ├── diagnostic_evaluator.py    # Evaluador exacto con descomposición matemática
│   └── graph_builder.py           # ARCHIVO MUTABLE DEL FAST-LOOP
├── fast_loop/
│   ├── evaluate_fast.py           # Runner de evaluación de alta velocidad (<1 seg)
│   ├── orchestrator.py            # Orquestador de selección natural (commit/revert)
│   └── mock_cache.py              # Generador de benchmark biológico sintético 3D+t
├── slow_loop/
│   ├── cluster_client.py          # Cliente SSH para las 4x RTX 4090
│   ├── slow_loop_controller.py    # Supervisor de reentrenamiento profundo
│   └── cache_extractor.py         # Extractor de predicciones neuronales a caché
├── intelligence/
│   └── external_intel.py          # Scraper y sintetizador de foros y literatura
└── integration/
    ├── export_to_kaggle.py        # Generador de módulos de inferencia para Kaggle
    └── export_to_biohub.py        # Puente hacia el repositorio canónico de Biohub
```

---

## 🚀 Guía Rápida de Uso

### 1. Ejecutar una Evaluación Rápida de Diagnóstico
```bash
python3 fast_loop/evaluate_fast.py
```

### 2. Ejecutar un Ciclo de Hipótesis Científica (Fast-Loop)
```bash
python3 fast_loop/orchestrator.py \
  --hypothesis "Reducir SAFE_DIV_MAX_UM a 7.5um para podar falsas divisiones" \
  --target-metric "division_jaccard"
```
- Si mejora el score: se commitea automáticamente y se registra en `results.tsv`.
- Si empeora: se revierte `core/graph_builder.py` y se registra el post-mortem en `MEMORY_FAILURES.md`.

### 3. Actualizar la Ingestión de Inteligencia Externa
```bash
python3 intelligence/external_intel.py
```

### 4. Consultar el Estado del Clúster de GPUs Remotas
```bash
python3 slow_loop/cluster_client.py
```

### 5. Exportar la Mejor Configuración a Kaggle o Biohub
```bash
python3 integration/export_to_kaggle.py
python3 integration/export_to_biohub.py --dry-run
```
