# EXP-060: DeepGraph Wide Window-5 Anisotropic (Wide UNet 48-192 + T=5 Temporal Context + Z 3.25um + Cosine Annealing)

## 1. Resumen y Motivación Científica
- **ID de Experimento**: `EXP-060`
- **Linaje Padre**: `EXP-056` (Récord histórico de Kaggle Public Score **0.933**)
- **Carril**: `adjacent_exploration` (Receta canónica: `deepgraph_multiscale_window`)
- **Objetivo**: Superar el récord de 0.933 extendiendo el horizonte temporal del grafo de tracking biológico:
  1. **Horizonte Temporal Extendido ($T=5$)**: En el seguimiento de linajes celulares embrionarios durante el desarrollo, los procesos de división mitótica y cruce de células vecinas típicamente transcurren a lo largo de 4 a 6 marcos temporales. Con una ventana temporal de $T=4$ (utilizada en EXP-056), los eventos prolongados sufren pérdida de contexto de las células madre/hijas y fragmentación de pistas. Al expandir la ventana temporal a $T=5$, los bloques de cross-attention del Node Transformer capturan la cinemática multiescala completa y la persistencia de trayectorias.
  2. **Arquitectura Wide UNet Anisótropa Probada**: Se preserva la columna vertebral de alta resolución de EXP-056 con canales `[48, 96, 192]`, salida de 48 canales y `pool_kernel_um = 3.25 µm` para resolver la compresión axial del eje Z en microscopía 3D.
  3. **Node Transformer de Alta Capacidad**: `hidden_dim = 192`, `n_heads = 6`, `dropout = 0.1`.
  4. **Esquema de Optimización Cosine Annealing**: `lr = 0.001 \rightarrow 10^{-5}`, regularizado con `weight_decay = 0.0003` para convergencia óptima en 50 épocas.
  5. **Post-procesamiento ILP y Anti-Timeout**: Parámetros ganadores de división (`SAFE_DIV_MAX_UM = 7.0`, `SAFE_DIV_SISTER_MAX_UM = 12.0`, `GLOBAL_FRAC_CAP = 0.00375`, `ILP_DIVISION_WEIGHT = 1.2`) con guardia ILP de 60s y desactivación del validador pesado para inferencia en <15 min en Kaggle.

## 2. Hardware y Ejecución Remota
- **Nodo Asignado**: `A1688` (NVIDIA GeForce RTX 4090, 24,564 MiB VRAM libre, entorno Conda `train-88`).
- **Modo de Entrenamiento**: Full-fit (199 datasets pareados), doble semilla (`[3407, 9479]`), 50 épocas por semilla.
- **Inferencia / Kaggle**: Pipeline automático con ensamble dual-seed y submission para Public Score.
