# EXP-059: DeepGraph Wide HighRes Anisotropic

## 1. Resumen y Motivación
- **ID de Experimento**: `EXP-059`
- **Linaje Padre**: `EXP-056` (Wide Capacity 48-192) y `EXP-054` (HighRes Anisotropic Spatial)
- **Objetivo**: Superar el récord de 0.9338 / 0.934 integrando:
  1. **Capacidad Representacional Ampliada**: `unet_layers=[48, 96, 192]` con 48 canales de salida (+50% de parámetros visuales sobre el baseline de 32).
  2. **Grid Espacial Fino**: `downsample=(1, 2, 2)` que preserva la resolución axial $Z$ y el detalle subcelular $XY$.
  3. **Pooling Físico Anisótropo**: `pool_kernel_um=3.50 µm` calibrado al tamaño de células embrionarias.
  4. **Ventana Temporal Balanceada**: $T=4$ con Node Transformer (`d_model=192`, `nhead=6`).
  5. **Blindaje de Memoria Dinámica**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` para asignación sin fragmentación.
  6. **Optimización**: `AdamW` + `CosineAnnealingLR` ($10^{-3} \rightarrow 10^{-5}$) con `weight_decay=0.0003`.

## 2. Hardware y Ejecución
- **Nodo**: `A1560` (NVIDIA GeForce RTX 4090, 24 GB VRAM).
- **Consumo de Memoria**: ~21.2 GB VRAM.
- **Semillas**: Doble semilla estandarizada `[3407, 9479]`, 50 épocas por semilla en modo Full-Fit (199 datasets).

## 3. Métricas Finales de Entrenamiento (Full-Fit 199 Datasets)
- **Semilla 1 (Primary, seed 3407)**:
  - `detection_loss`: **0.017439** (récord de detección en alta resolución)
  - `edge_loss`: **0.000554**
  - Duración: 50 épocas (~621 s/época)
- **Semilla 2 (Secondary, seed 9479)**:
  - `detection_loss`: **0.019317**
  - `edge_loss`: **0.000537**
  - Duración: 50 épocas (~621 s/época)

## 4. Conclusiones y Decisión
- **Estado**: **`SUCCEEDED`** y reconciliado con éxito.
- **Validación**: Excelente estabilidad y velocidad de convergencia (10.3 horas totales por corrida dual). Mínimo de detección sobresaliente de 0.0174.
- **Almacenamiento Desacoplado**: Checkpoints respaldados automáticamente en Hugging Face Hub (`hf://Matycaba2016/biohub-checkpoints/experiments/EXP-059_DeepGraph_Wide_HighRes_Anisotropic/`) y purgados del almacenamiento local.
- **Decisión**: **`PROCEED`**. Corrida cerrada formalmente y disponible para ensamble y benchmark.
