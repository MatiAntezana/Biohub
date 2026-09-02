---
tags: [benchmark, runtime, gpu]
---

# ⏱️ Benchmark de Tiempos y Restricciones de Recursos

## ⚠️ Restricciones del Entorno Kaggle
- **Tiempo límite de ejecución:** $\le 12	ext{ horas}$ (CPU o GPU).
- **Acceso a Internet:** Deshabilitado durante la inferencia en submit.
- **Formato de salida:** `submission.csv` con columnas `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id`.
- **Memoria RAM disponible:** 30 GB (Kaggle Standard) / 16 GB VRAM GPU (T4 o P100) / 32 GB VRAM (2x T4).

---

## 📊 Tabla de Presupuesto de Tiempo por Muestra

| Operación | Tiempo Máximo por Clip (100 frames) | Tiempo Total en Test (aprox. 15-20 clips) |
|---|---|---|
| I/O y Descompresión Zarr | $\le 15	ext{ s}$ | $\le 5	ext{ min}$ |
| Inferencia 3D U-Net / Detección | $\le 60	ext{ s}$ | $\le 20	ext{ min}$ |
| Extracción de Picos y Centroides | $\le 10	ext{ s}$ | $\le 3	ext{ min}$ |
| Asociación Temporal / ILP | $\le 45	ext{ s}$ | $\le 15	ext{ min}$ |
| Formateo y Exportación CSV | $\le 5	ext{ s}$ | $\le 2	ext{ min}$ |
| **Total Estimado** | **$pprox 2.2	ext{ min/clip}$** | **$pprox 45	ext{ min}$ (muy seguro dentro de las 12h)** |
