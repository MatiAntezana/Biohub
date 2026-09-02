---
id: EXP-020
tags: [experimento, inception3d, dilated_conv, multiscale_receptive_field, split_80_20]
fecha: 2026-08-25
autor: Matias & Antigravity AI
score_cv: 0.6150
score_lb: null
split: "80% / 20% (159 Train / 40 Val vols)"
estado: Finalizado
decision: INTEGRAR AL POOL DE ENSAMBLE MULTIMODELO (Score 0.6150)
---

# 🧪 EXP-020: 3D Inception-UNet con Convoluciones Dilatadas Multiescala (Esquema 80/20)

## 🎯 1. Hipótesis & Estrategia
- **Fundamento:** Introducir bloques **Inception 3D (`InceptionBlock3D`)** con 4 ramas paralelas por nivel:
  1. Convolución $1 \times 1 \times 1$ para mapeo de canales.
  2. Convolución $1 \times 3 \times 3$ anisotrópica.
  3. Convolución $3 \times 3 \times 3$ con dilatación $d=2$ para capturar contexto celular amplio.
  4. Convolución estándar $3 \times 3 \times 3$ para centroides locales.
- **Hardware:** NVIDIA GeForce RTX 4090 (24 GB VRAM) en nodo remoto `A1560-UBU`.

---

## 📊 2. Resultados de Validación Oficial (Holdout 20%)

| Métrica | Valor Obtenido | Comentarios |
|---|:---:|---|
| **Score Total (Adjusted Edge Jaccard)** | **`0.6150`** | Sólido rendimiento en 80/20 |
| **Node Recall** | **`0.8825` (88.25%)** | Detección limpia de células |
| **Edge Jaccard** | **`0.6043`** | Consistencia temporal |
| **Division Penalties (FP)** | **`0`** | Zero divisiones espurias |

---

## 🔍 3. Análisis Científico
1. **Velocidad y Diversidad:**
   * Entrenó a gran velocidad (314s por época).
   * La mezcla de convoluciones dilatadas ($d=2$) y convoluciones anisotrópicas $(1, 3, 3)$ aportó una representación espacial única que difiere significativamente de las arquitecturas clásicas `AnisotropicUNet3D` y `ResUNet3D`.
2. **Rol en el Ensamble Multimodelo:**
   * Su diversidad estructural provee una señal de probabilidad complementaria para ensamblar con los modelos principales.

---

## 🏁 4. Conclusión & Acción
- **Decisión:** **APROBADO E INTEGRADO AL POOL DE ENSAMBLE**.
- **Checkpoint:** `/home/alumno1/Documents/Biohub/checkpoints/exp020_inception3d_80_20.pt` guardado en `A1560-UBU`.
