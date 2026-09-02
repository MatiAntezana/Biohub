---
id: EXP-008
tags: [experimento, resnet3d, se_channel_attention, split_80_20, autopsia]
fecha: 2026-08-25
autor: Matias & Antigravity AI
score_cv: 0.0000
score_lb: null
split: "80% / 20% (159 Train / 40 Val vols)"
estado: Descartado
decision: DESCARTADO - REEMPLAZADO POR EXP-009 (AnisotropicUNet3D con Concatenación en Split 80/20)
---

# 🧪 EXP-008: 3D SE-ResNet-UNet (Esquema 80/20)

## 🎯 1. Hipótesis & Estrategia
- **Hipótesis:** Probar el **Esquema 80% Train (159 vols) / 20% Val (40 vols)** utilizando bloques residuales con Squeeze-and-Excitation y saltos residuales directos en el decoder.
- **Configuración:** 15,117 parches 3D entrenados durante 20 épocas en RTX 4090.

---

## 🔬 2. Resultados
- **Score CV:** `0.0000` (Pesos del modelo divergieron a `NaN`).
- **Causa Raíz:** En el decoder, la instrucción `d3 = self.dec3(self.up3(b) + e3)` sumó tensores sin proyección ni normalización, provocando una acumulación de gradientes que desestabilizó la red hacia la época final.

---

## 🏁 3. Conclusión & Acción
- **Lección:** Las conexiones U-Net **deben usar concatenación (`torch.cat([up, skip], dim=1)`)**, que es la que garantizó el récord de **`0.6700`** en `EXP-005`.
- **Nuevo Experimento Inmediato (`EXP-009`):** Entrenar la arquitectura probada `AnisotropicUNet3D` (con concatenación pura) bajo el **Esquema 80% Train / 20% Val**.
