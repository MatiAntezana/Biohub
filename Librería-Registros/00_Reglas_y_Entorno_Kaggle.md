---
tags: [kaggle, infraestructura, reglas, restricciones]
updated: 2026-08-25
autor: Matias & Antigravity AI
---

# 📜 Guía de Infraestructura y Reglas de Envío en Kaggle

## ⚠️ 1. Restricción Estricta de Hardware (P100 Prohibida)

> [!CAUTION]
> **Regla de la Competencia Biohub:**
> *"Your Notebook cannot use P100 GPUs in this competition."*
> Los organizadores bloquean cualquier intento de submission oficial si el notebook fue ejecutado con una GPU Tesla P100.

---

## ⚙️ 2. Aceleradores Permitidos y Configuración Óptima

| Acelerador | Configuración `kernel-metadata.json` | Tiempo de Inferencia | Consumo de Cuota | Estado de Compatibilidad |
|---|---|:---:|:---:|:---:|
| **CPU Multi-Hilo** | `"enable_gpu": false, "enable_tpu": false` | $\approx 3\text{--}4\text{ min}$ | **0 Horas (Ilimitado)** | 🟢 **100% Permitido (Recomendado)** |
| **GPU Tesla T4x2** | `"enable_gpu": true` *(con selección T4)* | $\approx 2\text{ min}$ | 30 Horas/semana | 🟢 **Permitido** |
| **TPU v3-8** | `"enable_tpu": true` | $\approx 3\text{ min}$ | 20 Horas/semana | 🟢 **Permitido** |
| **GPU Tesla P100** | Por defecto en GPU Kaggle | Fallback lento | Cuota GPU | 🔴 **PROHIBIDO POR REGLAS** |

---

## 🛠️ 3. Buenas Prácticas de Inferencia en Producción

1. **Gestión de Memoria en CPU/GPU:**
   * Utilizar `torch.inference_mode()`.
   * Liberar memoria tras cada frame: `del vol, heat; gc.collect()`.
   * Mantener el consumo de RAM constante por debajo de 1.5 GB.
2. **Procedimiento de Envío al Leaderboard:**
   * Empaquetar pesos champion en dataset adjunto (`/kaggle/input/biohub-champion-weights/`).
   * Ejecutar notebook y verificar `KernelWorkerStatus.COMPLETE`.
   * En el modal de submission seleccionar la versión generada con CPU / T4 y confirmar `submission.csv`.
