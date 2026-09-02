# EXP-028: 3D Wide Attention U-Net (64 Canales + Spatial Attention Gates) en Esquema 90/10

- **Fecha de Inicio**: 2026-08-26
- **Estado**: 🟡 Listo para Lanzar (Nodo `A1560-UBU` - RTX 4090)
- **Autor / Agente**: Antigravity Autonomous Flywheel

---

## 1. Hipótesis Científica
En embriones biológicos 3D, más del 98% de los vóxeles corresponden a líquido extracelular y medio de fondo oscuro que no contiene células. Las compuertas de atención espacial 3D (**Spatial Attention Gates**) filtran y atenúan selectivamente las activaciones de fondo antes de fusionar las conexiones de salto (*skip connections*) en el decoder:
$$\alpha = \sigma(\psi(W_g * g + W_x * x))$$
$$\hat{x} = \alpha \odot x$$
Esto permite concentrar el gradiente exclusivamente en los núcleos celulares y las mitosis, eliminando picos falsos y mejorando drásticamente la precisión del rastreo temporal.

---

## 2. Configuración Experimental
- **Arquitectura**: `AttentionUNet3D` (64 canales base, 3 niveles de Attention Gates 3D con convoluciones $1\times 1\times 1$).
- **Esquema de Datos**: 90% Entrenamiento (179 volúmenes / 17,021 parches 3D) y 10% Holdout (20 volúmenes).
- **Función de Pérdida**: Focal-Weighted MSE Loss ($\alpha = 10.0$).
- **Optimizador**: AdamW ($lr = 10^{-3}$, weight decay $= 10^{-4}$) con Cosine Annealing.
- **Hardware**: GPU NVIDIA GeForce RTX 4090 24GB (Nodo `A1560-UBU`).
