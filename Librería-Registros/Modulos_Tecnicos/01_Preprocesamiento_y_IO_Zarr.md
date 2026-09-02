---
tags: [modulo, zarr, preprocesamiento]
---

# 📦 Módulo 01: Preprocesamiento y Manejo de Zarr v3

## 📐 Especificaciones Críticas
- **Estructura:** Volúmenes 3D+t en formato Zarr v3. Array en `0/` con shape `(T, Z, Y, X) = (100, 64, 256, 256)`.
- **Tipo de dato:** `uint16`.
- **Compresión:** Chunks `(1, 64, 256, 256)` comprimidos con `blosc2` / `zstd` en `0/c/{t}/0/0/0`.
- **Escala física:** 
  $$	ext{Scale}_Z = 1.625\,\mu	ext{m/vóxel}, \quad 	ext{Scale}_Y = 0.40625\,\mu	ext{m/vóxel}, \quad 	ext{Scale}_X = 0.40625\,\mu	ext{m/vóxel}$$
  *Observación:* La resolución en Z es $4	imes$ más gruesa que en X/Y (anisotropía $4:1$).

---

## 🛠️ Técnicas Clave de Optimización de I/O
1. **Descompresión en C/Blosc2:** Leer directamente los buffers binarios con `blosc2.decompress(f.read())` en lugar de cargadores pesados.
2. **Normalización por Percentiles:** Clip entre percentil 1% y 99.8% para suprimir ruido de fondo y destellos de fluorescencia.
3. **Rescaling / Padding:** Evaluar si interpolar el eje Z a isotropía ($0.40625\,\mu	ext{m}$) o usar kernels anisotrópicos `(3, 7, 7)`.
