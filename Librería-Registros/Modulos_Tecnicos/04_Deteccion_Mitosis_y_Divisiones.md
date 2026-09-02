---
tags: [modulo, mitosis, divisiones]
---

# 🧬 Módulo 04: Detección de Mitosis y Divisiones Celulares

## 🎯 Objetivo
Identificar con alta precisión los eventos de bifurcación biológica ($1 	ext{ célula madre} 	o 2 	ext{ células hijas}$) para maximizar el `division_jaccard`.

> [!IMPORTANT]
> En la métrica oficial:
> $$	ext{Score} = 	ext{adjusted\_edge\_jaccard} + 0.1 	imes 	ext{division\_jaccard}$$
> Un modelo sin detector de mitosis pierde hasta $0.10$ puntos directos en el Leaderboard.

---

## 🔍 Criterios Geométricos y Temporales de Mitosis
1. **Conservación de Volumen/Intensidad:** La célula madre antes de dividirse suele ser más brillante/grande; las dos hijas en $t+1$ suman una masa similar.
2. **Divergencia Angular Simétrica:** Las células hijas se separan en direcciones aproximadamente opuestas ($	heta pprox 180^\circ$ en su eje de clivaje).
3. **Proximidad Espacial:** Ambas hijas nacen a muy corta distancia ($< 6\,\mu	ext{m}$) del centroide de la madre en $t$.
4. **Validación de Linaje Posterior:** Ambas ramas hijas deben continuar existiendo durante al menos $K$ frames subsiguientes (no desaparecer inmediatamente).
