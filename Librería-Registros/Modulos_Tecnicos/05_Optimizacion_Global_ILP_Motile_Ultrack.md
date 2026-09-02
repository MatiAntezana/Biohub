---
tags: [modulo, ilp, motile, ultrack, optimizacion]
---

# 🧠 Módulo 05: Optimización Global por ILP (Motile & Ultrack)

## 🎯 Formulación Matemática del ILP
En lugar de tomar decisiones voraces (greedy) frame a frame, se construye un grafo espacio-temporal global $G = (V, E)$ y se resuelve el problema de Programación Lineal Entera:

$$\min \sum_{u \in V} c_u x_u + \sum_{(u,v) \in E} c_{uv} y_{uv} + \sum_{u \in V} c_u^{	ext{div}} d_u$$

Sujeto a:
1. **Conservación de Flujo de Entrada:** $\sum_{p \in 	ext{in}(u)} y_{pu} \le x_u \quad orall u \in V$ (a lo sumo 1 célula padre).
2. **Conservación de Flujo de Salida:** $\sum_{v \in 	ext{out}(u)} y_{uv} \le 2 x_u \quad orall u \in V$ (a lo sumo 2 células hijas).
3. **Restricción de División:** Si hay 2 aristas salientes activas, se activa la variable binaria de división $d_u = 1$.
4. **Variables Binarias:** $x_u, y_{uv}, d_u \in \{0, 1\}$.

---

## 🛠️ Herramientas
- **`motile`** (Funke Lab): Framework de Python para formular y resolver ILP en tracking con solucionadores de código abierto (ILP solvers como SCIP / HiGHS).
- **`ultrack`** (Royer Lab / Biohub): Pipeline completo que une segmentaciones candidatas jerárquicas con ILP global.
