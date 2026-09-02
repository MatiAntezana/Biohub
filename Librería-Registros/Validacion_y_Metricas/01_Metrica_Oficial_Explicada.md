---
tags: [validacion, metrica, formula]
---

# 📐 Métrica Oficial de Evaluación Explicada

## 🏆 Fórmula Combinada
$$	ext{Score} = 	ext{Adjusted\_Edge\_Jaccard} + 0.1 	imes 	ext{Division\_Jaccard}$$

---

### 1. Adjusted Edge Jaccard
- **Matching de Nodos:** Por cada timepoint $t$, los nodos predichos se emparejan con los nodos de ground-truth mediante asignación bipartita óptima sobre distancia Euclidiana escalada ($\le 7.0\,\mu	ext{m}$).
- **True Positive (TP) de Arista:** Una arista predicha $(u 	o v)$ es TP si y solo si tanto $u$ como $v$ fueron emparejados con nodos de ground truth $(u^*, v^*)$ que a su vez están conectados por una arista en el ground truth.
- **Edge Jaccard:** $rac{	ext{TP}}{	ext{TP} + 	ext{FP} + 	ext{FN}}$.
- **Penalización por Sobre-predicción:** Se ajusta en base al ratio de nodos predichos vs el número estimado total de células.

---

### 2. Division Jaccard
- Una división es cualquier nodo con 2 o más aristas salientes.
- Para cada división de ground truth, se verifica si el grafo predicho tiene una componente conexa que cubra el estado pre-split y toque a ambos linajes hijos.
- Se computan TP, FP, FN micro-promediados entre todas las muestras.
