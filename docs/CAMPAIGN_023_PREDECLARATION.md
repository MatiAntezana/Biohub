# Predeclaración Formal — Campaña 023: Regularización Axial Pura y Poda Geométrica de Horquillas Espurias de Citocinesis (Rama E + M)

**Fecha**: 2026-09-14  
**Diseñador y Operador**: Antigravity  
**Estado**: Predeclarada antes de la ejecución  
**Nodo de Cómputo Autorizado**: `A1555`  
**Perfil de Recursos**: `cpu_evaluation_v1` (CUDA oculta, 2 threads CPU, address space 8 GiB)  
**Corrida Base / Fuente**: `b0970-selector-corpus-011` (SHA256 validado)  
**Corrida de Tests Previa Requerida**: `b0970-tests-023-rel1`  
**Identificador de Corrida Asignado**: `b0970-temporal-023`  

---

## 1. Contexto Científico y Diagnóstico de la Campaña 022

La Campaña 022 (`b0970-temporal-022`) demostró con exactitud que:
1. La regularización axial simétrica pura ($w_z = 0.80$, clamp $= 0.6\,\mu\text{m}$, umbral $= 0.8\,\mu\text{m}$) proporciona una ganancia limpia de **$+0.001788$** ($0.901274 \to 0.903063$) con **cero regresiones en todos y cada uno de los 8 datasets de desarrollo**.
2. Las guardias de densidad espacial y choque de cuadro eliminaron por completo las regresiones históricas en `6bba_09961292` y `44b6_d5e7d891`.
3. Sin embargo, la regularización de coordenadas espaciales sobre trayectorias existentes está saturada: el score de asociación de bordes ajustado ya se encuentra en $J_{\text{adj\_edge}} \approx 0.8900$.
4. El cuello de botella determinante del score global es el **score de división** ($J_{\text{div}} = 0.130435$):
   $$\text{Score} = J_{\text{adj\_edge}} + 0.1 \times J_{\text{div}}$$
   En los 8 datasets de desarrollo:
   - $\text{TP}_{\text{div}} = 3$
   - $\text{FP}_{\text{div}} = 8$
   - $\text{FN}_{\text{div}} = 12$
   - $J_{\text{div}} = \frac{3}{3 + 8 + 12} = \mathbf{0.130435}$
   - Contribución al score global: solo $0.1 \times 0.130435 = \mathbf{0.013044}$.

### Descubrimiento Forense en el Grafo de Referencia
El grafo de referencia de `b0970-selector-corpus-011` contiene **591 horquillas predichas** across los 8 datasets.
De estas 591 horquillas, **135 horquillas tienen coseno de divergencia positivo ($\cos(\theta) > 0.00$)**!
En citocinesis celular eucariota, el huso mitótico y los microtúbulos astrales arrastran las cromátidas hermanas hacia polos opuestos a lo largo del eje de división. Una horquilla donde ambas células hijas se desplazan en el mismo hemisferio ($\cos(\theta) \ge 0$) es biológicamente imposible: representa un falso enlace entre células independientes que viajan en la misma dirección.
Además:
- En `44b6_ddf577ad` hay 0 divisiones en Ground Truth, pero el grafo predice 95 horquillas (24 de ellas con $\cos > 0$), produciendo 3 divisiones falsas positivas evaluables.
- En `44b6_e57ff5c6` hay 0 divisiones en Ground Truth, pero el grafo predice 57 horquillas (18 con $\cos > 0$), produciendo 1 división falsa positiva evaluable.
- En `6bba_09961292` hay 2 divisiones falsas positivas evaluables.
- **Ninguna de las divisiones verdaderas recuperadas en Ground Truth tiene $\cos(\theta) > -0.40$** (todas tienen $\cos \le -0.483$, confirmando su fuerte collinearidad opuesta).

---

## 2. Mecanismo Físico de la Campaña 023

La Campaña 023 implementa una arquitectura híbrida de **Regularización Axial Simétrica + Poda Geométrica de Citocinesis**:

### Mecanismo 1: Regularización Axial Simétrica de Enlaces Lineales
- Preserva la regularización de rebotes ópticos monocuadro en Z con inversión de signo idéntica ($z_p = z_s$):
  $$z_t^* = z_t - w_z \cdot \Delta z_p, \quad \text{clamp } 0.6\,\mu\text{m}$$
- Mantiene la ganancia base demostrada de $+0.001788$ y la inmunidad a regresiones.

### Mecanismo 2: Poda Geométrica de Horquillas Espurias de Citocinesis
Para cada horquilla predicha $m \to (d_1, d_2)$ con madre $m$ a tiempo $t$ e hijas $d_1, d_2$ a tiempo $t+1$:
1. Vectores de desplazamiento físico:
   $$\vec{v}_1 = (d_1 - m) \odot \text{scale}, \quad \vec{v}_2 = (d_2 - m) \odot \text{scale}$$
2. Coseno del ángulo de divergencia entre hijas:
   $$\cos(\theta) = \frac{\vec{v}_1 \cdot \vec{v}_2}{\max(\|\vec{v}_1\| \cdot \|\vec{v}_2\|, 10^{-6})}$$
3. Asimetría de distancia de hijas: $\text{asymm} = |\|\vec{v}_1\| - \|\vec{v}_2\||$.
4. Distancia física entre hermanas: $\text{dist}_{\text{sister}} = \|\vec{v}_1 - \vec{v}_2\|$.
5. **Criterio de Poda**: Si la horquilla viola los límites de citocinesis biológica ($\cos(\theta) > \text{threshold}$ o $\text{asymm} > \text{max\_asymm}$):
   - La horquilla se clasifica como bifurcación espuria / intrusión tracking.
   - Se determina cuál de las dos hijas es la continuación legítima de la trayectoria previa mediante inercia direccional:
     Si la madre $m$ tiene predecesor $p$ a $t-1$ ($\vec{u}_{\text{in}} = (m - p) \odot \text{scale}$):
     $$\text{align}_k = \frac{\vec{u}_{\text{in}} \cdot \vec{v}_k}{\|\vec{u}_{\text{in}}\| \cdot \|\vec{v}_k\|}, \quad S_k = \text{align}_k - \frac{\|\vec{v}_k\|}{12\,\mu\text{m}}$$
     Se conserva la arista a la hija con mayor $S_k$, y se poda la arista a la hija intrusa.
     Si la madre no tiene predecesor, se conserva la hija con menor distancia euclídea $\|\vec{v}_k\|$.
   - La hija intrusa podada pasa a ser inicio de trayectoria independiente ($d_2 \to \dots$).

---

## 3. Variantes Predeclaradas

| Variante | Nombre | Regularización Z | Coseno Umbral ($\cos >$) | Max Asimetría | Min Distancia Hermanas | Criterio de Selección |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Control** | `b0970-selector-corpus-011` | Ninguna | N/A | N/A | N/A | Referencia congelada V19 |
| **Variante 1** | Conservadora | $w_z=0.80, \text{cl}=0.6$ | $> 0.20$ ($< 78^\circ$) | Desactivado | Desactivado | Inercia Direccional $S_k$ |
| **Variante 2** | Hemisférica Cero | $w_z=0.80, \text{cl}=0.6$ | $> 0.00$ ($< 90^\circ$) | Desactivado | Desactivado | Inercia Direccional $S_k$ |
| **Variante 3** | Citocinesis Calibrada | $w_z=0.80, \text{cl}=0.6$ | $> -0.10$ ($< 96^\circ$) | $> 8.0\,\mu\text{m}$ | $< 3.0\,\mu\text{m}$ | Inercia Direccional + Similitud |

---

## 4. Invariantes y Gates de Decisión

1. **Invariante de Etiquetas**: Totalmente label-free durante inferencia y propuesta. Ground Truth prohibido durante la ejecución de la poda y regularización. Se abre exclusivamente en la evaluación métrica oficial final.
2. **Invariante Topológica**:
   - $\text{in\_degree} \le 1$ para todo nodo (la poda solo reduce el in-degree de la arista eliminada de 1 a 0).
   - $\text{out\_degree} \le 2$ para todo nodo (la madre pasa de 2 a 1).
   - Sin bucles dirigidos ni aristas hacia atrás en el tiempo.
3. **Cota de Filas**: Conteo de nodos estrictamente invariante en 235,674 filas ($234{,}000 < \text{filas} < 242{,}000$).
4. **Gates de Desarrollo (Configs/protocol.json)**:
   - $\Delta \text{score}_{\text{dev}} \ge +0.00300$
   - $\Delta \text{score}_{44\text{b}6} \ge 0.00000$ (cero regresiones en 44b6)
   - $\Delta \text{score}_{6\text{bba}} \ge 0.00000$ (cero regresiones en 6bba)
   - Cero regresiones individuales por dataset.
5. **Criterio de Promoción y Kaggle**:
   - Si una variante supera el gate de desarrollo $\ge +0.00300$ con cero regresiones, se habilita inmediatamente el empaquetado del bundle con `ops/bundle_build.py` y submission privada a Kaggle con `ops/kaggle.py`.
   - Si ninguna variante alcanza $+0.00300$, se preservan los artefactos y se avanza hacia la siguiente hipótesis autónoma (Campaña 024: Acoplamiento de Citocinesis Aditiva para División FN).
