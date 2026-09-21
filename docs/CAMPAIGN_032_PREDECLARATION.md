# Predeclaración Oficial — Campaña 032: Distance-Dominant Keeper Selection & Spatial Proximity Invariant (DDKS-SPI)

**Fecha**: 2026-09-14  
**Responsable**: Antigravity (Protocolo de Continuidad Autónoma / Contrato de Operación y Continuidad de Biohub)  
**Código Hash Predeclarado**: `8ff0e718a26f257be1929017788679ef2925920ae3be311a275a427543b11c1c`  
**Nodo de Ejecución Autorizado**: `A1555`  
**Perfil de Recursos**: `cpu_evaluation_v1` (2 hilos CPU, 8 GiB límite espacio virtual, CUDA oculta)

---

## 1. Identidad y Punto de Partida

- **Campaña/fecha/responsable**: Campaña 032 / 2026-09-14 / Antigravity (operación autónoma).
- **Hipótesis**: Selección de Rama por Dominancia de Proximidad Espacial (DDKS) e Invariante de Proximidad Espacial (SPI) con Protección Biológica Dual Previa.
- **Evidencia Causal Directa**:
  - En la Campaña 030 (panel de confirmación, 8 datasets nunca expuestos), el score aumentó netamente de $0.952252 \to 0.952844$ ($\Delta = +0.000592 > 0$), pero el gate estricto de confirmación ($+0.00200$) falló por una sola arista verdadera perdida en `44b6_2a2eff9f` ($TP_{\text{edge}}: 200 \to 199$).
  - La Campaña 031 falsó la hipótesis de que las bifurcaciones espurias son espolones transitorios ($L \le 2$). Más del 50% de las ramas de bifurcaciones espurias tienen longitudes persistentes ($L > 10$, mediana 14–17 frames), porque representan cruces erróneos con trayectos de otras células reales.
  - La inspección forense en `44b6_2a2eff9f` reveló que de las 22 bifurcaciones podadas en 029, **12 de ellas sufrieron una inversión patológica**:
    Una de las hijas era una célula biológica estacionaria situada a $d \le 0.9$ µm (con persistencia de hasta 51 frames), mientras que la otra hija había saltado a $d \ge 5.0$–$6.5$ µm.
    La función lineal de puntuación de 029:
    $$\text{score}_i = -\frac{d_{v_i}}{10.0} + 1.0 \cdot \text{align}_i + 0.6 \cdot \text{head}_i$$
    penalizó el salto de 6 µm con solo $-0.6$ puntos, mientras que la célula saltadora recibió hasta $+1.0$ punto por alineamiento cinemático con la deriva del fluido. La célula saltadora obtuvo una puntuación mayor que la célula estacionaria de 0.9 µm (cuyo movimiento de difusión Browniana no tiene velocidad alineada unidireccional), provocando la poda de la célula estacionaria verdadera de 51 frames y la pérdida de la arista.
- **Corrección de Arquitectura de Protección-Primero**:
  - Toda bifurcación candidata se somete primero a la guarda de protección biológica dual ($L_1 \ge 2, L_2 \ge 2, \text{head}_1 > 0.3, \text{head}_2 > 0.3$). Si ambas ramas son trayectos persistentes y coherentes hacia adelante, **ninguna se poda** (preserva al 100% divisiones biológicas genuinas como la madre 834 en `6bba_48816121`).
  - Solo si la bifurcación no cumple la protección dual, se aplica la selección de proximidad espacial (SPI/DDKS), asegurando que la célula contigua real sea retenida incondicionalmente sobre el cruce saltador.
- **Run fuente**: `b0970-selector-corpus-011` (desarrollo) y `b0970-prepare-005` (confirmación).
- **Nodo de ejecución**: `A1555` (RTX 4090 ociosa, 29.2 GiB RAM, 168 GiB disco libre).

---

## 2. Hipótesis Falsable y Formulación Matemática

### Ley Biofísica de Movimiento Browniano / Difusión
En medio citoplasmático/viscoso, el desplazamiento de una célula en el intervalo $\Delta t$ sigue una distribución de difusión gaussiana:
$$P(\Delta \vec{x}) \propto \exp\left(-\frac{\|\Delta \vec{x}\|^2}{2\sigma^2}\right)$$
Para una escala de desplazamiento típico $\sigma \approx 1.0$ µm:
- Para $d_1 = 0.9$ µm: $P(d_1) \propto \exp(-0.405) \approx 0.667$.
- Para $d_2 = 6.0$ µm: $P(d_2) \propto \exp(-18.0) \approx 1.5 \times 10^{-8}$.
La probabilidad física de que $d_1$ sea la continuación biológica es **44 millones de veces superior** a la de $d_2$. Ningún alineamiento de velocidad de arrastre (+1.0) puede justificar biológicamente elegir un salto de 6 µm sobre una célula contigua de 0.9 µm.

### Reglas de Decisión Predeclaradas:
1. **Guarda de Protección Biológica Dual (Prioritaria)**:
   Si $\cos\theta \le \cos_{\text{max}}$ y $L_1 \ge 2, L_2 \ge 2, \text{head}_1 > 0.3, \text{head}_2 > 0.3$, la bifurcación se conserva íntegra (`continue`).
2. **Invariante de Proximidad Espacial (SPI)**:
   Ante asimetría espacial marcada ($\min(d_1, d_2) \le 1.5$ µm y $\max(d_1, d_2) \ge 2.5$ µm, o razón $d_{\text{far}} / d_{\text{close}} \ge 2.0$ con $d_{\text{far}} \ge 2.5$ µm):
   $$\text{keeper} = \arg\min(d_1, d_2) \quad \text{(incondicionalmente)}$$
   $$\text{prune\_candidate} = \arg\max(d_1, d_2)$$
3. **Costo Cuadrático de Distancia (QDC)**:
   Para bifurcaciones intermedias que no activan el umbral estricto de SPI, el costo lineal $(-d/10.0)$ se reemplaza por:
   $$\text{cost}_{\text{dist}}(d) = -1.0 \times \left(\frac{d}{d_0}\right)^2, \quad d_0 = 2.0\,\mu\text{m}$$
   lo que penaliza $d=5.0$ µm con $-6.25$ puntos, impidiendo cualquier anulación por alineamiento cinemático.

---

## 3. Variantes Predeclaradas

- **Variante 1 (DDKS-SPI-V1 - Control Verificado)**:
  - Invariante Duro de Proximidad Espacial (SPI): si $\min(d) \le 1.5\,\mu\text{m}$ y $\max(d) \ge 2.5\,\mu\text{m}$ (o ratio $\ge 2.0$), el keeper es estrictamente la hija más cercana.
  - Guarda biológica dual en todos los ángulos ($\cos_{\text{max}} = 1.0$).
- **Variante 2 (DDKS-SPI-Cytokinesis-V2 - Campeona Cytokinesis-Directed)**:
  - Guarda de protección biológica dual restringida a ángulos genuinos de citocinesis ($\cos\theta \le 0.00$).
  - Las bifurcaciones agudas no biológicas ($\cos\theta > 0$) entre células paralelas contiguas no quedan blindadas y son podadas por SPI / dominancia de proximidad.
- **Variante 3 (DDKS-SPI-Mild-V3)**:
  - Guarda de protección biológica dual con umbral intermedio ($\cos\theta \le 0.20$), permitiendo ligeras desviaciones de perpendicularidad mientras poda cruces agudos.
  - Costo cuadrático de distancia para horquillas intermedias.

---

## 4. Gates y Protocolo de Verificación

- **Desarrollo**:
  - $\Delta \text{score}_{\text{dev}} \ge +0.00300$
  - Cero regresiones de grupo: $\Delta \text{score}_{44\text{b}6} \ge 0.0$, $\Delta \text{score}_{6\text{bba}} \ge 0.0$
  - Peor delta individual $\ge -0.01500$
- **Confirmación** (si supera desarrollo):
  - $\Delta \text{score}_{\text{conf}} \ge +0.00200$
  - Cero regresiones de grupo
  - Peor delta individual $\ge -0.01500$
- **Invariantes**:
  - Generación de candidatos 100% libre de etiquetas (label-free).
  - Ground Truth se accede exclusivamente durante la función de evaluación pareada.
  - In-degree $\le 1$, out-degree $\le 2$ preservados en todo el grafo.
