# Reporte Formal de Campaña 039 — Calibrated Cytokinesis-Directed Guard (CCDG)

## 1. Resumen Ejecutivo

- **Campaña**: 039
- **Fecha**: 2026-09-15
- **Responsable**: Antigravity (Protocolo de Continuidad Autónoma / Contrato de Operación y Continuidad de Biohub)
- **Hipótesis**: Calibración angular de la compuerta de citocinesis a $\cos_{3d} \le +0.20$ con persistencia $\ge 2$ para otorgar inmunidad a divisiones verdaderas inclinadas por flujo tisular (shear drift) mientras se podan cruces paralelos espurios ($\cos > 0.20$).
- **Veredicto de Desarrollo**: **APROBADO** ($\Delta = \mathbf{+0.003150} \ge +0.003$, score $0.904424$ vs ref $0.901274$).
- **Veredicto de Confirmación**: **RECHAZADA** ($\Delta = +0.001644 < +0.002000$, score $0.953897$ vs ref $0.952252$).
- **Campeona de Confirmación Vigente**: **Campaña 032 Variante 2 ($0.955029$)**.

---

## 2. Métricas Clave de Evaluación

### Desarrollo (`b0970-temporal-039-rel1`)
| Métrica | Referencia | Campaña 039 (V1) | Delta |
| :--- | :---: | :---: | :---: |
| **Score Macro** | 0.901274 | **0.904424** | **+0.003150** |
| Division TP | 3 | 3 | 0 |
| Division FP | 8 | 6 | -2 |
| Division FN | 12 | 12 | 0 |
| Edge Jaccard (44b6) | 0.834231 | 0.838541 | +0.004310 |
| Edge Jaccard (6bba) | 0.918834 | 0.919292 | +0.000458 |
| Bifurcaciones Podadas | - | 97 (68 SPI) | - |

### Confirmación (`b0970-confirm-039-rel1`)
| Métrica | Referencia (`prepare-005`) | Campaña 032 V2 | Campaña 039 (V1) | Delta vs Ref |
| :--- | :---: | :---: | :---: | :---: |
| **Score Macro** | 0.952252 | **0.955029** | 0.953897 | +0.001644 |
| Division TP | 4 | 4 | **4 (100% protegida)** | 0 |
| Division FP | 7 | 7 | **5 (-2 FP)** | -2 |
| Division FN | 12 | 12 | 12 | 0 |
| Edge Jaccard | 0.931604 | 0.931758 | 0.931591 | -0.000013 |
| Adj Edge Jaccard | 0.934861 | 0.935105 | 0.934849 | -0.000012 |

---

## 3. Hallazgos Biofísicos y Causa Raíz de Rechazo

1. **Éxito en la Protección de Citocinesis Real**:
   - En la Campaña 038, la compuerta estricta $\cos \le 0.00$ podó erróneamente la división real de `44b6_587a1e22` debido a la deriva de cizalla tisular ($\cos \approx +0.08$).
   - En la Campaña 039, la calibración a $\cos \le +0.20$ **protegió con 100% de éxito dicha división**, manteniendo $\text{division\_tp} = 4/4$ en confirmación.
2. **Causa del Déficit en Confirmación frente a 032**:
   - En `44b6_d2f34f90`, 032 podó con éxito la bifurcación espuria de la madre 245, cuya hija 2 saltaba una distancia unphysica de $4.143\,\mu\text{m}$ (con $\cos = +0.0476$).
   - Al bajar el umbral de persistencia a $\text{persistence\_min\_frames} = 2$ e incorporar $\cos \le +0.20$, la bifurcación 245 quedó inmunizada en 039, impidiendo la poda de esa arista FP.
   - En consecuencia, en `44b6_d2f34f90` el score fue de $0.99712$ en 039 vs $1.00062$ en 032 (-0.00350), lo que impidió que la ganancia macro alcanzara el umbral de $\Delta \ge +0.002000$.

---

## 4. Estado y Próximos Pasos

- Campaña 032 Variante 2 (`b0970-confirm-032-rel5`, score **0.955029**) se mantiene como la **Campeona de Confirmación vigente**.
- Todos los receipts y artefactos de la Campaña 039 han sido recuperados y preservados bajo `artifacts/A1555/b0970-temporal-039-rel1` y `artifacts/A1555/b0970-confirm-039-rel1`.
- La submission `56244048` de la Campaña 032 sigue en evaluación en Kaggle.
