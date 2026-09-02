---
tags: [validacion, cv, split]
---

# 🧪 Setup de Validación Cruzada Local (CV)

## ⚠️ Prevención de Data Leakage
Los organizadores especificaron que **Train y Test son disjuntos a nivel de embrión** (`embryo-disjoint`).
- La nomenclatura de carpetas es `{embryo_id}_{field_of_view}` (ej: `44b6_0049_0438_1330_1273`).
- **REGLA ABSOLUTA:** Ningún embrión debe estar presente simultáneamente en el split de entrenamiento y en el de validación. Usar `GroupKFold(groups=embryo_ids)` o Leave-One-Embryo-Out.

---

## 📁 Estructura del Script de Evaluación Local
Crear un módulo `src/evaluation.py` que:
1. Cargue el archivo `sample_submission.csv` o predicción local en memoria.
2. Lea los archivos `.geff` correspondientes del split de validación.
3. Ejecute la asignación bipartita (matching a $7.0\,\mu	ext{m}$) y calcule `Edge Jaccard` + `Division Jaccard`.
4. Devuelva un reporte detallado de TP, FP, FN por muestra y score total global.
