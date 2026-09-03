# EXP-055 — DeepGraph Multi-Scale Temporal Window (T=5)

- **Fecha de Inicio**: 2026-09-02
- **Fecha de Cierre**: 2026-09-03
- **Nodo de Entrenamiento**: A1553 (RTX 4090 24GB)
- **Estado**: SUCCEEDED / CLOSED
- **Hipótesis Científica**:
  Ampliar la ventana temporal del modelo a 5 cuadros temporales (=5$) preservando la resolución espacial en Z (, 2, 2$) permite resolver bifurcaciones y mitóticas con mayor precisión, reduciendo drásticamente la pérdida de detección y el descarte espurio.
- **Resultados Métricos**:
  - Semilla 1: detection_loss = 0.01700, edge_loss = 0.000275
  - Semilla 2: detection_loss = 0.01669, edge_loss = 0.000276
  - Récord Histórico: Récord absoluto de mínima pérdida de detección del proyecto (0.01669).
- **Destino de Artefactos**:
  - Checkpoints sincronizados a Hugging Face Hub (Matycaba2016/biohub-checkpoints).
  - Candidato prioritario para ensamble dual T=5 con EXP-057.
