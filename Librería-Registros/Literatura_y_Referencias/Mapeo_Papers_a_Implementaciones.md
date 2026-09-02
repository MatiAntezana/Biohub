---
tags: [literatura, referencias, papers]
---

# 📚 Mapeo de Papers Científicos a Implementaciones del Pipeline

Esta nota conecta directamente los 12 papers descargados en `../papers/` con las tareas concretas de nuestro repositorio de experimentos.

---

## 🧬 1. Métodos Oficiales y del Host (Biohub / Royer Lab)
- **Paper:** `01_Ultrack_Large_Scale_Cell_Tracking_Developing_Organisms.pdf`
- **Autores:** Jordão Bragantini, Prachee Avasthi, Loïc A. Royer (*Chan Zuckerberg Biohub*).
- **Código Oficial:** `https://github.com/royerlab/ultrack`
- **Cómo usarlo en la competencia:** Es el baseline SOTA diseñado específicamente por los organizadores para este tipo de datos de pez cebra. Modela candidatos de segmentación superpuestos y resuelve la asignación global con un solver ILP.

---

## 🎯 2. Detección y Estimación de Movimiento Conjunta (MPM)
- **Paper:** `02_CVPR2020_MPM_Joint_Representation_Motion_Position_Map.pdf` (CVPR 2020)
- **Autores:** Junya Hayashida, Kazuya Nishimura, Ryoma Bise.
- **Paper Complementario:** `10_Hayashida_Cell_Division_Tracking_MICCAI.pdf` (MICCAI)
- **Cómo usarlo:** Entrenar una red 3D para predecir conjuntamente mapas de posición de centroides y vectores de desplazamiento temporal hacia atrás y adelante. Facilita el matching en zonas densas y la detección de bifurcaciones de mitosis.

---

## 🤖 3. Modelos de Asociación Basados en Transformers
- **Paper:** `03_Trackastra_Tracking_Association_Transformers.pdf` (ECCV 2024 / Weigert Lab)
- **Código:** `https://github.com/weigertlab/trackastra`
- **Cómo usarlo:** Permite desacoplar el detector 3D del tracker. Dado un conjunto de centroides o máscaras candidatas en una ventana temporal, Trackastra predice las probabilidades de aristas del grafo mediante atención espacial y temporal.

---

## ⚙️ 4. Optimización Global e Integer Linear Programming (ILP)
- **Papers:**
  - `07_Conservation_Tracking_ILP_Global_Optimization.pdf` (ICCV)
  - `09_Motile_Multi_Object_Tracker_ILP.pdf` (Funke Lab / Janelia)
- **Código:** `https://github.com/funkelab/motile`
- **Cómo usarlo:** Garantiza restricciones biológicas duras (no cruce de linajes, a lo sumo 2 hijas por mitosis, conservación de flujo) encontrando el óptimo global del grafo DAG.

---

## 🔬 5. Segmentación Celular 3D
- **Papers:**
  - `08_StarDist3D_Cell_Detection_Polyhedra.pdf` (WACV)
  - `12_Cellpose_Cell_Segmentation.pdf` (Nature Methods)
  - `11_ELEKTRONN3_3D_CNN_Segmentation.pdf`
- **Cómo usarlo:** Generar máscaras y centroides iniciales robustos para alimentar a los módulos de tracking.
