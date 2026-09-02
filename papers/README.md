# Research Papers: 3D Cell Tracking & Lineage Reconstruction

Esta carpeta contiene los papers científicos oficiales y de acceso abierto fundamentales para resolver la competencia **Biohub - Cell Tracking During Development** (detección, seguimiento temporal y reconstrucción de linajes celulares en embriones de pez cebra en 4D).

---

## Índice de Papers Descargados

| # | Archivo | Título / Autores | Publicación / DOI | Enfoque Principal |
|---|---|---|---|---|
| 01 | [01_Ultrack_Large_Scale_Cell_Tracking_Developing_Organisms.pdf](./01_Ultrack_Large_Scale_Cell_Tracking_Developing_Organisms.pdf) | **Ultrack: Large-scale cell tracking in developing organisms using overlapping segmentations**<br>*Jordão Bragantini, Prachee Avasthi, Loïc A. Royer* | Nature Methods / bioRxiv<br>`10.1101/2023.08.08.551978` | **Método oficial de Biohub**. Segmentaciones superpuestas candidatas + formulación ILP para resolver segmentación y tracking conjunto. |
| 02 | [02_CVPR2020_MPM_Joint_Representation_Motion_Position_Map.pdf](./02_CVPR2020_MPM_Joint_Representation_Motion_Position_Map.pdf) | **MPM: Joint Representation of Motion and Position Map for Cell Tracking**<br>*Junya Hayashida, Kazuya Nishimura, Ryoma Bise* | CVPR 2020 (IEEE/CVF) | Estimación conjunta de mapa de posiciones de centroides y vectores de movimiento hacia adelante/atrás para matching temporal. |
| 03 | [03_Trackastra_Tracking_Association_Transformers.pdf](./03_Trackastra_Tracking_Association_Transformers.pdf) | **Trackastra: Tracking by Association with Transformers**<br>*Benjamin Gallusser, Martin Weigert* | ECCV 2024 / bioRxiv<br>`2405.19794` | Asociación temporal aprendida mediante Transformers sobre ventanas temporales (detección frame a frame + grafo de enlaces). |
| 04 | [04_ByoTrack_Unified_Framework_Biological_Object_Tracking.pdf](./04_ByoTrack_Unified_Framework_Biological_Object_Tracking.pdf) | **ByoTrack: A Unified Framework for Biological Object Tracking**<br>*Raphaël Reme, Anatole Chessel, Jean-Christophe Olivo-Marin, Thibault Lagache* | IEEE TPAMI / arXiv<br>`2303.07807` | Framework unificado de tracking biológico y optimización global de trayectorias. |
| 05 | [05_LapTrack_Linear_Assignment_Particle_Tracking.pdf](./05_LapTrack_Linear_Assignment_Particle_Tracking.pdf) | **LapTrack: Linear Assignment Particle Tracking with Versatile Feature Spaces**<br>*Yutaro Fukai, Kyogo Kawaguchi* | Bioinformatics 2023<br>`10.1093/bioinformatics/btad107` | Tracking rápido por Linear Assignment Problem (LAP) que soporta división celular, gaps temporales y distancias multidimensionales. |
| 06 | [06_EmbedTrack_Cell_Segmentation_Tracking_Embeddings.pdf](./06_EmbedTrack_Cell_Segmentation_Tracking_Embeddings.pdf) | **EmbedTrack: Simultaneous Cell Segmentation and Tracking Through Learning Embeddings**<br>*Katharina Löffler, Ralf Mikut* | IEEE Transactions on Medical Imaging<br>`2103.03615` | Aprendizaje de embeddings métricos espacio-temporales para clusterizar instancias y propagar identidades a lo largo del tiempo. |
| 07 | [07_Conservation_Tracking_ILP_Global_Optimization.pdf](./07_Conservation_Tracking_ILP_Global_Optimization.pdf) | **Conservation Tracking: Global Optimization for Cell Tracking with Lineage Divisions**<br>*Florian Jug, Evgeny Levinkov, Bjoern Andres, Gene Myers* | ICCV | Formulación de Programación Lineal Entera (ILP) con conservación de flujo para modelar trayectorias y eventos de mitosis. |
| 08 | [08_StarDist3D_Cell_Detection_Polyhedra.pdf](./08_StarDist3D_Cell_Detection_Polyhedra.pdf) | **StarDist 3D: Cell Detection in 3D Microscopy with Star-Convex Polyhedra**<br>*Martin Weigert, Uwe Schmidt, et al.* | WACV (IEEE) | Detección 3D robusta de objetos estrellados convexos (ideal para núcleos/células esféricas u ovoides de pez cebra). |
| 09 | [09_Motile_Multi_Object_Tracker_ILP.pdf](./09_Motile_Multi_Object_Tracker_ILP.pdf) | **Motile: Multi-Object Tracker with Integer Linear Programming**<br>*Funke Lab / Janelia Research Campus* | bioRxiv / arXiv<br>`2212.03061` | Motor de optimización ILP modular y escalable para grafos espacio-temporales en microscopía. |
| 10 | [10_Hayashida_Cell_Division_Tracking_MICCAI.pdf](./10_Hayashida_Cell_Division_Tracking_MICCAI.pdf) | **Cell Tracking with Cell Division Detection Using Motion and Position Map**<br>*Junya Hayashida, Ryoma Bise* | MICCAI | Detección explícita de eventos de bifurcación/división mitótica combinada con mapas de movimiento. |
| 11 | [11_ELEKTRONN3_3D_CNN_Segmentation.pdf](./11_ELEKTRONN3_3D_CNN_Segmentation.pdf) | **ELEKTRONN3: 3D CNNs for Dense Cellular Segmentation**<br>*Martin Drawitsch et al.* | arXiv<br>`2006.01258` | Arquitecturas 3D U-Net y estrategias de entrenamiento volumétrico eficiente en GPU. |
| 12 | [12_Cellpose_Cell_Segmentation.pdf](./12_Cellpose_Cell_Segmentation.pdf) | **Cellpose: A generalist algorithm for cellular segmentation**<br>*Carsen Stringer, Tim Wang, Michalis Michaelos, Marius Pachitariu* | Nature Methods | Predicción de mapas de flujo de gradiente para separar células contiguas y densas sin sobrefusionar. |

---

## Síntesis Estratégica para la Solución del Problema

El pipeline competitivo estándar se divide en 3 etapas complementarias:

### 1. Detección y Segmentación 3D (Frame a Frame)
- **Modelos:** 3D U-Net, StarDist 3D o representación de mapas de calor de centroides (Heatmaps Gaussianos / MPM).
- **Entrada:** Volúmenes $Z \times Y \times X = 64 \times 256 \times 256$ descomprimidos con `blosc2`.
- **Salida:** Centroides de células $(t, z, y, x)$ escalados a unidades físicas ($Z=1.625, Y=0.40625, X=0.40625\,\mu\text{m}$).

### 2. Asociación Temporal y Detección de Mitosis
- **Enfoque Heurístico / LAP (Baseline rápido):** `LapTrack` o Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) con cota de distancia física ($\le 15\,\mu\text{m}$).
- **Enfoque Basado en Aprendizaje:** `Trackastra` (Transformers sobre candidatos) o `MPM` (vectores de flujo hacia adelante y atrás).
- **Detección de Mitosis:** Identificar nodos padre con 2 nodos hijos en $t+1$ que minimicen la distancia angular y la suma de distancias Euclideanass.

### 3. Optimización Global del Grafo de Linaje (ILP / DAG)
- **Herramientas:** `motile` o `ultrack` para resolver un Integer Linear Program:
  $$\min \sum_{e \in E} c_e x_e + \sum_{v \in V} c_v y_v$$
  sujeto a restricciones de conservación de flujo (grado de entrada $\le 1$, grado de salida $\le 2$).
