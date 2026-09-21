# Predeclaración — Campaña 024: Poda Asimétrica de Mitosis y Reconexión Antiparalela con Persistencia Temporal (Rama E)

**Fecha**: 2026-09-14  
**Responsable**: Antigravity  
**Fase de Trabajo**: Rama E (Continuaciones y Contexto Temporal)  
**Run Fuente**: `b0970-selector-corpus-011` en `A1555`  
**Nodo de Ejecución**: `A1555` (RTX 4090 ociosa, 29.2 GiB RAM, 168 GiB disco, carga 0.08)  
**Nodo de Pruebas**: `A1557` (343/343 tests auditados y aprobados)  
**Perfil de Recursos**: `cpu_evaluation_v1` (2 hilos CPU, aislamiento de espacio de direcciones 8 GiB, CUDA oculta, sin GPU)

---

## 1. Identidad y Punto de Partida

- **Hipótesis**: Rama E (Continuaciones y Contexto Temporal).
- **Evidencia Previa (Campaña 023)**:
  - La poda de bifurcaciones espurias que violan la colinealidad de citocinesis ($\cos > 0.20$) superó el Development Gate formal con **$+0.003456$** (Variante 1) y **$+0.004007$** (Variante 2).
  - La Variante 1 obtuvo **cero regresiones en todos y cada uno de los 8 datasets de desarrollo**.
  - Los falsos positivos de división se redujeron de 8 a 5 en la Variante 2.
- **Cuello de Botella Medido en Campaña 023**:
  1. Aún persisten 5 falsos positivos de división en el grafo predicho. La auditoría geométrica demuestra que muchas bifurcaciones falsas poseen extrema asimetría de distancia ($\text{ratio} = \min / \max < 0.15$), donde la célula madre simplemente avanzó $0.4\,\mu\text{m}$ y una célula vecina que transitaba a $6\,\mu\text{m}$ fue conectada erróneamente como segunda hija.
  2. En el panel de desarrollo, existen **12 divisiones verdaderas no recuperadas (False Negatives)**. En al menos 2 casos confirmados (`6bba_07e24132` y `6bba_48816121`), la división es un caso puro de `additive_second_child`: la madre ya está unida a una de las hijas, mientras que la segunda hija es el inicio huérfano de una trayectoria persistente ($>30$ cuadros) situada en dirección estrictamente antiparalela ($\cos \le -0.72$).

---

## 2. Hipótesis Falsable y Mecanismo Biológico

### Mecanismo 1: Poda de Bifurcaciones por Asimetría Extrema y Guarda de Persistencia
En la citocinesis eucariota normal, el plano de clivaje separa la célula en dos masas citoplasmáticas comparables; los cinetocoros y centrosomas se mueven a velocidades y distancias aproximadamente simétricas del punto de división original:
$$\text{ratio} = \frac{\min(\|\vec{v}_1\|, \|\vec{v}_2\|)}{\max(\|\vec{v}_1\|, \|\vec{v}_2\|)} \ge 0.20$$
Cuando $\text{ratio} < 0.15$ y $\max > 3.0\,\mu\text{m}$, la supuesta "segunda hija" es un transeúnte espacial. Podar la arista hacia la hija lejana y retener la hija cercana restituye la trayectoria unilineal correcta y suprime el falso positivo de división. Además, en casos limítrofes, la hija que forma parte de una trayectoria de mayor persistencia temporal ($\text{forward\_len}$) debe ser priorizada como el verdadero linaje continuo.

### Mecanismo 2: Reconexión Antiparalela de Citocinesis con Filtro de Persistencia y Morfología 3D
Para madres con un solo enlace saliente ($m \to d_1$), una célula huérfana $d_2$ ($\text{in-degree} = 0$) en $t+1$ es una verdadera hija mitótica si y solo si:
1. Presenta divergencia colineal antiparalela estricta: $\cos(\vec{v}_1, \vec{v}_2) \le -0.85$ ($\theta \ge 148^\circ$).
2. Posee distancias balanceadas: $\text{ratio} \ge 0.35$.
3. Ambas distancias físicas son $\le 8.0\,\mu\text{m}$ y la distancia entre hermanas es $\le 12.0\,\mu\text{m}$.
4. Ambas células pertenecen a linajes maduros y persistentes: la madre tiene al menos 10 cuadros de historia previa ($\text{backward\_len}(m) \ge 10$) y la hija huérfana se extiende por al menos 10 cuadros futuros ($\text{forward\_len}(d_2) \ge 10$).
5. Presentan alta correlación en sus descriptores de parche 3D ($\text{corr}(desc_{d_1}, desc_{d_2}) \ge 0.40$).

### Criterio de Falsación
- La hipótesis será rechazada si:
  1. La poda asimétrica remueve divisiones verdaderas o deteriora el Jaccard ajustado de aristas en algún dataset.
  2. La reconexión aditiva introduce más falsos positivos de división que los verdaderos positivos que recupera, provocando una caída neta en el score de desarrollo o regresión en algún grupo.

---

## 3. Variantes Predeclaradas

- **Control**: Referencia intacta de `b0970-selector-corpus-011` (Score oficial: $0.901274$).
- **Variante 1: Poda de Colinealidad + Asimetría Extrema (Pruning Puro Conservador)**:
  - Regularizador axial simétrico ($w_z = 0.80$, clamp $\le 0.6\,\mu\text{m}$, guardias de reversión y simetría).
  - Poda de bifurcaciones con $\cos(\theta) > 0.20$.
  - Poda adicional de bifurcaciones con $\text{ratio} < 0.15$ y $\max(\|\vec{v}_1\|, \|\vec{v}_2\|) > 3.0\,\mu\text{m}$.
  - Selección de preservada: La hija más cercana es retenida; la arista hacia la hija distante se remueve.
- **Variante 2: Poda Estricta con Guarda de Persistencia Temporal**:
  - Regularizador axial simétrico.
  - Poda de bifurcaciones con $\cos(\theta) > 0.10$.
  - Poda de bifurcaciones con $\text{ratio} < 0.20$ y $\max > 3.5\,\mu\text{m}$.
  - Selección de preservada con guarda de persistencia: Si una hija persiste al menos 5 cuadros más que la otra, es retenida prioritariamente; en caso contrario, se aplica la combinación de alineamiento y correlación morfológica.
- **Variante 3: Poda Variante 1 + Reconexión Antiparalela Ultra-Confiable de Citocinesis**:
  - Aplica todas las reglas de la Variante 1.
  - Añade aristas $(m, d_2)$ para madres solteras ($m \to d_1$) y huérfanas $d_2$ que cumplan simultáneamente:
    - $\cos(\vec{v}_1, \vec{v}_2) \le -0.85$
    - $\text{ratio} \ge 0.35$
    - $\max(d_1, d_2) \le 8.0\,\mu\text{m}$ y sister distance $\le 12.0\,\mu\text{m}$
    - $\text{backward\_len}(m) \ge 10$ y $\text{forward\_len}(d_2) \ge 10$
    - $\text{corr}(desc_{d_1}, desc_{d_2}) \ge 0.40$

---

## 4. Gates y Reglas de Decisión

1. **Development Gate**:
   - $\Delta \text{score}_{\text{dev}} \ge +0.00300$.
   - $\Delta \text{edge\_jaccard} \ge 0.00000$.
   - $\Delta \text{division\_jaccard} \ge 0.00000$.
   - Cero regresiones por grupo ($\Delta \text{score}_{44\text{b}6} \ge 0$, $\Delta \text{score}_{6\text{bba}} \ge 0$).
   - Ningún dataset individual con $\Delta \text{adj\_edge\_jaccard} < -0.01500$.
2. **Criterio de Selección de Mejor Variante**:
   - Entre las variantes que pasen todos los gates sin regresión en ningún dataset individual ($\Delta \ge 0$), se seleccionará la de mayor $\Delta \text{score}_{\text{dev}}$.
3. **Disposición**:
   - Si una variante supera todos los criterios: `eligible_for_confirmation_and_package`.
   - Si ninguna variante supera el gate: `rejected_no_kaggle_submission`.
