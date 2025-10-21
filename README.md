# Proyecto de Inteligencia Artificial: Predicción de Diabetes y Costos de Seguro Médico

Este documento describe paso a paso el desarrollo del proyecto que implementa dos modelos de Machine Learning: uno para predecir la **probabilidad de diabetes** y otro para **estimar los costos asociados a seguros médicos**. Ambos fueron entrenados, optimizados y evaluados en Python utilizando **scikit-learn**.

---

##  1. Contexto del Proyecto

El propósito del proyecto fue aplicar técnicas de Inteligencia Artificial para resolver dos problemas del ámbito de la salud y economía:

1. **Predicción de diabetes:** Determinar la probabilidad de que una persona presente diabetes según características clínicas y demográficas.
2. **Predicción de costos de seguro médico:** Estimar los costos o primas médicas esperadas según factores personales y de estilo de vida.

---

##  2. Preparación y Preprocesamiento de Datos

### 2.1 Conjunto de datos de diabetes

* Dataset tipo **Pima Indians Diabetes**.
* Variables: `glucose`, `bmi`, `age`, `blood_pressure`, `insulin`, `skin_thickness`, `pregnancies`, `diabetes_pedigree_function`.
* Limpieza de valores nulos y ceros incorrectos.
* Escalado de variables numéricas con **StandardScaler**.
* División en entrenamiento y prueba (80/20).

### 2.2 Conjunto de datos de seguros médicos

* Dataset tipo **Medical Cost Personal Dataset (Kaggle)**.
* Variables: `age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`.
* Codificación categórica con **OneHotEncoder**.
* Escalado de variables numéricas.
* Transformación logarítmica de `charges` para reducir sesgo.

---

## 3. Modelado

Se entrenaron dos tipos de modelos para cada problema:

* **Diabetes:** `LogisticRegression` y `RandomForestClassifier`.
* **Seguros médicos:** `LinearRegression` y `RandomForestRegressor`.

El entrenamiento se realizó utilizando un **pipeline con ColumnTransformer** para integrar la codificación, escalado y modelo.

---

##  4. Optimización de Hiperparámetros

Se utilizó **RandomizedSearchCV** con validación cruzada (KFold o StratifiedKFold según el caso).
Parámetros ajustados:

* Para RandomForest: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`.
* Para Regresión Logística: `C`, `penalty`, `class_weight`.

> Además, en el modelo de diabetes se aplicó **class_weight='balanced'** para corregir el desbalance de clases.

---

##  5. Evaluación de Modelos

### 5.1 Modelo de Diabetes

* Métricas: **ROC-AUC**, **F1-Score**, **Precision**, **Recall**, **Brier Score**.
* Curvas ROC y calibración utilizadas para seleccionar el umbral ideal.

### 5.2 Modelo de Seguros Médicos

* Métricas: **MAE**, **RMSE**, **R²**.
* En caso de usar log-transformación, las métricas se ajustaron a la escala original.

---

##  6. Umbral Ideal para el Modelo de Diabetes

El **umbral ideal** se definió a partir del punto donde se maximiza **Youden’s J (TPR - FPR)** o el mejor **F1-Score**.
Generalmente, se encuentra entre **0.35 y 0.50**.

* Si el objetivo es **no omitir casos de diabetes**, se prefiere un umbral más bajo (≈0.35) para aumentar la sensibilidad.
* Si se busca **mayor precisión**, se eleva el umbral (≈0.50).

---

##  7. Factores que Influyen en el Costo del Seguro Médico

Según el modelo **RandomForestRegressor**, los factores más influyentes son:

1. **Ser fumador (`smoker`)** → eleva significativamente los costos.
2. **Índice de masa corporal (`bmi`)** → a mayor IMC, mayores costos.
3. **Edad (`age`)** → los costos aumentan progresivamente con la edad.
4. **Interacciones** (`smoker × bmi`, `age × bmi`).
5. **Número de hijos (`children`)** → influencia leve.
6. **Región y sexo** → factores de menor peso.

---

##  8. Comparativa entre Modelos RandomForest

| Característica | Diabetes (Clasificador RF) | Seguro Médico (Regresor RF) |
| -------------- | -------------------------- | --------------------------- |
| Edad           | Alta importancia           | Alta importancia            |
| IMC            | Alta importancia           | Alta importancia            |
| Glucosa        | Muy alta                   | No aplica                   |
| Insulina       | Media                      | No aplica                   |
| Fumador        | No aplica                  | Muy alta                    |
| Hijos          | No aplica                  | Baja-media                  |
| Región / Sexo  | No aplica                  | Baja                        |

---

##  9. Técnica de Optimización que Mejora Ambos Modelos

La técnica más eficaz fue **RandomizedSearchCV** combinada con **validación cruzada (CV)** y preprocesamiento estandarizado.
Otras mejoras:

* En **diabetes**: balance de clases, calibración de probabilidades, y ajuste de umbral.
* En **seguros**: transformación logarítmica de la variable objetivo, eliminación de outliers extremos, y optimización de profundidad del bosque.

---

## 10. Contexto de los Datos

Los datasets utilizados son de carácter **educativo y público** (Kaggle).
Provienen principalmente de estudios en **EE.UU.**, por lo tanto, **no representan necesariamente la realidad chilena**.

Este contexto implica que los modelos no deben usarse con fines clínicos ni comerciales reales, sino como ejemplo académico del uso de IA en predicción de salud y economía.

---

##  11. Análisis del Sesgo en los Modelos

### Modelo de Diabetes

* **Desbalance de clases:** la mayoría de los casos son no diabéticos.
* Riesgo de subdiagnóstico si se mantiene umbral 0.5.
* Mitigado mediante `class_weight='balanced'` y ajuste de umbral.

### Modelo de Seguro Médico

* **Sesgo demográfico:** mayor representación de ciertas regiones y edades.
* Los fumadores tienen una influencia desproporcionada sobre los resultados.

### En general

Los modelos pueden sufrir **sesgo de muestreo** y **sesgo de población**, por lo que deben validarse con datos locales antes de aplicarlos en entornos reales.

---

##  12. Estructura del Proyecto

```
├── data/
│   ├── diabetes.csv
│   └── insurance.csv
├── src/
│   ├── preprocessing.py
│   ├── train_diabetes.py
│   ├── train_insurance.py
│   ├── evaluate_models.py
│   └── api.py
├── models/
│   ├── diabetes_rf.pkl
│   ├── insurance_rf.pkl
├── notebooks/
│   ├── Exploración.ipynb
│   ├── Entrenamiento.ipynb
│   └── Evaluación.ipynb
├── requirements.txt
└── README.md
```

---

##  13. Conclusión

El proyecto demuestra cómo aplicar IA para resolver problemas reales en salud y seguros. Los resultados confirman que:

* La **predicción de diabetes** requiere calibración y ajuste de umbral para obtener sensibilidad adecuada.
* Los **costos médicos** se explican principalmente por factores de riesgo como tabaquismo, IMC y edad.
* La **optimización de hiperparámetros** y el **balance de datos** son fundamentales para mejorar el rendimiento.


