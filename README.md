Tarea 4 - Clasificación.
# Clasificación de Escenas Acuáticas con Machine Learning

## Introducción

Este proyecto desarrolla y optimiza un modelo de machine learning para clasificar escenas del dataset LaRS (Lakes, Rivers, and Seas). El objetivo principal es predecir el tipo de escena (`scene_type`), distinguiendo entre `river_like` y `sea_like`, utilizando únicamente los metadatos descriptivos de las imágenes.

El notebook documenta un proceso completo de ciencia de datos, desde la carga y limpieza inicial, pasando por una fase de experimentación sistemática para seleccionar las características más relevantes, hasta la optimización y la interpretación del modelo final.

-----

## Dataset

Se utilizó el dataset [LaRS v1.0.0](https://lojzezust.github.io/lars-dataset/). Las características de entrada (features) se extrajeron de las anotaciones en formato JSON y consisten en etiquetas descriptivas como:

  * **`reflections`**: Nivel de reflejos en el agua.
  * **`waves`**: Estado del oleaje.
  * **`lighting`**: Condiciones de iluminación.
  * **`special_*`**: Indicadores binarios de condiciones especiales (e.g., `special_extra_dark`).

-----

## Metodología

El análisis se estructuró en varios pasos clave para asegurar la construcción de un modelo robusto y eficiente.

### 1\. Preprocesamiento de Datos

  * **Carga y Consolidación:** Se cargaron los datos de los archivos JSON (los archivos de image_annotations.json) de los conjuntos de entrenamiento, validación y prueba de la carpeta lars_v1.0.0_annotations, unificándolos en un único DataFrame de Pandas.
  * **Limpieza:** Se eliminaron filas con datos nulos para garantizar la calidad del dataset.
  * **Codificación Numérica:** Se utilizó **One-Hot Encoding** (`pd.get_dummies`) para convertir las características categóricas (como `reflections`) a un formato numérico que el modelo pudiera procesar.

### 2\. Experimentación y Selección de Características

Se llevó a cabo un proceso iterativo para encontrar el conjunto de características óptimo:

1.  **Modelo Completo:** Se entrenó un modelo inicial con todas las características disponibles.

2.  **Análisis con SHAP:** Se utilizó la librería `SHAP` (`TreeExplainer`) para analizar la importancia y el impacto de cada característica. Este análisis reveló que la categoría `lighting` tenía una contribución casi nula.

3.  **Modelo Simplificado:** Se entrenó un segundo modelo eliminando la característica `lighting`. La comparación de métricas demostró que este modelo no solo era más simple, sino que tenía un **rendimiento superior**.

4.  **Modelo Ultra-Simplificado:** Se probó un tercer modelo usando solo las 5-6 características más importantes. Aunque eficiente, su rendimiento fue ligeramente inferior, confirmando que el **Modelo Simplificado** representaba el punto óptimo.

### 3\. Optimización del Modelo

El algoritmo `RandomForestClassifier` fue seleccionado para la optimización final. Se utilizó `GridSearchCV` para realizar una búsqueda exhaustiva de los mejores hiperparámetros (`n_estimators` y `max_depth`), asegurando el máximo rendimiento del modelo ganador.

-----

##  Resultados y Conclusiones

El modelo final seleccionado es un `RandomForestClassifier` optimizado, entrenado sobre el conjunto de datos simplificado (sin la característica `lighting`).

  * **Rendimiento:** El modelo alcanzó un **Accuracy del 78.5%**, demostrando una mejora notable en la capacidad de clasificar correctamente la clase minoritaria (`river_like`) en comparación con el modelo completo.
  * **Interpretación:** El análisis con SHAP reveló que el modelo basa sus decisiones en una lógica coherente:
      * La **ausencia de reflejos** (`reflections_none`) y el agua en calma (`waves_still`) son los indicadores más fuertes para predecir un río.
      * Una **escena muy oscura** (`special_extra_dark`) y los reflejos fuertes (`reflections_heavy`) son los principales indicadores de un mar.

Este proyecto demuestra que una cuidadosa selección de características, guiada por herramientas de interpretabilidad como SHAP, es crucial para construir modelos que no solo son precisos, sino también eficientes y robustos.

-----

## Cómo Ejecutar

Para replicar este análisis, sigue estos pasos:

1.  Clona el repositorio:

    ```bash
    git clone [URL-DE-TU-REPOSITORIO]
    cd [NOMBRE-DE-TU-REPOSITORIO]
    ```

2.  Asegúrate de tener las librerías necesarias instaladas. Puedes instalarlas usando pip:

    ```bash
    pip install pandas scikit-learn matplotlib seaborn shap
    ```

3.  Ejecuta el Jupyter Notebook `Clasification_dataset_LaRS.ipynb`. Asegúrate de que los datos del dataset LaRS estén en la ruta correcta especificada en el notebook.
