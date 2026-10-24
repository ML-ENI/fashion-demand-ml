# Enunciado del trabajo de Machine Learning — Predicción de demanda de moda (H&M)

**Asignaturas:** PDRL y MLLB  
**Proyecto:** Fashion Demand ML  
**Fuente de datos:** H&M Personalized Fashion Recommendations  
**Alcance:** regresión, clasificación y aprendizaje no supervisado sobre un caso de estudio común.

> **Cómo interpretar este documento.** Se mantiene el orden de las ocho carpetas del ejemplo OSA mostrado por el profesor, pero **el caso que debemos desarrollar es nuestro proyecto de moda**, no la predicción de apnea. Los materiales docentes aportados explican el flujo general, la necesidad de justificar objetivos e hipótesis y la metodología de EDA, modelos y experimentación. Las propuestas concretas para H&M, especialmente en validación anidada, importancia de variables y aprendizaje no supervisado, son una **adaptación metodológica**; no constituyen una transcripción literal de las instrucciones de esas carpetas, cuyo contenido no se ha proporcionado.

## Objetivo general y planteamiento

Desarrollar un sistema de Machine Learning capaz de estudiar y anticipar la **demanda semanal de artículos de moda de H&M** a partir de su historial de transacciones y de los metadatos de producto. El proyecto tendrá tres tareas conectadas:

1. **Regresión:** estimar cuántas unidades de un artículo se venderán en la semana siguiente.
2. **Clasificación:** predecir si un artículo tendrá **demanda alta** o **no alta** en la semana siguiente.
3. **Aprendizaje no supervisado:** identificar grupos de artículos con características y comportamientos comerciales similares.

La unidad de análisis será **artículo-semana (`article-week`)**. Para cada artículo y semana `t`, las variables predictoras se construirán exclusivamente con información conocida **hasta el momento de predicción**, y la variable objetivo supervisada será la demanda de la semana `t+1`. El umbral que define «demanda alta» se justificará a partir de los datos y de los objetivos del problema, sin fijarlo arbitrariamente de antemano.

---

## Organización de notebooks por fase

Cada fase tendrá un notebook principal, salvo el EDA, que se divide en dos para estudiar por separado las variables objetivo de regresión y clasificación. Se utilizará una numeración fija para que el orden de ejecución sea evidente.

| Fase de Moodle | Notebook propuesto | Resultado principal |
|---|---|---|
| 1. OSA Data | `01_data_understanding.ipynb` | Inventario de datos, relaciones, cobertura y calidad inicial |
| 2. DATA PREPARATION | `02_data_preparation.ipynb` | Dataset `article-week`, características históricas y objetivos |
| 3. EDA Regression & Classification | `03_eda_regression.ipynb` y `04_eda_classification.ipynb` | Análisis exploratorio de ambas tareas |
| 4. REGRESSION | `05_regression.ipynb` | Baselines y modelos de regresión comparados |
| 5. CLASSIFICATION | `06_classification.ipynb` | Baselines y modelos de clasificación comparados |
| 6. Hyper-Parameters – NESTED CROSSVALIDATION | `07_nested_temporal_cv.ipynb` | Búsqueda y evaluación temporal de hiperparámetros |
| 7. Single Tree and RF FEATURE IMPORTANCE | `08_tree_rf_feature_importance.ipynb` | Interpretación de árboles e importancia de variables |
| 8. UNSUPERVISED | `09_unsupervised.ipynb` | Clustering y caracterización de segmentos |

**Ruta recomendada:** guardar los nueve notebooks en `notebooks/`, o mantener las subcarpetas `notebooks/eda/` y `notebooks/experiments/` del repositorio si se prefiere. Los nombres son **una propuesta organizativa para nuestro proyecto**, no nombres exigidos literalmente por Moodle. Los notebooks deben importar funciones reutilizables desde `src/` cuando proceda, en lugar de duplicar bloques extensos de código.

## 1. OSA Data → Datos del proyecto de moda

**Finalidad:** comprender el problema, formular hipótesis y estudiar las fuentes de datos antes de modelar.

- Plantear la motivación: prever demanda puede ayudar a priorizar inventario, reposición y planificación comercial. Precisar que el dataset permite estimar **ventas observadas**, que no equivalen necesariamente a demanda real si existen roturas de stock u otras limitaciones.
- Definir el objetivo de cada una de las tres tareas y formular hipótesis comprobables. Ejemplos: la demanda reciente contiene información predictiva; la categoría y los atributos del producto aportan señal adicional; existen perfiles de artículos con distintas trayectorias de venta.
- Describir los archivos principales de H&M:

  | Archivo | Contenido | Papel en el proyecto |
  |---|---|---|
  | `transactions_train.csv` | Fecha (`t_dat`), artículo (`article_id`), comprador, precio y canal | Reconstruir las ventas y variables históricas |
  | `articles.csv` | Tipo, grupo, sección, color y demás metadatos del artículo | Añadir atributos del producto |
  | `customers.csv` | Datos del cliente, como edad y estado de membresía | Fuente **opcional** para agregados históricos |

- Explicar las claves de unión, granularidad, cobertura temporal, unidades, limitaciones y riesgos de sesgo. Distinguir claramente el objetivo original de la competición H&M (recomendación personalizada) de **nuestro objetivo** (predicción agregada de ventas por artículo-semana).
- Revisar bibliografía o soluciones relacionadas con forecasting de demanda, datos comerciales y aprendizaje supervisado/no supervisado.

**Notebook de la fase:** `01_data_understanding.ipynb` — cargar muestras de los CSV, comprobar esquema, granularidad, cobertura temporal, claves de unión y estadísticas iniciales; presentar las hipótesis del proyecto en Markdown.

**Entregable de la fase:** definición del problema, hipótesis, inventario de fuentes, esquema de datos y limitaciones iniciales.

## 2. DATA PREPARATION → Preparación de datos

**Finalidad:** construir una tabla reproducible adecuada para modelar las tres tareas.

1. Leer y validar las fuentes: columnas, tipos, fechas, identificadores, duplicados, valores ausentes, precios anómalos y coherencia de los registros.
2. Agregar las transacciones por **artículo y semana**. Definir de manera explícita qué se cuenta como venta y cómo se gestionan las semanas sin registros: un cero solo representa «sin ventas observadas» cuando el artículo pertenece al universo de observación pertinente; no se debe inventar disponibilidad o stock.
3. Generar variables históricas: ventas de semanas anteriores (*lags*), medias y variabilidad móviles, tendencia, semanas con actividad, clientes únicos, precios históricos, canales y variables de calendario, según la información realmente disponible.
4. Incorporar los metadatos de `articles.csv` y, opcionalmente, agregados de `customers.csv` calculados solo con transacciones pasadas.
5. Crear la variable de **regresión** (`next_week_demand`) desplazando las ventas semanales una semana hacia el futuro.
6. Derivar la variable de **clasificación** (demanda alta/no alta) mediante un criterio **explícito y justificable**. Si el umbral se estima a partir de la distribución, hacerlo únicamente con entrenamiento y mantenerlo fijo para validación y test.
7. Preparar las variables que se usarán en **clustering**; excluir el objetivo futuro y las variables que provocarían fuga temporal. No es necesario usar exactamente las mismas transformaciones que en los modelos supervisados.
8. Definir particiones **cronológicas** de entrenamiento, validación y prueba. Reservar el test final para una única evaluación; ajustar codificación, escalado, imputación y selección de variables exclusivamente con datos permitidos en cada partición.

**Atención:** en este problema **no** se debe realizar un `train_test_split` aleatorio como opción principal. Cada característica ha de poder calcularse en la fecha en la que supuestamente se predice el futuro.

**Notebook de la fase:** `02_data_preparation.ipynb` — ejecutar y verificar el ETL, generar la tabla artículo-semana, las variables históricas y los objetivos, y exportar datasets procesados con los cortes temporales documentados.

**Entregable de la fase:** pipeline ETL/feature engineering, tabla `article-week` y descripción de las particiones temporales, con comprobaciones explícitas de *data leakage*.

## 3. EDA Regression & Classification → Análisis exploratorio

**Finalidad:** comprender los patrones relevantes para ambas tareas antes de entrenar modelos.

### 3.1. EDA general

Estudiar la cantidad de artículos, transacciones y semanas, la evolución temporal de las ventas, la concentración de la demanda, la frecuencia de semanas sin ventas y la composición por categorías, secciones y canales. Examinar anomalías, datos ausentes, cambios temporales y posibles problemas de cobertura.

### 3.2. EDA orientado a regresión

Analizar la distribución de `next_week_demand`, su asimetría y dispersión. Comparar la demanda futura con las ventas anteriores, tendencias, medias móviles, precio y características del artículo. Utilizar gráficos temporales, histogramas, dispersión, boxplots y correlaciones (por ejemplo, Pearson/Spearman cuando correspondan). Estudiar redundancia y multicolinealidad entre predictores, pero sin asumir que una correlación baja descarta automáticamente utilidad predictiva.

### 3.3. EDA orientado a clasificación

Estudiar la proporción de artículos-semana con demanda alta/no alta, la distribución de las variables por clase y su evolución temporal. Identificar posible desequilibrio de clases, diferencias por categoría y cambios entre períodos de entrenamiento y prueba. Justificar el umbral de demanda alta y comprobar qué implicaciones tiene para el problema.

**Notebook de la fase:** `03_eda_regression.ipynb` — analizar distribución y evolución de ventas, lags, relaciones entre variables y objetivo continuo. `04_eda_classification.ipynb` — estudiar el umbral de demanda alta, balance de clases y diferencias entre grupos y períodos. El EDA general se puede incluir al comienzo del primero y reutilizar sus resultados en el segundo.

**Entregable de la fase:** notebooks y gráficos comentados; conclusiones concretas que fundamenten modelos, transformaciones y métricas, sin atribuir causalidad a relaciones meramente observacionales.

## 4. REGRESSION → Predicción de ventas de la próxima semana

**Finalidad:** predecir un valor numérico: unidades vendidas por artículo en la semana siguiente.

- Establecer una referencia sencilla (*baseline*), por ejemplo predecir con ventas recientes o con una media histórica calculada sin información futura.
- Implementar y comparar varios modelos adecuados, razonando su elección: regresión lineal regularizada, árboles de decisión, Random Forest y métodos de boosting, según recursos y viabilidad.
- Entrenar con información del pasado y evaluar sobre intervalos posteriores. Utilizar validación temporal para comparar configuraciones y seleccionar hiperparámetros.
- Medir el error mediante **MAE** y **RMSE**, con **R²** como información complementaria cuando proceda. Explicar la interpretación de las unidades y comparar frente al baseline.
- Analizar errores en artículos con demanda baja/alta, diferentes categorías y períodos. Comprobar si el modelo produce valores imposibles (por ejemplo, predicciones negativas) y cómo se gestionan.

**Notebook de la fase:** `05_regression.ipynb` — entrenar baselines y regresores, comparar métricas por ventana temporal y analizar errores y predicciones.

**Entregable de la fase:** modelos y baseline, tabla comparativa de métricas, gráficos de predicción/error y análisis crítico del mejor enfoque.

## 5. CLASSIFICATION → Predicción de demanda alta/no alta

**Finalidad:** anticipar si la demanda de la próxima semana superará el criterio definido.

- Usar la **misma unidad artículo-semana** y las mismas reglas de disponibilidad temporal que en regresión.
- Construir un baseline (por ejemplo, clase mayoritaria o regla basada en demanda pasada) y comparar varios modelos: regresión logística, árbol, Random Forest y boosting, entre otros si están justificados.
- Separar claramente el umbral de **definición de clase** del umbral de **decisión del clasificador**; este último puede ajustarse con validación, nunca con test.
- Evaluar mediante **matriz de confusión**, **precision**, **recall** y **F1**; añadir **ROC-AUC** y, en caso de desbalance, **PR-AUC** cuando sea pertinente. Interpretar falsos positivos y negativos en términos de planificación comercial.
- Comprobar la estabilidad de resultados por períodos y categorías. Evitar usar `next_week_demand` ni ninguna transformación directa de la variable objetivo como predictor.

**Notebook de la fase:** `06_classification.ipynb` — entrenar baselines y clasificadores, producir matriz de confusión, precision/recall/F1 y curvas pertinentes.

**Entregable de la fase:** comparación de clasificadores, métricas y curvas pertinentes, matriz de confusión y discusión del coste de los errores.

## 6. Hyper-Parameters selection – NESTED CROSSVALIDATION → Selección rigurosa de hiperparámetros

**Finalidad:** comparar configuraciones de manera objetiva evitando estimaciones optimistas de rendimiento.

- Distinguir los datos empleados para **ajustar hiperparámetros** de los destinados a **estimar rendimiento**.
- Adaptar la idea de **validación cruzada anidada** al problema temporal, utilizando ventanas cronológicas: bucle interno para búsqueda de hiperparámetros y bucle externo para estimación fuera de muestra. No mezclar semanas futuras con entrenamiento de semanas anteriores.
- Elegir un espacio de búsqueda razonable (por ejemplo, profundidad de árbol, hojas mínimas, número de estimadores, regularización y tasa de aprendizaje, según modelo) y justificar el presupuesto computacional.
- Ejecutar la búsqueda en pipelines que incluyan las transformaciones aprendidas. Comparar resultados entre ventanas y mostrar dispersión, no solo el mejor valor.
- Una vez cerrado el diseño, entrenar la configuración elegida con el período de desarrollo permitido y evaluar sobre el **test temporal final**, aún no utilizado.

**Notebook de la fase:** `07_nested_temporal_cv.ipynb` — definir ventanas temporales internas/externas, ejecutar la búsqueda reproducible de hiperparámetros y resumir resultados para regresión y clasificación (secciones independientes dentro del notebook).

**Entregable de la fase:** diseño del esquema temporal/anidado, espacios de hiperparámetros, tablas de resultados por ventana, configuración seleccionada y evaluación independiente.

> **Aclaración metodológica:** esta sección adapta el nombre de la carpeta docente a un problema de forecasting; la validación cruzada aleatoria convencional no sería adecuada.

## 7. Single Tree and RF FEATURE IMPORTANCE → Interpretabilidad e importancia de variables

**Finalidad:** comprender qué señales usan los modelos y hasta qué punto son estables e interpretables.

- Entrenar un **árbol individual** suficientemente simple para inspeccionar reglas, divisiones y comportamiento. Representar sus decisiones sin confundir asociaciones predictivas con causas.
- Analizar un **Random Forest** y comparar su rendimiento y comportamiento con el árbol simple.
- Obtener medidas de importancia de variables, señalando las limitaciones de la importancia basada en impureza (sesgo por cardinalidad, correlación entre predictores, etc.).
- Contrastar, cuando resulte útil, con **permutation importance** calculada en un conjunto temporal de validación/test permitido y con una métrica claramente especificada.
- Comparar señales como ventas pasadas, tendencias, precio y atributos de producto. Estudiar si la importancia cambia entre regresión y clasificación o entre períodos.

**Notebook de la fase:** `08_tree_rf_feature_importance.ipynb` — visualizar un árbol simple, comparar Random Forest y calcular e interpretar importancia de variables para las dos tareas supervisadas.

**Entregable de la fase:** visualización del árbol, resultados del Random Forest, gráficos de importancia e interpretación cautelosa de las variables relevantes.

## 8. UNSUPERVISED → Segmentación de artículos de moda

**Finalidad:** descubrir estructura descriptiva en los artículos sin utilizar etiquetas futuras.

- Decidir qué se quiere agrupar: preferentemente **artículos** caracterizados por sus atributos y estadísticas de comportamiento comercial calculadas en una ventana histórica definida; documentar la unidad de clustering.
- Construir una representación que combine, de forma razonada, metadatos de producto, nivel de ventas, regularidad, tendencia, variabilidad, precio y otras señales históricas disponibles.
- Tratar escalas y variables categóricas de manera apropiada; razonar si procede aplicar **PCA** para exploración/reducción de dimensión.
- Probar métodos de agrupamiento como **K-Means** y clustering jerárquico o **DBSCAN**, siempre que sean técnicamente pertinentes para los datos y su tamaño.
- Elegir configuraciones mediante criterios adecuados (por ejemplo, silhouette y estabilidad, interpretándolos con cautela) y visualizar los grupos.
- Describir cada segmento en términos de categorías de producto y comportamiento: artículos de venta regular, de alta variabilidad, de demanda concentrada, etc., **solo si esos patrones aparecen realmente en los resultados**.
- Relacionar críticamente los segmentos encontrados con las tareas de predicción, sin incorporar el objetivo futuro a la construcción de los grupos.

**Notebook de la fase:** `09_unsupervised.ipynb` — construir descriptores históricos por artículo, probar reducción de dimensión y clustering, evaluar y perfilar grupos.

**Entregable de la fase:** representación utilizada, métodos comparados, análisis de calidad/estabilidad, visualizaciones y perfiles interpretados de los segmentos.

---

## Integración de los resultados y documentación final

Las tres tareas deben formar **un solo caso de estudio coherente** y compartir, siempre que sea razonable, las fuentes, definiciones y variables históricas. El informe global debe exponer problema e hipótesis, EDA, diseño experimental, modelos, refinamiento iterativo, limitaciones y conclusiones basadas en resultados ejecutados. Conviene registrar también librerías, versiones, semillas, cortes temporales, configuraciones e historial de experimentos para facilitar la reproducción.

### Diferencia entre los dos informes

| Informe | Enfoque principal |
|---|---|
| **PDRL** | Motivación científica, formulación de regresión/clasificación/clustering, fundamentos de modelos, hipótesis, justificación de métricas y validación, limitaciones e interpretación crítica. |
| **MLLB** | Implementación real: lectura de CSV, ETL, generación de `article-week`, notebooks EDA, pipelines, entrenamiento, ajuste de hiperparámetros, evaluación, visualizaciones, registro de experimentos y reproducibilidad. |

No es necesario repetir los mismos párrafos en ambos documentos. Deben describir **el mismo trabajo desde dos perspectivas complementarias**. Se documentará también el uso de asistentes de IA cuando corresponda, explicando su función y qué decisiones se verificaron personalmente.

### Lista de comprobación de entrega

- [ ] Problema, unidad artículo-semana, objetivos e hipótesis claramente definidos.
- [ ] Los nueve notebooks propuestos existen, están numerados y se pueden ejecutar en el orden indicado.
- [ ] Datos documentados y proceso de preparación reproducible.
- [ ] Fuga temporal prevenida y particiones cronológicas justificadas.
- [ ] EDA de regresión y clasificación con conclusiones interpretadas.
- [ ] Regresión comparada con baseline y evaluada con métricas apropiadas.
- [ ] Clasificación comparada con baseline, clases justificadas y errores analizados.
- [ ] Selección de hiperparámetros y validación temporal/anidada documentadas.
- [ ] Árbol simple, Random Forest e importancia de variables analizados.
- [ ] Aprendizaje no supervisado realizado e interpretado.
- [ ] Resultados, limitaciones, conclusiones, código y reproducibilidad documentados.
- [ ] Distinción clara entre contenidos de PDRL y MLLB.

## Referencias utilizadas para construir este enunciado

- Material docente aportado: `MLLB_OSA_CaseStudy_Intro.pptx`, `MLLB_1_EDA_Regression_Classification.pptx`, `MLLB_2_EDA_Regression.pptx`, `MLLB_1_prompting_Classification.pptx` y captura de las ocho carpetas de Moodle.
- Definición del proyecto del repositorio: `Fashion Demand ML` (`README.md`).
- Convenciones comunes de los informes: `DESIGN_RUBRIC.md`.

**Nota:** las carpetas de Moodle sirven como **secuencia de actividades**; los conceptos OSA/IAH, variables clínicas y clasificación healthy/severe del ejemplo original **no forman parte del proyecto de moda**.
