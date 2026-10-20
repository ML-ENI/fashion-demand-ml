# Revisión de preparación de datos — 20 de octubre de 2026

## Alcance y decisiones

Esta revisión corresponde a las copias locales de `fashion-demand-ml` y
`fashion-vision-dl`. Las rutas y números del informe aportado proceden de otro
ordenador; no se consideran resultados comprobados de este equipo.

El modelo de demanda se define como **ventas registradas de la semana siguiente
para artículos vendidos en la semana actual**. El panel sigue siendo disperso de
forma deliberada. No predice todo el catálogo, productos sin ventas actuales,
demanda latente ni disponibilidad. Crear otro universo requiere reglas de surtido
o actividad basadas solo en información disponible en el momento de predecir.

## Correcciones incorporadas

| Observación | Tratamiento en versión 2 |
|---|---|
| Selección de artículos con venta actual | Definición explícita en script, README, resumen y notebook. |
| Cero en t+1 ambiguo | Se define como cero registros de venta en semana cubierta; nunca como prueba de stock. |
| Semanas inicial y final parciales | Solo se modelan semanas actuales con cobertura diaria completa. |
| Objetivo en semana parcial | Solo se generan filas cuando la semana siguiente está completa. |
| Historial fuera del fichero | Se exigen doce semanas completas previas. |
| Corte con max-rows | Se descarta como completo el último día leído y se publica PREVIEW, no CURRENT. |
| Mezcla año natural/semana ISO | Se guardan iso_year e iso_week. |
| Continuidad | Se rechazan días ausentes dentro del intervalo seleccionado. |
| Partición solo en notebook | split se guarda por target_week; primero 70/15/15 de filas y después los cortes entre semanas más próximos, con parámetros versionados. |
| customer_id sin uso | Se usa para contar filas idénticas, incluso entre bloques; no se borran. |
| IDs, canales, precios, fechas | Contrato y comprobaciones por bloque; se rechazan campos obligatorios vacíos, IDs desconocidos, precios no finitos/negativos, canales inválidos y fechas fuera de rango. |
| Precisión | Agregación de sumas en float64 y comprobación con tolerancia de 1e-12. |
| Nombres de imagen repetidos | Error explícito con las dos rutas. |
| Imágenes ilegibles | Verificación, decodificación completa, tamaño y modo; exclusión con motivo. |
| Duplicados de contenido | Hash del archivo y hash de los píxeles; agrupación transitiva junto a product_code. |
| Split de imágenes no factible | Se calcula la asignación por clase y por grupo; se exige un grupo en cada partición. |
| Balance y selección | Filtros de clases y tamaño mínimo; pesos inversos calculados solo en entrenamiento. El entrenador deberá aplicar los pesos. |
| Resumen de clases | Distingue metadatos, imágenes válidas, clases utilizables, exclusiones y distribución por partición. |
| articles.csv duplicado | Se utiliza la copia canónica del repo de demanda cuando ambos repos son vecinos; --articles-path permite fijarla. |
| Contrato común | Mismo data_contract.py versionado en ambos repos, con pruebas del esquema. Debe mantenerse sincronizado. |
| Trazabilidad | SHA-256 de entradas, código y salidas; parámetros, versiones, corte y esquema. |
| Escrituras parciales | Publicación en directorio nuevo; CURRENT cambia atómicamente tras completar todos los archivos. |
| Parquet | Opción --parquet y requirements-parquet.txt. CSV sigue disponible como exportación. |
| Pruebas | Agregación entre bloques, ceros, fechas, ISO, datos inválidos, splits, imágenes corruptas/duplicadas y fallo de escritura. |

## Dos matices sobre el informe recibido

**No es obligatoria una semana de separación en este protocolo.** Las particiones
se definen por semana objetivo y se predice al terminar la semana actual. El
último objetivo de entrenamiento puede coincidir con información ya conocida en
la primera fecha de predicción de validación. Eso no introduce información futura.
`--gap-weeks 1` permite una evaluación con separación extra. La decisión se registra.
Si el objetivo fuera predecir varios meses desde una única fecha, habría que cambiar
el protocolo y las variables: este experimento es de predicción semanal sucesiva.

**Los mismos artículos pueden aparecer en periodos distintos del modelo de ventas.**
Se evalúa el futuro, no generalización a productos nuevos. En visión sí se impide
que grupos de producto o contenido idéntico crucen las particiones. Los hashes
no detectan por sí solos todas las imágenes casi idénticas ni campañas comunes.

En este ordenador las dos copias originales de articles.csv contienen 105.542 filas
y 25 columnas y son iguales al leerlas como tabla. Sus bytes no son idénticos:

- Demanda: `25f21882053517f7684ea43f524225a9481a7bb9aab9cf0abfc7fe8c4576a72b`.
- Visión: `98d50274824544672fc704258f9140a08200a44af5548dfd5f2c0e1a16d8f0d9`.

Por tanto, la afirmación del informe de que coinciden por SHA-256 no describe estas
copias locales. No se eliminan copias ni se reescriben los datos de origen.

## Limitaciones que no resuelve el código

- La presencia de registros diarios no garantiza exhaustividad del origen.
- No hay inventario, altas/bajas de catálogo ni demanda perdida por rotura de stock.
- No se puede afirmar que una transacción repetida sea un error; puede ser otra unidad.
- No están disponibles aquí las imágenes reales de H&M. La validación visual se prueba
  con imágenes pequeñas generadas para pruebas, no con todo el conjunto real.
- Generar pesos no equivale a entrenar aplicándolos. El entrenamiento sigue pendiente.
- Las proporciones de visión son aproximadas en número de imágenes, porque la unidad
  de separación son grupos indivisibles. Se reportan sus tamaños reales.
- La opción Parquet requiere instalar su dependencia adicional; la ejecución CSV
  no verifica el funcionamiento del motor Parquet.

## Uso

1. Ejecutar `python -m unittest discover -s tests -v` en cada repo.
2. Demanda: `python src/data/process_data.py`.
3. Visión: añadir las imágenes y ejecutar el mismo comando en su repo.
4. Leer `data/processed/CURRENT.json`, que apunta a la ejecución completa vigente.
5. Demanda: ejecutar el notebook `notebooks/eda/01_revision_datos.ipynb`.

Los CSV antiguos de la raíz de processed quedan como históricos: no son la salida
vigente de versión 2. No se deben mezclar con el notebook nuevo.

## Actualización: porcentajes antes que fechas

Por petición del equipo, se sustituye el corte fijo de mayo/julio por un objetivo
70/15/15 de filas. Se examinan conjuntamente todos los pares de cortes entre semanas
que dejan tres grupos no vacíos y se minimiza la suma de errores cuadráticos de sus
proporciones. Las semanas no se fragmentan y no se mezcla el orden temporal.
Si se usa gap, las proporciones se calculan sobre las filas restantes; las excluidas
se informan aparte. Las fechas dejan de ser entradas y pasan a ser resultados.
