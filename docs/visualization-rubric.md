# Rúbrica de visualización

Esta es la fuente de verdad para cualquier gráfico del proyecto `fashion-demand-ml`. Debe leerse antes de crear, modificar, revisar o exportar una visualización.

## 1. Jerarquía visual

- Cada gráfico tiene un único título, centrado, en negrita y separado del área de datos.
- No se incluyen subtítulos dentro de la figura. La definición, metodología y contexto se explican en el texto que acompaña al gráfico.
- Los nombres de los ejes están en negrita y en gris oscuro.
- Ticks, fechas, valores, leyendas y anotaciones usan peso normal.
- Una figura tiene una sola nota de fuente, situada al pie y sin competir con los ejes.
- Se evita cualquier solapamiento entre títulos, leyendas, etiquetas, anotaciones, fuentes y paneles.

## 2. Tipografía y estilo base

- Familia principal: `Arial`.
- Fallbacks: `Liberation Sans` y `DejaVu Sans`.
- Título de gráfico: 14–16 pt, negrita.
- Etiquetas de ejes: 10 pt, negrita.
- Texto auxiliar y leyendas: 8–10 pt, peso normal.
- Fondo: blanco roto `#FAFBFC`.
- Texto principal: `#17212B`.
- Texto secundario: `#667085`.
- Rejilla: `#DDE3EA`, fina y discreta.
- Se eliminan los bordes superior y derecho cuando no aportan información.

## 3. Color

Colores semánticos:

| Uso | Color |
|---|---|
| Azul principal | `#2F6BFF` |
| Verde azulado | `#16A085` |
| Coral | `#E76F51` |
| Dorado | `#E9A23B` |
| Violeta | `#7357C8` |

Colores fijos para particiones de datos:

| Partición | Color |
|---|---|
| `train` | Morado `#6F4EAF` |
| `validation` | Naranja `#E17C05` |
| `test` | Cian `#00A6D6` |

- Esta correspondencia es obligatoria en todos los gráficos que comparen o señalen particiones, incluidas líneas, barras, puntos, bandas, fronteras temporales y leyendas.
- Estos tres colores son exclusivos de las particiones y distinguen conjuntos de datos; no representan rendimiento positivo, negativo o neutral.
- Si hace falta distinguir otra dimensión dentro de una partición, se conserva su color y se varían el estilo de línea, el marcador, la trama o la transparencia.

Paleta categórica extendida, optimizada para maximizar la distancia perceptual entre colores sobre el fondo del proyecto:

| Orden | Color | Orden | Color |
|---:|---|---:|---|
| 1 | `#2F6BFF` | 11 | `#AA0033` |
| 2 | `#00AA00` | 12 | `#999900` |
| 3 | `#FF0011` | 13 | `#0055AA` |
| 4 | `#FF00AA` | 14 | `#CC7777` |
| 5 | `#775522` | 15 | `#336600` |
| 6 | `#0099CC` | 16 | `#CC66DD` |
| 7 | `#884477` | 17 | `#993300` |
| 8 | `#008866` | 18 | `#9988CC` |
| 9 | `#DD7700` | 19 | `#FF2277` |
| 10 | `#AA11DD` | 20 | `#889955` |

- El color siempre representa una variable o resalta un hallazgo; nunca es decoración.
- Positivo: verde azulado. Negativo: coral. Referencia o baseline: gris.
- En comparaciones categóricas se usa la paleta extendida en el orden indicado y se asignan hasta veinte colores antes de repetirlos.
- Los significados semánticos de positivo, negativo y referencia prevalecen sobre el orden categórico.
- Se mantiene el mismo color para la misma categoría dentro de una figura.
- No se usan escalas arcoíris para variables continuas, degradados arbitrarios ni gráficos 3D.
- La información importante no puede depender únicamente del color.

## 4. Títulos, ejes, fechas y números

- El título comunica el hallazgo, no solo el tipo de gráfico.
- Los ejes indican magnitud y unidad: por ejemplo, `Demanda semanal`, `Precio (€)` o `MAE · menor es mejor`.
- Las fechas siguen el orden español:
  - fecha completa: `dd/mm/aaaa`;
  - mes y año: `mm/aaaa`.
- En series temporales múltiples, todos los paneles muestran valores en el eje X.
- Se incluyen suficientes referencias temporales para interpretar la evolución. Como guía, usar frecuencia trimestral en periodos de 12–24 meses; nunca dejar solo dos o tres fechas si caben más sin solapamiento.
- Las barras parten de cero. Toda escala logarítmica se declara en el eje.
- El redondeo refleja la precisión real; se evitan decimales ficticios.

## 5. Leyendas y anotaciones

- Las leyendas se colocan dentro de un único contenedor blanco, con borde gris fino, esquinas redondeadas y espaciado interior suficiente.
- La leyenda no tapa datos relevantes. Si no existe un hueco seguro, se coloca fuera del área de datos y se reserva espacio para ella.
- No se crean cápsulas o recuadros independientes para cada título de panel.
- Las anotaciones se limitan a hallazgos importantes y no repiten todos los valores disponibles.
- Los valores directos se sitúan junto a la marca correspondiente y conservan peso normal, salvo que sean el resultado principal.

## 6. Paneles múltiples

- Los paneles comparten escalas cuando la comparación lo requiere.
- Todos los paneles temporales muestran su eje X y usan la misma frecuencia de fechas.
- La cuadrícula debe quedar equilibrada. Ninguna fila termina con un único panel aislado.
- Para cinco paneles se usa una distribución `2 + 3`; para números impares se reorganizan los anchos para mantener al menos dos paneles por fila.
- Se reduce el espacio vertical entre el título general y los paneles, sin provocar solapamientos.
- Los nombres de panel se muestran como títulos sencillos, centrados y en negrita, sin cajas decorativas.

## 7. Elección del gráfico

| Pregunta | Opción principal | Complemento riguroso | Evitar |
|---|---|---|---|
| ¿Cómo cambia en el tiempo? | Línea | Banda de incertidumbre; paneles equilibrados | Eje dual, demasiadas líneas, pocas fechas |
| ¿Qué categoría es mayor? | Barras ordenadas o lollipop | Intervalo de confianza | Tarta con muchas categorías |
| ¿Cómo se distribuye? | Histograma + KDE o ECDF | Intervalos robustos y ridgelines | Solo la media, puntos saturados |
| ¿Hay relación? | Scatter o hexbin | Regresión con intervalo de confianza | Inferir causalidad |
| ¿Cómo se compone? | Barra 100 % o área apilada | Tabla breve de porcentajes | 3D |
| ¿Dónde están los patrones? | Heatmap | Etiquetas selectivas | Rainbow colormap |
| ¿Predice bien una regresión? | Real vs. predicho y residuos | Error por segmento | Solo R² |
| ¿Clasifica bien? | Confusión, PR/ROC y calibración | Umbral y prevalencia | Solo accuracy |
| ¿Se separan los clústeres? | PCA, silhouette y perfiles | Estabilidad entre semillas | PCA como prueba de separación |
| ¿Cómo entrenó la red? | Curvas de aprendizaje | Variación entre semillas | Solo el mejor epoch |
| ¿Qué explica un cambio? | Barras divergentes de contribución | Base, resultado y cambio neto | Waterfall difícil de reconstruir |

## 8. Rigor estadístico

- Toda estimación muestra incertidumbre o variabilidad cuando sea relevante.
- Se informa la población, el tamaño muestral `n`, el periodo y la transformación aplicada cuando afecten la interpretación.
- Correlación no se presenta como causalidad.
- PCA se describe como proyección, no como prueba de separación.
- Las comparaciones de modelos incluyen un baseline y variabilidad entre particiones o semillas.
- En clasificación se indican prevalencia y umbral; con clases desequilibradas se prioriza Precision–Recall.
- Las contribuciones a un cambio deben ser aditivas y sumar exactamente el cambio neto.

## 9. Patrones aprobados

- Distribución resumida: percentiles 5–95 como línea fina, IQR como banda gruesa y mediana como punto.
- Distribución completa: ridgeline limpio, relleno transparente y mediana marcada; evitar nubes de puntos si saturan.
- Cambio explicado: barras horizontales divergentes alrededor de cero, con positivos en verde, negativos en coral y valores directos.
- Serie temporal: observación tenue, tendencia principal destacada y banda de incertidumbre semitransparente.
- Comparación categórica: orden lógico o por magnitud y etiquetas directas cuando mejoren la lectura.

## 10. Validación y exportación

Antes de dar una figura por terminada:

1. Ejecutar la celda o script completo con semilla fija.
2. Revisar visualmente la salida renderizada, no solo el código.
3. Confirmar que no hay solapamientos ni texto cortado.
4. Verificar títulos, pesos tipográficos, formatos de fecha, unidades y leyenda.
5. Comprobar que los paneles múltiples están equilibrados y muestran ejes suficientes.
6. Exportar en SVG para informes y PNG a 240–300 dpi para presentaciones.

El notebook local `notebooks/eda/visualizacion_graficos.ipynb` contiene ejemplos ejecutables, pero esta rúbrica Markdown es la especificación canónica.
