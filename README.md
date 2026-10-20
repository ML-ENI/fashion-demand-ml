# Fashion Demand ML

Machine Learning project for forecasting next-week demand of H&M fashion articles using historical transactions, product metadata, regression, classification and clustering.

## Overview

This project studies short-term fashion demand using the **H&M Personalized Fashion Recommendations** dataset.

The original transaction-level data are transformed into an **article-week** dataset so that historical information available up to week `t` can be used to model demand at week `t+1`.

The project is developed for the **PDRL & MLLB** Machine Learning reports.

## Objectives

The same case study is addressed from three complementary Machine Learning perspectives:

- **Regression** — predict the number of sales of each article during the following week.
- **Classification** — identify articles expected to exhibit high next-week demand.
- **Unsupervised learning** — discover groups of articles with similar product and commercial behaviour.

## Dataset

Main input files:

| File | Purpose |
|---|---|
| `transactions_train.csv` | Historical purchases, dates, prices and sales channels |
| `articles.csv` | Product metadata and garment characteristics |
| `customers.csv` | Customer information for optional aggregated features |

The raw H&M dataset is not redistributed in this repository.

## Data Pipeline

```text
Raw transactions
      ↓
Data validation
      ↓
Weekly aggregation
      ↓
Feature engineering
      ↓
Product metadata integration
      ↓
Temporal train / validation / test split
      ↓
Regression / Classification / Clustering
      ↓
Evaluation and comparison
```

The fundamental modelling unit is:

```text
article-week
```

and the supervised forecasting target is:

```text
next-week demand
```

## Candidate Features

The modelling dataset may include:

- lagged weekly sales,
- rolling demand statistics,
- unique-customer counts,
- price statistics,
- calendar information,
- product type,
- garment group,
- section,
- colour and other product metadata.

All temporal features are constructed using only information available before the prediction target.

## Project Structure

```text
fashion-demand-ml/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── eda/
│   └── experiments/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── evaluation/
├── results/
│   ├── figures/
│   ├── metrics/
│   └── models/
├── README.md
└── requirements.txt
```

## Experimental Design

The dataset is split chronologically:

```text
Training → Validation → Test
```

Random splitting is avoided because this is a forecasting problem and future information must not leak into model development.

## Evaluation

Candidate metrics include:

- **Regression:** MAE, RMSE and complementary R².
- **Classification:** precision, recall, F1-score, confusion matrix and, when appropriate, ROC-AUC or PR-AUC.
- **Unsupervised learning:** cluster cohesion/separation, stability and profile interpretation.

## Reproducibility

The project is designed to keep data preparation, feature generation, model training and evaluation reproducible.

Relevant experiments should record:

- model family,
- hyperparameters,
- feature set,
- temporal split,
- random seed,
- evaluation metrics.

Generate the processed article-week dataset with:

```bash
python src/data/process_data.py
```

The output is a gzip-compressed sparse weekly panel with calendar-correct lags,
rolling demand features and next-week demand. Run with `--help` for test and
path options.

## Reports

The same project is documented from two perspectives:

- **PDRL:** theory, formulation, assumptions, model behaviour and critical interpretation.
- **MLLB:** implementation, pipelines, tools, tuning, evaluation and reproducibility.

## License

This repository contains project code and documentation only. The H&M dataset remains subject to its original terms of use.


## Preparación validada de datos (versión 2)

El comando `python src/data/process_data.py` valida entradas y publica una ejecución
en `data/processed/runs/`. **Lee `data/processed/CURRENT.json` para localizar la
ejecución vigente**. Los CSV antiguos de la raíz se conservan como archivos heredados;
no contienen las correcciones de la versión 2. Nunca uses el resultado de PREVIEW
para un experimento completo.

Cada ejecución incluye `summary.json`, `provenance.json`, `schema.json` y
`checksums.json`. Se registran parámetros, versiones, hashes de entradas y código.
El cambio de CURRENT es atómico: un fallo no publica una ejecución a medias.

La copia canónica de `articles.csv`, cuando ambos repos están juntos, es
`../fashion-demand-ml/data/raw/articles.csv`. Se puede indicar otra ubicación con
`--articles-path`. En un clon independiente se usa `data/raw/articles.csv`.
No se borran copias locales; el hash y la ruta efectiva quedan registrados.
`src/data/data_contract.py` es el mismo contrato versionado en ambos repos;
si se modifica, hay que actualizar y comprobar ambas copias.

Para añadir Parquet, instala `python -m pip install -r requirements-parquet.txt`
y ejecuta el procesador con `--parquet`. Se conserva CSV comprimido como exportación.
Sin esa opción no se requiere pyarrow. Las lecturas CSV deben conservar los IDs como texto.

Pruebas: `python -m unittest discover -s tests -v`.
Consulta `docs/data-preparation-audit.md` para decisiones y limitaciones.


El universo se limita a artículos vendidos en la semana actual. El objetivo son
ventas registradas, no demanda latente ni stock. Se usan solo semanas con cobertura
diaria, 12 semanas anteriores completas y una semana objetivo completa.
El calendario se guarda como `iso_year` y `iso_week`. `split` se genera por
`target_week`. Primero se fijan proporciones de filas: 70 % entrenamiento,
15 % validación y 15 % prueba. Después se buscan los dos cortes entre semanas
que minimizan conjuntamente la suma de desviaciones cuadráticas respecto a esas
proporciones. Cada semana completa pertenece a un solo grupo y los tres grupos
son cronológicos y no vacíos. No se seleccionan cortes mirando las ventas objetivo.
Las fechas resultantes, porcentajes solicitados y reales se guardan en summary y
provenance. Se configuran con `--train-fraction 0.70 --validation-fraction 0.15
--test-fraction 0.15`; deben sumar 1. Los porcentajes son de filas, no de semanas.
La evaluación predice una semana después de observar la actual. `--gap-weeks 1`
permite una separación adicional como experimento, pero no es obligatoria para
este protocolo. Los grupos `gap` no se entrenan ni evalúan.

Las transacciones idénticas se cuentan y conservan porque pueden representar varias
unidades. `customer_id` se utiliza en esta auditoría. Se agregan precios en float64.
Los rangos `--data-start`, `--data-end` y `--data-cutoff` son explícitos y quedan
registrados. `--max-rows` nunca publica CURRENT y desconfía del último día leído.
