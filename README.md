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

## Reports

The same project is documented from two perspectives:

- **PDRL:** theory, formulation, assumptions, model behaviour and critical interpretation.
- **MLLB:** implementation, pipelines, tools, tuning, evaluation and reproducibility.

## License

This repository contains project code and documentation only. The H&M dataset remains subject to its original terms of use.
