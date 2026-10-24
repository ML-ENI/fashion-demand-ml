#!/usr/bin/env python3
"""Preparar ventas observadas de t+1 para artículos vendidos en t.

No se estima demanda latente ni disponibilidad: cero significa cero registros
de venta en una semana cubierta por el archivo. Los artículos sin ventas en t
no pertenecen al universo de predicción de este experimento.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_contract import ARTICLE_COLUMNS, articles_path, load_articles, metadata, publish

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--articles-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--max-rows", type=int, help="Solo diagnóstico: publica PREVIEW, nunca CURRENT")
    parser.add_argument("--data-start", default="2018-09-20")
    parser.add_argument("--data-end", default="2020-09-22")
    parser.add_argument("--data-cutoff", help="Último día incluido; por defecto, final observado")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--high-demand-quantile", type=float, default=0.75,
                        help="Cuantil de ventas positivas de train que define demanda alta")
    parser.add_argument("--gap-weeks", type=int, default=0,
                        help="Semanas objetivo excluidas antes de cada corte; 0 para evaluación semanal")
    parser.add_argument("--parquet", action="store_true", help="Añade Parquet; requiere pyarrow")
    return parser.parse_args()


def aggregate_transactions(path, articles, chunk_size, max_rows=None,
                           data_start="2018-09-20", data_end="2020-09-22", cutoff=None):
    """Valida cada bloque y combina sumas, no medias de medias sin ponderar."""
    if chunk_size < 1 or (max_rows is not None and max_rows < 1):
        raise ValueError("chunk-size y max-rows deben ser positivos")
    lower, upper = pd.Timestamp(data_start), pd.Timestamp(data_end)
    if lower > upper:
        raise ValueError("data-start debe ser anterior a data-end")
    cutoff = pd.Timestamp(cutoff) if cutoff else upper
    if not lower <= cutoff <= upper:
        raise ValueError("data-cutoff fuera del rango configurado")
    columns = ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
    reader = pd.read_csv(path, usecols=columns, chunksize=chunk_size, nrows=max_rows,
                         dtype={"t_dat": "string", "customer_id": "string", "article_id": "string",
                                "price": "float64", "sales_channel_id": "Int64"})
    known_ids = set(articles["article_id"])
    parts, days, raw_days = [], set(), set()
    previous_day = pd.DataFrame(columns=columns)
    last_date = None
    rows_read = included = duplicate_rows = 0
    with reader:
        for number, chunk in enumerate(reader, 1):
            if chunk[columns].isna().any().any():
                raise ValueError(f"Bloque {number}: hay campos obligatorios vacíos")
            for col in ["article_id", "customer_id"]:
                if chunk[col].str.strip().eq("").any():
                    raise ValueError(f"Bloque {number}: {col} vacío")
            if not chunk["article_id"].isin(known_ids).all():
                raise ValueError(f"Bloque {number}: artículos desconocidos")
            dates = pd.to_datetime(chunk["t_dat"], format="%Y-%m-%d", errors="raise")
            if not dates.between(lower, upper).all():
                raise ValueError(f"Bloque {number}: fechas fuera del rango configurado")
            # El fichero H&M está ordenado por fecha. Se comprueba, no se presupone.
            if not dates.is_monotonic_increasing or (last_date is not None and dates.iloc[0] < last_date):
                raise ValueError("Ordena las transacciones por fecha antes de procesarlas")
            last_date = dates.iloc[-1]
            raw_days.update(dates.unique())
            if not np.isfinite(chunk["price"]).all() or chunk["price"].lt(0).any():
                raise ValueError(f"Bloque {number}: precios negativos o no finitos")
            if not chunk["sales_channel_id"].isin([1, 2]).all():
                raise ValueError(f"Bloque {number}: canal distinto de 1 o 2")
            rows_read += len(chunk)
            chunk = chunk.loc[dates <= cutoff].copy()
            dates = dates.loc[chunk.index]
            if chunk.empty:
                continue
            # customer_id se utiliza para auditar filas idénticas entre bloques.
            # Se conservan: podrían ser compras de varias unidades, no errores.
            combined = pd.concat([previous_day, chunk], ignore_index=True) if len(previous_day) else chunk
            duplicate_rows += int(combined.duplicated(columns).sum())
            previous_day = combined.loc[combined["t_dat"] == chunk["t_dat"].iloc[-1], columns].drop_duplicates()
            included += len(chunk)
            days.update(dates.unique())
            chunk["week"] = dates - pd.to_timedelta(dates.dt.weekday, unit="D")
            chunk["is_online"] = chunk["sales_channel_id"].eq(2)
            parts.append(chunk.groupby(["week", "article_id"], sort=False).agg(
                sales=("article_id", "size"), price_total=("price", "sum"),
                min_price=("price", "min"), max_price=("price", "max"),
                online_sales=("is_online", "sum")
            ).reset_index())
            print(f"Bloque {number}: {rows_read:,} transacciones revisadas", flush=True)
    if not parts:
        raise ValueError("No hay transacciones en el periodo seleccionado")
    daily = pd.DatetimeIndex(sorted(days))
    missing_days = pd.date_range(daily.min(), daily.max()).difference(daily)
    if len(missing_days):
        raise ValueError(f"Días sin registros: {missing_days.strftime('%Y-%m-%d').tolist()}")
    full = pd.concat(parts, ignore_index=True).groupby(["week", "article_id"], sort=True).agg(
        sales=("sales", "sum"), price_total=("price_total", "sum"),
        min_price=("min_price", "min"), max_price=("max_price", "max"),
        online_sales=("online_sales", "sum")
    ).reset_index()
    full["mean_price"] = full.pop("price_total") / full["sales"]
    full["online_share"] = full["online_sales"] / full["sales"]
    if not ((full["mean_price"] >= full["min_price"] - 1e-12)
            & (full["mean_price"] <= full["max_price"] + 1e-12)).all():
        raise ValueError("Media de precios fuera del intervalo mínimo/máximo")
    # --max-rows puede cortar un día incluso si termina en domingo.
    # Ese último día leído nunca se considera completo en un diagnóstico parcial.
    if max_rows is not None:
        final_read_day = pd.Timestamp(max(raw_days))
        daily = daily[daily < final_read_day]
    complete = {pd.Timestamp(week) for week, count in
                pd.Series(1, index=daily).groupby(daily.to_period("W-SUN").start_time).sum().items()
                if count == 7}
    info = {
        "rows_read": rows_read, "source_transactions": included,
        "first_date": str(pd.Timestamp(min(days)).date()),
        "last_date": str(pd.Timestamp(max(days)).date()),
        "data_cutoff": str(min(cutoff, pd.Timestamp(max(days))).date()),
        "identical_transaction_rows_preserved": duplicate_rows,
        "complete_weeks": sorted(str(x.date()) for x in complete),
        "partial_week_rows_excluded": int((~full["week"].isin(complete)).sum()),
        "preview": max_rows is not None,
    }
    return full, complete, info


def add_temporal_features(weekly, complete_weeks, history_weeks=12):
    """Construye solo filas con semana actual, historial y objetivo cubiertos."""
    complete = set(complete_weeks)
    eligible = {w for w in complete
                if w + pd.Timedelta(weeks=1) in complete
                and all(w - pd.Timedelta(weeks=i) in complete for i in range(1, history_weeks + 1))}
    result = weekly.loc[weekly["week"].isin(eligible)].copy()
    lookup = weekly.set_index(["week", "article_id"])
    if lookup.index.duplicated().any():
        raise ValueError("Hay parejas artículo-semana duplicadas")
    def value_at(column, offset, fill_value=None, dtype=None):
        keys = pd.MultiIndex.from_arrays([
            result["week"] + pd.Timedelta(weeks=offset), result["article_id"]
        ], names=["week", "article_id"])
        values = lookup[column].reindex(keys)
        if fill_value is not None:
            values = values.fillna(fill_value)
        return values.to_numpy(dtype=dtype)

    # Solo rellenamos ventas con cero en semanas cuya cobertura está comprobada.
    for lag in range(1, history_weeks + 1):
        result[f"lag_{lag}_sales"] = value_at("sales", -lag, fill_value=0, dtype="int64")
    result["target_next_week_sales"] = value_at("sales", 1, fill_value=0, dtype="int64")
    result["target_week"] = result["week"] + pd.Timedelta(weeks=1)
    result["target_is_observed"] = True
    result["history_is_observed"] = True
    for window in [4, 12]:
        lag_columns = [f"lag_{i}_sales" for i in range(1, window + 1)]
        result[f"rolling_{window}w_sales_mean"] = result[lag_columns].mean(axis=1)
        result[f"rolling_{window}w_sales_std"] = result[lag_columns].std(axis=1, ddof=0)
        result[f"rolling_{window}w_active_weeks"] = result[lag_columns].gt(0).sum(axis=1).astype("int8")

    result["sales_trend_4w"] = (
        result[["lag_1_sales", "lag_2_sales"]].mean(axis=1)
        - result[["lag_3_sales", "lag_4_sales"]].mean(axis=1)
    )
    historical_prices = []
    historical_online_sales = []
    for lag in range(1, 5):
        price_column = f"lag_{lag}_mean_price"
        result[price_column] = value_at("mean_price", -lag)
        historical_prices.append(price_column)
        historical_online_sales.append(
            value_at("online_sales", -lag, fill_value=0, dtype="int64")
        )
    result["rolling_4w_mean_price"] = result[historical_prices].mean(axis=1)
    result["price_change_from_4w_mean"] = result["mean_price"] - result["rolling_4w_mean_price"]
    prior_sales = result[[f"lag_{lag}_sales" for lag in range(1, 5)]].sum(axis=1).to_numpy()
    prior_online_sales = np.vstack(historical_online_sales).sum(axis=0)
    result["rolling_4w_online_share"] = np.divide(
        prior_online_sales, prior_sales,
        out=np.full(len(result), np.nan, dtype="float64"), where=prior_sales > 0)
    # Los dos componentes usan el mismo calendario ISO.
    calendar = result["week"].dt.isocalendar()
    result["iso_year"] = calendar.year.astype("int16")
    result["iso_week"] = calendar.week.astype("int16")
    return result


def add_classification_target(data, quantile=0.75):
    """Define demanda alta con ventas positivas de train y fija el umbral."""
    if not 0 < quantile < 1:
        raise ValueError("high-demand-quantile debe estar entre 0 y 1")
    train = data.loc[data["split"].eq("train"), "target_next_week_sales"]
    positive_train = train[train.gt(0)]
    if positive_train.empty:
        raise ValueError("No hay ventas positivas en train para definir demanda alta")
    threshold = int(max(1, positive_train.quantile(quantile, interpolation="higher")))
    result = data.copy()
    result["target_high_demand"] = result["target_next_week_sales"].ge(threshold).astype("int8")
    prevalence = result.groupby("split")["target_high_demand"].mean()
    details = {
        "definition": "target_next_week_sales >= threshold",
        "threshold_units": threshold,
        "training_quantile": float(quantile),
        "reference_population": "positive target_next_week_sales rows in train only",
        "prevalence_by_split": {str(k): float(v) for k, v in prevalence.items()},
    }
    return result, details


def build_clustering_table(data):
    """Resume artículos con señales históricas de train, sin objetivos futuros."""
    train = data.loc[data["split"].eq("train")].copy()
    if train.empty:
        raise ValueError("Train está vacío; no se puede construir clustering")
    grouped = train.groupby("article_id", sort=True)
    clusters = grouped.agg(
        observation_start=("week", "min"),
        observation_end=("week", "max"),
        article_weeks=("week", "size"),
        mean_current_week_sales=("sales", "mean"),
        median_current_week_sales=("sales", "median"),
        std_current_week_sales=("sales", "std"),
        max_current_week_sales=("sales", "max"),
        mean_12w_sales=("rolling_12w_sales_mean", "mean"),
        mean_12w_sales_volatility=("rolling_12w_sales_std", "mean"),
        mean_sales_trend_4w=("sales_trend_4w", "mean"),
        mean_active_weeks_12w=("rolling_12w_active_weeks", "mean"),
        mean_price=("mean_price", "mean"),
        price_std=("mean_price", "std"),
        total_sales=("sales", "sum"),
        total_online_sales=("online_sales", "sum"),
    ).reset_index()
    clusters["std_current_week_sales"] = clusters["std_current_week_sales"].fillna(0)
    clusters["price_std"] = clusters["price_std"].fillna(0)
    clusters["historical_active_share"] = clusters.pop("mean_active_weeks_12w") / 12
    clusters["mean_online_share"] = clusters.pop("total_online_sales") / clusters.pop("total_sales")
    metadata_columns = ARTICLE_COLUMNS[1:]
    metadata = grouped[metadata_columns].first().reset_index()
    return clusters.merge(metadata, on="article_id", how="left", validate="one_to_one")


def assign_splits(data, train_fraction=0.70, validation_fraction=0.15,
                  test_fraction=0.15, gap_weeks=0):
    """Elige primero proporciones de FILAS y después cortes entre semanas.

    Se examinan todos los pares de cortes cronológicos que dejan tres grupos
    no vacíos. Se minimiza la suma de desviaciones cuadráticas respecto a las
    proporciones solicitadas. No se usa el valor de las ventas para elegir.
    Si se solicita gap, los porcentajes se calculan sobre las filas no excluidas.
    """
    fractions = np.asarray([train_fraction, validation_fraction, test_fraction], dtype=float)
    if (not np.isfinite(fractions).all() or (fractions <= 0).any()
            or not np.isclose(fractions.sum(), 1.0, rtol=0, atol=1e-10)):
        raise ValueError("Las tres proporciones deben ser positivas y sumar 1")
    if not isinstance(gap_weeks, (int, np.integer)) or gap_weeks < 0:
        raise ValueError("gap-weeks debe ser un entero no negativo")
    if data["target_week"].isna().any():
        raise ValueError("Hay semanas objetivo vacías")
    weeks = data.groupby("target_week", sort=True).size()
    if len(weeks) < 3:
        raise ValueError("Se necesitan al menos tres semanas objetivo para dividir")
    dates = pd.DatetimeIndex(weeks.index)
    if (dates.weekday != 0).any() or not dates.equals(dates.normalize()):
        raise ValueError("Las semanas objetivo deben ser lunes sin hora")

    # prefix[k] cuenta todas las filas de las primeras k semanas.
    prefix = np.concatenate(([0], weeks.to_numpy(dtype="int64").cumsum()))
    best = None
    for i in range(1, len(dates) - 1):
        # Si hay separación, excluimos las semanas inmediatamente anteriores al corte.
        train_end = int(dates.searchsorted(dates[i] - pd.Timedelta(weeks=gap_weeks)))
        for j in range(i + 1, len(dates)):
            validation_end = int(dates.searchsorted(dates[j] - pd.Timedelta(weeks=gap_weeks)))
            counts = np.array([prefix[train_end], prefix[validation_end] - prefix[i],
                               prefix[-1] - prefix[j]], dtype="int64")
            if (counts <= 0).any():
                continue
            actual = counts / counts.sum()
            error = float(np.square(actual - fractions).sum())
            # En caso de empate, se eligen los cortes más antiguos: resultado reproducible.
            candidate = (error, i, j)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError("No hay suficientes semanas para tres grupos y el gap solicitado")
    _, i, j = best
    validation_start, test_start = dates[i], dates[j]

    result = data.copy()
    result["split"] = "train"
    result.loc[result["target_week"] >= validation_start, "split"] = "validation"
    result.loc[result["target_week"] >= test_start, "split"] = "test"
    for cut in [validation_start, test_start]:
        gap = result["target_week"].between(cut - pd.Timedelta(weeks=gap_weeks), cut, inclusive="left")
        result.loc[gap, "split"] = "gap"

    names = ["train", "validation", "test"]
    counts = result["split"].value_counts().reindex(names, fill_value=0)
    percentages = 100 * counts / counts.sum()
    details = {
        "strategy": "percentage_first_chronological_whole_weeks",
        "percentage_unit": "article-week rows, excluding optional gap rows",
        "requested_percentages": dict(zip(names, (fractions * 100).tolist())),
        "actual_percentages": {s: float(percentages[s]) for s in names},
        "validation_start": str(validation_start.date()),
        "test_start": str(test_start.date()),
        "gap_weeks": int(gap_weeks),
        "gap_rows": int(result["split"].eq("gap").sum()),
        "cut_selection": "minimum sum of squared fraction errors; earliest cuts break ties",
    }
    return result, details

def main():
    args = parse_args()
    source = articles_path(ROOT, args.raw_dir, args.articles_path)
    articles = load_articles(source)
    weekly, complete, info = aggregate_transactions(
        args.raw_dir / "transactions_train.csv", articles, args.chunk_size,
        args.max_rows, args.data_start, args.data_end, args.data_cutoff)
    features = add_temporal_features(weekly, complete)
    # Un diagnóstico muy corto puede no contener tres semanas modelables.
    if args.max_rows is not None and features["target_week"].nunique() < 3:
        dataset = features.copy()
        dataset["split"] = "preview"
        split_details = {"strategy": "preview_insufficient_weeks"}
    else:
        dataset, split_details = assign_splits(features, args.train_fraction,
            args.validation_fraction, args.test_fraction, args.gap_weeks)
    dataset = dataset.merge(articles[ARTICLE_COLUMNS], on="article_id", how="left", validate="many_to_one")
    dataset, classification_details = add_classification_target(
        dataset, args.high_demand_quantile)
    clustering = build_clustering_table(dataset)
    counts = {str(k): int(v) for k, v in dataset["split"].value_counts().items()}
    if args.max_rows is None and any(counts.get(s, 0) == 0 for s in ["train", "validation", "test"]):
        raise ValueError("Alguna partición está vacía. Revisa los cortes temporales.")
    summary = {**info, "article_week_rows": len(dataset), "split_counts": counts,
               "articles_with_sales": int(dataset["article_id"].nunique()),
               "candidate_policy": "articles with at least one recorded sale in current week",
               "target_definition": "recorded next-week sales; not latent demand or stock availability",
               "evaluation": "weekly rolling origin; current week has ended before prediction",
               "split_by": "target_week", "history_weeks": 12,
               "split_configuration": split_details,
               "classification_target": classification_details,
               "clustering_definition": {
                   "unit": "article",
                   "source_partition": "train only",
                   "future_targets_used": False,
                   "rows": len(clustering),
               }}
    provenance = metadata(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        {"articles": source, "transactions": args.raw_dir / "transactions_train.csv"},
        [Path(__file__), Path(__file__).with_name("data_contract.py")])
    provenance["split_configuration"] = split_details
    final = publish(args.output_dir, {"article_week": dataset, "articles_clean": articles,
                                     "article_clustering": clustering},
                    {"summary": summary, "provenance": provenance}, args.parquet, args.max_rows is not None)
    print(json.dumps(summary, indent=2))
    print("Ejecución publicada:", final)


if __name__ == "__main__":
    main()
