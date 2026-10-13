#!/usr/bin/env python3
"""Build modelling-ready article/week data from the raw H&M CSV files.

The transaction file is read in chunks so the 3.2 GB source does not need to
fit in memory. Article IDs are always strings: their leading zero is meaningful.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = ROOT / "data" / "raw"
DEFAULT_OUTPUT = ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Only for quick tests; omit to process the complete transaction file.",
    )
    return parser.parse_args()


def clean_articles(path: Path) -> pd.DataFrame:
    articles = pd.read_csv(
        path,
        dtype={"article_id": "string", "product_code": "string", "index_code": "string"},
    )
    if articles["article_id"].duplicated().any():
        raise ValueError("articles.csv contains duplicated article_id values")
    articles["detail_desc"] = articles["detail_desc"].fillna("")
    return articles


def aggregate_transactions(path: Path, chunk_size: int, max_rows: int | None) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    rows_read = 0
    reader = pd.read_csv(
        path,
        usecols=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"],
        dtype={
            "t_dat": "string",
            "customer_id": "string",
            "article_id": "string",
            "price": "float32",
            "sales_channel_id": "int8",
        },
        chunksize=chunk_size,
        nrows=max_rows,
    )
    for number, chunk in enumerate(reader, start=1):
        dates = pd.to_datetime(chunk.pop("t_dat"), errors="raise")
        # Weeks start on Monday; this avoids pandas' Sunday-ending week convention.
        chunk["week"] = dates - pd.to_timedelta(dates.dt.weekday, unit="D")
        chunk["is_online"] = chunk["sales_channel_id"].eq(2)
        grouped = (
            chunk.groupby(["week", "article_id"], observed=True, sort=False)
            .agg(
                sales=("article_id", "size"),
                mean_price=("price", "mean"),
                min_price=("price", "min"),
                max_price=("price", "max"),
                online_sales=("is_online", "sum"),
            )
            .reset_index()
        )
        parts.append(grouped)
        rows_read += len(chunk)
        print(f"chunk {number}: {rows_read:,} transactions read", flush=True)

    if not parts:
        raise ValueError("transactions_train.csv contains no rows")

    weekly = pd.concat(parts, ignore_index=True)
    # A week can straddle two input chunks, so consolidate the partial aggregates.
    weighted_price = weekly["mean_price"] * weekly["sales"]
    weekly["price_total"] = weighted_price
    weekly = (
        weekly.groupby(["week", "article_id"], observed=True, sort=True)
        .agg(
            sales=("sales", "sum"),
            price_total=("price_total", "sum"),
            min_price=("min_price", "min"),
            max_price=("max_price", "max"),
            online_sales=("online_sales", "sum"),
        )
        .reset_index()
    )
    weekly["mean_price"] = weekly.pop("price_total") / weekly["sales"]
    weekly["online_share"] = weekly["online_sales"] / weekly["sales"]
    return weekly


def add_temporal_features(weekly: pd.DataFrame) -> pd.DataFrame:
    result = weekly.copy()
    demand_lookup = weekly[["article_id", "week", "sales"]]
    max_week = weekly["week"].max()

    for lag in range(1, 13):
        previous = demand_lookup.rename(columns={"sales": f"lag_{lag}_sales"}).copy()
        previous["week"] = previous["week"] + pd.Timedelta(weeks=lag)
        result = result.merge(previous, on=["article_id", "week"], how="left", validate="one_to_one")
        result[f"lag_{lag}_sales"] = result[f"lag_{lag}_sales"].fillna(0).astype("int32")

    next_week = demand_lookup.rename(columns={"sales": "target_next_week_sales"}).copy()
    next_week["week"] = next_week["week"] - pd.Timedelta(weeks=1)
    result = result.merge(next_week, on=["article_id", "week"], how="left", validate="one_to_one")
    result["target_next_week_sales"] = result["target_next_week_sales"].fillna(0).astype("int32")
    # The final observed week has no known target and must never enter training.
    result["target_is_observed"] = result["week"] < max_week
    result["rolling_4w_sales_mean"] = result[[f"lag_{i}_sales" for i in range(1, 5)]].mean(axis=1)
    result["rolling_12w_sales_mean"] = result[[f"lag_{i}_sales" for i in range(1, 13)]].mean(axis=1)
    result["week_of_year"] = result["week"].dt.isocalendar().week.astype("int16")
    result["year"] = result["week"].dt.year.astype("int16")
    return result


def dataset_report(datasets: list[tuple[str, pd.DataFrame]]) -> str:
    """Return a human-readable inventory similar to data/raw/dataset.txt."""
    sections: list[str] = []
    for filename, data in datasets:
        lines = [
            "=" * 80,
            f"ARCHIVO: {filename}",
            "=" * 80,
            f"Número de filas: {len(data):,}",
            "",
            "CABECERAS:",
        ]
        lines.extend(f"  {number}: {column}" for number, column in enumerate(data.columns))
        lines.extend(["", "PRIMERA FILA:"])
        if data.empty:
            lines.append("  El dataset no contiene datos.")
        else:
            lines.extend(f"  {column}: {value}" for column, value in data.iloc[0].items())
        sections.append("\n".join(lines))
    return "\n\n".join(sections) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    articles = clean_articles(args.raw_dir / "articles.csv")
    weekly = aggregate_transactions(args.raw_dir / "transactions_train.csv", args.chunk_size, args.max_rows)
    dataset = add_temporal_features(weekly)

    metadata_columns = [
        "article_id", "product_type_name", "product_group_name", "graphical_appearance_name",
        "colour_group_name", "perceived_colour_value_name", "index_group_name", "section_name",
        "garment_group_name",
    ]
    dataset = dataset.merge(articles[metadata_columns], on="article_id", how="left", validate="many_to_one")
    if dataset["garment_group_name"].isna().any():
        raise ValueError("Some transaction article IDs are absent from articles.csv")

    articles.to_csv(args.output_dir / "articles_clean.csv.gz", index=False, compression="gzip")
    dataset.to_csv(args.output_dir / "article_week.csv.gz", index=False, compression="gzip")
    (args.output_dir / "dataset_processed.txt").write_text(
        dataset_report(
            [("articles_clean.csv.gz", articles), ("article_week.csv.gz", dataset)]
        ),
        encoding="utf-8",
    )
    summary = {
        "source_transactions": int(dataset["sales"].sum()),
        "article_week_rows": len(dataset),
        "articles_with_sales": int(dataset["article_id"].nunique()),
        "first_week": dataset["week"].min().date().isoformat(),
        "last_week": dataset["week"].max().date().isoformat(),
        "note": "Sparse panel: rows are article-weeks with at least one sale.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
