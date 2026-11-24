"""Forecast errors by group, with bootstrap intervals over articles, and conformal prediction intervals by group.

Rows from the same article are not independent, so every interval resamples **articles**, not rows.
"""

import numpy as np
import pandas as pd


def _article_draws(per_article, seed, repeats):
    values = per_article.to_numpy()
    return np.array([values[np.random.default_rng(seed + draw).integers(0, len(values), len(values))].sum(axis=0)
                     for draw in range(repeats)])


def group_metrics(rows_all, group_column, order, seed, repeats=200, actual="target_next_week_sales",
                  model="model_abs_error", persistence="persistence_abs_error", signed="model_signed_error",
                  high_demand="target_high_demand"):
    """WAPE, skill over persistence, bias, and shares per group, with 95 % article-bootstrap intervals."""
    total_error = rows_all[model].sum()
    results = []
    for group in order:
        rows = rows_all.loc[rows_all[group_column].eq(group)]
        per_article = rows.groupby("article_id")[[actual, model, persistence, signed]].sum()
        per_article.insert(0, "rows", rows.groupby("article_id").size())
        draws = _article_draws(per_article, seed, repeats)
        actual_sum, model_sum, persistence_sum = rows[actual].sum(), rows[model].sum(), rows[persistence].sum()
        results.append({
            "group": group, "articles": len(per_article), "rows": len(rows),
            "rows (%)": 100 * len(rows) / len(rows_all), "abs. error (%)": 100 * model_sum / total_error,
            "mean next-week sales": rows[actual].mean(),
            "model WAPE (%)": 100 * model_sum / actual_sum,
            "WAPE CI low (%)": 100 * np.percentile(draws[:, 2] / draws[:, 1], 2.5),
            "WAPE CI high (%)": 100 * np.percentile(draws[:, 2] / draws[:, 1], 97.5),
            "persistence WAPE (%)": 100 * persistence_sum / actual_sum,
            "skill (%)": 100 * (1 - model_sum / persistence_sum),
            "skill CI low (%)": 100 * np.percentile(1 - draws[:, 2] / draws[:, 3], 2.5),
            "skill CI high (%)": 100 * np.percentile(1 - draws[:, 2] / draws[:, 3], 97.5),
            "mean bias (units)": rows[signed].mean(),
            "high-demand prevalence (%)": 100 * rows[high_demand].mean(),
        })
    return pd.DataFrame(results).set_index("group")


def paired_mae_change(rows, base_column, new_column, seed, repeats=200):
    """Change of MAE (%) from ``base_column`` to ``new_column``, with a 95 % paired article-bootstrap interval."""
    per_article = rows.groupby("article_id")[[base_column, new_column]].sum()
    observed = 100 * (per_article[new_column].sum() / per_article[base_column].sum() - 1)
    draws = _article_draws(per_article, seed, repeats)
    changes = 100 * (draws[:, 1] / draws[:, 0] - 1)
    return observed, *np.percentile(changes, [2.5, 97.5])


def grouped_wape(rows, group_column, error_column, order, actual="target_next_week_sales"):
    """WAPE (%) per group."""
    sums = rows.groupby(group_column)[[error_column, actual]].sum().reindex(order)
    return 100 * sums[error_column] / sums[actual]


def static_half_widths(calibration, group_column, error_column, level):
    """Split conformal half-widths: the ``level`` quantile of absolute errors, overall and per group."""
    return calibration[error_column].quantile(level), calibration.groupby(group_column)[error_column].quantile(level)


def rolling_half_widths(history, forecast_weeks, group_column, error_column, level, window_weeks, min_rows=200):
    """Half-width per (forecast week, group) from the errors of the previous ``window_weeks`` weeks.

    For a forecast made at the end of week *w*, the errors of rows with current week *w* − ``window_weeks`` … *w* − 1
    are known (their targets are weeks up to *w*), so the interval uses only information available at forecast time.
    Groups with fewer than ``min_rows`` rows in the window fall back to the window's overall quantile.
    """
    results = []
    for week in forecast_weeks:
        window = history.loc[history["week"].lt(week) & history["week"].ge(week - pd.Timedelta(weeks=window_weeks))]
        overall = window[error_column].quantile(level)
        by_group = window.groupby(group_column)[error_column].agg(["size", lambda values: values.quantile(level)])
        by_group.columns = ["rows", "half_width"]
        half_width = by_group["half_width"].where(by_group["rows"] >= min_rows, overall)
        results.append(pd.DataFrame({"week": week, group_column: half_width.index, "half_width": half_width.to_numpy()}))
    return pd.concat(results, ignore_index=True)


def interval_coverage(rows, prediction, half_width, actual="target_next_week_sales"):
    """Whether each actual value falls in [prediction − half_width, prediction + half_width] (clipped at 0), and the width."""
    low = (rows[prediction] - half_width).clip(lower=0)
    high = rows[prediction] + half_width
    return rows[actual].between(low, high), high - low
