"""Confidence intervals and significance tests for comparing classifiers on the same rows.

Rows of the same article are correlated (an article appears in many weeks), so the bootstrap resamples
**articles** (cluster bootstrap). Each draw is expressed as integer row weights, which is exactly equivalent
to replicating the rows of every drawn article and lets scikit-learn metrics use ``sample_weight``.

Model comparisons are **paired**: every draw scores all models on the same resampled articles, so the
interval of a difference already accounts for the strong correlation between models' errors.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score)


def _threshold_metric(function):
    def metric(y_true, probability, weight, threshold):
        return function(y_true, probability >= threshold, sample_weight=weight, zero_division=0)
    return metric


CLASSIFICATION_METRICS = {
    "precision": _threshold_metric(precision_score),
    "recall": _threshold_metric(recall_score),
    "F1": _threshold_metric(f1_score),
    "PR-AUC": lambda y, p, w, t: average_precision_score(y, p, sample_weight=w),
    "ROC-AUC": lambda y, p, w, t: roc_auc_score(y, p, sample_weight=w),
    "Brier": lambda y, p, w, t: brier_score_loss(y, p, sample_weight=w),
}
LOWER_IS_BETTER = {"Brier"}


def cluster_bootstrap_weights(groups, repeats, seed):
    """Yield one row-weight vector per draw: how many times each row's group was drawn."""
    codes, uniques = pd.factorize(pd.Series(groups), sort=False)
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        drawn = rng.integers(0, len(uniques), len(uniques))
        yield np.bincount(drawn, minlength=len(uniques))[codes].astype(float)


def bootstrap_p_value(differences):
    """Two-sided percentile-bootstrap p-value for H0: difference = 0."""
    differences = np.asarray(differences, dtype=float)
    share_below = np.mean(differences <= 0)
    share_above = np.mean(differences >= 0)
    return float(min(1.0, 2 * min(share_below, share_above)))


def holm_adjust(p_values):
    """Holm–Bonferroni adjusted p-values (family-wise error control), in the input order."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    scaled = np.maximum.accumulate((len(p_values) - np.arange(len(p_values))) * p_values[order])
    adjusted = np.empty_like(p_values)
    adjusted[order] = np.minimum(scaled, 1.0)
    return adjusted


def paired_bootstrap_comparison(y_true, probabilities, groups, reference=None, pairs=None, threshold=0.5,
                                metrics=None, repeats=500, seed=42, level=0.95):
    """Cluster-bootstrap intervals per model and paired differences between models.

    Compare every model against ``reference`` or give explicit ``pairs`` of ``(model, other)``.

    Returns two tables:

    - ``intervals``: one row per (model, metric) with the point estimate and its interval;
    - ``differences``: one row per (model, other, metric) with ``model − other``, its interval, the
      bootstrap p-value and a Holm-adjusted p-value across all comparisons in the table.
    """
    metrics = list(metrics or CLASSIFICATION_METRICS)
    y_true = np.asarray(y_true, dtype=int)
    probabilities = {name: np.asarray(values, dtype=float) for name, values in probabilities.items()}
    if pairs is None:
        if reference is None:
            raise ValueError("Give either a reference model or explicit pairs.")
        pairs = [(model, reference) for model in probabilities if model != reference]
    unknown = {name for pair in pairs for name in pair} - set(probabilities)
    if unknown:
        raise KeyError(f"Models not among the probabilities: {sorted(unknown)}")
    alpha = (1 - level) / 2

    def score_all(weight):
        return {(model, metric): CLASSIFICATION_METRICS[metric](y_true, values, weight, threshold)
                for model, values in probabilities.items() for metric in metrics}

    observed = score_all(None)
    draws = pd.DataFrame([score_all(weight) for weight in cluster_bootstrap_weights(groups, repeats, seed)])

    interval_rows, difference_rows = [], []
    for model in probabilities:
        for metric in metrics:
            values = draws[(model, metric)]
            interval_rows.append({"model": model, "metric": metric, "estimate": observed[(model, metric)],
                                  "CI low": values.quantile(alpha), "CI high": values.quantile(1 - alpha)})
    for model, other in pairs:
        for metric in metrics:
            change = draws[(model, metric)] - draws[(other, metric)]
            difference_rows.append({
                "model": model, "other": other, "metric": metric,
                "difference": observed[(model, metric)] - observed[(other, metric)],
                "CI low": change.quantile(alpha), "CI high": change.quantile(1 - alpha),
                "p-value": bootstrap_p_value(change),
                "better": "lower" if metric in LOWER_IS_BETTER else "higher",
            })
    intervals = pd.DataFrame(interval_rows).set_index(["model", "metric"])
    differences = pd.DataFrame(difference_rows)
    if not differences.empty:
        differences["Holm p-value"] = holm_adjust(differences["p-value"])
        differences["significant (Holm, 5 %)"] = differences["Holm p-value"] < 0.05
        differences = differences.set_index(["model", "other", "metric"])
    return intervals, differences


def mcnemar_test(y_true, predicted_a, predicted_b):
    """McNemar test on paired correct/incorrect decisions (exact binomial for few discordant pairs).

    Assumes independent rows, so with repeated articles it is an optimistic, row-level reference;
    the cluster bootstrap above is the primary evidence.
    """
    y_true = np.asarray(y_true, dtype=int)
    correct_a = np.asarray(predicted_a, dtype=int) == y_true
    correct_b = np.asarray(predicted_b, dtype=int) == y_true
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    discordant = only_a + only_b
    if discordant == 0:
        return {"only A correct": 0, "only B correct": 0, "statistic": 0.0, "p-value": 1.0, "method": "none"}
    if discordant < 25:
        p_value = stats.binomtest(only_a, discordant, 0.5).pvalue
        return {"only A correct": only_a, "only B correct": only_b, "statistic": float(min(only_a, only_b)),
                "p-value": float(p_value), "method": "exact binomial"}
    statistic = (abs(only_a - only_b) - 1) ** 2 / discordant
    return {"only A correct": only_a, "only B correct": only_b, "statistic": float(statistic),
            "p-value": float(stats.chi2.sf(statistic, df=1)), "method": "chi² with continuity correction"}
