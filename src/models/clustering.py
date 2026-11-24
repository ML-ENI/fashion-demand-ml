"""K-Means model selection with a pre-registered rule: stability first, then separation."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score


def stability_ari(matrix, k, labels, seed, resamples=10, fraction=0.8, n_init=10):
    """ARI between ``labels`` and K-Means refitted on ``resamples`` random subsamples of ``matrix``.

    Each refit uses the same procedure as the full fit (``n_init`` starts), so instability is not an
    artefact of fewer starts.
    """
    scores = []
    for repeat in range(resamples):
        subsample = np.random.default_rng(seed + repeat).choice(len(matrix), int(fraction * len(matrix)), replace=False)
        refit = KMeans(n_clusters=k, n_init=n_init, random_state=seed + repeat).fit(matrix[subsample])
        scores.append(adjusted_rand_score(labels, refit.predict(matrix)))
    return np.array(scores)


def select_k(matrix, k_range, min_clusters, seed, resamples=10, fraction=0.8, min_stability=0.90,
             silhouette_rows=10_000, n_init=10):
    """Fit K-Means for every k and apply the rule: k >= ``min_clusters``, mean stability ARI >= ``min_stability``,
    then the highest silhouette.

    Returns the selection table (one row per k), the fitted models by k, and the selected k.
    """
    rows_for_silhouette = np.random.default_rng(seed).choice(len(matrix), min(silhouette_rows, len(matrix)), replace=False)
    models, rows = {}, []
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(matrix)
        models[k] = model
        stability = stability_ari(matrix, k, model.labels_, seed, resamples, fraction, n_init)
        rows.append({
            "k": k, "inertia": model.inertia_,
            "silhouette": silhouette_score(matrix[rows_for_silhouette], model.labels_[rows_for_silhouette]),
            "stability ARI mean": stability.mean(), "stability ARI min": stability.min(),
            "stability ARI max": stability.max(),
            "smallest cluster (%)": 100 * np.bincount(model.labels_).min() / len(matrix),
        })
    table = pd.DataFrame(rows).set_index("k")
    eligible = table.loc[(table.index >= min_clusters) & table["stability ARI mean"].ge(min_stability)]
    if eligible.empty:
        raise ValueError("No candidate k satisfies the stability rule.")
    return table, models, int(eligible["silhouette"].idxmax())
