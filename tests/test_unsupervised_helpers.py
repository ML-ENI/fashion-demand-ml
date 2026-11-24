import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.cache import cached
from src.evaluation.grouped import (group_metrics, grouped_wape, interval_coverage, paired_mae_change,
                                    rolling_half_widths, static_half_widths)
from src.models.clustering import davies_bouldin_by_k, select_k
from src.visualization.labels import place_line_labels


class SelectKTests(unittest.TestCase):
    def test_selects_the_number_of_separated_blobs(self):
        rng = np.random.default_rng(0)
        centres = np.array([[0, 0], [10, 0], [0, 10]])
        matrix = np.vstack([rng.normal(centre, 0.5, size=(200, 2)) for centre in centres])
        table, models, k = select_k(matrix, range(2, 6), min_clusters=2, seed=0, resamples=3, silhouette_rows=600,
                                    n_init=3)
        self.assertEqual(k, 3)
        self.assertEqual(list(table.index), [2, 3, 4, 5])
        self.assertGreater(table.loc[3, "silhouette"], 0.8)
        davies_bouldin = davies_bouldin_by_k(matrix, models)
        self.assertEqual(int(davies_bouldin.idxmin()), 3)
        self.assertLess(davies_bouldin.loc[3], 0.3)

    def test_rule_fails_loudly_when_no_k_is_eligible(self):
        matrix = np.random.default_rng(1).normal(size=(100, 2))
        with self.assertRaises(ValueError):
            select_k(matrix, range(2, 4), min_clusters=2, seed=0, resamples=2, min_stability=1.01, n_init=2)


class CacheTests(unittest.TestCase):
    def test_reuses_result_for_same_key_and_recomputes_for_new_key(self):
        calls = []

        def compute():
            calls.append(1)
            return {"value": len(calls)}

        with tempfile.TemporaryDirectory() as directory:
            first = cached(directory, "step", {"a": 1}, compute)
            again = cached(directory, "step", {"a": 1}, compute)
            changed = cached(directory, "step", {"a": 2}, compute)
            refreshed = cached(directory, "step", {"a": 1}, compute, refresh=True)
        self.assertEqual(first, again)
        self.assertEqual(changed["value"], 2)
        self.assertEqual(refreshed["value"], 3)
        self.assertEqual(len(calls), 3)


def toy_rows():
    return pd.DataFrame({
        "article_id": ["a", "a", "b", "b", "c", "c"],
        "group": ["x", "x", "x", "y", "y", "y"],
        "target_next_week_sales": [10, 10, 20, 5, 5, 5],
        "model_abs_error": [2, 2, 4, 1, 1, 3],
        "persistence_abs_error": [4, 4, 4, 2, 2, 2],
        "model_signed_error": [-2, 2, -4, 1, -1, 3],
        "target_high_demand": [0, 0, 1, 0, 0, 0],
    })


class GroupedMetricTests(unittest.TestCase):
    def test_group_metrics_shares_and_wape(self):
        table = group_metrics(toy_rows(), "group", ["x", "y"], seed=0, repeats=20)
        self.assertAlmostEqual(table["rows (%)"].sum(), 100)
        self.assertAlmostEqual(table["abs. error (%)"].sum(), 100)
        self.assertAlmostEqual(table.loc["x", "model WAPE (%)"], 100 * 8 / 40)
        self.assertAlmostEqual(table.loc["x", "skill (%)"], 100 * (1 - 8 / 12))
        wape = grouped_wape(toy_rows(), "group", "model_abs_error", ["x", "y"])
        self.assertAlmostEqual(wape["y"], 100 * 5 / 15)

    def test_paired_change_is_zero_for_identical_errors(self):
        rows = toy_rows().assign(copy=lambda frame: frame["model_abs_error"])
        change, low, high = paired_mae_change(rows, "model_abs_error", "copy", seed=0, repeats=20)
        self.assertEqual((change, low, high), (0, 0, 0))


class ConformalTests(unittest.TestCase):
    def test_static_and_coverage(self):
        calibration = pd.DataFrame({"group": ["x"] * 5 + ["y"] * 5, "error": [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]})
        overall, by_group = static_half_widths(calibration, "group", "error", 0.8)
        self.assertGreater(by_group["y"], by_group["x"])
        rows = pd.DataFrame({"prediction": [10.0, 10.0], "target_next_week_sales": [12, 30]})
        covered, width = interval_coverage(rows, "prediction", pd.Series([5.0, 5.0]))
        self.assertEqual(covered.tolist(), [True, False])
        self.assertEqual(width.tolist(), [10.0, 10.0])

    def test_rolling_window_uses_only_previous_weeks(self):
        weeks = pd.date_range("2020-01-06", periods=6, freq="7D")
        history = pd.DataFrame({"week": np.repeat(weeks, 10), "group": "x",
                                "error": np.repeat([1, 1, 1, 1, 100, 100], 10).astype(float)})
        result = rolling_half_widths(history, [weeks[4]], "group", "error", level=0.8, window_weeks=4, min_rows=1)
        self.assertEqual(result["half_width"].iloc[0], 1.0)   # week 4 itself (errors of 100) is not used


class LineLabelTests(unittest.TestCase):
    def test_one_label_per_line_and_no_box_overlap(self):
        x = np.arange(10)
        curves = np.array([x * 10, x * 5 + 20, np.full(10, 80.0)])
        fig, ax = plt.subplots(figsize=(8, 5))
        for curve in curves:
            ax.plot(x, curve)
        ax.set_xlim(-0.5, 9.5)
        ax.set_ylim(0, 102)
        place_line_labels(ax, x, curves, ["first", "second", "third"], ["C0", "C1", "C2"])
        fig.canvas.draw()
        boxes = [text.get_bbox_patch().get_window_extent() for text in ax.texts]
        self.assertEqual(len(boxes), 3)
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertFalse(boxes[i].overlaps(boxes[j]))
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
