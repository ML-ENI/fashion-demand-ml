import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.significance import (bootstrap_p_value, cluster_bootstrap_weights, holm_adjust,
                                         mcnemar_test, paired_bootstrap_comparison)


def make_data(rows=4_000, articles=400, seed=0):
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, articles, rows)
    y = rng.binomial(1, 0.3, rows)
    good = np.clip(y * 0.6 + rng.normal(0.2, 0.15, rows), 0, 1)
    twin = np.clip(good + rng.normal(0, 0.01, rows), 0, 1)
    noise = rng.uniform(0, 1, rows)
    return y, groups, {"good": good, "twin": twin, "noise": noise}


class ClusterBootstrapTests(unittest.TestCase):
    def test_weights_keep_whole_groups_and_total_groups(self):
        groups = np.array(["a", "a", "b", "c", "c", "c"])
        for weight in cluster_bootstrap_weights(groups, repeats=20, seed=1):
            self.assertEqual(weight[0], weight[1])
            self.assertEqual(weight[3], weight[5])
            drawn_groups = weight[0] + weight[2] + weight[3]
            self.assertEqual(drawn_groups, 3)

    def test_weights_match_row_replication(self):
        y, groups, probabilities = make_data(rows=500, articles=50)
        weight = next(cluster_bootstrap_weights(groups, repeats=1, seed=3))
        predicted = probabilities["good"] >= 0.5
        replicated = np.repeat(np.arange(len(y)), weight.astype(int))
        self.assertAlmostEqual(f1_score(y, predicted, sample_weight=weight),
                               f1_score(y[replicated], predicted[replicated]))


class PairedComparisonTests(unittest.TestCase):
    def test_detects_real_difference_and_not_a_negligible_one(self):
        y, groups, probabilities = make_data()
        intervals, differences = paired_bootstrap_comparison(
            y, probabilities, groups, reference="good", metrics=["F1", "PR-AUC", "Brier"], repeats=200, seed=0)
        for metric in ["F1", "PR-AUC", "Brier"]:
            row = intervals.loc[("good", metric)]
            self.assertLessEqual(row["CI low"], row["estimate"])
            self.assertGreaterEqual(row["CI high"], row["estimate"])
        self.assertTrue(differences.loc[("noise", "good", "F1"), "significant (Holm, 5 %)"])
        self.assertGreater(differences.loc[("noise", "good", "Brier"), "difference"], 0)
        self.assertFalse(differences.loc[("twin", "good", "PR-AUC"), "significant (Holm, 5 %)"])

    def test_explicit_pairs(self):
        y, groups, probabilities = make_data(rows=1_000, articles=100)
        _, differences = paired_bootstrap_comparison(y, probabilities, groups, pairs=[("twin", "noise")],
                                                     metrics=["F1"], repeats=50, seed=0)
        self.assertEqual(list(differences.index), [("twin", "noise", "F1")])
        self.assertGreater(differences.iloc[0]["difference"], 0)

    def test_unknown_reference_raises(self):
        y, groups, probabilities = make_data(rows=200, articles=20)
        with self.assertRaises(KeyError):
            paired_bootstrap_comparison(y, probabilities, groups, reference="missing", repeats=5)


class HelperTests(unittest.TestCase):
    def test_bootstrap_p_value(self):
        self.assertEqual(bootstrap_p_value(np.ones(100)), 0.0)
        self.assertAlmostEqual(bootstrap_p_value(np.linspace(-1, 1, 101)), 1.0)

    def test_holm_matches_hand_computation(self):
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])

    def test_mcnemar(self):
        y = np.zeros(200, dtype=int)
        a = np.zeros(200, dtype=int)
        b = np.r_[np.ones(60, dtype=int), np.zeros(140, dtype=int)]
        result = mcnemar_test(y, a, b)
        self.assertEqual(result["only A correct"], 60)
        self.assertLess(result["p-value"], 1e-6)
        self.assertEqual(mcnemar_test(y, a, a)["p-value"], 1.0)
        small = mcnemar_test(y, a, np.r_[np.ones(3, dtype=int), np.zeros(197, dtype=int)])
        self.assertEqual(small["method"], "exact binomial")


if __name__ == "__main__":
    unittest.main()
