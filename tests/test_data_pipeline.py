import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/data"))
import data_contract as common
import process_data as demand

class DemandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "transactions.csv"
        self.articles = pd.DataFrame({"article_id": ["0001", "0002"]})
        self.raw = pd.DataFrame({
            "t_dat": pd.date_range("2018-09-20", "2019-02-03").strftime("%Y-%m-%d"),
            "customer_id": "customer", "article_id": "0001", "price": 0.1,
            "sales_channel_id": 2,
        })

    def tearDown(self):
        self.temp.cleanup()

    def aggregate(self, raw=None, chunks=17, **kwargs):
        (self.raw if raw is None else raw).to_csv(self.path, index=False)
        return demand.aggregate_transactions(self.path, self.articles, chunks, **kwargs)

    def test_chunk_boundaries_and_leading_zero(self):
        extra = self.raw.iloc[[10]].copy()
        extra["price"] = 0.3
        raw = pd.concat([self.raw, extra]).sort_values("t_dat", kind="stable")
        first, _, _ = self.aggregate(raw, 11)
        second, _, _ = self.aggregate(raw, 1000)
        pd.testing.assert_frame_equal(first, second, check_exact=False, atol=1e-14, rtol=1e-14)
        self.assertEqual(first["sales"].sum(), len(raw))
        self.assertEqual(set(first["article_id"]), {"0001"})

    def test_partial_weeks_history_iso_and_target(self):
        weekly, complete, _ = self.aggregate()
        data = demand.add_temporal_features(weekly, complete)
        self.assertEqual(data.week.min(), pd.Timestamp("2018-12-17"))
        self.assertEqual(data.week.max(), pd.Timestamp("2019-01-21"))
        row = data.loc[data.week == "2018-12-31"].iloc[0]
        self.assertEqual((row.iso_year, row.iso_week), (2019, 1))
        self.assertTrue(data.target_next_week_sales.eq(7).all())
        self.assertTrue(data.rolling_4w_sales_std.eq(0).all())
        self.assertTrue(data.rolling_12w_active_weeks.eq(12).all())
        self.assertTrue(data.sales_trend_4w.eq(0).all())
        self.assertTrue(data.rolling_4w_online_share.eq(1).all())
        self.assertNotIn(pd.Timestamp("2018-09-17"), complete)

    def test_absent_article_is_zero_only_in_covered_week(self):
        # Continúa habiendo datos diarios del otro artículo: cobertura completa.
        raw = self.raw.copy()
        raw.loc[raw.t_dat.between("2019-01-14", "2019-01-20"), "article_id"] = "0002"
        weekly, complete, _ = self.aggregate(raw)
        data = demand.add_temporal_features(weekly, complete)
        row = data.loc[(data.week == "2019-01-07") & (data.article_id == "0001")].iloc[0]
        self.assertEqual(row.target_next_week_sales, 0)
        self.assertFalse(((data.week == "2019-01-14") & (data.article_id == "0001")).any())

    def test_max_rows_does_not_trust_final_sunday(self):
        _, complete, info = self.aggregate(max_rows=len(self.raw))
        self.assertNotIn(pd.Timestamp("2019-01-28"), complete)
        self.assertTrue(info["preview"])

    def test_duplicate_transactions_are_counted_not_removed(self):
        raw = pd.concat([self.raw.iloc[:1], self.raw], ignore_index=True)
        weekly, _, info = self.aggregate(raw, 1)
        self.assertEqual(info["identical_transaction_rows_preserved"], 1)
        self.assertEqual(weekly.sales.sum(), len(raw))

    def test_missing_day_rejected(self):
        with self.assertRaisesRegex(ValueError, "Días sin registros"):
            self.aggregate(self.raw.drop(index=20))

    def test_invalid_fields_rejected(self):
        for column, value in [("article_id", None), ("article_id", "9999"),
                              ("customer_id", ""), ("price", -1), ("price", np.inf),
                              ("sales_channel_id", 3), ("t_dat", "2040-01-01")]:
            raw = self.raw.copy()
            raw.loc[0, column] = value
            with self.subTest(column=column, value=value), self.assertRaises((ValueError, TypeError)):
                self.aggregate(raw)

    def test_temporal_split(self):
        weekly, complete, _ = self.aggregate()
        data, config = demand.assign_splits(demand.add_temporal_features(weekly, complete))
        self.assertEqual(set(data.split), {"train", "validation", "test"})
        self.assertEqual(config["requested_percentages"], {"train": 70.0, "validation": 15.0, "test": 15.0})
        for left, right in [("train", "validation"), ("validation", "test")]:
            self.assertLess(data.loc[data.split == left, "target_week"].max(),
                            data.loc[data.split == right, "target_week"].min())

class PercentageSplitTests(unittest.TestCase):
    def make_data(self, counts):
        dates = pd.date_range("2020-01-06", periods=len(counts), freq="W-MON")
        return pd.DataFrame({"target_week": dates.repeat(counts), "article_id": range(sum(counts))})

    def test_exact_percentages_when_week_sizes_allow(self):
        data, config = demand.assign_splits(self.make_data([40, 40, 10, 10]), .8, .1, .1)
        self.assertEqual(data.split.value_counts().to_dict(), {"train": 80, "validation": 10, "test": 10})
        self.assertEqual(config["validation_start"], "2020-01-20")
        self.assertEqual(config["test_start"], "2020-01-27")

    def test_nearest_whole_week_split_not_random(self):
        data = self.make_data([50, 30, 8, 7, 5])
        result, config = demand.assign_splits(data, .8, .1, .1)
        self.assertEqual(result.split.value_counts().to_dict(), {"train": 80, "test": 12, "validation": 8})
        self.assertTrue(result.groupby("target_week").split.nunique().eq(1).all())
        shuffled, again = demand.assign_splits(data.sample(frac=1, random_state=7), .8, .1, .1)
        self.assertEqual(config, again)
        pd.testing.assert_series_equal(result.set_index("article_id").split.sort_index(),
                                       shuffled.set_index("article_id").split.sort_index())

    def test_invalid_percentages_and_small_data(self):
        for fractions in [(0.8, 0.2, 0.2), (0, 0.5, 0.5), (np.nan, 0.1, 0.1)]:
            with self.subTest(fractions=fractions), self.assertRaises(ValueError):
                demand.assign_splits(self.make_data([10, 10, 10]), *fractions)
        with self.assertRaises(ValueError):
            demand.assign_splits(self.make_data([10, 10]))

    def test_classification_threshold_uses_train_only(self):
        data = pd.DataFrame({
            "split": ["train"] * 5 + ["validation", "test"],
            "target_next_week_sales": [0, 1, 2, 4, 8, 1_000, 2_000],
        })
        result, details = demand.add_classification_target(data, quantile=.5)
        self.assertEqual(details["threshold_units"], 4)
        self.assertEqual(result.target_high_demand.tolist(), [0, 0, 0, 1, 1, 1, 1])
        changed = data.copy()
        changed.loc[changed.split.ne("train"), "target_next_week_sales"] = 1_000_000
        _, repeated = demand.add_classification_target(changed, quantile=.5)
        self.assertEqual(details["threshold_units"], repeated["threshold_units"])

    def test_clustering_table_uses_training_history_without_targets(self):
        rows = pd.DataFrame({
            "article_id": ["001", "001", "002", "001"],
            "week": pd.to_datetime(["2020-01-06", "2020-01-13", "2020-01-06", "2020-02-03"]),
            "split": ["train", "train", "train", "validation"],
            "sales": [2, 4, 3, 999], "rolling_12w_sales_mean": [1., 2., 1.5, 999.],
            "rolling_12w_sales_std": [.5, 1., .75, 999.],
            "sales_trend_4w": [0., 1., -1., 999.],
            "rolling_12w_active_weeks": [6, 8, 7, 12],
            "mean_price": [.1, .2, .3, 99.], "online_sales": [1, 4, 0, 999],
            "target_next_week_sales": [5, 6, 7, 999], "target_high_demand": [0, 1, 1, 1],
        })
        for column in common.ARTICLE_COLUMNS[1:]:
            rows[column] = "category"
        clusters = demand.build_clustering_table(rows)
        self.assertEqual(set(clusters.article_id), {"001", "002"})
        self.assertEqual(clusters.loc[clusters.article_id.eq("001"), "article_weeks"].iloc[0], 2)
        self.assertFalse(any(column.startswith("target_") for column in clusters.columns))

    def test_optional_gap_keeps_whole_weeks_and_all_rows(self):
        data = self.make_data([10] * 20)
        result, config = demand.assign_splits(data, gap_weeks=1)
        self.assertEqual(len(data), len(result))
        self.assertEqual(config["gap_rows"], 20)
        self.assertTrue(result.groupby("target_week").split.nunique().eq(1).all())
        self.assertAlmostEqual(sum(config["actual_percentages"].values()), 100)


class PublicationTests(unittest.TestCase):
    def test_atomic_publication_and_preview(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            first = common.publish(out, {"test": pd.DataFrame({"id": ["001"]})}, {"summary": {"ok": True}})
            before = (out / "CURRENT.json").read_bytes()
            with patch.object(pd.DataFrame, "to_csv", side_effect=RuntimeError("interrupted")):
                with self.assertRaises(RuntimeError):
                    common.publish(out, {"test": pd.DataFrame({"id": ["002"]})}, {})
            self.assertEqual(before, (out / "CURRENT.json").read_bytes())
            common.publish(out, {}, {"summary": {"preview": True}}, preview=True)
            self.assertEqual(before, (out / "CURRENT.json").read_bytes())
            self.assertTrue((first / "checksums.json").is_file())

    def test_identifier_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "articles.csv"
            row = {col: "A" for col in common.ARTICLE_COLUMNS}
            row.update(article_id="00001", product_code="0001")
            pd.DataFrame([row]).to_csv(path, index=False)
            self.assertEqual(common.load_articles(path).article_id.iloc[0], "00001")
            row["article_id"] = ""
            pd.DataFrame([row]).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                common.load_articles(path)

if __name__ == "__main__":
    unittest.main()
