"""Models, evaluation metrics, lending economics, the training workflow and model persistence."""

import json
import unittest

import numpy as np

from creditrisk import datagen, persistence, report, schema as S, training
from creditrisk import evaluation as E
from creditrisk import models as M
from creditrisk import preprocessing as P
from creditrisk.exceptions import DataError, ModelError, NotTrainedError
from tests.helpers import TempDirTest, shared


class TestMetrics(unittest.TestCase):
    def test_confusion_counts_and_rates(self):
        actual = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        prob = [0.9, 0.8, 0.4, 0.1, 0.7, 0.3, 0.2, 0.1, 0.05, 0.0]
        m = E.classification_metrics(actual, prob, 0.5)
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (2, 1, 2, 5))
        self.assertAlmostEqual(m["recall"], 50.0)
        self.assertAlmostEqual(m["precision"], 100 * 2 / 3)
        self.assertAlmostEqual(m["accuracy"], 70.0)
        self.assertAlmostEqual(m["approval_rate"], 70.0)
        self.assertAlmostEqual(m["bad_rate_approved"], 100 * 2 / 7)
        self.assertEqual((m["declined"], m["approved"]), (3, 7))

    def test_nothing_declined_does_not_divide_by_zero(self):
        m = E.classification_metrics([1, 0, 1], [0.1, 0.1, 0.1], 0.9)
        self.assertEqual((m["precision"], m["f1"], m["recall"]), (0.0, 0.0, 0.0))
        everyone = E.classification_metrics([1, 0], [0.9, 0.9], 0.5)
        self.assertEqual((everyone["approved"], everyone["bad_rate_approved"]), (0, 0.0))

    def test_ranking_metrics(self):
        self.assertAlmostEqual(E.ranking_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])["roc_auc"], 1.0)
        self.assertAlmostEqual(E.ranking_metrics([0, 1, 0, 1], [0.5] * 4)["roc_auc"], 0.5)
        self.assertEqual(E.ranking_metrics([1, 1], [0.4, 0.6])["roc_auc"], 0.0)

    def test_age_groups(self):
        self.assertEqual([E.age_group(a) for a in (21, 29, 30, 39, 40, 49, 50, 70)],
                         ["under 30", "under 30", "30-39", "30-39", "40-49", "40-49", "50 or older", "50 or older"])


class TestLendingEconomics(unittest.TestCase):
    def test_profit_arithmetic(self):
        actual = np.array([0, 0, 1, 0])
        prob = np.array([0.05, 0.10, 0.30, 0.50])
        amount = np.array([10.0, 20.0, 5.0, 8.0])
        r = E.lending_result(actual, prob, amount, 0.4, margin=0.1, loss=0.5)       # approves the first three
        self.assertEqual((r["approved"], r["defaults"]), (3, 1))
        self.assertAlmostEqual(r["profit"], 0.1 * 10 + 0.1 * 20 - 0.5 * 5)
        self.assertAlmostEqual(r["lent"], 35.0)
        self.assertAlmostEqual(r["default_rate"], 100 / 3)
        self.assertAlmostEqual(r["return_on_lent"], 100 * 0.5 / 35)

    def test_declining_everyone_earns_nothing_and_approving_everyone_takes_all_risk(self):
        actual, prob, amount = np.array([0, 1]), np.array([0.2, 0.9]), np.array([4.0, 4.0])
        none = E.lending_result(actual, prob, amount, 0.0)
        self.assertEqual((none["approved"], none["profit"], none["return_on_lent"]), (0, 0.0, 0.0))
        everyone = E.lending_result(actual, prob, amount, 1.1, margin=0.1, loss=0.5)
        self.assertAlmostEqual(everyone["profit"], 0.4 - 2.0)

    def test_break_even_threshold_is_where_expected_profit_turns_negative(self):
        t = S.break_even(0.12, 0.6)
        for p, sign in ((t - 0.02, 1), (t + 0.02, -1)):
            expected = (1 - p) * 0.12 - p * 0.6
            self.assertEqual(np.sign(expected), sign)
        self.assertAlmostEqual((1 - t) * 0.12 - t * 0.6, 0.0)

    def test_threshold_choice_uses_break_even_and_checks_training_profit(self):
        rng = np.random.default_rng(0)
        prob = rng.beta(1, 6, 3000)
        actual = (rng.random(3000) < prob).astype(int)
        amount = rng.uniform(1, 10, 3000)
        info = E.choose_approval_threshold(actual, prob, amount, 0.12, 0.6)
        self.assertAlmostEqual(info["approve_below"], 1 / 6)
        self.assertTrue(info["policy_beats_approving_all"])
        self.assertGreater(info["training_profit"], info["training_profit_approve_all"])
        self.assertGreater(info["training_profit"], 0.8 * info["empirical_best_profit"])   # the simple rule is near the best
        cheap = E.choose_approval_threshold(actual, prob, amount, 0.05, 0.9)
        self.assertLess(cheap["approve_below"], info["approve_below"])                      # bigger losses: stricter

    def test_policy_table(self):
        rng = np.random.default_rng(1)
        prob = rng.beta(1, 5, 1500)
        actual = (rng.random(1500) < prob).astype(int)
        amount = rng.uniform(1, 8, 1500)
        rows = {r["policy"]: r for r in E.policy_table(actual, prob, amount, 1 / 6)}
        self.assertEqual(rows["Decline everyone"]["profit"], 0.0)
        self.assertEqual(rows["Approve everyone"]["approved"], 1500)
        self.assertGreater(rows["Use the model"]["profit"], rows["Approve everyone"]["profit"])
        self.assertGreater(rows["Perfect foresight (impossible)"]["profit"], rows["Use the model"]["profit"])
        self.assertEqual(rows["Perfect foresight (impossible)"]["defaults"], 0)

    def test_profit_curve_peaks_near_the_break_even(self):
        rng = np.random.default_rng(2)
        prob = rng.beta(1, 5, 4000)
        actual = (rng.random(4000) < prob).astype(int)
        amount = np.full(4000, 5.0)
        curve = E.profit_curve(actual, prob, amount)
        best = max(curve, key=lambda point: point[1])
        self.assertLess(abs(best[0] - 1 / 6), 0.06)
        self.assertEqual(curve[0][1], 0.0)

    def test_bands_add_up(self):
        rng = np.random.default_rng(3)
        prob = rng.random(1000) * 0.5
        actual = (rng.random(1000) < prob).astype(int)
        bands = E.band_table(actual, prob, 0.3)
        self.assertEqual([b["decision"] for b in bands], ["Approve", "Manual review", "Decline"])
        self.assertEqual(sum(b["applicants"] for b in bands), 1000)
        self.assertEqual(sum(b["defaults"] for b in bands), int(actual.sum()))
        self.assertLess(bands[0]["rate"], bands[1]["rate"])
        self.assertLess(bands[1]["rate"], bands[2]["rate"])

    def test_calibration_and_roc_tables(self):
        prob = np.linspace(0.01, 0.99, 400)
        actual = (np.random.default_rng(2).random(400) < prob).astype(int)
        for row in E.calibration_table(actual, prob):
            self.assertLess(abs(row["predicted"] - row["observed"]), 20)
        roc = E.roc_points(actual, prob)
        self.assertEqual(roc[0], (0.0, 0.0))
        self.assertEqual(roc[-1], (1.0, 1.0))
        self.assertTrue(all(b[0] >= a[0] and b[1] >= a[1] for a, b in zip(roc, roc[1:])))


class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cleaned, _ = P.clean(datagen.generate(1500, seed=11))
        cls.X, cls.y = P.split_xy(cleaned)

    def test_five_candidates_with_a_baseline(self):
        names = list(M.candidates(fast=True))
        self.assertEqual(len(names), 5)
        self.assertIn(M.BASELINE, names)

    def test_comparison_is_sorted_and_beats_the_baseline(self):
        results = M.compare(self.X, self.y, cv=3, fast=True)
        aucs = [r["roc_auc"] for r in results]
        self.assertEqual(aucs, sorted(aucs, reverse=True))
        baseline = next(r for r in results if r["model"] == M.BASELINE)
        self.assertAlmostEqual(baseline["roc_auc"], 0.5, places=2)
        best = M.best_candidate(results)
        self.assertNotEqual(best["model"], M.BASELINE)
        self.assertGreater(best["roc_auc"], 0.75)

    def test_tuning_returns_known_parameters(self):
        _, params = M.tune("Logistic Regression", self.X, self.y, cv=3, iterations=3)
        self.assertIn("C", params)
        self.assertEqual(M.tune(M.BASELINE, self.X, self.y)[1], {})
        pipe = M.pipeline_with_params("Logistic Regression", params).fit(self.X, self.y)
        self.assertEqual(pipe.named_steps["model"].C, params["C"])

    def test_calibration_choice_is_never_worse(self):
        pipe = M.build_pipeline(M.candidates(fast=True)["Random Forest"])
        model, oof, info = M.choose_calibration(pipe, self.X, self.y, cv=3)
        self.assertLessEqual(info["brier_after"], info["brier_before"] + 1e-4)
        self.assertEqual(len(oof), len(self.y))
        self.assertTrue(((oof >= 0) & (oof <= 1)).all())
        model.fit(self.X, self.y)
        self.assertEqual(model.predict_proba(self.X.head(4)).shape, (4, 2))


class TestTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result, _, cls.raw = shared()
        cls.meta = cls.result.metadata

    def test_the_model_is_good_and_close_to_the_ceiling(self):
        test = self.meta["metrics"]["test"]
        self.assertGreater(test["roc_auc"], 0.78)
        self.assertGreater(test["roc_auc"], self.meta["metrics"]["baseline_cv"]["roc_auc"] + 0.2)
        ceiling = self.meta["ceiling"]["roc_auc"]
        self.assertLess(test["roc_auc"], ceiling + 0.05)                       # nothing can beat the truth by much
        self.assertGreater(test["roc_auc"], ceiling - 0.08)

    def test_test_set_is_untouched_and_stratified(self):
        rows = self.meta["rows"]
        self.assertEqual(rows["train"] + rows["test"], rows["clean"])
        self.assertAlmostEqual(rows["test"] / rows["clean"], 0.2, delta=0.01)
        frame = self.result.test_frame
        self.assertEqual(len(frame), rows["test"])
        self.assertAlmostEqual(frame["actual"].mean(), self.result.reference[S.TARGET].mean(), delta=0.02)

    def test_approval_threshold_is_the_break_even_point(self):
        eco = self.meta["economics"]
        self.assertAlmostEqual(self.meta["thresholds"]["approve_below"], S.break_even(eco["margin"], eco["loss"]))
        self.assertAlmostEqual(self.meta["thresholds"]["review_from"], self.meta["thresholds"]["approve_below"] / 2)
        self.assertTrue(eco["policy_beats_approving_all"])

    def test_the_rule_beats_approving_everyone_on_unseen_loans(self):
        policies = {p["policy"]: p for p in self.meta["policies"]}
        model, everyone = policies["Use the model"], policies["Approve everyone"]
        self.assertGreater(model["profit"], everyone["profit"])
        self.assertGreater(model["profit"], 0)
        self.assertLess(model["default_rate"], everyone["default_rate"])
        self.assertLess(model["profit"], policies["Perfect foresight (impossible)"]["profit"])
        self.assertEqual(policies["Decline everyone"]["profit"], 0.0)

    def test_bands_separate_applicants(self):
        low, review, high = self.meta["bands"]
        self.assertLess(low["rate"], review["rate"])
        self.assertLess(review["rate"], high["rate"])
        self.assertLess(low["rate"], 8)
        self.assertEqual(low["applicants"] + review["applicants"] + high["applicants"], self.meta["rows"]["test"])

    def test_importance_includes_the_obvious_drivers(self):
        top = [i["feature"] for i in self.meta["importance"][:6]]
        self.assertIn("credit_score", top)
        self.assertGreater(self.meta["importance"][0]["importance"], 0.03)

    def test_metadata_is_complete_and_json_safe(self):
        for key in ("model", "calibration", "metrics", "economics", "thresholds", "policies", "bands", "segments", "roc",
                    "profit_curve", "comparison", "cleaning", "training_ranges", "typical", "environment", "calibration_table"):
            self.assertIn(key, self.meta)
        json.dumps(self.meta, default=persistence._json_default)
        self.assertEqual(set(self.meta["typical"]), set(S.RAW_FEATURES))

    def test_segments_cover_employment_purpose_and_age(self):
        segments = self.meta["segments"]
        self.assertEqual({s["group"] for s in segments["employment"]}, set(S.EMPLOYMENT))
        self.assertEqual({s["group"] for s in segments["purpose"]}, set(S.PURPOSES))
        self.assertEqual([s["group"] for s in segments["age"]], E.AGE_GROUPS)
        self.assertEqual(sum(s["n"] for s in segments["employment"]), self.meta["rows"]["test"])
        self.assertLessEqual(sum(s["n"] for s in segments["age"]), self.meta["rows"]["test"])

    def test_age_is_not_used_by_the_model(self):
        X = self.result.test_frame[S.RAW_FEATURES].head(40)
        with_age = X.assign(age=np.arange(40, 80))
        flipped = X.assign(age=np.arange(80, 40, -1))
        np.testing.assert_allclose(self.result.pipeline.predict_proba(with_age), self.result.pipeline.predict_proba(flipped))
        self.assertNotIn(S.MONITOR, self.result.reference.columns)

    def test_training_without_an_age_column_skips_the_age_check(self):
        raw = datagen.generate(1500, seed=7).drop(columns=[S.MONITOR])
        result = training.train(raw, fast=True, tune=False, cv=3, source="synthetic")
        self.assertNotIn("age", result.metadata["segments"])

    def test_same_seed_same_result(self):
        raw = datagen.generate(1500, seed=9)
        a = training.train(raw, fast=True, tune=False, cv=3, source="synthetic")
        b = training.train(raw, fast=True, tune=False, cv=3, source="synthetic")
        self.assertEqual(a.metadata["metrics"]["test"]["roc_auc"], b.metadata["metrics"]["test"]["roc_auc"])
        self.assertEqual(a.metadata["policies"][2]["profit"], b.metadata["policies"][2]["profit"])

    def test_bad_settings_and_data_are_refused(self):
        with self.assertRaisesRegex(DataError, "margin"):
            training.train(self.raw, margin=1.5)
        with self.assertRaisesRegex(DataError, "loss"):
            training.train(self.raw, loss=0)
        with self.assertRaises(DataError):
            training.train(self.raw.head(50))

    def test_progress_is_reported(self):
        steps = []
        training.train(datagen.generate(900, seed=4), fast=True, tune=False, cv=3, progress=steps.append)
        self.assertGreaterEqual(len(steps), 8)
        self.assertTrue(steps[0].startswith("Cleaning"))

    def test_harsher_losses_make_the_rule_stricter(self):
        raw = datagen.generate(1500, seed=6)
        mild = training.train(raw, fast=True, tune=False, cv=3, loss=0.4)
        harsh = training.train(raw, fast=True, tune=False, cv=3, loss=0.9)
        self.assertLess(harsh.metadata["thresholds"]["approve_below"], mild.metadata["thresholds"]["approve_below"])
        self.assertLess(harsh.metadata["metrics"]["test"]["approval_rate"], mild.metadata["metrics"]["test"]["approval_rate"])


class TestPersistence(TempDirTest):
    def test_round_trip_gives_identical_predictions(self):
        result, model_dir, raw = shared()
        loaded = persistence.load(model_dir)
        X = result.test_frame[S.RAW_FEATURES].head(30)
        np.testing.assert_allclose(result.pipeline.predict_proba(X), loaded.pipeline.predict_proba(X))
        self.assertTrue(persistence.is_trained(model_dir))
        self.assertEqual(loaded.metadata["model"], result.metadata["model"])
        self.assertEqual(len(loaded.reference), len(result.reference))

    def test_no_model_gives_instructions(self):
        with self.assertRaisesRegex(NotTrainedError, "Train one first"):
            persistence.load(self.dir / "empty")

    def test_tampering_is_detected(self):
        target = self.copy_model()
        with open(target / persistence.MODEL_FILE, "ab") as fh:
            fh.write(b"x")
        with self.assertRaisesRegex(ModelError, "checksum"):
            persistence.load(target)

    def test_missing_files_and_damaged_metadata(self):
        target = self.copy_model()
        (target / persistence.REFERENCE_FILE).unlink()
        with self.assertRaisesRegex(ModelError, "missing"):
            persistence.load(target)
        target = self.copy_model()
        (target / persistence.META_FILE).write_text("{not json", encoding="utf-8")
        with self.assertRaisesRegex(ModelError, "damaged"):
            persistence.load(target)
        target = self.copy_model()
        meta = json.loads((target / persistence.META_FILE).read_text(encoding="utf-8"))
        meta.pop("files")
        (target / persistence.META_FILE).write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaisesRegex(ModelError, "checksums"):
            persistence.load(target)

    def test_version_mismatch_is_a_warning(self):
        target = self.copy_model()
        meta = json.loads((target / persistence.META_FILE).read_text(encoding="utf-8"))
        meta["environment"]["sklearn"] = "0.0.1"
        (target / persistence.META_FILE).write_text(json.dumps(meta), encoding="utf-8")
        self.assertTrue(any("0.0.1" in w for w in persistence.load(target).warnings))


class TestReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = shared()[0].metadata

    def test_text_report_has_the_key_sections(self):
        text = report.model_text(self.meta)
        for part in ("HOW WELL IT RANKS", "LENDING RULE", "WHAT EACH POLICY EARNS", "RISK BANDS", "WHAT THE MODEL RELIES ON",
                     "WHAT THIS MEANS", "must not be used to make real lending decisions", "Approve everyone", "Fairness check"):
            self.assertIn(part, text)
        self.assertIn("MODEL COMPARISON", report.comparison_text(self.meta))
        self.assertIn("chosen", report.comparison_text(self.meta))

    def test_html_report_is_self_contained_and_accessible(self):
        page = report.render_html(self.meta)
        self.assertTrue(page.startswith("<!doctype html>"))
        self.assertEqual(page.count("<svg"), page.count("</svg>"))
        self.assertGreaterEqual(page.count('role="img"'), 5)
        self.assertGreaterEqual(page.count('<details class="tv">'), 5)
        self.assertIn("prefers-color-scheme:dark", page)
        self.assertNotIn("http://", page.replace("http://www.w3.org", ""))
        self.assertIn("By age group", page)
        self.assertIn("By employment type", page)

    def test_html_report_escapes_text_from_the_data(self):
        meta = json.loads(json.dumps(self.meta, default=persistence._json_default))
        meta["source"] = "<script>alert(1)</script>"
        page = report.render_html(meta)
        self.assertNotIn("<script>alert(1)", page)
        self.assertIn("&lt;script&gt;alert(1)", page)

    def test_html_report_without_age_checks(self):
        meta = json.loads(json.dumps(self.meta, default=persistence._json_default))
        meta["segments"].pop("age")
        page = report.render_html(meta)
        self.assertNotIn("By age group", page)
        self.assertNotIn("Fairness check by age group", "\n".join(report._verdict(meta)))

    def test_old_models_without_results_are_refused(self):
        with self.assertRaises(ModelError):
            report.render_html({"metrics": {}})

    def test_save_html(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path = report.save_html(self.meta, Path(tmp) / "sub" / "r.html")
            self.assertTrue(path.exists() and path.stat().st_size > 5000)


if __name__ == "__main__":
    unittest.main()
