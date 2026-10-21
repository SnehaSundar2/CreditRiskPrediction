"""Input validation and the prediction service."""

import math
import unittest

import numpy as np
import pandas as pd

from creditrisk import schema as S
from creditrisk import validators as v
from creditrisk.exceptions import DataError, ValidationError
from creditrisk.predictor import MAX_BATCH, require_predictor
from tests.helpers import HIGH_RISK, LOW_RISK, MINIMAL, TempDirTest, shared


class TestParsing(unittest.TestCase):
    def test_money_units(self):
        self.assertEqual(v.parse_number("annual_income", "8.5")[0], 8.5)
        self.assertEqual(v.parse_number("annual_income", "8.5 lakh")[0], 8.5)
        self.assertEqual(v.parse_number("annual_income", "₹8.5 lakhs")[0], 8.5)
        self.assertEqual(v.parse_number("loan_amount", "Rs. 2L")[0], 2.0)
        self.assertEqual(v.parse_number("loan_amount", "250 k")[0], 2.5)

    def test_crore_is_converted_and_must_fit_the_range(self):
        number, note = v.parse_number("loan_amount", "0.4 crore")
        self.assertEqual(number, 40)
        self.assertIn("lakh", note)
        with self.assertRaisesRegex(ValueError, "between"):
            v.parse_number("annual_income", "2 crore")

    def test_rupees_are_recognised(self):
        number, note = v.parse_number("annual_income", "850000")
        self.assertAlmostEqual(number, 8.5)
        self.assertIn("rupees", note)
        number, note = v.parse_number("loan_amount", 400000)
        self.assertAlmostEqual(number, 4.0)
        self.assertEqual(v.parse_number("loan_amount", 4)[1], None)

    def test_terms_rates_and_other_units(self):
        self.assertEqual(v.parse_number("loan_term_months", "36")[0], 36)
        number, note = v.parse_number("loan_term_months", "3 years")
        self.assertEqual(number, 36)
        self.assertIn("months", note)
        self.assertEqual(v.parse_number("interest_rate", "11.5%")[0], 11.5)
        self.assertEqual(v.parse_number("credit_utilisation", "40 %")[0], 40)
        self.assertEqual(v.parse_number("employment_years", "6 years")[0], 6)
        self.assertAlmostEqual(v.parse_number("employment_years", "18 months")[0], 1.5)
        self.assertEqual(v.parse_number("monthly_debt_payments", "12k")[0], 12)
        self.assertEqual(v.parse_number("monthly_debt_payments", "0.1 lakh")[0], 10)
        self.assertEqual(v.parse_number("credit_score", "720")[0], 720)

    def test_bad_numbers_have_clear_messages(self):
        cases = [("annual_income", "abc", "must be a number"), ("annual_income", "0.2", "between 1 and 100"),
                 ("credit_score", "950", "between 300 and 900"), ("credit_score", "720.5", "whole number"),
                 ("loan_term_months", "2", "between 6 and 120"), ("interest_rate", "40", "between 5 and 30"),
                 ("annual_income", True, "must be a number"), ("annual_income", "inf", "must be a number"),
                 ("annual_income", float("nan"), "normal number"), ("annual_income", "8 kg", "unit"),
                 ("credit_score", "700 points", "plain number"), ("loan_term_months", "3 weeks", "months or years"),
                 ("credit_utilisation", "30 apples", "percentage")]
        for feature, value, message in cases:
            with self.assertRaisesRegex(ValueError, message, msg=f"{feature}={value!r}"):
                v.parse_number(feature, value)

    def test_categories_suggest_corrections(self):
        self.assertEqual(v.parse_category("employment_type", "salary"), "Salaried")
        with self.assertRaisesRegex(ValueError, "Did you mean 'Salaried'"):
            v.parse_category("employment_type", "Salaried!")
        with self.assertRaisesRegex(ValueError, "Choose one of: Home, Education"):
            v.parse_category("loan_purpose", "holiday")

    def test_a_minimal_record_is_complete(self):
        record, notes, assumed = v.parse_features(MINIMAL)
        self.assertEqual(set(record), set(S.RAW_FEATURES))
        self.assertTrue(math.isnan(record["interest_rate"]))
        self.assertEqual(set(assumed), set(S.RAW_FEATURES) - set(S.REQUIRED_INPUT))
        self.assertEqual(notes, [])

    def test_keys_are_forgiving_and_dataset_columns_are_ignored(self):
        record, _, _ = v.parse_features({"Annual Income": "7", "loan-amount": "3", "LOAN_TERM_MONTHS": "24", "credit_score": "650",
                                         "employment_type": "business", "application_id": "X1", "age": "44"})
        self.assertEqual((record["annual_income"], record["employment_type"]), (7.0, "Business owner"))
        self.assertNotIn("age", record)

    def test_every_problem_is_reported_at_once(self):
        with self.assertRaises(ValidationError) as ctx:
            v.parse_features({"annual_income": "x", "employment_type": "pirate", "loan_amount": 999, "owns_home": "perhaps", "colour": "red"})
        errors = ctx.exception.errors
        for field in ("annual_income", "employment_type", "loan_amount", "owns_home", "loan_term_months", "credit_score", "colour"):
            self.assertIn(field, errors)

    def test_missing_required_fields(self):
        with self.assertRaises(ValidationError) as ctx:
            v.parse_features({"annual_income": 5})
        self.assertEqual(set(ctx.exception.errors), {"loan_amount", "loan_term_months", "credit_score", "employment_type"})
        with self.assertRaises(ValidationError):
            v.parse_features({**MINIMAL, "credit_score": ""})

    def test_unknown_fields_get_a_hint(self):
        with self.assertRaisesRegex(ValidationError, "Did you mean 'credit_utilisation'"):
            v.parse_features({**MINIMAL, "credit_utilization": 20})

    def test_late_payments_need_credit_accounts(self):
        with self.assertRaisesRegex(ValidationError, "no credit accounts"):
            v.parse_features({**MINIMAL, "credit_accounts": 0, "late_payments": 2})
        v.parse_features({**MINIMAL, "credit_accounts": 0, "late_payments": 0})

    def test_non_dict_is_refused(self):
        with self.assertRaises(ValidationError):
            v.parse_features([1, 2, 3])

    def test_settings_validators(self):
        self.assertEqual(v.validate_port("8050"), 8050)
        for bad in ("80", "99999", "x"):
            with self.assertRaises(ValidationError):
                v.validate_port(bad)
        self.assertEqual(v.validate_int(" 5 ", "N", 1, 10), 5)
        with self.assertRaises(ValidationError):
            v.validate_int("11", "N", 1, 10)
        self.assertEqual(v.validate_number("0.12", "M", 0.01, 1), 0.12)
        with self.assertRaises(ValidationError):
            v.validate_number("nan", "M", 0.01, 1)


class TestPredictions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = require_predictor(shared()[1])

    def test_risky_and_safe_applicants(self):
        high, low = self.predictor.predict(HIGH_RISK), self.predictor.predict(LOW_RISK)
        self.assertEqual((high.band, high.decision), ("High", "Decline"))
        self.assertEqual((low.band, low.decision), ("Low", "Approve"))
        self.assertGreater(high.probability, 0.5)
        self.assertLess(low.probability, 0.05)
        self.assertLess(high.expected_profit_share, 0)
        self.assertGreater(low.expected_profit_share, 0)

    def test_expected_profit_matches_the_economics(self):
        p = self.predictor.predict(MINIMAL)
        m, l = self.predictor.margin, self.predictor.loss
        self.assertAlmostEqual(p.expected_profit_share, (1 - p.probability) * m - p.probability * l)
        at_break_even = (1 - self.predictor.threshold) * m - self.predictor.threshold * l
        self.assertAlmostEqual(at_break_even, 0.0)

    def test_risk_moves_with_each_obvious_factor(self):
        base = self.predictor.predict(MINIMAL).probability
        for change in ({"credit_score": 540}, {"late_payments": 4, "credit_accounts": 5}, {"credit_utilisation": 90},
                       {"loan_amount": 14}, {"annual_income": 3}, {"employment_type": "Other"}, {"loan_purpose": "Business"}):
            self.assertGreater(self.predictor.predict({**MINIMAL, **change}).probability, base, change)
        for change in ({"credit_score": 830}, {"has_cosigner": "yes"}, {"loan_purpose": "Home"}):
            self.assertLess(self.predictor.predict({**MINIMAL, **change}).probability, base, change)

    def test_probabilities_stay_inside_zero_and_one(self):
        for person in (HIGH_RISK, LOW_RISK, MINIMAL, {**HIGH_RISK, "credit_score": 300, "late_payments": 12}):
            p = self.predictor.predict(person).probability
            self.assertTrue(0 < p < 1)

    def test_decisions_follow_the_thresholds(self):
        t = self.predictor.threshold
        seen = set()
        for score in (820, 760, 700, 650, 600, 550, 500):
            p = self.predictor.predict({**MINIMAL, "credit_score": score, "late_payments": 2 if score < 650 else 0, "credit_accounts": 4})
            expected = "Approve" if p.probability < t / 2 else "Manual review" if p.probability < t else "Decline"
            self.assertEqual(p.decision, expected)
            seen.add(p.decision)
        self.assertEqual(seen, {"Approve", "Manual review", "Decline"})

    def test_reasons_are_the_largest_risk_raisers(self):
        p = self.predictor.predict(HIGH_RISK)
        self.assertTrue(p.reasons)
        self.assertTrue(all(f["effect"] > 0 for f in p.reasons))
        self.assertLessEqual(len(p.reasons), 4)
        labels = [f["label"] for f in p.reasons]
        self.assertIn("Credit score", labels)
        self.assertEqual([f["effect"] for f in p.reasons], sorted((f["effect"] for f in p.reasons), reverse=True))
        self.assertEqual(self.predictor.predict(LOW_RISK).reasons, [])

    def test_factors_are_sorted_and_signed(self):
        p = self.predictor.predict(HIGH_RISK)
        sizes = [abs(f["effect"]) for f in p.factors]
        self.assertEqual(sizes, sorted(sizes, reverse=True))
        score = next(f for f in p.factors if f["feature"] == "credit_score")
        self.assertGreater(score["effect"], 0)
        good = next(f for f in self.predictor.predict(LOW_RISK).factors if f["feature"] == "credit_score")
        self.assertLess(good["effect"], 0)

    def test_what_ifs_only_offer_improvements(self):
        p = self.predictor.predict(HIGH_RISK)
        self.assertTrue(p.what_ifs)
        self.assertTrue(all(w["change"] < 0 and w["probability"] < p.probability for w in p.what_ifs))
        self.assertEqual([w["change"] for w in p.what_ifs], sorted(w["change"] for w in p.what_ifs))
        self.assertIn("utilisation", " ".join(w["scenario"] for w in p.what_ifs))
        healthy = self.predictor.predict(LOW_RISK)
        self.assertNotIn("credit_utilisation", [w["feature"] for w in healthy.what_ifs])       # already low

    def test_unknown_details_are_assumed_and_not_explained(self):
        p = self.predictor.predict(MINIMAL)
        self.assertIn("interest_rate", p.assumed)
        self.assertTrue(all(f["feature"] not in p.assumed for f in p.factors))
        self.assertEqual(self.predictor.predict(HIGH_RISK).assumed, [])

    def test_unit_notes_are_passed_on(self):
        p = self.predictor.predict({**MINIMAL, "annual_income": "850000", "loan_term_months": "4 years"})
        self.assertEqual(len(p.notes), 2)
        self.assertEqual(p.record["loan_term_months"], 48)
        self.assertAlmostEqual(p.record["annual_income"], 8.5)

    def test_repayment_figures(self):
        p = self.predictor.predict({**MINIMAL, "interest_rate": 12, "loan_amount": 10, "loan_term_months": 12, "monthly_debt_payments": 5})
        self.assertAlmostEqual(p.emi, 88.849, places=2)
        self.assertAlmostEqual(p.burden, (88.849 + 5) * 12 / 800, places=3)

    def test_warnings(self):
        far = self.predictor.predict({**MINIMAL, "loan_amount": 59, "annual_income": 3})
        self.assertTrue(any("ten times" in w for w in far.warnings))
        self.assertEqual(self.predictor.predict(MINIMAL).warnings, [])

    def test_to_dict_is_json_ready(self):
        import json
        d = self.predictor.predict(HIGH_RISK).to_dict()
        json.dumps(d)
        for key in ("default_risk_percent", "band", "decision", "main_reasons", "factors", "what_if", "break_even_percent",
                    "expected_profit_percent_of_loan", "monthly_instalment_rupees", "disclaimer", "assumed_typical_values_for"):
            self.assertIn(key, d)
        self.assertEqual(d["decision"], "Decline")
        self.assertIn("must not be used", d["disclaimer"])

    def test_bad_input_raises_validation_error(self):
        with self.assertRaises(ValidationError):
            self.predictor.predict({"annual_income": 0.1})

    def test_what_if_tool(self):
        results = self.predictor.what_if(MINIMAL, "credit_score", [800, 700, 600])
        probabilities = [p for _, p in results]
        self.assertEqual([x for x, _ in results], [800, 700, 600])
        self.assertEqual(probabilities, sorted(probabilities))
        self.assertEqual([x for x, _ in self.predictor.what_if(MINIMAL, "Has Cosigner", ["no", "yes"])], [0, 1])
        for bad in (("colour", [1]), ("credit_score", []), ("credit_score", [100]), ("credit_score", list(range(500, 530)))):
            with self.assertRaises(ValidationError):
                self.predictor.what_if(MINIMAL, *bad)

    def test_suggestions(self):
        self.assertEqual(self.predictor.suggestions("employment_type"), S.EMPLOYMENT)
        self.assertEqual(self.predictor.suggestions("owns_home"), ["no", "yes"])
        scores = self.predictor.suggestions("credit_score")
        self.assertEqual(len(scores), 5)
        self.assertEqual(scores, sorted(scores))
        self.assertTrue(all(isinstance(x, int) for x in scores))

    def test_health_summary(self):
        h = self.predictor.health()
        self.assertEqual(h["status"], "ok")
        self.assertTrue(0.5 < h["roc_auc"] < 1)
        self.assertAlmostEqual(h["break_even_percent"], 16.7, places=1)


class TestBatch(TempDirTest):
    @classmethod
    def setUpClass(cls):
        cls.predictor = require_predictor(shared()[1])

    def frame(self):
        return pd.DataFrame([
            {S.ID: "a", **HIGH_RISK},
            {S.ID: "b", **LOW_RISK},
            {S.ID: "c", "annual_income": "9 lakh", "loan_amount": "3 lakh", "loan_term_months": "2 years", "credit_score": "710", "employment_type": "salary"},
            {S.ID: "d", "annual_income": "0.1", "loan_amount": "3", "loan_term_months": "24", "credit_score": "710", "employment_type": "Salaried"},
            {S.ID: "e", "annual_income": "9", "loan_amount": "3", "loan_term_months": "24", "credit_score": "710", "employment_type": "Astronaut"},
        ])

    def test_rows_are_scored_independently(self):
        out = self.predictor.predict_frame(self.frame())
        self.assertEqual(list(out["status"][:3]), ["ok", "ok", "ok"])
        self.assertIn("between 1 and 100", out.loc[3, "status"])
        self.assertIn("row 5", out.loc[3, "status"])
        self.assertIn("not supported", out.loc[4, "status"])
        self.assertTrue(np.isnan(out.loc[3, "default_risk_percent"]))
        self.assertEqual(out.loc[0, "decision"], "Decline")
        self.assertEqual(out.loc[1, "decision"], "Approve")
        self.assertEqual(out[S.ID].tolist(), list("abcde"))

    def test_batch_matches_single_predictions(self):
        out = self.predictor.predict_frame(self.frame())
        self.assertAlmostEqual(out.loc[0, "default_risk_percent"], self.predictor.predict(HIGH_RISK).percent, delta=0.06)

    def test_extra_dataset_columns_are_carried_along_and_ignored(self):
        frame = self.frame().assign(age=40, default="No")
        out = self.predictor.predict_frame(frame)
        self.assertIn("age", out.columns)
        self.assertEqual(list(out["status"][:3]), ["ok", "ok", "ok"])

    def test_csv_round_trip(self):
        source = self.dir / "applications.csv"
        self.frame().to_csv(source, index=False)
        target = self.dir / "out" / "scored.csv"
        result = self.predictor.predict_csv(source, target)
        saved = pd.read_csv(target)
        self.assertEqual(len(saved), len(result))
        self.assertIn("default_risk_percent", saved.columns)

    def test_csv_problems(self):
        with self.assertRaisesRegex(DataError, "not found"):
            self.predictor.predict_csv(self.dir / "nope.csv", self.dir / "o.csv")
        empty = self.dir / "empty.csv"
        empty.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(DataError, "Could not read"):
            self.predictor.predict_csv(empty, self.dir / "o.csv")

    def test_batch_size_limit(self):
        with self.assertRaisesRegex(DataError, "At most"):
            self.predictor.predict_frame(pd.DataFrame([MINIMAL] * (MAX_BATCH + 1)))

    def test_file_with_no_valid_rows(self):
        out = self.predictor.predict_frame(pd.DataFrame([{"annual_income": 5}]))
        self.assertEqual(out.loc[0, "status"][:6], "row 2:")
        self.assertEqual(out.loc[0, "decision"], "")
