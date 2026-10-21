"""Schema helpers, synthetic data, loading and cleaning."""

import unittest

import numpy as np
import pandas as pd

from creditrisk import datagen, schema as S
from creditrisk import preprocessing as P
from creditrisk.exceptions import DataError
from tests.helpers import TempDirTest


def tiny(**overrides):
    """A 500-row clean dataset (object columns, like a freshly read CSV) with some columns replaced."""
    df = datagen.generate(500, seed=5, dirty=False).astype(object)
    for column, value in overrides.items():
        df[column] = value
    return df


class TestSchema(unittest.TestCase):
    def test_categories_accept_messy_spellings(self):
        self.assertEqual(S.canonical_category("employment_type", "  salary "), "Salaried")
        self.assertEqual(S.canonical_category("employment_type", "Self_Employed"), "Self-employed")
        self.assertEqual(S.canonical_category("employment_type", "retired"), "Other")
        self.assertEqual(S.canonical_category("loan_purpose", "Home Loan"), "Home")
        self.assertEqual(S.canonical_category("loan_purpose", "car"), "Vehicle")
        self.assertIsNone(S.canonical_category("loan_purpose", "holiday"))
        self.assertIsNone(S.canonical_category("loan_purpose", None))
        self.assertIsNone(S.canonical_category("loan_purpose", float("nan")))

    def test_yes_no_and_labels(self):
        for yes in ("Yes", "y", "TRUE", 1, "1", 1.0):
            self.assertEqual(S.parse_yes_no(yes), 1, yes)
        for no in ("No", "n", "false", 0, "0", 0.0):
            self.assertEqual(S.parse_yes_no(no), 0, no)
        for bad in ("maybe", "", 2, float("nan")):
            self.assertIsNone(S.parse_yes_no(bad), bad)
        for yes in ("Yes ", 1, 1.0, "defaulted", "TRUE", True, "written off"):
            self.assertEqual(S.canonical_label(yes), 1, yes)
        for no in ("no", 0, "repaid", "False", False, "fully paid"):
            self.assertEqual(S.canonical_label(no), 0, no)
        for bad in ("pending", None, float("nan"), 2, ""):
            self.assertIsNone(S.canonical_label(bad), bad)

    def test_break_even_and_decisions(self):
        self.assertAlmostEqual(S.break_even(0.12, 0.6), 1 / 6)
        self.assertAlmostEqual(S.break_even(0.2, 0.2), 0.5)
        t = 0.20
        self.assertEqual(S.decision(0.05, t), "Approve")
        self.assertEqual(S.decision(0.10, t), "Manual review")
        self.assertEqual(S.decision(0.199, t), "Manual review")
        self.assertEqual(S.decision(0.20, t), "Decline")
        self.assertEqual(S.risk_band(0.099, t), "Low")

    def test_money_formatting(self):
        self.assertEqual(S.format_lakh(4.5), "₹4.50 lakh")
        self.assertEqual(S.format_lakh(0.5), "₹50,000")
        self.assertTrue(S.format_lakh(120).endswith("crore"))
        self.assertEqual(S.format_lakh(-0.5), "-₹50,000")
        self.assertTrue(S.format_lakh(-18.8).startswith("-₹"))

    def test_monthly_payment(self):
        emi = float(S.monthly_payment(10, 12, 12))          # 10 lakh at 12% for a year, in thousand rupees a month
        self.assertAlmostEqual(emi, 88.849, places=2)
        self.assertAlmostEqual(float(S.monthly_payment(1, 0, 10)), 10.0)         # no interest: simply split
        values = S.monthly_payment(np.array([1.0, 2.0]), np.array([10.0, 10.0]), np.array([12, 12]))
        self.assertAlmostEqual(values[1], 2 * values[0])
        self.assertGreater(float(S.monthly_payment(5, 20, 36)), float(S.monthly_payment(5, 10, 36)))

    def test_every_feature_is_described_and_labelled(self):
        for feature in S.RAW_FEATURES:
            self.assertTrue(S.describe(feature))
            self.assertIn(feature, S.LABELS)
        for feature, (improve, text) in S.MODIFIABLE.items():
            self.assertIn(feature, S.RAW_FEATURES)
            self.assertTrue(text)

    def test_no_personal_attributes_are_features(self):
        for banned in ("gender", "sex", "religion", "caste", "marital_status", "age", "address", "name"):
            self.assertNotIn(banned, S.RAW_FEATURES)
        self.assertEqual(S.MONITOR, "age")

    def test_improvement_functions_only_improve(self):
        self.assertEqual(S.MODIFIABLE["loan_amount"][0](8), 6.0)
        self.assertIsNone(S.MODIFIABLE["loan_amount"][0](0.3))
        self.assertEqual(S.MODIFIABLE["has_cosigner"][0](0), 1)
        self.assertIsNone(S.MODIFIABLE["has_cosigner"][0](1))
        self.assertEqual(S.MODIFIABLE["credit_utilisation"][0](80), 30.0)
        self.assertIsNone(S.MODIFIABLE["credit_utilisation"][0](20))
        self.assertIsNone(S.MODIFIABLE["loan_term_months"][0](120))


class TestGenerator(unittest.TestCase):
    def test_same_seed_same_data(self):
        a, b = datagen.generate(300, seed=3), datagen.generate(300, seed=3)
        pd.testing.assert_frame_equal(a, b)
        self.assertFalse(a.equals(datagen.generate(300, seed=4)))

    def test_clean_data_is_valid_and_has_a_plausible_default_rate(self):
        df = datagen.generate(3000, dirty=False)
        self.assertEqual(list(df.columns), [S.ID] + S.RAW_FEATURES + [S.MONITOR, S.TARGET])
        self.assertTrue(0.10 < (df[S.TARGET] == "Yes").mean() < 0.25)
        for column, (kind, low, high, _) in S.NUMERIC.items():
            self.assertTrue(df[column].between(low, high).all(), column)
        self.assertTrue(df[S.MONITOR].between(*S.AGE_RANGE).all())
        self.assertEqual(df[S.ID].nunique(), len(df))
        self.assertTrue((df.loc[df["credit_accounts"] == 0, "late_payments"] == 0).all())

    def test_dirty_data_has_problems(self):
        df = datagen.generate(1500)
        self.assertGreater(len(df), 1500)
        self.assertTrue(df.isna().any().any())
        self.assertTrue((pd.to_numeric(df["annual_income"], errors="coerce") > 1000).any())      # typed in rupees

    def test_true_risk_moves_the_right_way(self):
        def p(**kw):
            base = dict(annual_income=8, loan_amount=4, loan_term_months=48, interest_rate=11, credit_score=700,
                        employment_years=5, monthly_debt_payments=5, late_payments=0, credit_utilisation=30,
                        owns_home=0, has_cosigner=0, employment_type="Salaried", loan_purpose="Vehicle")
            base.update(kw)
            return float(datagen.default_probability(*[np.array([base[k]]) for k in base])[0])
        self.assertGreater(p(credit_score=550), p(credit_score=750))
        self.assertGreater(p(late_payments=4), p(late_payments=0))
        self.assertGreater(p(loan_amount=12), p(loan_amount=2))
        self.assertGreater(p(annual_income=3), p(annual_income=15))
        self.assertGreater(p(credit_utilisation=90), p(credit_utilisation=10))
        self.assertGreater(p(has_cosigner=0), p(has_cosigner=1))
        self.assertGreater(p(loan_purpose="Personal"), p(loan_purpose="Home"))
        self.assertGreater(p(employment_type="Other"), p(employment_type="Salaried"))
        self.assertTrue(0 < p() < 1)


class TestLoading(TempDirTest):
    def write(self, df, name="data.csv"):
        path = self.dir / name
        df.to_csv(path, index=False)
        return path

    def test_loads_a_valid_file_and_normalises_headers(self):
        df = datagen.generate(500, dirty=False).rename(columns={"credit_score": "Credit Score", "loan_amount": "Loan-Amount"})
        out = P.load_dataset(self.write(df))
        self.assertIn("credit_score", out.columns)
        self.assertIn("loan_amount", out.columns)
        self.assertEqual(len(out), 500)

    def test_id_and_age_are_optional(self):
        out = P.load_dataset(self.write(datagen.generate(500, dirty=False).drop(columns=[S.ID, S.MONITOR])))
        self.assertNotIn(S.ID, out.columns)
        self.assertNotIn(S.MONITOR, out.columns)
        out = P.load_dataset(self.write(datagen.generate(500, dirty=False)))
        self.assertIn(S.MONITOR, out.columns)

    def test_clear_errors(self):
        with self.assertRaisesRegex(DataError, "not found"):
            P.load_dataset(self.dir / "nope.csv")
        empty = self.dir / "empty.csv"
        empty.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(DataError, "empty"):
            P.load_dataset(empty)
        with self.assertRaisesRegex(DataError, "missing column"):
            P.load_dataset(self.write(datagen.generate(500, dirty=False).drop(columns=["credit_score", "loan_purpose"])))
        with self.assertRaisesRegex(DataError, "only 10 rows"):
            P.load_dataset(self.write(datagen.generate(500, dirty=False).head(10)))

    def test_not_utf8(self):
        path = self.dir / "latin.csv"
        path.write_bytes(b"annual_income,loan_amount\n\xe9\xe8,x\n")
        with self.assertRaisesRegex(DataError, "UTF-8"):
            P.load_dataset(path)


class TestCleaning(unittest.TestCase):
    def test_clean_data_passes_through_untouched(self):
        df = datagen.generate(600, dirty=False)
        cleaned, report = P.clean(df)
        self.assertEqual(len(cleaned), 600)
        self.assertEqual(report.steps, [])
        self.assertEqual(report.rows_dropped, 0)

    def test_labels_become_zero_one_and_bad_labels_are_dropped(self):
        df = tiny()
        df.loc[0, S.TARGET] = "pending"
        df.loc[1, S.TARGET] = np.nan
        df.loc[2, S.TARGET] = "repaid"
        df.loc[3, S.TARGET] = "1"
        cleaned, report = P.clean(df)
        self.assertEqual(len(cleaned), 498)
        self.assertTrue(set(cleaned[S.TARGET]) <= {0, 1})
        self.assertIn("missing or unrecognised outcome label", report.steps[0][0])
        self.assertEqual(report.steps[0][1], 2)

    def test_spellings_are_standardised_and_missing_employment_drops_the_row(self):
        df = tiny()
        df.loc[0, "employment_type"] = "salary"
        df.loc[1, "employment_type"] = "BUSINESS"
        df.loc[2, "employment_type"] = np.nan
        df.loc[3, "loan_purpose"] = "car"
        cleaned, report = P.clean(df)
        self.assertEqual(len(cleaned), 499)
        self.assertEqual(set(cleaned["employment_type"]), set(S.EMPLOYMENT))
        self.assertEqual(cleaned.loc[cleaned[S.ID] == df.loc[3, S.ID], "loan_purpose"].iloc[0], "Vehicle")
        self.assertTrue(any("unknown or missing employment type" in s[0] for s in report.steps))

    def test_money_typed_in_rupees_is_converted_to_lakh(self):
        df = tiny()
        df.loc[0, "annual_income"] = 850000
        df.loc[1, "loan_amount"] = 250000
        cleaned, report = P.clean(df)
        self.assertAlmostEqual(cleaned.loc[0, "annual_income"], 8.5)
        self.assertAlmostEqual(cleaned.loc[1, "loan_amount"], 2.5)
        self.assertTrue(any("rupees, not lakh" in s[0] for s in report.steps))

    def test_unreadable_numbers_and_out_of_range_become_missing(self):
        df = tiny()
        df.loc[0, "credit_score"] = "n/a"
        df.loc[1, "credit_score"] = 9999
        df.loc[2, "interest_rate"] = 80
        df.loc[3, "loan_term_months"] = 600
        df.loc[4, "credit_utilisation"] = 150
        cleaned, report = P.clean(df)
        self.assertEqual(len(cleaned), 500)
        for index, column in ((0, "credit_score"), (1, "credit_score"), (2, "interest_rate"), (3, "loan_term_months"), (4, "credit_utilisation")):
            self.assertTrue(np.isnan(cleaned.loc[index, column]), (index, column))
        self.assertTrue(any("Out-of-range credit_score" in s[0] for s in report.steps))

    def test_age_is_only_blanked_when_impossible(self):
        df = tiny()
        df.loc[0, S.MONITOR] = 150
        df.loc[1, S.MONITOR] = 5
        cleaned, report = P.clean(df)
        self.assertTrue(np.isnan(cleaned.loc[0, S.MONITOR]) and np.isnan(cleaned.loc[1, S.MONITOR]))
        self.assertTrue(cleaned[S.MONITOR].dropna().between(18, 90).all())
        without_age, _ = P.clean(tiny().drop(columns=[S.MONITOR]))
        self.assertNotIn(S.MONITOR, without_age.columns)

    def test_late_payments_need_credit_accounts(self):
        df = tiny()
        df.loc[0, ["credit_accounts", "late_payments"]] = [0, 4]
        cleaned, report = P.clean(df)
        self.assertTrue(np.isnan(cleaned.loc[0, "late_payments"]))
        self.assertTrue(any("no credit accounts" in s[0] for s in report.steps))

    def test_work_years_must_fit_in_adult_life(self):
        df = tiny()
        df.loc[0, [S.MONITOR, "employment_years"]] = [24, 30]
        df.loc[1, [S.MONITOR, "employment_years"]] = [50, 30]
        cleaned, _ = P.clean(df)
        self.assertTrue(np.isnan(cleaned.loc[0, "employment_years"]))
        self.assertEqual(cleaned.loc[1, "employment_years"], 30)

    def test_yes_no_variants(self):
        df = tiny()
        df.loc[0, "owns_home"] = "Yes"
        df.loc[1, "owns_home"] = "N"
        df.loc[2, "owns_home"] = "maybe"
        cleaned, _ = P.clean(df)
        self.assertEqual(cleaned.loc[0, "owns_home"], 1)
        self.assertEqual(cleaned.loc[1, "owns_home"], 0)
        self.assertTrue(np.isnan(cleaned.loc[2, "owns_home"]))

    def test_duplicates_and_repeated_ids(self):
        df = tiny()
        df = pd.concat([df, df.head(5)], ignore_index=True)
        changed = df.head(3).copy()
        changed["credit_score"] = changed["credit_score"] + 20
        df = pd.concat([df, changed], ignore_index=True)
        cleaned, report = P.clean(df)
        self.assertEqual(len(cleaned), 500)
        self.assertEqual(cleaned[S.ID].nunique(), 500)
        labels = [s[0] for s in report.steps]
        self.assertIn("Duplicate rows", labels)
        self.assertTrue(any("more than once" in label for label in labels))

    def test_input_is_not_modified(self):
        df = datagen.generate(600)
        before = df.copy(deep=True)
        P.clean(df)
        pd.testing.assert_frame_equal(df, before)

    def test_too_little_data_or_one_class_is_refused(self):
        with self.assertRaisesRegex(DataError, "usable rows remain"):
            P.clean(datagen.generate(500, dirty=False).head(100))
        df = tiny()
        df[S.TARGET] = "No"
        df.loc[:10, S.TARGET] = "Yes"
        with self.assertRaisesRegex(DataError, "at least"):
            P.clean(df)

    def test_report_summary(self):
        _, report = P.clean(datagen.generate(800))
        self.assertGreater(report.rows_dropped, 0)
        self.assertEqual(report.rows_out, sum(report.class_counts.values()))
        self.assertIn("rows read", report.lines()[0])
        self.assertEqual(report.to_dict()["rows_in"], report.rows_in)


class TestFeatures(unittest.TestCase):
    def test_engineered_features_use_inputs_only(self):
        df = pd.DataFrame({"annual_income": [10.0, np.nan], "loan_amount": [5.0, 2.0], "loan_term_months": [24, 24],
                           "interest_rate": [12.0, 12.0], "monthly_debt_payments": [10.0, 5.0]})
        out = P.add_features(df)
        self.assertAlmostEqual(out.loc[0, "loan_to_income"], 0.5)
        self.assertAlmostEqual(out.loc[0, "debt_to_income"], 10 * 12 / 1000)
        emi = float(S.monthly_payment(5, 12, 24))
        self.assertAlmostEqual(out.loc[0, "emi_estimate"], emi)
        self.assertAlmostEqual(out.loc[0, "payment_burden"], (emi + 10) * 12 / 1000)
        self.assertTrue(np.isnan(out.loc[1, "loan_to_income"]))
        self.assertNotIn(S.TARGET, out.columns)

    def test_missing_rate_gives_missing_instalment(self):
        out = P.add_features(pd.DataFrame({"annual_income": [8.0], "loan_amount": [4.0], "loan_term_months": [36],
                                           "interest_rate": [np.nan], "monthly_debt_payments": [2.0]}))
        self.assertTrue(np.isnan(out.loc[0, "emi_estimate"]))

    def test_preprocessor_handles_missing_and_unseen_values(self):
        cleaned, _ = P.clean(datagen.generate(700))
        X, y = P.split_xy(cleaned)
        from creditrisk import models as M
        pipe = M.build_pipeline(M.candidates()["Logistic Regression"]).fit(X, y)
        odd = X.head(3).copy()
        odd.loc[odd.index[0], "loan_purpose"] = "Holiday"
        odd.loc[odd.index[1], "credit_score"] = np.nan
        odd.loc[odd.index[2], "interest_rate"] = np.nan
        p = pipe.predict_proba(odd)[:, 1]
        self.assertTrue(((p > 0) & (p < 1)).all())

    def test_split_keeps_the_age_column_out_of_the_features(self):
        cleaned, _ = P.clean(datagen.generate(700))
        X, _ = P.split_xy(cleaned)
        self.assertNotIn(S.MONITOR, X.columns)
        self.assertNotIn(S.ID, X.columns)
        self.assertEqual(list(X.columns), S.RAW_FEATURES)
