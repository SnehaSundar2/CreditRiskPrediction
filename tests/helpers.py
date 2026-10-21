"""Shared fixtures. Training takes a few seconds, so ONE small model is trained for the whole test run."""

import atexit
import shutil
import tempfile
import unittest
from pathlib import Path

from creditrisk import datagen, persistence, training

_SHARED = {}

# a low income, a big personal loan, a poor score, three late payments and heavy card use: very likely to default
HIGH_RISK = {"annual_income": 4, "loan_amount": 6, "loan_term_months": 36, "interest_rate": 17, "credit_score": 560,
             "employment_type": "Other", "loan_purpose": "Personal", "employment_years": 1, "monthly_debt_payments": 9,
             "credit_accounts": 5, "late_payments": 3, "credit_utilisation": 85, "owns_home": "no", "has_cosigner": "no"}
# a steady salaried applicant with a high score, a small vehicle loan, no late payments and low card use: very safe
LOW_RISK = {"annual_income": 18, "loan_amount": 4, "loan_term_months": 48, "interest_rate": 9.5, "credit_score": 790,
            "employment_type": "Salaried", "loan_purpose": "Vehicle", "employment_years": 9, "monthly_debt_payments": 2,
            "credit_accounts": 6, "late_payments": 0, "credit_utilisation": 12, "owns_home": "yes", "has_cosigner": "no"}
MINIMAL = {"annual_income": 8, "loan_amount": 4, "loan_term_months": 48, "credit_score": 700, "employment_type": "Salaried"}


def shared():
    """(TrainingResult, saved model directory, raw training DataFrame) - trained once, in fast mode."""
    if not _SHARED:
        folder = Path(tempfile.mkdtemp(prefix="creditrisk_tests_"))
        atexit.register(shutil.rmtree, folder, ignore_errors=True)
        raw = datagen.generate(2500)
        result = training.train(raw, fast=True, tune=False, cv=3, source="synthetic test data")
        persistence.save(result, folder / "model")
        _SHARED.update(result=result, model_dir=folder / "model", raw=raw, folder=folder)
    return _SHARED["result"], _SHARED["model_dir"], _SHARED["raw"]


class TempDirTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def copy_model(self):
        """A private copy of the trained model that a test may damage (each call makes a fresh copy)."""
        _, model_dir, _ = shared()
        self._copies = getattr(self, "_copies", 0) + 1
        target = self.dir / f"model_copy_{self._copies}"
        shutil.copytree(model_dir, target)
        return target
