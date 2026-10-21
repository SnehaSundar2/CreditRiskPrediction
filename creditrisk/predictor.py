"""Prediction service: validate an application, estimate the default risk, decide, explain and show "what if".

Explanations are computed by replacing one feature at a time with the typical (training median / most common)
value and measuring how the estimated default risk changes. They describe what the MODEL associates with
default in its training data; they are not causes. The "main reasons" for a high risk follow the same idea as
the reason codes lenders must give when they decline an application.
"""

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import persistence
from . import schema as S
from . import validators as v
from .exceptions import DataError, ValidationError

log = logging.getLogger("creditrisk")

DECISION_MESSAGE = {
    "Approve": "Default risk is below half of the break-even level: the loan is expected to earn money.",
    "Manual review": "Default risk is between half of the break-even level and the break-even level: a loan officer should look at it.",
    "Decline": "Default risk is at or above the break-even level: the loan is expected to lose money.",
}
MIN_EFFECT = 0.005         # ignore factors that move the risk by less than half a percentage point
MAX_BATCH = 5000


@dataclass
class Prediction:
    probability: float
    band: str
    decision: str
    threshold: float
    typical_probability: float
    expected_profit_share: float                      # expected profit as a share of the loan amount
    emi: float                                        # estimated monthly instalment, thousand rupees
    burden: float                                     # all repayments as a share of income
    factors: list = field(default_factory=list)       # [{feature, label, value, effect}] sorted by size
    what_ifs: list = field(default_factory=list)      # [{feature, scenario, probability, change}]
    assumed: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    record: dict = field(default_factory=dict)
    disclaimer: str = S.DISCLAIMER

    @property
    def percent(self):
        return 100 * self.probability

    @property
    def message(self):
        return DECISION_MESSAGE[self.decision]

    @property
    def reasons(self):
        """The details that raise the risk most (positive effects), largest first."""
        return [f for f in self.factors if f["effect"] > 0][:4]

    def to_dict(self):
        return {"default_risk_percent": round(self.percent, 1), "band": self.band, "decision": self.decision,
                "message": self.message, "break_even_percent": round(100 * self.threshold, 1),
                "typical_applicant_percent": round(100 * self.typical_probability, 1),
                "expected_profit_percent_of_loan": round(100 * self.expected_profit_share, 1),
                "monthly_instalment_rupees": round(self.emi * 1000), "repayments_share_of_income_percent": round(100 * self.burden, 1),
                "main_reasons": [{**f, "effect_points": round(100 * f["effect"], 1)} for f in self.reasons],
                "factors": [{**f, "effect_points": round(100 * f["effect"], 1)} for f in self.factors],
                "what_if": [{**w, "default_risk_percent": round(100 * w["probability"], 1), "change_points": round(100 * w["change"], 1)}
                            for w in self.what_ifs],
                "assumed_typical_values_for": self.assumed, "notes": self.notes, "warnings": self.warnings,
                "disclaimer": self.disclaimer}


def format_value(feature, value):
    if feature in S.BINARY:
        return "yes" if value == 1 else "no"
    if feature in S.CATEGORICAL:
        return str(value)
    if feature in ("annual_income", "loan_amount"):
        return S.format_lakh(value)
    unit = {"loan_term_months": " months", "interest_rate": "%", "credit_utilisation": "%", "employment_years": " years",
            "monthly_debt_payments": "k/month"}.get(feature, "")
    return f"{value:g}{unit}"


class Predictor:
    def __init__(self, loaded):
        self.pipeline = loaded.pipeline
        self.meta = loaded.metadata
        self.reference = loaded.reference
        self.warnings = list(loaded.warnings)
        self.typical = self.meta["typical"]
        self.threshold = self.meta["thresholds"]["approve_below"]
        self.margin = self.meta["economics"]["margin"]
        self.loss = self.meta["economics"]["loss"]

    # ------------------------------------------------------------------ single applicant
    def predict(self, raw):
        record, notes, assumed = v.parse_features(raw)
        frame = pd.DataFrame([record])
        warnings = self._range_warnings(record)
        warnings += self._consistency_warnings(record)

        known = [f for f in S.RAW_FEATURES if f not in assumed]
        variants = [frame]
        for feature in known:
            changed = frame.copy()
            changed[feature] = self.typical[feature]
            variants.append(changed)
        scenarios = []
        for feature, (improve, text) in S.MODIFIABLE.items():
            if feature not in known:
                continue
            new_value = improve(record[feature])
            if new_value is not None:
                changed = frame.copy()
                changed[feature] = new_value
                variants.append(changed)
                scenarios.append((feature, text))
        probabilities = self.pipeline.predict_proba(pd.concat(variants, ignore_index=True))[:, 1]
        p = float(probabilities[0])
        factors = []
        for feature, p_typical in zip(known, probabilities[1:1 + len(known)]):
            effect = p - float(p_typical)
            if abs(effect) >= MIN_EFFECT:
                factors.append({"feature": feature, "label": S.LABELS[feature],
                                "value": format_value(feature, record[feature]), "effect": effect})
        factors.sort(key=lambda f: -abs(f["effect"]))
        what_ifs = []
        for (feature, text), p_new in zip(scenarios, probabilities[1 + len(known):]):
            if p - float(p_new) >= MIN_EFFECT:
                what_ifs.append({"feature": feature, "scenario": f"Default risk {text}",
                                 "probability": float(p_new), "change": float(p_new) - p})
        what_ifs.sort(key=lambda w: w["change"])
        typical_record = pd.DataFrame([{f: self.typical[f] for f in S.RAW_FEATURES}])
        typical_p = float(self.pipeline.predict_proba(typical_record)[0, 1])
        emi, burden = self._repayments(record)
        band = S.risk_band(p, self.threshold)
        return Prediction(p, band, S.DECISIONS[band], self.threshold, typical_p, (1 - p) * self.margin - p * self.loss,
                          emi, burden, factors[:8], what_ifs, assumed, notes, warnings, record)

    def _repayments(self, record):
        """(monthly instalment of this loan in thousand rupees, all repayments as a share of income)."""
        rate = record["interest_rate"] if not _nan(record["interest_rate"]) else self.typical["interest_rate"]
        debt = record["monthly_debt_payments"] if not _nan(record["monthly_debt_payments"]) else self.typical["monthly_debt_payments"]
        emi = float(S.monthly_payment(record["loan_amount"], rate, record["loan_term_months"]))
        return emi, (emi + debt) * 12 / (record["annual_income"] * 100)

    def _range_warnings(self, record):
        out = []
        for feature, (low, high) in self.meta["training_ranges"].items():
            value = record[feature]
            if _nan(value):
                continue
            span = max(high - low, 1e-9)
            if value < low - 0.1 * span or value > high + 0.1 * span:
                out.append(f"{S.LABELS[feature]} {value:g} is outside the range the model learned from "
                           f"({low:g} to {high:g}), so the estimate is less reliable.")
        return out

    @staticmethod
    def _consistency_warnings(record):
        out = []
        if record["loan_amount"] > 10 * record["annual_income"]:
            out.append("The loan is more than ten times the yearly income; such requests are very unusual.")
        return out

    # ------------------------------------------------------------------ many applicants
    def predict_frame(self, df):
        """Score a DataFrame of raw rows. Returns a copy with default_risk_percent, decision, status; bad rows get a message."""
        if len(df) > MAX_BATCH:
            raise DataError(f"At most {MAX_BATCH} applications can be scored at once (this file has {len(df)}).")
        df = df.copy()
        df.columns = [v.normalise_key(c) for c in df.columns]
        out = df.copy()
        out["default_risk_percent"], out["decision"], out["status"] = np.nan, "", "ok"
        records, positions = [], []
        for i, (_, row) in enumerate(df.iterrows()):
            try:
                record, _, _ = v.parse_features({k: row[k] for k in S.RAW_FEATURES if k in df.columns})
                records.append(record)
                positions.append(i)
            except ValidationError as exc:
                out.iloc[i, out.columns.get_loc("status")] = f"row {i + 2}: {exc}"[:200]
        if records:
            p = self.pipeline.predict_proba(pd.DataFrame(records))[:, 1]
            for i, prob in zip(positions, p):
                out.iloc[i, out.columns.get_loc("default_risk_percent")] = round(100 * float(prob), 1)
                out.iloc[i, out.columns.get_loc("decision")] = S.decision(float(prob), self.threshold)
        return out

    def predict_csv(self, source, target):
        path = Path(str(source).strip().strip('"'))
        if not path.is_file():
            raise DataError(f"File not found: {path}")
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", dtype=object, keep_default_na=False, na_values=[""])
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, OSError) as exc:
            raise DataError(f"Could not read {path.name}: {exc}") from None
        result = self.predict_frame(df)
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(target, index=False)
        return result

    # ------------------------------------------------------------------ what-if tool
    def what_if(self, raw, feature, values):
        """The same application with different values of one detail: [(value, probability)]."""
        name = v.normalise_key(feature)
        if name not in S.RAW_FEATURES:
            raise ValidationError(f"Unknown detail '{str(feature)[:30]}'.", {name or "field": "Unknown field."})
        if not values:
            raise ValidationError("Give at least one value to try.")
        if len(values) > 12:
            raise ValidationError("Try at most 12 values at once.")
        records = [v.parse_features({**raw, name: value})[0] for value in values]
        probabilities = self.pipeline.predict_proba(pd.DataFrame(records))[:, 1]
        return [(records[i][name], float(p)) for i, p in enumerate(probabilities)]

    def suggestions(self, feature):
        """A few sensible values to compare for a detail."""
        if feature in S.CATEGORICAL:
            return list(S.CATEGORICAL[feature])
        if feature in S.BINARY:
            return ["no", "yes"]
        low, high = self.meta["training_ranges"][feature]
        kind = S.NUMERIC[feature][0]
        values = [low + (high - low) * k / 4 for k in range(5)]
        return [int(round(x)) if kind == "int" else round(x, 1) for x in values]

    def health(self):
        m = self.meta
        return {"status": "ok", "model": m["model"], "trained_at": m["trained_at"], "source": m["source"],
                "roc_auc": round(m["metrics"]["test"]["roc_auc"], 3), "break_even_percent": round(100 * self.threshold, 1),
                "disclaimer": S.DISCLAIMER}


def _nan(value):
    return isinstance(value, float) and math.isnan(value)


def require_predictor(directory):
    return Predictor(persistence.load(directory))
