"""Data preprocessing: loading, cleaning, feature engineering and the scikit-learn transformer.

Two kinds of preprocessing, kept deliberately separate:

* clean()               - fixes or removes BAD DATA (typos, impossible values, unit mix-ups, repeated applications) before
                          the split. Every rule looks at one row (or at exact repeats); none learns statistics from
                          other rows, so nothing from the test set can influence training.
* build_preprocessor()  - LEARNS from data (medians for imputing, means for scaling, categories for one-hot
                          encoding). It lives inside the model pipeline and is fitted on the TRAINING rows only.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import schema as S
from .exceptions import DataError

log = logging.getLogger("creditrisk")

MAX_FILE_BYTES = 50_000_000
RUPEE_FLOOR = 1000                      # an income or loan above this (in lakh) must have been typed in rupees


def normalise_headers(columns):
    return [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in columns]


def load_dataset(path):
    """Read a CSV and check it has the columns the project needs (application_id and age are optional)."""
    path = Path(str(path).strip().strip('"'))
    if not path.exists() or not path.is_file():
        raise DataError(f"Dataset file not found: {path}")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise DataError(f"{path.name} is larger than {MAX_FILE_BYTES // 1_000_000} MB.")
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=object, keep_default_na=False, na_values=[""])
    except pd.errors.EmptyDataError:
        raise DataError(f"{path.name} is empty.") from None
    except UnicodeDecodeError:
        raise DataError(f"{path.name} is not UTF-8 text. In Excel use 'Save As > CSV UTF-8'.") from None
    except (pd.errors.ParserError, OSError) as exc:
        raise DataError(f"Could not read {path.name}: {exc}") from None
    df.columns = normalise_headers(df.columns)
    missing = [c for c in S.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"{path.name} is missing column(s): {', '.join(missing)}. "
                        f"Required columns: {', '.join(S.REQUIRED_COLUMNS)} ({S.ID} and {S.MONITOR} are optional).")
    if len(df) < S.MIN_ROWS:
        raise DataError(f"{path.name} has only {len(df)} rows; at least {S.MIN_ROWS} are needed to train a "
                        "reliable model.")
    columns = [c for c in (S.ID, S.MONITOR) if c in df.columns] + S.REQUIRED_COLUMNS
    return df[columns].copy()


@dataclass
class CleaningReport:
    rows_in: int = 0
    rows_out: int = 0
    steps: list = field(default_factory=list)        # (label, count, outcome)
    missing_before: dict = field(default_factory=dict)
    missing_after: dict = field(default_factory=dict)
    class_counts: dict = field(default_factory=dict)

    def add(self, label, count, outcome):
        if count:
            self.steps.append((label, int(count), outcome))

    @property
    def rows_dropped(self):
        return self.rows_in - self.rows_out

    def lines(self):
        out = [f"{self.rows_in} rows read → {self.rows_out} kept ({self.rows_dropped} removed)"]
        out += [f"  • {label}: {count} - {outcome}" for label, count, outcome in self.steps]
        if self.class_counts:
            total = sum(self.class_counts.values())
            out.append(f"  Kept: {self.class_counts['default']} defaulted and {self.class_counts['repaid']} repaid "
                       f"({100 * self.class_counts['default'] / total:.1f}% default)")
        return out

    def to_dict(self):
        return {"rows_in": self.rows_in, "rows_out": self.rows_out,
                "steps": [{"label": a, "count": b, "outcome": c} for a, b, c in self.steps],
                "missing_before": self.missing_before, "missing_after": self.missing_after,
                "class_counts": self.class_counts}


def _blank(value):
    return isinstance(value, str) and not value.strip()


def clean(df):
    """Return (clean_df with int `default`, CleaningReport). The input is not modified."""
    has_id = S.ID in df.columns
    df = df.copy()
    report = CleaningReport(rows_in=len(df))
    report.missing_before = {c: int(n) for c, n in df.isna().sum().items() if n}

    # 1. the label: a person whose outcome is unknown cannot teach the model anything
    label = df[S.TARGET].map(S.canonical_label)
    report.add("Rows with a missing or unrecognised outcome label", label.isna().sum(), "removed")
    df = df[label.notna()].copy()
    df[S.TARGET] = label[label.notna()].astype(int)

    # 2. categories: fix spelling; a missing employment type is not guessable, other gaps are filled in later
    for col in S.CATEGORICAL_FEATURES:
        original = df[col]
        canon = original.map(lambda v, c=col: S.canonical_category(c, v))
        changed = canon.notna() & (canon != original.map(lambda v: v if isinstance(v, str) else None))
        report.add(f"{S.LABELS[col]} spelling standardised", changed.sum(),
                   "e.g. 'salary' → 'Salaried'" if col == "employment_type" else "e.g. 'car' → 'Vehicle'")
        unrecognised = canon.isna() & original.notna()
        df[col] = canon
        if col == "employment_type":
            report.add("Rows with an unknown or missing employment type", canon.isna().sum(), "removed")
            df = df[canon.notna()]
        else:
            report.add(f"Unrecognised {S.LABELS[col].lower()} values", unrecognised.sum(), "set to missing, filled in later")

    # 3. numbers: text -> numbers, yes/no -> 1/0
    for col in S.NUMERIC_FEATURES:
        before = df[col].isna() | df[col].map(_blank)
        df[col] = pd.to_numeric(df[col], errors="coerce")
        report.add(f"Non-numeric text in {col}", (df[col].isna() & ~before).sum(), "set to missing")
    for col in S.BINARY_FEATURES:
        flags = df[col].map(S.parse_yes_no)
        report.add(f"Unrecognised {S.LABELS[col].lower()} values", (flags.isna() & df[col].notna()).sum(), "set to missing")
        df[col] = flags
    for col in S.NUMERIC_FEATURES:
        df[col] = df[col].mask(np.isinf(df[col]))

    # 4. unit mix-up: money typed in rupees (850000) instead of lakh rupees (8.5)
    for col in ("annual_income", "loan_amount"):
        rupees = df[col] > RUPEE_FLOOR
        report.add(f"{S.LABELS[col]} that looks like rupees, not lakh", rupees.sum(), "converted to lakh (÷ 100,000)")
        df.loc[rupees, col] = df.loc[rupees, col] / 100000.0

    # 4b. the optional age column is only used for fairness checks; unusable ages are blanked
    if S.MONITOR in df.columns:
        df[S.MONITOR] = pd.to_numeric(df[S.MONITOR], errors="coerce")
        bad_age = df[S.MONITOR].notna() & ~df[S.MONITOR].between(*S.AGE_RANGE)
        report.add("Ages outside 18-90", bad_age.sum(), "set to missing (age is only used for fairness checks)")
        df.loc[bad_age, S.MONITOR] = np.nan

    # 5. out-of-range values are probably typos: blank them and impute later
    for col in S.NUMERIC_FEATURES:
        low, high = S.NUMERIC[col][1:3]
        bad = df[col].notna() & ~df[col].between(low, high)
        report.add(f"Out-of-range {col}", bad.sum(), f"set to missing (valid: {low}-{high})")
        df.loc[bad, col] = np.nan

    # 6. combinations that cannot be true
    no_history = (df["credit_accounts"] == 0) & (df["late_payments"] > 0)
    report.add("Late payments recorded for someone with no credit accounts", no_history.sum(), "late payments set to missing")
    df.loc[no_history, "late_payments"] = np.nan
    if S.MONITOR in df.columns:
        too_long = df[S.MONITOR].notna() & (df["employment_years"] > df[S.MONITOR] - 16)
        report.add("More years in work than adult life allows (age minus 16)", too_long.sum(), "years in work set to missing")
        df.loc[too_long, "employment_years"] = np.nan

    # 7. repeats
    dup = df.duplicated()
    report.add("Duplicate rows", dup.sum(), "removed (kept the first copy)")
    df = df[~dup]
    if has_id:
        same_id = df[S.ID].notna() & df[S.ID].duplicated()
        report.add("Application id appearing more than once with different details", same_id.sum(), "removed (kept the first)")
        df = df[~same_id]

    df = df.reset_index(drop=True)
    counts = df[S.TARGET].value_counts()
    report.class_counts = {"default": int(counts.get(1, 0)), "repaid": int(counts.get(0, 0))}
    report.rows_out = len(df)
    report.missing_after = {c: int(n) for c, n in df.isna().sum().items() if n and c != S.ID}
    log.info("Cleaning: %d → %d rows", report.rows_in, report.rows_out)
    if len(df) < S.MIN_ROWS:
        raise DataError(f"Only {len(df)} usable rows remain after cleaning (need {S.MIN_ROWS}). "
                        "Check that the column names and units match the schema.")
    for name, count in report.class_counts.items():
        if count < S.MIN_PER_CLASS:
            kind = "that defaulted" if name == "default" else "that were repaid"
            raise DataError(f"Only {count} loans {kind} remain after cleaning; at least {S.MIN_PER_CLASS} of each "
                            "kind are needed to train and test a model.")
    return df, report


def add_features(df):
    """Engineered features. They use only other input columns - never the outcome label (that would be leakage)."""
    out = pd.DataFrame(df).copy()
    income_thousand = out["annual_income"] * 100                              # thousand rupees a year
    emi = pd.Series(S.monthly_payment(out["loan_amount"], out["interest_rate"], out["loan_term_months"]), index=out.index)
    emi = emi.where(out["loan_amount"].notna() & out["interest_rate"].notna() & out["loan_term_months"].notna())
    out["emi_estimate"] = emi                                                  # monthly instalment of this loan, thousand rupees
    out["loan_to_income"] = out["loan_amount"] / out["annual_income"]
    out["debt_to_income"] = out["monthly_debt_payments"] * 12 / income_thousand
    out["payment_burden"] = (emi + out["monthly_debt_payments"]) * 12 / income_thousand   # all repayments as a share of income
    return out


def build_preprocessor():
    """Impute -> scale numbers, impute binary flags, impute -> one-hot encode categories."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    binary = SimpleImputer(strategy="most_frequent")
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([
        ("num", numeric, S.NUMERIC_FEATURES + S.ENGINEERED),
        ("bin", binary, S.BINARY_FEATURES),
        ("cat", categorical, S.CATEGORICAL_FEATURES),
    ])


def feature_adder():
    return FunctionTransformer(add_features)


def split_xy(df):
    return df[S.RAW_FEATURES].copy(), df[S.TARGET].astype(int).copy()
