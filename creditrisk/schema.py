"""The dataset schema: every feature, its type, valid range and allowed values.

Training-data cleaning and prediction-input validation both read from here, so the two can never disagree
about what a valid application looks like.

Fairness note: the model deliberately does NOT use gender, religion, caste, marital status, address or any other
personal attribute, and it does not use age. It uses only the loan terms and the applicant's financial history.
Age may be present in a dataset, but only so the report can CHECK the results for age groups.

All money is in lakh rupees (1 lakh = 100,000) except monthly debt payments, which are in thousand rupees.
"""

TARGET = "default"               # 1 = the loan went into default within two years, 0 = repaid as agreed
ID = "application_id"            # optional column in a dataset; never used as a feature
MONITOR = "age"                  # optional column used ONLY for fairness checks, never as a feature

DISCLAIMER = ("This is a learning demo trained on generated (synthetic) data. It must not be used to make real "
              "lending decisions.")

EMPLOYMENT = ["Salaried", "Self-employed", "Business owner", "Other"]
PURPOSES = ["Home", "Education", "Vehicle", "Personal", "Business"]

# name -> (type, low, high, description). Numeric limits are hard limits for valid data.
NUMERIC = {
    "annual_income": ("float", 1, 100, "Yearly income in lakh rupees"),
    "loan_amount": ("float", 0.2, 60, "Amount requested in lakh rupees"),
    "loan_term_months": ("int", 6, 120, "Repayment period in months"),
    "interest_rate": ("float", 5, 30, "Yearly interest rate offered, in percent"),
    "credit_score": ("int", 300, 900, "Credit score (300 to 900)"),
    "employment_years": ("float", 0, 45, "Years in the current kind of work"),
    "monthly_debt_payments": ("float", 0, 300, "Monthly payments on existing loans, in thousand rupees"),
    "credit_accounts": ("int", 0, 25, "Number of credit accounts (loans and cards) ever opened"),
    "late_payments": ("int", 0, 12, "Payments more than 30 days late in the last 2 years"),
    "credit_utilisation": ("float", 0, 100, "Share of the credit-card limit in use, in percent"),
}
BINARY = {
    "owns_home": "Owns the home they live in (yes/no)",
    "has_cosigner": "Has a co-signer who guarantees the loan (yes/no)",
}
CATEGORICAL = {"employment_type": EMPLOYMENT, "loan_purpose": PURPOSES}

NUMERIC_FEATURES = list(NUMERIC)
BINARY_FEATURES = list(BINARY)
CATEGORICAL_FEATURES = list(CATEGORICAL)
RAW_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = RAW_FEATURES + [TARGET]
REQUIRED_INPUT = ["annual_income", "loan_amount", "loan_term_months", "credit_score", "employment_type"]
AGE_RANGE = (18, 90)

ENGINEERED = ["loan_to_income", "debt_to_income", "payment_burden", "emi_estimate"]

MIN_ROWS = 400
MIN_PER_CLASS = 50

# Lending economics, as a share of the loan amount. They decide the approval threshold.
PROFIT_MARGIN = 0.12             # net interest income earned over the loan when it is repaid
LOSS_GIVEN_DEFAULT = 0.60        # share of the loan lost when it defaults

LABELS = {
    "annual_income": "Annual income", "loan_amount": "Loan amount", "loan_term_months": "Loan term",
    "interest_rate": "Interest rate", "credit_score": "Credit score", "employment_years": "Years in work",
    "monthly_debt_payments": "Existing monthly debt payments", "credit_accounts": "Credit accounts",
    "late_payments": "Late payments (2 years)", "credit_utilisation": "Credit-card utilisation",
    "owns_home": "Home ownership", "has_cosigner": "Co-signer", "employment_type": "Employment type",
    "loan_purpose": "Loan purpose",
}
UNITS = {"annual_income": "lakh", "loan_amount": "lakh", "loan_term_months": "months", "interest_rate": "%",
         "credit_score": "", "employment_years": "years", "monthly_debt_payments": "thousand/month",
         "credit_utilisation": "%"}

# Details an applicant can change, with a function giving the improved value (or None when nothing would improve).
MODIFIABLE = {
    "loan_amount": (lambda v: round(v * 0.75, 2) if v > 0.4 else None, "with a 25% smaller loan"),
    "has_cosigner": (lambda v: 1 if v == 0 else None, "with a co-signer"),
    "credit_utilisation": (lambda v: 30.0 if v > 30 else None, "with card utilisation at 30%"),
    "monthly_debt_payments": (lambda v: round(v * 0.7, 1) if v > 1 else None, "after paying down 30% of existing debt payments"),
    "loan_term_months": (lambda v: min(v + 12, 120) if v < 120 else None, "with a term 12 months longer"),
}

ALIASES = {
    "employment_type": {"salary": "Salaried", "salaried": "Salaried", "employee": "Salaried", "job": "Salaried",
                        "self employed": "Self-employed", "selfemployed": "Self-employed", "freelance": "Self-employed",
                        "freelancer": "Self-employed", "business": "Business owner", "business owner": "Business owner",
                        "owner": "Business owner", "entrepreneur": "Business owner", "retired": "Other",
                        "student": "Other", "unemployed": "Other", "other": "Other"},
    "loan_purpose": {"house": "Home", "home loan": "Home", "housing": "Home", "mortgage": "Home", "study": "Education",
                     "education loan": "Education", "college": "Education", "car": "Vehicle", "bike": "Vehicle",
                     "auto": "Vehicle", "vehicle loan": "Vehicle", "personal loan": "Personal", "consumer": "Personal",
                     "wedding": "Personal", "medical": "Personal", "business loan": "Business", "startup": "Business",
                     "working capital": "Business"},
}

DEFAULT_YES = {"yes", "y", "1", "true", "default", "defaulted", "bad", "charged off", "written off"}
DEFAULT_NO = {"no", "n", "0", "false", "repaid", "paid", "good", "current", "fully paid"}


def describe(feature):
    if feature in NUMERIC:
        return NUMERIC[feature][3]
    if feature in BINARY:
        return BINARY[feature]
    return {"employment_type": "Employment type (Salaried, Self-employed, Business owner, Other)",
            "loan_purpose": "Loan purpose (Home, Education, Vehicle, Personal, Business)"}.get(feature, feature)


def break_even(margin=PROFIT_MARGIN, loss=LOSS_GIVEN_DEFAULT):
    """The default chance above which a loan is expected to lose money: margin / (margin + loss)."""
    return margin / (margin + loss)


def risk_band(probability, approve_below):
    """'High' at or above the break-even chance, 'Medium' from half of it, otherwise 'Low'."""
    if probability >= approve_below:
        return "High"
    if probability >= approve_below / 2:
        return "Medium"
    return "Low"


DECISIONS = {"Low": "Approve", "Medium": "Manual review", "High": "Decline"}


def decision(probability, approve_below):
    return DECISIONS[risk_band(probability, approve_below)]


def _key(text):
    return " ".join(str(text).replace("_", " ").replace("-", " ").split()).lower()


def _missing(value):
    return value is None or (isinstance(value, float) and value != value)


def canonical_category(feature, value):
    """'  salary ' -> 'Salaried'. Returns None when the value is missing or not recognised."""
    if _missing(value):
        return None
    lookup = {_key(name): name for name in CATEGORICAL[feature]}
    lookup.update({_key(alias): name for alias, name in ALIASES.get(feature, {}).items()})
    return lookup.get(_key(value))


def parse_yes_no(value):
    """'Yes' / 'y' / true / 1 -> 1;  'no' / 'n' / false / 0 -> 0;  anything else -> None."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value == value:
        return {1: 1, 0: 0}.get(int(value)) if value in (0, 1) else None
    return {"yes": 1, "y": 1, "true": 1, "1": 1, "no": 0, "n": 0, "false": 0, "0": 0}.get(str(value).strip().lower())


def canonical_label(value):
    """'Yes ' / 1 / 'defaulted' -> 1;  'No' / 0 / 'repaid' -> 0;  anything else (or missing) -> None."""
    if _missing(value):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and value == int(value):
        value = int(value)
    key = _key(value)
    if key in DEFAULT_YES:
        return 1
    if key in DEFAULT_NO:
        return 0
    return None


def format_lakh(amount):
    """4.5 -> '₹4.5 lakh'; 120 -> '₹1.2 crore'; 0.5 -> '₹50,000'; negatives keep their sign."""
    sign, amount = ("-" if amount < 0 else ""), abs(amount)
    if amount >= 100:
        return f"{sign}₹{amount / 100:.2f} crore".replace(".00 ", " ")
    if amount >= 1:
        return f"{sign}₹{amount:.2f} lakh".replace(".00 ", " ")
    return f"{sign}₹{amount * 100000:,.0f}"


def monthly_payment(amount_lakh, annual_rate_pct, months):
    """Monthly instalment in THOUSAND rupees for a loan in lakh rupees (standard amortisation). Works on arrays."""
    import numpy as np
    amount = np.asarray(amount_lakh, float) * 100.0                       # thousand rupees
    months = np.maximum(np.asarray(months, float), 1.0)
    r = np.asarray(annual_rate_pct, float) / 1200.0
    safe = np.where(r > 1e-9, r, 1.0)
    return np.where(r > 1e-9, amount * safe / (1 - (1 + safe) ** -months), amount / months)
