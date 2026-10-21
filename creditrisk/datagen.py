"""Synthetic loan data for the project.

Real loan records are private, so the project ships a generator that builds a lender's portfolio with a KNOWN
default rule. Each applicant's true chance of defaulting is computed from their credit score, repayment burden,
history and loan terms; whether they actually default is then a random draw from that chance. Two things follow:

* no model can ever be perfect (even the true rule only ranks borrowers so well: the "ceiling"), and
* the evaluation can show how close a model gets to that ceiling.

The data is clearly SYNTHETIC: it teaches the workflow and must never be used to judge a real person.
The generated file is also deliberately DIRTY (blanks, impossible values, amounts typed in rupees instead of
lakh, messy spellings, repeated applications) so the preprocessing stage has real work to do.

Like every lender's data, it only contains loans that were APPROVED in the past, so it says nothing about people
who were turned down ("selection bias"); the documentation points this out.
"""

import numpy as np
import pandas as pd

from . import schema as S

INTERCEPT = -3.6                        # tuned so that about one loan in six defaults
EMPLOYMENT_EFFECT = {"Salaried": 0.0, "Self-employed": 0.25, "Business owner": 0.15, "Other": 0.6}
PURPOSE_EFFECT = {"Home": -0.4, "Education": 0.1, "Vehicle": -0.1, "Personal": 0.35, "Business": 0.45}


def default_probability(annual_income, loan_amount, loan_term_months, interest_rate, credit_score, employment_years,
                        monthly_debt_payments, late_payments, credit_utilisation, owns_home, has_cosigner,
                        employment_type, loan_purpose):
    """The 'ground truth' rule: the chance (0-1) that each loan defaults. Vectorised over arrays / columns."""
    income, amount = np.asarray(annual_income, float), np.asarray(loan_amount, float)
    burden = (S.monthly_payment(amount, interest_rate, loan_term_months) + np.asarray(monthly_debt_payments, float)) * 12 / (income * 100)
    z = (INTERCEPT
         - 0.011 * (np.asarray(credit_score, float) - 680)
         + 2.6 * np.minimum(burden, 1.5)                                 # share of income already committed to repayments
         + 0.30 * np.asarray(late_payments, float)
         + 0.016 * (np.asarray(credit_utilisation, float) - 30)
         - 0.04 * np.minimum(np.asarray(employment_years, float), 15)
         + np.array([EMPLOYMENT_EFFECT[e] for e in employment_type])
         + np.array([PURPOSE_EFFECT[p] for p in loan_purpose])
         - 0.5 * np.asarray(has_cosigner, float) - 0.3 * np.asarray(owns_home, float))
    return 1 / (1 + np.exp(-z))


def generate(rows=3000, seed=2026, dirty=True):
    """Return a DataFrame with application_id, the schema's feature columns, age and default ('Yes'/'No')."""
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(37, 10, rows), 21, 70).astype(int)
    employment = rng.choice(S.EMPLOYMENT, rows, p=[0.55, 0.2, 0.1, 0.15])
    income = np.clip(np.exp(rng.normal(np.log(7.0), 0.55, rows)) * np.where(employment == "Business owner", 1.3, 1.0), 1.5, 100).round(2)
    purpose = rng.choice(S.PURPOSES, rows, p=[0.22, 0.12, 0.2, 0.3, 0.16])
    factor = np.select([purpose == "Home", purpose == "Education", purpose == "Vehicle", purpose == "Business"],
                       [rng.uniform(1.5, 5.0, rows), rng.uniform(0.3, 1.2, rows), rng.uniform(0.4, 1.3, rows), rng.uniform(0.5, 2.5, rows)],
                       default=rng.uniform(0.1, 0.9, rows))
    amount = np.clip(income * factor, 0.3, 60).round(2)
    term = np.where(purpose == "Home", rng.choice([60, 84, 120], rows), rng.choice([12, 24, 36, 48, 60], rows)).astype(int)
    score = np.clip(rng.normal(690, 75, rows) + 6 * np.log(income), 300, 900).astype(int)
    rate = np.clip(10.5 + (720 - score) * 0.016 + rng.normal(0, 1.0, rows) + (purpose == "Personal") * 2, 6, 29).round(1)
    work = np.clip(rng.uniform(0, 1, rows) * (age - 20) * 0.8 - (employment == "Other") * 3, 0, 40).round(1)
    debt = np.clip(income * 100 / 12 * rng.beta(1.5, 5, rows) * (rng.random(rows) < 0.8), 0, 300).round(1)
    accounts = np.clip(rng.poisson(3 + (age - 21) / 12, rows), 0, 25)
    late = np.clip(rng.poisson(0.2 + np.clip(700 - score, 0, None) / 120), 0, 12)
    late = np.where(accounts == 0, 0, late)
    utilisation = np.clip(rng.beta(2, 4, rows) * 100 + (650 - score) * 0.1, 0, 100).round(1)
    owns = (rng.random(rows) < 0.25 + 0.35 * (age > 32) + 0.1 * (income > 10)).astype(int)
    cosigner = (rng.random(rows) < 0.10).astype(int)
    p = default_probability(income, amount, term, rate, score, work, debt, late, utilisation, owns, cosigner, employment, purpose)
    defaulted = rng.random(rows) < p
    df = pd.DataFrame({
        S.ID: [f"LN{100000 + i}" for i in range(rows)], "annual_income": income, "loan_amount": amount,
        "loan_term_months": term, "interest_rate": rate, "credit_score": score, "employment_years": work,
        "monthly_debt_payments": debt, "credit_accounts": accounts, "late_payments": late,
        "credit_utilisation": utilisation, "owns_home": owns, "has_cosigner": cosigner, "employment_type": employment,
        "loan_purpose": purpose, S.MONITOR: age, S.TARGET: np.where(defaulted, "Yes", "No")})
    return make_dirty(df, rng) if dirty else df


def _bump(score):
    """A slightly different credit score for a repeated application (impossible and unreadable scores stay as they are)."""
    try:
        value = int(score)
    except (TypeError, ValueError):
        return score
    return value if value in (0, 25, 1200, 9999) else min(value + 20, 900)


def make_dirty(df, rng):
    """Inject realistic data-quality problems (about 10% of rows are affected)."""
    df = df.copy().astype(object)
    n = len(df)

    def rows(count):
        return rng.choice(n, count, replace=False)

    for col, rate in (("interest_rate", 0.03), ("employment_years", 0.04), ("monthly_debt_payments", 0.03),
                      ("credit_accounts", 0.03), ("late_payments", 0.03), ("credit_utilisation", 0.05),
                      ("owns_home", 0.02), ("has_cosigner", 0.02), ("loan_purpose", 0.02), (S.MONITOR, 0.03),
                      ("credit_score", 0.01)):
        df.loc[rng.random(n) < rate, col] = np.nan                                       # missing values
    for idx in rows(8):                                                                  # money typed in rupees, not lakh
        df.loc[idx, "annual_income"] = int(round(float(df.loc[idx, "annual_income"]) * 100000))
    for idx in rows(8):
        df.loc[idx, "loan_amount"] = int(round(float(df.loc[idx, "loan_amount"]) * 100000))
    for idx in rows(5):
        df.loc[idx, "annual_income"] = float(rng.choice([-4, 0, 0.1]))
    for idx in rows(5):
        df.loc[idx, "credit_score"] = int(rng.choice([0, 25, 1200, 9999]))
    for idx in rows(4):
        df.loc[idx, "interest_rate"] = float(rng.choice([-1, 0, 80]))
    for idx in rows(4):
        df.loc[idx, "loan_term_months"] = int(rng.choice([0, 3, 600]))
    for idx in rows(4):
        df.loc[idx, "credit_utilisation"] = float(rng.choice([150, -20]))
    for idx in rows(4):
        df.loc[idx, S.MONITOR] = int(rng.choice([5, 150]))
    for idx in rows(8):                                                                  # unreadable text
        df.loc[idx, str(rng.choice(["annual_income", "credit_score", "loan_amount"]))] = str(rng.choice(["n/a", "high", "??", "5L"]))
    for idx in rows(6):                                                                  # late payments with no credit history
        df.loc[idx, ["credit_accounts", "late_payments"]] = [0, 3]
    for idx in rows(5):                                                                  # more years working than adult life
        df.loc[idx, [S.MONITOR, "employment_years"]] = [24, 30]
    spellings = {"employment_type": ["salary", "Employee", "self employed", "BUSINESS", "retired", "freelancer"],
                 "loan_purpose": ["house", "Home Loan", "car", "wedding", "business loan", "study"]}
    for col, options in spellings.items():
        for idx in np.flatnonzero(rng.random(n) < 0.05):
            df.loc[idx, col] = str(rng.choice(options))
    for idx in rows(5):
        df.loc[idx, "employment_type"] = np.nan                                          # unusable: employment type is essential
    for col in ("owns_home", "has_cosigner"):                                            # yes/no written in many ways
        for idx in np.flatnonzero((rng.random(n) < 0.20) & df[col].notna().to_numpy()):
            value = df.loc[idx, col]
            if value in (0, 1):
                df.loc[idx, col] = str(rng.choice(["Yes", "Y", "true", "1"] if value == 1 else ["No", "N", "false", "0"]))
    for idx in np.flatnonzero(rng.random(n) < 0.15):                                     # outcome spellings
        df.loc[idx, S.TARGET] = str(rng.choice(["yes ", "1", "True", "defaulted"] if df.loc[idx, S.TARGET] == "Yes"
                                               else ["no ", "0", "repaid", "False"]))
    for idx in rows(5):
        df.loc[idx, S.TARGET] = np.nan
    for idx in rows(4):
        df.loc[idx, S.TARGET] = "pending"
    dupes = df.sample(int(n * 0.015), random_state=1)                                    # exact repeats
    reused = df.sample(6, random_state=2).copy()                                         # same id, different details
    reused["credit_score"] = reused["credit_score"].map(_bump)
    return pd.concat([df, dupes, reused], ignore_index=True).sample(frac=1, random_state=3).reset_index(drop=True)
