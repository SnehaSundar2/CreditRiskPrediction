# Credit Risk Prediction

A machine-learning project that estimates the chance that a loan applicant will default, and turns that estimate into an **approve / manual review / decline** decision using the economics of lending. It covers the full workflow: **data preprocessing → model training → evaluation → prediction interface**.

Developed by **Sneha S**, Skyrovix virtual internship (Intern ID SKX-2026-6425).

> **This is a learning demo trained on generated (synthetic) data. It must not be used to make real lending decisions.** The notice is repeated in every interface and report.

## What it does

- **Cleans messy loan data** and logs every fix: blanks, impossible values (credit score 9999, loan term 600 months), income or loan amounts typed in rupees instead of lakh, text such as "n/a", messy spellings (`salary`, `car`), yes/no written many ways, late payments for someone with no credit accounts, more years in work than adult life allows, repeated applications.
- **Compares models fairly** with cross-validation on training data: a baseline, logistic regression, decision tree, random forest and gradient boosting, all in the same pipeline (feature engineering → imputing/scaling/encoding → model). The best is lightly tuned and its probabilities are calibrated only if that improves the Brier score.
- **Takes the decision rule from the economics of lending.** A repaid loan earns 12% of its amount and a default loses 60% (both adjustable), so a loan is worth making only when the default chance is below 12 / (12 + 60) = **16.7%**. That is the decline threshold; from half of it the application goes to manual review. The rule is checked on training data and scored once on the test set.
- **Evaluates honestly:** ROC AUC, PR AUC, Brier score, approval rate, defaults among approved loans, confusion matrix, what each policy earns (approve everyone, decline everyone, the model, a crystal ball), risk bands, calibration, a profit-against-threshold curve and the best possible score on the generated data.
- **Explains every decision:** the main reasons the risk is high (like the reason codes lenders give), what drives the estimate compared with a typical applicant, and "what if" scenarios (a 25% smaller loan, a co-signer, card use at 30%, …).
- **Checks fairness:** the model uses no gender, religion, caste, marital status, address or **age**. Age may be in a dataset only so the report can check approval rates by age group, employment type and loan purpose.
- **Interfaces:** interactive menu, command line, local web form, JSON API and batch CSV scoring.

## Quick start

```bash
pip install -r requirements.txt
python main.py --train                  # about 20 seconds
python main.py                          # interactive menu
python main.py --predict annual_income="4 lakh" loan_amount=6 loan_term_months="3 years" credit_score=560 employment_type=Other late_payments=3 credit_accounts=4
python main.py --score sample_data/applications_to_score.csv
python main.py --serve                  # web page at http://127.0.0.1:8050
python main.py --report reports/model_report.html
```

Money is in lakh rupees (1 lakh = 100,000). Required: `annual_income`, `loan_amount`, `loan_term_months`, `credit_score`, `employment_type`. For anything else left out, typical values are assumed and the result says so. Units are understood (`8.5 lakh`, `₹850000`, `0.4 crore`, `3 years`, `11.5%`).

Your own data: a CSV with the columns `annual_income, loan_amount, loan_term_months, interest_rate, credit_score, employment_years, monthly_debt_payments, credit_accounts, late_payments, credit_utilisation, owns_home, has_cosigner, employment_type, loan_purpose, default` (plus optional `application_id` and `age`) and at least 400 rows, then `python main.py --train --data my_loans.csv`. Change the economics with `--margin 0.10 --loss 0.65`. `python main.py --generate sample_data/loans.csv` writes a synthetic file; `sample_data/loans_dirty.csv` is a ready-made messy one.

JSON API: `POST /api/predict` with `{"annual_income": "8 lakh", "loan_amount": 4, "loan_term_months": 48, "credit_score": 700, "employment_type": "Salaried"}`; `GET /api/health`.

## Results (generated data, 2,986 applications after cleaning, test set 598)

| Measure | Value |
|---|---|
| ROC AUC (best possible on this data: 0.836) | **0.838** |
| PR AUC (guessing: 0.157) | 0.552 |
| Brier score | 0.099 |
| Approved at the 16.7% rule | 71% of applicants; **5.9%** of them defaulted (15.7% if everyone were approved) |
| Defaulters declined / declines that were right | 73% / 40% |
| Profit on the test loans: approve everyone / model / crystal ball | **−₹18.8 lakh** / **₹3.16 crore** / ₹6.50 crore |
| Default rate in the Low / Medium / High band | 3.3% / 10.3% / 40.1% |

Logistic regression won the cross-validation (AUC 0.820 against 0.810 for random forest); the generated default rule is close to logistic, so a simple model suits it, and with real data another model could win. Accuracy (79%) is not the headline: approving everybody scores 84% while losing money.

**Honest limitations:** many declined applicants would have repaid (precision 40%), the price of avoiding losses; approval rates differ by employment type (61% for "Other" to 79% for business owners) and slightly by age group (68% to 75%), which the report flags for monitoring; and a real lender only has outcomes for loans it approved, so a model trained on such data knows little about people it turned down.

## Project layout

```
main.py                 command line and entry point
creditrisk/
  schema.py             features, ranges, units, economics, decisions
  datagen.py            synthetic loan portfolio with a known default rule + dirty data
  preprocessing.py      loading, cleaning, feature engineering, scikit-learn transformer
  models.py             candidates, cross-validated comparison, tuning, calibration
  evaluation.py         metrics, lending economics, bands, fairness segments, importance
  training.py           the workflow and metadata
  persistence.py        model saving/loading with SHA-256 checks
  validators.py         input parsing with units and per-field messages
  predictor.py          decisions, reasons, what-if, batch scoring
  report.py             text and HTML report (SVG charts, light/dark)
  web.py / cli.py       local web form + JSON API, interactive menu
tests/                  138 tests
docs/PROJECT_REPORT.md
```

## Testing

```bash
python -m unittest discover -s tests -t .
```

138 tests (data and cleaning 33, models/economics/training/persistence/reports 41, validation and prediction 38, web over real HTTP, menu and command line 26), all passing. Requires Python 3.10+.
