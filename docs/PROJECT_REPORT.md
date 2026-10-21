# Project Report – Credit Risk Prediction

**Intern:** Sneha S · **Intern ID:** SKX-2026-6425 · **Programme:** Skyrovix virtual internship · **Due:** 08 Dec 2026

> A learning demo on synthetic data. It must not be used to make real lending decisions.

## 1. Objective
Build a credit risk prediction system that estimates the chance a loan defaults and supports an approve / review / decline decision, with a documented workflow covering data preprocessing, model training, evaluation and a prediction interface, plus testing and a submission-ready project.

## 2. Workflow

1. **Data.** Real loan records are private, so a generator builds a lender's portfolio (income, loan amount, term, rate, credit score, employment, existing debt, credit history, utilisation, home ownership, co-signer, purpose, and age for fairness checks) with a *known* default rule; whether a loan defaults is a random draw from its true chance. The best possible score is therefore known, and data problems are injected on purpose.
2. **Preprocessing** (`clean`, before the split, row by row so nothing leaks from test data): labels standardised (unknown outcomes removed), spellings fixed, unreadable text and out-of-range values blanked, money typed in rupees converted to lakh, impossible combinations blanked (late payments with no accounts, work years longer than adult life), repeats removed, rows without an employment type removed. Sample run: 3,051 rows read → 2,986 kept, 65 removed, every step logged. Learned steps (median imputing, scaling, one-hot encoding) sit inside the model pipeline and are fitted on training rows only. Engineered features: loan-to-income, debt-to-income, estimated instalment and total repayments as a share of income.
3. **Split.** 80% train / 20% test, stratified so both keep the same default share (16%).
4. **Model comparison.** Five candidates in the same pipeline, 5-fold cross-validation on training data, ranked by ROC AUC. The best is tuned (randomised search) and kept only if cross-validation improves; calibration is applied only if it lowers the Brier score, because the decision threshold is a probability.
5. **Decision rule.** A repaid loan earns 12% of its amount and a default loses 60%, so the expected profit of a loan, (1 − p) × 12% − p × 60%, is positive only when p is below **16.7%**. That is the decline threshold; from half of it (8.3%) an application goes to manual review. Out-of-fold training predictions confirm the rule beats approving everyone before it is used.
6. **Evaluation, once, on the untouched test set**, including money, fairness segments and the effect of each policy.
7. **Explanation.** For each applicant, one detail at a time is replaced by the typical value; the details that raise the risk most are listed as the main reasons. "What if" scenarios do the same with improved values.
8. **Interfaces.** Menu, command line, local web form, JSON API, batch CSV; shared validation with a message per field.

## 3. Results (test set, 598 applications)

| Model (cross-validated ROC AUC) | AUC | PR AUC | Brier |
|---|---|---|---|
| **Logistic regression (tuned)** | **0.820** | 0.506 | 0.103 |
| Logistic regression | 0.818 | 0.501 | 0.103 |
| Random forest | 0.810 | 0.481 | 0.106 |
| Gradient boosting | 0.806 | 0.465 | 0.108 |
| Decision tree | 0.747 | 0.380 | 0.119 |
| Baseline (nobody defaults) | 0.500 | 0.156 | 0.132 |

| Test-set measure | Value |
|---|---|
| ROC AUC | 0.838 (training 0.828, best possible 0.836: no overfitting) |
| Approved at the 16.7% rule | 426 of 598 (71%); 25 of them defaulted (5.9%) |
| Declined | 172 applicants; 69 really defaulted (40%), 103 would have repaid |
| Defaulters declined | 73% of all defaulters |
| Accuracy | 79% (approving everybody: 84%) |
| Calibration | already honest, so calibration was tried but not applied |

| Policy | Approved | Defaults among approved | Profit |
|---|---|---|---|
| Approve everyone | 598 | 15.7% | −₹18.8 lakh |
| Decline everyone | 0 | – | ₹0 |
| **Use the model** | 426 | 5.9% | **₹3.16 crore** |
| Perfect foresight (impossible) | 504 | 0% | ₹6.50 crore |

| Risk band | Decision | Applicants | Defaulted |
|---|---|---|---|
| Low | Approve | 270 (45%) | 3.3% |
| Medium | Manual review | 156 (26%) | 10.3% |
| High | Decline | 172 (29%) | 40.1% |

**Findings**
- Approving everyone loses money; the rule turns the same applications into a profit of about half of what perfect foresight could earn.
- The bands separate applicants clearly: the High band defaults about twelve times as often as the Low band.
- Loan size, loan term, income and credit score drive the estimate; late payments and existing debt follow.
- Fairness check: the model uses no personal attributes and not age. Approval rates still differ by employment type (61% for "Other" to 79% for business owners) and slightly by age group (68% for under 30 to 75% for 50 or older). The first follows real differences in default rates in the data; both are reported for monitoring.

## 4. Prediction interface
- **Inputs:** five required details, nine optional; units and formats are understood (`8.5 lakh`, `₹850000`, `3 years`, `11.5%`); missing details are assumed typical and reported.
- **Outputs:** decision, estimated default chance, break-even level, expected result as a share of the loan, instalment and repayment burden, main reasons, comparison with a typical applicant, what-if scenarios, warnings for values outside the training range or a loan more than ten times income, and the disclaimer.
- **Safety:** web server on localhost only with Host check, size limits, escaped output, no scripts, strict Content-Security-Policy; model files verified by SHA-256 before loading; no personal attributes in the form.

## 5. Testing
138 automated tests: data and cleaning (33), models/economics/training/persistence/reports (41), validation and prediction (38), web over real HTTP, menu sessions and command line (26). They check, among other things, that cleaning removes each injected problem, that the profit arithmetic and break-even threshold are right, that the rule beats approving everyone on unseen loans, that age cannot change a prediction, that risk moves the right way with each factor, that tampered models are refused, and that script injection in form fields is escaped. Training with a fixed seed reproduces the same numbers.

## 6. Limitations and next steps
- Synthetic data: scores show how the methods behave, not how they would perform on real borrowers. Real lending would need real data, legal and fairness review, validation and regulatory approval.
- Selection bias: lenders only see outcomes of loans they approved (reject inference is outside this project).
- Possible extensions: segment-level monitoring of approval rates over time, loan-amount-specific thresholds, survival models for time to default, comparison with a bureau score.
