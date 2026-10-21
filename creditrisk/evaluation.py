"""Model evaluation for credit risk: ranking quality, honest probabilities and what a lending policy earns.

A credit model is only worth its money if acting on it pays. So besides the usual metrics this module takes the
approval threshold from the economics of lending (profit on a loan that is repaid against the loss on one that
defaults), compares the model with "approve everyone" and "decline everyone", and builds the band, calibration
and fairness tables. All money is in lakh rupees.
"""

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import schema as S


def classification_metrics(actual, probability, threshold):
    """Counts and percentages when every applicant at or above `threshold` is declined."""
    actual = np.asarray(actual, int)
    declined = np.asarray(probability, float) >= threshold
    bad = actual == 1
    tp, fp = int((declined & bad).sum()), int((declined & ~bad).sum())          # declined: really defaulters / would have repaid
    fn, tn = int((~declined & bad).sum()), int((~declined & ~bad).sum())        # approved: defaulted / repaid
    recall = 100 * tp / (tp + fn) if tp + fn else 0.0                          # share of defaulters that were declined
    precision = 100 * tp / (tp + fp) if tp + fp else 0.0                       # share of declines that were right
    approved = fn + tn
    return {
        "n": int(len(actual)), "threshold": float(threshold), "declined": tp + fp, "approved": approved,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": 100 * (tp + tn) / len(actual),
        "recall": recall, "precision": precision,
        "approval_rate": 100 * approved / len(actual),
        "bad_rate_approved": 100 * fn / approved if approved else 0.0,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def ranking_metrics(actual, probability):
    """Threshold-free scores: ROC AUC, average precision (PR AUC) and the Brier score (lower = more honest)."""
    actual, probability = np.asarray(actual, int), np.asarray(probability, float)
    if len(set(actual.tolist())) < 2:
        return {"roc_auc": 0.0, "average_precision": 0.0, "brier": float(brier_score_loss(actual, probability))}
    return {"roc_auc": float(roc_auc_score(actual, probability)),
            "average_precision": float(average_precision_score(actual, probability)),
            "brier": float(brier_score_loss(actual, probability))}


# ---------------------------------------------------------------------- lending economics
def lending_result(actual, probability, amount, threshold, margin=S.PROFIT_MARGIN, loss=S.LOSS_GIVEN_DEFAULT):
    """Outcome of approving every applicant whose default chance is BELOW `threshold`.

    A repaid loan earns `margin` x its amount; a defaulted one loses `loss` x its amount; a declined one earns nothing.
    Amounts and profit are in lakh rupees.
    """
    actual, probability, amount = np.asarray(actual, int), np.asarray(probability, float), np.asarray(amount, float)
    approved = probability < threshold
    profit = float((np.where(actual == 1, -loss, margin) * amount)[approved].sum())
    lent = float(amount[approved].sum())
    return {"approved": int(approved.sum()), "approved_share": 100 * float(approved.mean()), "lent": lent,
            "defaults": int(((actual == 1) & approved).sum()),
            "default_rate": 100 * float(actual[approved].mean()) if approved.any() else 0.0,
            "profit": profit, "return_on_lent": 100 * profit / lent if lent else 0.0,
            "profit_per_100_applied": 100 * profit / float(amount.sum()) if amount.sum() else 0.0}


def choose_approval_threshold(actual, probability, amount, margin=S.PROFIT_MARGIN, loss=S.LOSS_GIVEN_DEFAULT):
    """The default chance above which a loan is declined: the break-even point, margin / (margin + loss).

    With honest (calibrated) probabilities this is exactly the profit-maximising rule: a loan is worth making when
    the expected profit, (1 - p) x margin - p x loss, is positive. The out-of-fold training predictions are then used
    to CHECK that the policy makes money, and the empirically best threshold is reported for comparison (it chases
    noise, so it is not used).
    """
    actual, probability, amount = np.asarray(actual, int), np.asarray(probability, float), np.asarray(amount, float)
    t = S.break_even(margin, loss)
    on_policy = lending_result(actual, probability, amount, t, margin, loss)
    approve_all = lending_result(actual, probability, amount, 1.0000001, margin, loss)
    best_t, best = 0.0, 0.0
    for candidate in np.unique(probability):
        profit = lending_result(actual, probability, amount, candidate, margin, loss)["profit"]
        if profit > best + 1e-9:
            best_t, best = float(candidate), profit
    return {"approve_below": float(t), "break_even": float(t), "margin": float(margin), "loss": float(loss),
            "training_profit": on_policy["profit"], "training_profit_approve_all": approve_all["profit"],
            "policy_beats_approving_all": bool(on_policy["profit"] > approve_all["profit"]),
            "empirical_best": best_t, "empirical_best_profit": best}


def policy_table(actual, probability, amount, threshold, margin=S.PROFIT_MARGIN, loss=S.LOSS_GIVEN_DEFAULT):
    """Approve everyone / decline everyone / the model's rule / a crystal ball, side by side."""
    actual = np.asarray(actual, int)
    rows = [("Approve everyone", lending_result(actual, np.zeros(len(actual)), amount, 1, margin, loss)),
            ("Decline everyone", lending_result(actual, np.ones(len(actual)), amount, 0.5, margin, loss)),
            ("Use the model", lending_result(actual, probability, amount, threshold, margin, loss)),
            ("Perfect foresight (impossible)", lending_result(actual, actual.astype(float), amount, 0.5, margin, loss))]
    return [{"policy": name, **vals} for name, vals in rows]


def profit_curve(actual, probability, amount, margin=S.PROFIT_MARGIN, loss=S.LOSS_GIVEN_DEFAULT, steps=40, top=0.6):
    """[(threshold, profit in lakh)] for a chart of how the approval threshold matters."""
    return [(round(float(t), 3), round(lending_result(actual, probability, amount, t, margin, loss)["profit"], 2))
            for t in np.linspace(0, top, steps + 1)]


def band_table(actual, probability, threshold):
    """Low / Medium / High: how many applicants fall in each band, how many really defaulted, and the decision."""
    actual, probability = np.asarray(actual, int), np.asarray(probability, float)
    bands = np.array([S.risk_band(p, threshold) for p in probability])
    rows = []
    for name in ("Low", "Medium", "High"):
        mask = bands == name
        n = int(mask.sum())
        rows.append({"band": name, "decision": S.DECISIONS[name], "applicants": n, "share": 100 * n / len(actual),
                     "defaults": int(actual[mask].sum()), "rate": 100 * float(actual[mask].mean()) if n else 0.0,
                     "mean_probability": 100 * float(probability[mask].mean()) if n else 0.0})
    return rows


def calibration_table(actual, probability, bins=10):
    """Predicted chance against what really happened, per probability band. Empty bands are skipped."""
    actual, probability = np.asarray(actual, int), np.asarray(probability, float)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = which == b
        if mask.any():
            rows.append({"band": f"{edges[b] * 100:.0f}-{edges[b + 1] * 100:.0f}%", "applicants": int(mask.sum()),
                         "predicted": 100 * float(probability[mask].mean()), "observed": 100 * float(actual[mask].mean())})
    return rows


def roc_points(actual, probability, steps=40):
    """[(false-positive rate, true-positive rate)] for the ROC chart, thinned to about `steps` points."""
    actual, probability = np.asarray(actual, int), np.asarray(probability, float)
    order = np.argsort(-probability, kind="stable")
    bad, good = max(actual.sum(), 1), max((1 - actual).sum(), 1)
    tpr = np.concatenate([[0], np.cumsum(actual[order]) / bad])
    fpr = np.concatenate([[0], np.cumsum(1 - actual[order]) / good])
    keep = np.unique(np.linspace(0, len(tpr) - 1, steps + 1).round().astype(int))
    return [(round(float(fpr[i]), 4), round(float(tpr[i]), 4)) for i in keep]


def segment_rows(frame, by):
    """Per group: applicants, real default rate, mean predicted rate, approval rate and default rate among approved."""
    rows = []
    for key, group in frame.groupby(by, observed=True):
        approved = group["declined"] == 0
        rows.append({"group": str(key), "n": int(len(group)), "actual": float(100 * group["actual"].mean()),
                     "predicted": float(100 * group["probability"].mean()), "approval_rate": float(100 * approved.mean()),
                     "bad_rate_approved": float(100 * group.loc[approved, "actual"].mean()) if approved.any() else None})
    return rows


def age_group(age):
    return "under 30" if age < 30 else "30-39" if age < 40 else "40-49" if age < 50 else "50 or older"


AGE_GROUPS = ["under 30", "30-39", "40-49", "50 or older"]


def feature_importance(pipeline, X_test, y_test, seed=42, repeats=8):
    """Permutation importance: how much ROC AUC drops when one feature's values are shuffled (test set)."""
    result = permutation_importance(pipeline, X_test, y_test, scoring="roc_auc", n_repeats=repeats, random_state=seed,
                                    n_jobs=1)
    rows = [{"feature": f, "label": S.LABELS[f], "importance": max(0.0, float(m)), "std": float(s)}
            for f, m, s in zip(X_test.columns, result.importances_mean, result.importances_std)]
    return sorted(rows, key=lambda r: r["importance"], reverse=True)


def true_risk_ceiling(df_clean):
    """For SYNTHETIC data only: the scores of the TRUE default probabilities against what actually happened.

    No model can systematically beat this: the rest is pure luck in who actually defaults. Missing values are
    filled with the median / most common value, as the model's own pipeline does.
    """
    from .datagen import default_probability
    d = df_clean
    med = {c: d[c].median() for c in S.NUMERIC_FEATURES}
    p = default_probability(
        d["annual_income"].fillna(med["annual_income"]), d["loan_amount"].fillna(med["loan_amount"]),
        d["loan_term_months"].fillna(med["loan_term_months"]), d["interest_rate"].fillna(med["interest_rate"]),
        d["credit_score"].fillna(med["credit_score"]), d["employment_years"].fillna(med["employment_years"]),
        d["monthly_debt_payments"].fillna(med["monthly_debt_payments"]), d["late_payments"].fillna(0),
        d["credit_utilisation"].fillna(med["credit_utilisation"]), d["owns_home"].fillna(0), d["has_cosigner"].fillna(0),
        d["employment_type"], d["loan_purpose"].fillna("Personal"))
    return ranking_metrics(d[S.TARGET], p)
