"""Candidate models, cross-validated comparison, light hyper-parameter tuning and probability calibration.

Every candidate is wrapped in the SAME pipeline (feature engineering -> imputing/scaling/encoding -> classifier),
so the comparison is fair. Folds keep the share of borrowers who default equal (stratified). Models are ranked
by ROC AUC: how well they put borrowers who default above those who repay, whatever threshold is
used later.

A lending decision needs real PROBABILITIES ("a 12% chance of default"), not just a ranking, so the chosen model is
optionally wrapped in a calibrator when that improves the Brier score.
"""

import time
import warnings

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from . import preprocessing as P

BASELINE = "Baseline (nobody defaults)"
SCORING = {"roc_auc": "roc_auc", "average_precision": "average_precision", "brier": "neg_brier_score",
           "accuracy": "accuracy"}

TUNING_GRIDS = {
    "Logistic Regression": {"model__C": [0.03, 0.1, 0.3, 1, 3, 10]},
    "Decision Tree": {"model__max_depth": [3, 4, 5, 6, 8], "model__min_samples_leaf": [10, 20, 40]},
    "Random Forest": {"model__max_depth": [None, 6, 10], "model__min_samples_leaf": [3, 8, 15],
                      "model__max_features": ["sqrt", 0.5]},
    "Gradient Boosting": {"model__learning_rate": [0.03, 0.06, 0.1], "model__max_depth": [2, 3, 4],
                          "model__min_samples_leaf": [20, 40, 80], "model__l2_regularization": [0.0, 1.0]},
}


def build_pipeline(estimator):
    return Pipeline([("features", P.feature_adder()), ("prep", P.build_preprocessor()), ("model", estimator)])


def candidates(seed=42, fast=False):
    """name -> unfitted estimator (the pipeline is built around it)."""
    return {
        BASELINE: DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=3000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=80 if fast else 300, min_samples_leaf=8,
                                                n_jobs=1 if fast else -1, random_state=seed),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=100 if fast else 200, learning_rate=0.05,
                                                            max_depth=3, min_samples_leaf=40, random_state=seed),
    }


def _summarise(name, scores, seconds):
    out = {"model": name, "seconds": round(seconds, 1)}
    for key in SCORING:
        value = float(scores[f"test_{key}"].mean())
        out[key] = -value if key == "brier" else value
    out["roc_auc_std"] = float(scores["test_roc_auc"].std())
    return out


def compare(X, y, cv=5, seed=42, fast=False, progress=None):
    """Cross-validate every candidate on the TRAINING data only. Best (by ROC AUC) first."""
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    results = []
    for name, estimator in candidates(seed, fast).items():
        if progress:
            progress(f"Cross-validating {name} ...")
        started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_validate(build_pipeline(estimator), X, y, cv=folds, scoring=SCORING)
        results.append(_summarise(name, scores, time.perf_counter() - started))
    return sorted(results, key=lambda r: (r["roc_auc"], r["average_precision"]), reverse=True)


def best_candidate(results):
    """The best-scoring model that is not the baseline."""
    return next(r for r in results if r["model"] != BASELINE)


def tune(name, X, y, seed=42, cv=3, iterations=8, fast=False, progress=None):
    """Randomised search over a small grid. Returns (search, tuned_params) or (None, {}) if there is no grid."""
    grid = TUNING_GRIDS.get(name)
    if not grid:
        return None, {}
    if progress:
        progress(f"Tuning {name} ({iterations} settings × {cv} folds) ...")
    search = RandomizedSearchCV(build_pipeline(candidates(seed, fast)[name]), grid,
                                n_iter=int(min(iterations, np.prod([len(v) for v in grid.values()]))),
                                cv=StratifiedKFold(cv, shuffle=True, random_state=seed),
                                scoring="roc_auc", random_state=seed, n_jobs=1, refit=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X, y)
    return search, {k.replace("model__", "", 1): v for k, v in search.best_params_.items()}


def pipeline_with_params(name, params, seed=42, fast=False):
    """A fresh pipeline for `name` with tuned parameters applied."""
    estimator = candidates(seed, fast)[name]
    estimator.set_params(**params)
    return build_pipeline(estimator)


def calibrated(pipeline, n_rows, cv=3):
    """The pipeline wrapped in a probability calibrator (isotonic when there is plenty of data, else sigmoid)."""
    from sklearn.base import clone
    method = "isotonic" if n_rows >= 1500 else "sigmoid"
    return method, CalibratedClassifierCV(clone(pipeline), cv=cv, method=method)


def choose_calibration(pipeline, X, y, seed=42, cv=5):
    """Compare the Brier score of the raw and the calibrated pipeline on out-of-fold training predictions.

    Returns (final_unfitted_model, out_of_fold_probabilities, info). The calibrated version is used only if it
    is better, so a model that already gives honest probabilities (such as logistic regression) is left alone.
    """
    from sklearn.base import clone
    folds = StratifiedKFold(cv, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = cross_val_predict(clone(pipeline), X, y, cv=folds, method="predict_proba")[:, 1]
        method, wrapped = calibrated(pipeline, len(y))
        cal = cross_val_predict(wrapped, X, y, cv=folds, method="predict_proba")[:, 1]
    before, after = float(brier_score_loss(y, raw)), float(brier_score_loss(y, cal))
    if after < before - 1e-4:
        return wrapped, cal, {"method": method, "brier_before": before, "brier_after": after, "used": True}
    return clone(pipeline), raw, {"method": "none", "brier_before": before, "brier_after": after,
                                  "used": False, "tried": method}
