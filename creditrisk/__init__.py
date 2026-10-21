"""Credit Risk Prediction.

An educational machine-learning project: synthetic loan data and its cleaning, feature engineering, model
comparison with cross-validation, probability calibration, an approve/review/decline rule taken from the
economics of lending, honest evaluation on a held-out test set, plain-language reasons and "what if"
scenarios for every decision, fairness checks, and prediction interfaces (menu, local web form, JSON API,
batch CSV).

It is a demonstration of the workflow. It must not be used to make real lending decisions.
"""

import logging

__version__ = "1.0.0"

logging.getLogger("creditrisk").addHandler(logging.NullHandler())
