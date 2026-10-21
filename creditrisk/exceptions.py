"""Custom exception hierarchy.

Every deliberate error derives from CreditError so interfaces can catch a single base class
and show a friendly message instead of a traceback.
"""


class CreditError(Exception):
    """Base class for all project errors."""


class ValidationError(CreditError):
    """User input (a customer record or a setting) is invalid.

    `errors` maps a field name to its problem, so forms can show each message next to its field.
    """

    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = dict(errors or {})


class DataError(CreditError):
    """A dataset file is missing, unreadable or not usable for training."""


class ModelError(CreditError):
    """A model file is missing, damaged, tampered with or incompatible."""


class NotTrainedError(ModelError):
    """A prediction was requested but no trained model is available."""
