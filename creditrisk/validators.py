"""Validation of loan applications (prediction input) and of command-line settings.

parse_features() turns raw text values (from the menu, a web form, a JSON request or a CSV row) into a clean,
consistent application - or raises ValidationError listing EVERY problem at once, with a message per field.
It uses the same schema as training-data cleaning, so both agree on what is valid.
"""

import difflib
import math
import re

import numpy as np

from . import schema as S
from .exceptions import ValidationError

_NUMBER = re.compile(r"^(?:rs\.?|inr|₹)?\s*([-+]?\d+(?:\.\d+)?|[-+]?\.\d+)\s*([a-zA-Z%/]*)$", re.I)
MONEY = ("annual_income", "loan_amount")                 # in lakh rupees
_LAKH = {"lakh", "lakhs", "lac", "lacs", "l"}
_CRORE = {"crore", "crores", "cr"}
_THOUSAND = {"k", "thousand"}
_YEARS = {"y", "yr", "yrs", "year", "years"}
_MONTHS = {"m", "mo", "mos", "month", "months"}


def normalise_key(key):
    return str(key).strip().lower().replace(" ", "_").replace("-", "_")


def _is_blank(value):
    return value is None or (isinstance(value, float) and value != value) or (isinstance(value, str) and not value.strip())


def _convert(feature, number, unit):
    """Apply a unit written by the user. Returns (number, note|None) or raises ValueError."""
    label = S.LABELS[feature]
    note = None
    if feature in MONEY:
        if unit in _LAKH or unit == "":
            pass
        elif unit in _CRORE:
            number, note = number * 100, f"{label}: {number:g} crore = {number * 100:g} lakh."
        elif unit in _THOUSAND:
            number, note = number / 100, f"{label}: {number:g} thousand = {number / 100:g} lakh."
        else:
            raise ValueError(f"{label} must be a number in lakh rupees, e.g. 8.5 or '8.5 lakh' (you entered a unit '{unit}').")
        return number, note
    if feature == "monthly_debt_payments":
        if unit in _LAKH:
            return number * 100, f"{label}: {number:g} lakh = {number * 100:g} thousand."
        if unit in _THOUSAND | {"", "thousand/month", "k/month"}:
            return number, None
        raise ValueError(f"{label} must be a number in thousand rupees per month (you entered a unit '{unit}').")
    if feature == "loan_term_months":
        if unit in _YEARS:
            return number * 12, f"Loan term: {number:g} years = {number * 12:g} months."
        if unit in _MONTHS | {""}:
            return number, None
        raise ValueError(f"{label} must be in months or years, e.g. 36 or '3 years' (you entered a unit '{unit}').")
    if feature == "employment_years":
        if unit in _YEARS | {""}:
            return number, None
        if unit in _MONTHS:
            return number / 12, f"Years in work: {number:g} months = {number / 12:.1f} years."
        raise ValueError(f"{label} must be a number of years (you entered a unit '{unit}').")
    if feature in ("interest_rate", "credit_utilisation"):
        if unit in ("", "%", "percent", "pct", "p.a", "pa"):
            return number, None
        raise ValueError(f"{label} must be a percentage (you entered a unit '{unit}').")
    if unit != "":
        raise ValueError(f"{label} must be a plain number (you entered a unit '{unit}').")
    return number, None


def parse_number(feature, value):
    """'8.5 lakh' -> 8.5 ; '850000' (rupees) -> 8.5 ; '3 years' (term) -> 36. Returns (number, note|None)."""
    kind, low, high, _ = S.NUMERIC[feature]
    label = S.LABELS[feature]
    note = None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a number.")
    if isinstance(value, (int, float)):
        number, unit = float(value), ""
    else:
        text = str(value).strip().replace(",", "")
        match = _NUMBER.match(text)
        if not match:
            raise ValueError(f"{label} must be a number (you entered '{text[:30]}').")
        number, unit = float(match.group(1)), match.group(2).lower()
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a normal number.")
    number, note = _convert(feature, number, unit)
    if feature in MONEY and note is None and number > 1000:        # clearly rupees, not lakh
        number, note = number / 100000.0, f"{label} {value} looks like rupees; read as {number / 100000.0:g} lakh."
    if kind == "int":
        if number != int(number):
            raise ValueError(f"{label} must be a whole number (you entered {number:g}).")
        number = int(number)
    if not low <= number <= high:
        unit_text = f" {S.UNITS[feature]}" if S.UNITS.get(feature) else ""
        raise ValueError(f"{label} must be between {low:g} and {high:g}{unit_text} (you entered {number:g}).")
    return number, note


def parse_category(feature, value):
    canon = S.canonical_category(feature, value)
    if canon is None:
        allowed = S.CATEGORICAL[feature]
        guess = difflib.get_close_matches(str(value).strip().title(), allowed, n=1, cutoff=0.6)
        hint = f" Did you mean '{guess[0]}'?" if guess else ""
        raise ValueError(f"{S.LABELS[feature]} '{str(value).strip()[:30]}' is not supported.{hint} "
                         f"Choose one of: {', '.join(allowed)}.")
    return canon


def parse_features(raw):
    """Validate one application. Returns (record, notes, assumed).

    record   every feature; optional ones left out are np.nan (the model's pipeline fills in typical values)
    notes    friendly remarks about conversions that were applied
    assumed  names of the optional features that were not given
    Raises ValidationError with .errors = {field: message} for every problem found.
    """
    if not isinstance(raw, dict):
        raise ValidationError("An application must be given as field/value pairs.")
    given, errors = {}, {}
    for key, value in raw.items():
        name = normalise_key(key)
        if name == S.ID or name == S.MONITOR:
            continue                                # known columns of a dataset that the model does not use
        if name not in S.RAW_FEATURES:
            guess = difflib.get_close_matches(name, S.RAW_FEATURES, n=1, cutoff=0.6)
            errors[name or "field"] = f"Unknown field '{str(key)[:30]}'." + (f" Did you mean '{guess[0]}'?" if guess else "")
            continue
        given[name] = value
    for name in S.REQUIRED_INPUT:
        if _is_blank(given.get(name)):
            errors[name] = f"{S.LABELS[name]} is required."
    record, notes = {}, []
    for name in S.RAW_FEATURES:
        value = given.get(name)
        if _is_blank(value):
            record[name] = np.nan
            continue
        try:
            if name in S.NUMERIC:
                record[name], note = parse_number(name, value)
                if note:
                    notes.append(note)
            elif name in S.BINARY:
                flag = S.parse_yes_no(value)
                if flag is None:
                    raise ValueError(f"{S.LABELS[name]} must be yes or no (you entered '{str(value).strip()[:20]}').")
                record[name] = flag
            else:
                record[name] = parse_category(name, value)
        except ValueError as exc:
            errors[name] = str(exc)
    if not errors and not _is_blank(given.get("credit_accounts")) and not _is_blank(given.get("late_payments")) \
            and record["credit_accounts"] == 0 and record["late_payments"] > 0:
        errors["late_payments"] = "Late payments cannot be recorded for someone with no credit accounts."
    if errors:
        raise ValidationError("; ".join(errors.values()), errors)
    assumed = [n for n in S.RAW_FEATURES if n not in S.REQUIRED_INPUT and _is_blank(record[n])]
    return record, notes, assumed


def validate_port(value):
    try:
        port = int(str(value).strip())
    except (TypeError, ValueError):
        raise ValidationError("Port must be a whole number.") from None
    if not 1024 <= port <= 65535:
        raise ValidationError("Port must be between 1024 and 65535.")
    return port


def validate_int(value, field, low, high):
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        raise ValidationError(f"{field} must be a whole number.") from None
    if not low <= number <= high:
        raise ValidationError(f"{field} must be between {low} and {high}.")
    return number


def validate_number(value, field, low, high):
    try:
        number = float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        raise ValidationError(f"{field} must be a number.") from None
    if not math.isfinite(number) or not low <= number <= high:
        raise ValidationError(f"{field} must be between {low:g} and {high:g}.")
    return number
