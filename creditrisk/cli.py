"""Interactive menu-driven command-line interface.

All CreditError subclasses are caught here and shown as friendly messages; unexpected errors are
logged and the menu keeps running.
"""

import logging
from pathlib import Path

from . import datagen, persistence, report, schema as S, training
from . import preprocessing as P
from . import validators as v
from .exceptions import CreditError, ValidationError
from .predictor import require_predictor
from .web import serve

log = logging.getLogger("creditrisk")

MENU = """
============== CREDIT RISK PREDICTION ======================
 PREDICT                            MODEL
  1. Assess one loan application     5. Train a new model
  2. What-if: change one detail      6. Model report (accuracy, money)
  3. Score applications from a CSV   7. Compare models
  4. Start the web interface         8. Save an HTML report
 DATA
  9. Create a sample dataset (CSV)  10. Check a dataset's quality
  0. Exit
  (a learning demo on synthetic data - not for real lending)
============================================================"""

# (feature, prompt, hint) in the order they are asked; the first five are required
QUESTIONS = [
    ("annual_income", "Annual income", "lakh rupees, e.g. 8.5 or '8.5 lakh'"),
    ("loan_amount", "Loan amount", "lakh rupees, e.g. 4"),
    ("loan_term_months", "Loan term", "months, or e.g. '3 years'"),
    ("credit_score", "Credit score", "300-900"),
    ("employment_type", "Employment type", ", ".join(S.EMPLOYMENT)),
    ("loan_purpose", "Loan purpose", ", ".join(S.PURPOSES) + ", Enter = skip"),
    ("interest_rate", "Interest rate offered", "% a year, Enter = skip"),
    ("employment_years", "Years in current work", "Enter = skip"),
    ("monthly_debt_payments", "Existing monthly debt payments", "thousand rupees, Enter = skip"),
    ("credit_accounts", "Credit accounts opened", "Enter = skip"),
    ("late_payments", "Payments over 30 days late in 2 years", "Enter = skip"),
    ("credit_utilisation", "Credit-card utilisation", "% of limit, Enter = skip"),
    ("owns_home", "Owns their home? yes/no", "Enter = skip"),
    ("has_cosigner", "Has a co-signer? yes/no", "Enter = skip"),
]
SYNTHETIC_ROWS = 3000
PLAUSIBLE = {"annual_income": 8, "loan_amount": 4, "loan_term_months": 48, "credit_score": 700, "employment_type": "Salaried"}


def ask(prompt):
    return input(f"  {prompt}: ").strip()


def confirm(prompt, default=False):
    answer = ask(f"{prompt} [{'Y/n' if default else 'y/N'}]").lower()
    return default if not answer else answer in {"y", "yes"}


class CreditCLI:
    def __init__(self, model_dir="models", reports_dir="reports"):
        self.model_dir = Path(model_dir)
        self.reports_dir = Path(reports_dir)
        self._predictor = None
        self.actions = {"1": self.predict, "2": self.what_if, "3": self.predict_csv, "4": self.web,
                        "5": self.train, "6": self.model_report, "7": self.compare, "8": self.save_report,
                        "9": self.make_dataset, "10": self.check_quality}

    # ---------------------------------------------------------- plumbing
    def predictor(self):
        if self._predictor is None:
            self._predictor = require_predictor(self.model_dir)
            for warning in self._predictor.warnings:
                print(f"  ! {warning}")
        return self._predictor

    def run(self):
        print(f"\n  {S.DISCLAIMER}")
        if not persistence.is_trained(self.model_dir):
            print("\n  No trained model yet.")
            if confirm("Train one now on the built-in sample data (about half a minute)?", default=True):
                self._run_action(self.train_default)
        while True:
            print(MENU)
            choice = ask("Choose an option")
            if choice == "0":
                print("  Goodbye!")
                return
            action = self.actions.get(choice)
            if action is None:
                print("  ! Invalid option, please enter a number from the menu.")
                continue
            self._run_action(action)

    def _run_action(self, action):
        try:
            action()
        except CreditError as exc:
            log.warning("%s failed: %s", action.__name__, exc)
            print(f"  ! Error: {exc}")
        except (KeyboardInterrupt, EOFError):
            print("\n  Cancelled.")
        except Exception:  # safety net; the full traceback goes to the log file
            log.exception("Unexpected error in %s", action.__name__)
            print("  ! An unexpected error occurred. See creditrisk.log for details.")

    # ---------------------------------------------------------- prediction
    def ask_application(self):
        """Ask for each detail, re-asking (up to 3 times) when a value is invalid. Enter skips an optional one."""
        print("  Describe the application (Enter skips optional details; typical values are then assumed):")
        raw = {}
        context = dict(PLAUSIBLE)       # each answer is checked together with the earlier ones
        for name, prompt, hint in QUESTIONS:
            for attempt in range(3):
                value = ask(f"{prompt}  ({hint})")
                if value == "" and name not in S.REQUIRED_INPUT:
                    break
                try:
                    if value == "":
                        raise ValidationError(f"{S.LABELS[name]} is required.", {name: f"{S.LABELS[name]} is required."})
                    v.parse_features({**context, **raw, name: value})
                except ValidationError as exc:
                    print(f"    ! {exc.errors.get(name) or exc}")
                    if attempt == 2:
                        raise ValidationError("Too many invalid answers; assessment cancelled.") from None
                else:
                    raw[name] = context[name] = value
                    break
        return raw

    def show_prediction(self, p):
        print(f"\n  Decision:        {p.decision.upper()}   (default chance {p.percent:.1f}%, typical applicant {100 * p.typical_probability:.1f}%)")
        print(f"  {p.message}")
        print(f"  Expected result: {p.expected_profit_share * 100:+.1f}% of the loan amount · decline from {p.threshold * 100:.1f}% default chance")
        print(f"  Instalment about {S.format_lakh(p.emi / 100)} a month; all repayments take {p.burden * 100:.0f}% of income")
        for note in p.notes:
            print(f"  · {note}")
        for warning in p.warnings:
            print(f"  ! {warning}")
        if p.assumed:
            print("  Typical values were assumed for: " + ", ".join(S.LABELS[a].lower() for a in p.assumed))
        if p.reasons and p.decision != "Approve":
            print("\n  Main reasons the risk is high:")
            for f in p.reasons:
                print(f"   +{f['effect'] * 100:3.0f} points  {f['label']}: {f['value']}")
        if p.what_ifs:
            print("\n  What if? (patterns in the data, not promises):")
            for w in p.what_ifs:
                print(f"   {w['change'] * 100:+4.0f} points  {w['scenario']}  ({p.percent:.0f}% → {w['probability'] * 100:.0f}%)")
        if p.factors:
            print("\n  What drives this estimate (compared with a typical applicant):")
            top = max(abs(f["effect"]) for f in p.factors) or 1
            for f in p.factors:
                sign = "+" if f["effect"] >= 0 else "-"
                print(f"   {f['label']:<32} {f['value']:<14} {sign}{abs(f['effect']) * 100:4.0f} pts  {report.bar(abs(f['effect']), top, 12)}")
        print(f"\n  {p.disclaimer}")

    def predict(self):
        predictor = self.predictor()
        self.show_prediction(predictor.predict(self.ask_application()))

    def what_if(self):
        predictor = self.predictor()
        raw = self.ask_application()
        for i, name in enumerate(S.RAW_FEATURES, 1):
            print(f"   {i:>2}. {S.LABELS[name]}")
        choice = ask("Which detail to vary? (number or name)")
        name = S.RAW_FEATURES[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(S.RAW_FEATURES) else choice
        key = v.normalise_key(name)
        suggestions = predictor.suggestions(key) if key in S.RAW_FEATURES else []
        typed = ask(f"Values to try, comma-separated (Enter = {', '.join(map(str, suggestions)) or '?'})")
        values = [x.strip() for x in typed.split(",") if x.strip()] or suggestions
        results = predictor.what_if(raw, name, values)
        print(f"\n  Default chance of the same application with a different {S.LABELS[key].lower()}:")
        for value, p in results:
            print(f"   {str(value):<22} {p * 100:5.1f}%  {S.decision(p, predictor.threshold):<14} {report.bar(p, 1, 24)}")

    def predict_csv(self):
        predictor = self.predictor()
        print(f"  The CSV needs the columns {', '.join(S.REQUIRED_INPUT)}; other details are optional "
              f"({S.ID} is kept if present).")
        source = ask("CSV file with applications")
        default = Path("predictions") / f"{Path(source).stem}_scored.csv"
        target = ask(f"Save scores to [{default}]") or default
        result = predictor.predict_csv(source, target)
        failed = result[result["status"] != "ok"]
        decisions = result["decision"].value_counts()
        print(f"  Scored {len(result) - len(failed)} application(s); saved to {target}.")
        print(f"  Approve {decisions.get('Approve', 0)} · manual review {decisions.get('Manual review', 0)} · decline {decisions.get('Decline', 0)}.")
        if len(failed):
            print(f"  {len(failed)} row(s) could not be scored:")
            for line in failed["status"].head(15):
                print(f"    - {line}")

    def web(self):
        port = v.validate_port(ask("Port [8050]") or 8050)
        serve(self.predictor(), port)

    # ---------------------------------------------------------- training and reports
    def train_default(self):
        self._train(datagen.generate(SYNTHETIC_ROWS), "synthetic (generated)", fast=False, tune=True)

    def train(self):
        source = ask("CSV file to train on (Enter = generate the built-in sample data)")
        if source:
            df, label = P.load_dataset(source), Path(source.strip('"')).name
        else:
            rows = v.validate_int(ask(f"Applications to generate [{SYNTHETIC_ROWS}]") or SYNTHETIC_ROWS, "Rows", 500, 30000)
            df, label = datagen.generate(rows), "synthetic (generated)"
        margin = v.validate_number(ask(f"Profit on a repaid loan, as a share of the loan, 0-1 [{S.PROFIT_MARGIN:g}]") or S.PROFIT_MARGIN,
                                   "Profit margin", 0.01, 1)
        loss = v.validate_number(ask(f"Loss on a defaulted loan, as a share of the loan, 0-1 [{S.LOSS_GIVEN_DEFAULT:g}]") or S.LOSS_GIVEN_DEFAULT,
                                 "Loss given default", 0.01, 1)
        fast = confirm("Quick mode (smaller models, faster, a little less accurate)?")
        tune = False if fast else confirm("Tune the best model? (adds a few seconds)", default=True)
        self._train(df, label, fast, tune, margin=margin, loss=loss)

    def _train(self, df, label, fast, tune, **settings):
        print()
        result = training.train(df, fast=fast, tune=tune, source=label, progress=lambda m: print(f"   {m}"), **settings)
        persistence.save(result, self.model_dir)
        self._predictor = None
        print("\n  Data cleaning:")
        print("\n".join("  " + line for line in result.cleaning.lines()))
        print()
        print(report.comparison_text(result.metadata))
        print()
        print(report.model_text(result.metadata))
        print(f"\n  Model saved to {self.model_dir}/ - you can now assess applications.")

    def model_report(self):
        print(report.model_text(self.predictor().meta))

    def compare(self):
        print(report.comparison_text(self.predictor().meta))

    def save_report(self):
        default = self.reports_dir / "model_report.html"
        path = Path(ask(f"File name [{default}]") or default)
        if path.suffix.lower() not in (".html", ".htm"):
            path = path.with_suffix(".html")
        report.save_html(self.predictor().meta, path)
        print(f"  Report saved to {path} - open it in a web browser.")

    # ---------------------------------------------------------- data
    def make_dataset(self):
        rows = v.validate_int(ask(f"Number of applications [{SYNTHETIC_ROWS}]") or SYNTHETIC_ROWS, "Rows", 400, 50000)
        dirty = not ask("Include data problems for practising cleaning? [Y/n]").lower().startswith("n")
        default = Path("sample_data") / ("loans_dirty.csv" if dirty else "loans_clean.csv")
        path = Path(ask(f"File name [{default}]") or default)
        if path.exists():
            raise ValidationError(f"{path} already exists; choose another name.")
        path.parent.mkdir(parents=True, exist_ok=True)
        datagen.generate(rows, dirty=dirty).to_csv(path, index=False)
        print(f"  Saved {rows} applications to {path} (synthetic data).")

    def check_quality(self):
        df = P.load_dataset(ask("CSV file to check"))
        _, cleaning = P.clean(df)
        print("\n".join("  " + line for line in cleaning.lines()))
        if cleaning.missing_before:
            print("  Missing values before cleaning: " + ", ".join(f"{c} {n}" for c, n in cleaning.missing_before.items()))
            print("  (remaining gaps are filled during training using values learned from the training rows only)")
