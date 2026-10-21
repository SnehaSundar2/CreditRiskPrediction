"""Entry point for Credit Risk Prediction (a learning project on synthetic data; not for real lending decisions).

Usage:
    python main.py                                       # interactive menu
    python main.py --train                               # train on generated sample applications
    python main.py --train --data my_loans.csv           # train on your own CSV (see README for columns)
    python main.py --train --margin 0.10 --loss 0.65     # change the lending economics
    python main.py --predict annual_income="8 lakh" loan_amount=4 loan_term_months=48 credit_score=700 employment_type=Salaried
    python main.py --score sample_data/applications_to_score.csv   # score a CSV of applications
    python main.py --serve --port 8050                   # local web interface
    python main.py --report reports/model_report.html    # HTML report of the trained model
    python main.py --generate sample_data/loans.csv --rows 3000
"""

import argparse
import logging
import sys
from pathlib import Path

from creditrisk import datagen, persistence, report, schema as S, training
from creditrisk import preprocessing as P
from creditrisk import validators as v
from creditrisk.cli import SYNTHETIC_ROWS, CreditCLI
from creditrisk.exceptions import CreditError
from creditrisk.predictor import require_predictor
from creditrisk.web import serve


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):  # show ₹, block characters and arrows on Windows consoles
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Credit Risk Prediction (machine learning project; not for real lending decisions)")
    parser.add_argument("--model-dir", default="models", help="folder for the trained model (default models)")
    parser.add_argument("--train", action="store_true", help="train a model and save it")
    parser.add_argument("--data", metavar="CSV", help="with --train: your own dataset (default: generated sample data)")
    parser.add_argument("--rows", default=str(SYNTHETIC_ROWS), help="applications to generate (default 3000)")
    parser.add_argument("--fast", action="store_true", help="with --train: smaller models, much faster")
    parser.add_argument("--no-tune", action="store_true", help="with --train: skip hyper-parameter tuning")
    parser.add_argument("--margin", default=str(S.PROFIT_MARGIN), help="with --train: profit on a repaid loan as a share of the loan (default 0.12)")
    parser.add_argument("--loss", default=str(S.LOSS_GIVEN_DEFAULT), help="with --train: loss on a defaulted loan as a share of the loan (default 0.6)")
    parser.add_argument("--predict", nargs="+", metavar="KEY=VALUE", help="assess one application, e.g. annual_income=8 loan_amount=4 ...")
    parser.add_argument("--score", metavar="CSV", help="score every application in a CSV file")
    parser.add_argument("--output", metavar="CSV", help="with --score: where to save the scores")
    parser.add_argument("--serve", action="store_true", help="start the local web interface")
    parser.add_argument("--port", default="8050", help="web interface port (default 8050)")
    parser.add_argument("--report", metavar="HTML", help="save an HTML report of the trained model")
    parser.add_argument("--generate", metavar="CSV", help="write a synthetic dataset to a CSV file")
    parser.add_argument("--clean-data", action="store_true", help="with --generate: a dataset without data problems")
    args = parser.parse_args(argv)

    logging.basicConfig(filename="creditrisk.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cli = CreditCLI(args.model_dir)
    try:
        if args.generate:
            rows = v.validate_int(args.rows, "Rows", 400, 50000)
            path = Path(args.generate)
            if path.exists():
                print(f"Error: {path} already exists; choose another name.")
                return 1
            path.parent.mkdir(parents=True, exist_ok=True)
            datagen.generate(rows, dirty=not args.clean_data).to_csv(path, index=False)
            print(f"Saved {rows} synthetic applications to {path}.")
            return 0
        if args.train:
            margin = v.validate_number(args.margin, "Margin", 0.01, 1)
            loss = v.validate_number(args.loss, "Loss given default", 0.01, 1)
            if args.data:
                df, label = P.load_dataset(args.data), Path(args.data).name
            else:
                df, label = datagen.generate(v.validate_int(args.rows, "Rows", 500, 50000)), "synthetic (generated)"
            result = training.train(df, fast=args.fast, tune=not (args.fast or args.no_tune), source=label,
                                    progress=lambda m: print(f"  {m}"), margin=margin, loss=loss)
            persistence.save(result, args.model_dir)
            print("\n" + "\n".join(result.cleaning.lines()) + "\n")
            print(report.comparison_text(result.metadata) + "\n")
            print(report.model_text(result.metadata))
            print(f"\nModel saved to {args.model_dir}/")
            return 0
        if args.predict:
            raw = {}
            for pair in args.predict:
                if "=" not in pair:
                    print(f"Error: '{pair}' should look like key=value, e.g. credit_score=700.")
                    return 1
                key, value = pair.split("=", 1)
                raw[key] = value
            cli.show_prediction(require_predictor(args.model_dir).predict(raw))
            return 0
        if args.score:
            target = args.output or str(Path("predictions") / f"{Path(args.score).stem}_scored.csv")
            result = require_predictor(args.model_dir).predict_csv(args.score, target)
            failed = int((result["status"] != "ok").sum())
            print(f"Scored {len(result) - failed} application(s) and saved them to {target}"
                  + (f"; {failed} row(s) could not be scored." if failed else "."))
            return 0
        if args.report:
            print(f"Report saved to {report.save_html(require_predictor(args.model_dir).meta, args.report)}")
            return 0
        if args.serve:
            serve(require_predictor(args.model_dir), v.validate_port(args.port))
            return 0
        cli.run()
        return 0
    except CreditError as exc:
        print(f"Error: {exc}")
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
