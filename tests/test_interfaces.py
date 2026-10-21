"""The web interface (over real HTTP on a random local port), the interactive menu and main.py."""

import http.client
import io
import json
import threading
import urllib.parse
from contextlib import redirect_stdout
from unittest.mock import patch

import pandas as pd

import main as entry
from creditrisk import datagen, persistence, web
from creditrisk.cli import CreditCLI
from creditrisk.predictor import require_predictor
from tests.helpers import HIGH_RISK, LOW_RISK, MINIMAL, TempDirTest, shared


class TestWeb(TempDirTest):
    @classmethod
    def setUpClass(cls):
        cls.predictor = require_predictor(shared()[1])
        cls.server = web.make_server(cls.predictor, port=0)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        data = resp.read().decode("utf-8")
        conn.close()
        return resp.status, data, {k.lower(): v for k, v in resp.getheaders()}

    def post_form(self, **values):
        return self.request("POST", "/", urllib.parse.urlencode(values), {"Content-Type": "application/x-www-form-urlencoded"})

    def post_json(self, payload, path="/api/predict"):
        return self.request("POST", path, json.dumps(payload), {"Content-Type": "application/json"})

    def test_form_page_and_security_headers(self):
        status, body, headers = self.request("GET", "/")
        self.assertEqual(status, 200)
        for part in ("Credit Risk Prediction", "Assess loan", "must not be used to make real lending decisions",
                     'name="annual_income"', 'name="credit_score"', "no personal attributes"):
            self.assertIn(part, body)
        self.assertNotIn("<script", body.lower())
        self.assertIn("default-src 'none'", headers["content-security-policy"])
        self.assertEqual(headers["x-content-type-options"], "nosniff")
        self.assertEqual(self.request("GET", "/nope")[0], 404)
        self.assertEqual(self.request("POST", "/elsewhere", b"")[0], 404)

    def test_the_form_has_no_personal_attribute_fields(self):
        _, body, _ = self.request("GET", "/")
        for banned in ('name="age"', 'name="gender"', 'name="sex"', 'name="religion"'):
            self.assertNotIn(banned, body)

    def test_form_decline_with_reasons(self):
        status, body, _ = self.post_form(**{k: str(v) for k, v in HIGH_RISK.items()})
        self.assertEqual(status, 200)
        for part in ("Decline", "estimated chance of default", "Main reasons the risk is high", "What if?", "Credit score",
                     "What drives this estimate", "must not be used"):
            self.assertIn(part, body)
        self.assertIn('value="560"', body)                                  # the form keeps what was typed

    def test_form_approval_and_minimal_form(self):
        status, body, _ = self.post_form(**{k: str(v) for k, v in LOW_RISK.items()})
        self.assertEqual(status, 200)
        self.assertIn("Approve", body)
        self.assertNotIn("Main reasons the risk is high", body)
        status, body, _ = self.post_form(**{k: str(v) for k, v in MINIMAL.items()}, interest_rate="", loan_purpose="")
        self.assertEqual(status, 200)
        self.assertIn("typical values were assumed", body.lower())

    def test_units_in_the_form_are_understood(self):
        status, body, _ = self.post_form(annual_income="850000", loan_amount="4 lakh", loan_term_months="3 years", credit_score="700",
                                         employment_type="Salaried")
        self.assertEqual(status, 200)
        self.assertIn("looks like rupees", body)
        self.assertIn("3 years = 36 months", body)

    def test_mistakes_are_explained_next_to_the_field(self):
        status, body, _ = self.post_form(annual_income="0.1", loan_amount="4", loan_term_months="48", credit_score="700", employment_type="Salaried")
        self.assertEqual(status, 422)
        self.assertIn("between 1 and 100", body)
        self.assertIn('class="invalid"', body)
        status, body, _ = self.post_form(annual_income="", loan_amount="", loan_term_months="", credit_score="", employment_type="Salaried")
        self.assertEqual(status, 422)
        self.assertIn("Loan amount is required", body)
        self.assertIn("Credit score is required", body)

    def test_text_from_the_user_is_escaped(self):
        evil = '"><script>alert(1)</script><img src=x onerror=alert(2)>'
        status, body, _ = self.post_form(annual_income=evil, employment_type=evil, loan_amount="4", loan_term_months="48", credit_score="700")
        self.assertEqual(status, 422)
        self.assertNotIn("<script>alert", body)
        self.assertNotIn("<img src=x", body)
        self.assertIn("&lt;script&gt;", body)

    def test_json_api(self):
        status, body, headers = self.post_json(HIGH_RISK)
        data = json.loads(body)
        self.assertEqual(status, 200)
        self.assertIn("application/json", headers["content-type"])
        self.assertEqual(data["decision"], "Decline")
        self.assertTrue(data["main_reasons"] and data["what_if"])
        self.assertIn("must not be used", data["disclaimer"])
        data = json.loads(self.post_json(MINIMAL)[1])
        self.assertTrue(data["assumed_typical_values_for"])
        health = json.loads(self.request("GET", "/api/health")[1])
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["required"], S_REQUIRED)

    def test_json_errors_name_the_fields(self):
        status, body, _ = self.post_json({"annual_income": 0.1, "employment_type": "x"})
        data = json.loads(body)
        self.assertEqual(status, 422)
        self.assertIn("annual_income", data["fields"])
        self.assertIn("loan_amount", data["fields"])

    def test_bad_requests_are_refused(self):
        self.assertEqual(self.request("POST", "/api/predict", "{}", {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("POST", "/api/predict", "{nope", {"Content-Type": "application/json"})[0], 400)
        self.assertEqual(self.request("POST", "/api/predict", "x" * 30_000, {"Content-Type": "application/json"})[0], 413)
        self.assertEqual(self.request("POST", "/", "x" * 30_000, {"Content-Type": "application/x-www-form-urlencoded"})[0], 413)
        self.assertEqual(self.request("GET", "/", headers={"Host": "evil.example.com"})[0], 403)
        self.assertEqual(self.request("GET", "/", headers={"Host": f"localhost:{self.port}"})[0], 200)
        self.assertEqual(self.post_json(["not", "a", "dict"])[0], 422)


S_REQUIRED = ["annual_income", "loan_amount", "loan_term_months", "credit_score", "employment_type"]


def run_cli(cli, inputs):
    out = io.StringIO()
    with patch("builtins.input", side_effect=inputs), redirect_stdout(out):
        cli.run()
    return out.getvalue()


# answers for the 14 questions of ask_application(): the five required ones, then Enter to skip the rest
BASIC = ["8 lakh", "4", "4 years", "700", "salary"] + [""] * 9
RISKY = ["4", "6", "3 years", "560", "Other", "personal", "17", "1", "9", "5", "3", "85", "no", "no"]


class TestCLI(TempDirTest):
    @classmethod
    def setUpClass(cls):
        cls.model_dir = shared()[1]

    def make(self, model_dir=None):
        return CreditCLI(model_dir or self.model_dir, self.dir / "reports")

    def test_assess_a_risky_application(self):
        out = run_cli(self.make(), ["1"] + RISKY + ["0"])
        for part in ("Decision:", "DECLINE", "Main reasons the risk is high", "What if?", "What drives this estimate", "must not be used"):
            self.assertIn(part, out)

    def test_skipping_optional_questions(self):
        out = run_cli(self.make(), ["1"] + BASIC + ["0"])
        self.assertIn("Typical values were assumed for", out)
        self.assertIn("4 years = 48 months", out)

    def test_invalid_answers_are_asked_again(self):
        out = run_cli(self.make(), ["1", "abc", "0.1", "8", "4", "4 years", "700", "pirate", "Salaried"] + [""] * 9 + ["0"])
        self.assertIn("must be a number", out)
        self.assertIn("between 1 and 100", out)
        self.assertIn("not supported", out)
        self.assertIn("Decision:", out)

    def test_three_bad_answers_cancel(self):
        out = run_cli(self.make(), ["1", "a", "b", "c", "0"])
        self.assertIn("Too many invalid answers", out)
        self.assertIn("Goodbye", out)

    def test_required_answer_cannot_be_skipped(self):
        out = run_cli(self.make(), ["1", "", "", "", "0"])
        self.assertIn("Annual income is required", out)

    def test_what_if_menu(self):
        out = run_cli(self.make(), ["2"] + BASIC + ["credit_score", "800, 600", "0"])
        self.assertIn("different credit score", out)
        self.assertRegex(out, r"\n\s+800\s+\d+\.\d%\s+Approve")
        out = run_cli(self.make(), ["2"] + BASIC + ["5", "", "0"])                # number choice (credit score), default suggestions
        self.assertIn("different credit score", out)
        out = run_cli(self.make(), ["2"] + BASIC + ["colour", "1", "0"])
        self.assertIn("Unknown detail", out)

    def test_score_a_csv(self):
        source = self.dir / "apps.csv"
        pd.DataFrame([{"annual_income": 4, "loan_amount": 6, "loan_term_months": 36, "credit_score": 560, "employment_type": "Other", "late_payments": 3, "credit_accounts": 4},
                      {"annual_income": 18, "loan_amount": 3, "loan_term_months": 48, "credit_score": 790, "employment_type": "Salaried"},
                      {"annual_income": 0.1, "loan_amount": 3, "loan_term_months": 48, "credit_score": 700, "employment_type": "Salaried"}]).to_csv(source, index=False)
        target = self.dir / "scored.csv"
        out = run_cli(self.make(), ["3", str(source), str(target), "0"])
        self.assertIn("Scored 2 application(s)", out)
        self.assertIn("1 row(s) could not be scored", out)
        self.assertIn("Approve 1", out)
        self.assertIn("decline 1", out)
        self.assertEqual(len(pd.read_csv(target)), 3)
        out = run_cli(self.make(), ["3", str(self.dir / "missing.csv"), "", "0"])
        self.assertIn("not found", out)

    def test_reports_and_menu_errors(self):
        out = run_cli(self.make(), ["6", "7", "8", "", "99", "abc", "0"])
        self.assertIn("MODEL REPORT", out)
        self.assertIn("MODEL COMPARISON", out)
        self.assertIn("Report saved to", out)
        self.assertTrue((self.dir / "reports" / "model_report.html").exists())
        self.assertEqual(out.count("Invalid option"), 2)

    def test_dataset_tools(self):
        target = self.dir / "mine.csv"
        out = run_cli(self.make(), ["9", "500", "y", str(target), "10", str(target), "9", "500", "n", str(target), "0"])
        self.assertIn("Saved 500 applications", out)
        self.assertIn("rows read", out)
        self.assertIn("already exists", out)
        self.assertTrue(target.exists())

    def test_no_model_offers_training(self):
        out = run_cli(self.make(self.dir / "nomodel"), ["n", "1", "0"])
        self.assertIn("No trained model yet", out)
        self.assertIn("Train one first", out)

    def test_training_through_the_menu(self):
        folder = self.dir / "trainme"
        data = self.dir / "train.csv"
        datagen.generate(800, seed=8).to_csv(data, index=False)
        cli = self.make(folder)
        out = run_cli(cli, ["n", "5", str(data), "", "", "y", "0"])                 # default economics, quick mode
        self.assertIn("MODEL REPORT", out)
        self.assertIn("Model saved to", out)
        self.assertTrue(persistence.is_trained(folder))


class TestMain(TempDirTest):
    def run_main(self, *args):
        out = io.StringIO()
        with redirect_stdout(out), patch("logging.basicConfig"):
            code = entry.main(list(args))
        return code, out.getvalue()

    def test_predict_from_the_command_line(self):
        model = str(shared()[1])
        code, out = self.run_main("--model-dir", model, "--predict", "annual_income=4 lakh", "loan_amount=6", "loan_term_months=3 years",
                                  "credit_score=560", "employment_type=Other", "late_payments=3", "credit_accounts=4")
        self.assertEqual(code, 0)
        self.assertIn("Decision:", out)
        code, out = self.run_main("--model-dir", model, "--predict", "credit_score=700")
        self.assertEqual(code, 1)
        self.assertIn("Error:", out)
        code, out = self.run_main("--model-dir", model, "--predict", "credit_score")
        self.assertEqual(code, 1)
        self.assertIn("key=value", out)

    def test_score_and_report(self):
        model = str(shared()[1])
        source = self.dir / "p.csv"
        pd.DataFrame([MINIMAL]).to_csv(source, index=False)
        target = self.dir / "o.csv"
        code, out = self.run_main("--model-dir", model, "--score", str(source), "--output", str(target))
        self.assertEqual(code, 0)
        self.assertIn("Scored 1 application", out)
        html = self.dir / "r.html"
        code, _ = self.run_main("--model-dir", model, "--report", str(html))
        self.assertEqual(code, 0)
        self.assertTrue(html.exists())

    def test_generate_refuses_to_overwrite(self):
        target = self.dir / "gen.csv"
        code, out = self.run_main("--generate", str(target), "--rows", "500")
        self.assertEqual(code, 0)
        self.assertEqual(len(pd.read_csv(target)), len(datagen.generate(500)))
        code, out = self.run_main("--generate", str(target), "--rows", "500")
        self.assertEqual(code, 1)
        self.assertIn("already exists", out)
        code, out = self.run_main("--generate", str(self.dir / "x.csv"), "--rows", "5")
        self.assertEqual(code, 1)

    def test_train_from_the_command_line(self):
        data = self.dir / "t.csv"
        datagen.generate(800, seed=3).to_csv(data, index=False)
        model = self.dir / "m"
        code, out = self.run_main("--train", "--data", str(data), "--fast", "--model-dir", str(model), "--margin", "0.1", "--loss", "0.5")
        self.assertEqual(code, 0)
        self.assertIn("MODEL REPORT", out)
        self.assertIn("DECLINE AT 16.7% DEFAULT CHANCE (margin 10%, loss 50%", out)       # 0.1 / (0.1 + 0.5)
        self.assertTrue(persistence.is_trained(model))
        code, out = self.run_main("--train", "--data", str(data), "--fast", "--model-dir", str(model), "--margin", "2")
        self.assertEqual(code, 1)

    def test_prediction_without_a_model(self):
        code, out = self.run_main("--model-dir", str(self.dir / "none"), "--predict", "credit_score=700")
        self.assertEqual(code, 1)
        self.assertIn("Train one first", out)
