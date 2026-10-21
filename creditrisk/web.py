"""Local web interface for predictions (standard library only).

    GET  /              the form
    POST /              the form submitted → result page
    POST /api/predict   JSON in, JSON out
    GET  /api/health    model information

Safety measures: binds to 127.0.0.1 only; the Host header must be a local name (blocks DNS-rebinding);
request bodies are size-limited; every value is validated by the same code as the command line;
everything shown is HTML-escaped; the page has no scripts and a strict Content-Security-Policy.
Every page carries the "not for real lending decisions" notice.
"""

import html
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from . import schema as S
from .exceptions import CreditError, ValidationError

log = logging.getLogger("creditrisk")

MAX_BODY = 20_000
DRAIN_LIMIT = 1_000_000   # a rejected body up to this size is read and discarded before the error is sent
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; "
                               "base-uri 'none'; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", "Cache-Control": "no-store",
}
FORM_DEFAULTS = {"annual_income": "8", "loan_amount": "4", "loan_term_months": "48", "credit_score": "720",
                 "employment_type": "Salaried"}
BAND_CLASS = {"High": "risk", "Medium": "watch", "Low": "ok"}

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;
--accent:#2a78d6;--border:rgba(11,11,11,.12);--err:#b3261e;--errbg:#fdecea;--warnbg:#fff4e0;--okbg:#e6f4ea;--ok:#1e6b34}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;
--muted:#898781;--line:#2c2c2a;--accent:#3987e5;--border:rgba(255,255,255,.12);--err:#ff8a80;--errbg:#3d1717;--warnbg:#3a2c12;
--okbg:#16301d;--ok:#8fd9a0}}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1060px;margin:0 auto;padding:24px 16px 48px}h1{font-size:24px;margin:0}h2{font-size:17px;margin:0 0 8px}h3{font-size:14px;margin:16px 0 6px}
.sub{color:var(--ink2);font-size:13px}.badge{font-size:12px;border:1px solid var(--border);border-radius:999px;padding:2px 8px;margin-left:8px;color:var(--ink2)}
.layout{display:grid;grid-template-columns:minmax(300px,400px) 1fr;gap:16px;margin-top:20px;align-items:start}
@media (max-width:760px){.layout{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px}
form{display:grid;grid-template-columns:1fr 1fr;gap:10px 12px}label{display:block;font-size:13px;color:var(--ink2)}
.full{grid-column:1/-1}input,select{width:100%;margin-top:3px;padding:8px;border:1px solid var(--border);border-radius:8px;background:var(--page);
color:var(--ink);font:inherit}input:focus,select:focus{outline:2px solid var(--accent);outline-offset:1px}
.err{color:var(--err);font-size:12px;margin-top:2px}.invalid input,.invalid select{border-color:var(--err)}
button{grid-column:1/-1;padding:10px;border:0;border-radius:8px;background:var(--accent);color:#fff;font:inherit;font-weight:600;cursor:pointer}
button:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
.hero{font-size:48px;font-weight:600;line-height:1.1;margin:4px 0}.hero small{font-size:16px;color:var(--ink2);font-weight:400}
.chip{display:inline-block;font-size:13px;font-weight:600;border-radius:999px;padding:3px 12px;border:1px solid var(--border)}
.chip.ok{background:var(--okbg);color:var(--ok)}.chip.watch{background:var(--warnbg)}.chip.risk{background:var(--errbg);color:var(--err)}
.meter{position:relative;height:10px;border-radius:5px;background:var(--line);margin:16px 0 4px}
.meter i{position:absolute;top:0;bottom:0;left:0;border-radius:5px;background:var(--accent);opacity:.45}
.meter u{position:absolute;top:-5px;width:2px;height:20px;background:var(--ink2);text-decoration:none}
.meterlab{position:relative;height:16px;font-size:12px;color:var(--muted)}.meterlab span{position:absolute;white-space:nowrap}
.offer{border:1px solid var(--border);border-radius:8px;padding:10px 12px;margin:12px 0;font-size:14px}
.factor{display:grid;grid-template-columns:200px 1fr 90px;gap:8px;align-items:center;font-size:13px;margin:5px 0}
.factor .track{height:10px;background:var(--line);border-radius:5px;position:relative}.factor .fill{position:absolute;top:0;bottom:0;border-radius:5px;background:var(--accent)}
.factor .fill.neg{background:var(--muted)}.factor .amt{text-align:right;font-variant-numeric:tabular-nums}
ul.advice{margin:6px 0;padding-left:20px;font-size:14px}ul.advice li{margin:4px 0}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:6px 6px;border-bottom:1px solid var(--line)}th{color:var(--ink2)}
.num{text-align:right;font-variant-numeric:tabular-nums}.note{background:var(--warnbg);border-radius:8px;padding:8px 10px;margin:8px 0;font-size:13px}
.error{background:var(--errbg);color:var(--err);border-radius:8px;padding:10px;margin-bottom:12px}
footer{margin-top:24px;color:var(--muted);font-size:12px}@media (max-width:520px){.factor{grid-template-columns:1fr 70px}.factor .track{display:none}}
"""


def e(value):
    return html.escape(str(value), quote=True)


def select(name, options, current, blank=None):
    items = ([f'<option value=""{" selected" if not current else ""}>{e(blank)}</option>'] if blank else [])
    items += [f'<option{" selected" if o == current else ""}>{e(o)}</option>' for o in options]
    return f'<select name="{name}" id="{name}">{"".join(items)}</select>'


def number_input(name, current, required=True):
    kind, low, high, _ = S.NUMERIC[name]
    step = "1" if kind == "int" else "0.01" if name in ("annual_income", "loan_amount") else "0.1"
    return (f'<input type="number" name="{name}" id="{name}" min="{low}" max="{high}" step="{step}" '
            f'value="{e(current)}" inputmode="decimal"{" required" if required else ""}>')


def field(name, control, errors, wide=False):
    label = S.LABELS[name] + (f" ({S.UNITS[name]})" if S.UNITS.get(name) else "")
    message = f'<div class="err">{e(errors[name])}</div>' if name in errors else ""
    return f'<div class="{"full " if wide else ""}{"invalid" if name in errors else ""}"><label for="{name}">{e(label)}{control}</label>{message}</div>'


def form_html(values, errors):
    v = {**FORM_DEFAULTS, **values}
    blank = "not given"
    parts = [
        field("annual_income", number_input("annual_income", v.get("annual_income", "")), errors),
        field("loan_amount", number_input("loan_amount", v.get("loan_amount", "")), errors),
        field("loan_term_months", number_input("loan_term_months", v.get("loan_term_months", "")), errors),
        field("credit_score", number_input("credit_score", v.get("credit_score", "")), errors),
        field("employment_type", select("employment_type", S.EMPLOYMENT, v.get("employment_type", "")), errors),
        field("loan_purpose", select("loan_purpose", S.PURPOSES, v.get("loan_purpose", ""), blank), errors),
        field("interest_rate", number_input("interest_rate", v.get("interest_rate", ""), False), errors),
        field("employment_years", number_input("employment_years", v.get("employment_years", ""), False), errors),
        field("monthly_debt_payments", number_input("monthly_debt_payments", v.get("monthly_debt_payments", ""), False), errors),
        field("credit_accounts", number_input("credit_accounts", v.get("credit_accounts", ""), False), errors),
        field("late_payments", number_input("late_payments", v.get("late_payments", ""), False), errors),
        field("credit_utilisation", number_input("credit_utilisation", v.get("credit_utilisation", ""), False), errors),
        field("owns_home", select("owns_home", ["no", "yes"], v.get("owns_home", ""), blank), errors),
        field("has_cosigner", select("has_cosigner", ["no", "yes"], v.get("has_cosigner", ""), blank), errors),
    ]
    return (f'<form method="post" action="/" novalidate>{"".join(parts)}'
            '<div class="sub full">Money is in lakh rupees (1 lakh = 100,000). Only the first five are required; for anything left out, '
            'a typical value is assumed and the result says so.</div>'
            '<button type="submit">Assess loan</button></form>')


def result_html(p, meta):
    pct = p.percent
    review, decline = p.threshold * 50, min(p.threshold, 1) * 100
    out = [f'<h2>Credit risk</h2><span class="chip {BAND_CLASS[p.band]}">{e(p.decision)}</span>',
           f'<div class="hero">{pct:.0f}%<small> estimated chance of default</small></div>',
           f'<div class="meter" aria-hidden="true"><i style="width:{min(pct, 100):.1f}%"></i><u style="left:{review:.1f}%"></u>'
           f'<u style="left:{decline:.1f}%"></u></div>'
           f'<div class="meterlab"><span style="left:0">0%</span><span style="left:{min(review, 70):.1f}%">review from {review:.1f}%</span>'
           f'<span style="right:0">decline from {decline:.1f}%</span></div>',
           f'<div class="offer"><b>{e(p.decision)}.</b> {e(p.message)} Expected result about {p.expected_profit_share * 100:+.1f}% of the loan amount. '
           f'A typical applicant in the data has about {100 * p.typical_probability:.0f}% risk.</div>',
           f'<div class="sub">Estimated instalment {e(S.format_lakh(p.emi / 100))} a month; all repayments would take {p.burden * 100:.0f}% of income.</div>']
    for w in p.warnings:
        out.append(f'<div class="note">⚠ {e(w)}</div>')
    for n in p.notes:
        out.append(f'<div class="note">{e(n)}</div>')
    if p.assumed:
        names = ", ".join(S.LABELS[a].lower() for a in p.assumed)
        out.append(f'<div class="note">Not given, so typical values were assumed for: {e(names)}. Adding them makes the estimate more personal.</div>')
    if p.reasons and p.decision != "Approve":
        items = "".join(f'<li>{e(f["label"])}: {e(f["value"])} (+{f["effect"] * 100:.0f} points)</li>' for f in p.reasons)
        out.append(f"<h3>Main reasons the risk is high</h3><ul class='advice'>{items}</ul>")
    if p.what_ifs:
        items = "".join(f'<li>{e(w["scenario"])}: {pct:.0f}% → {w["probability"] * 100:.0f}% '
                        f'(<b>{w["change"] * 100:+.0f} points</b>)</li>' for w in p.what_ifs)
        out.append(f"<h3>What if?</h3><ul class='advice'>{items}</ul>"
                   "<div class='sub'>Patterns the model learned from generated data, not promises about any real borrower.</div>")
    if p.factors:
        top = max(abs(f["effect"]) for f in p.factors) or 1
        out.append("<h3>What drives this estimate (compared with a typical applicant)</h3>")
        for f in p.factors:
            width = max(3, 100 * abs(f["effect"]) / top)
            sign = "+" if f["effect"] >= 0 else "−"
            out.append(f'<div class="factor"><span>{e(f["label"])}: {e(f["value"])}</span>'
                       f'<div class="track"><div class="fill{" neg" if f["effect"] < 0 else ""}" style="width:{width:.0f}%"></div></div>'
                       f'<span class="amt">{sign}{abs(f["effect"]) * 100:.0f} points</span></div>')
        out.append('<div class="sub">“+” raises the estimated default chance compared with the typical value, “−” lowers it.</div>')
    out.append(f'<div class="note">{e(p.disclaimer)}</div>')
    return "".join(out)


def page(meta, values=None, errors=None, prediction=None, message=None):
    errors = errors or {}
    test = meta["metrics"]["test"]
    if prediction is not None:
        right = result_html(prediction, meta)
    elif message:
        right = f'<div class="error">{e(message)}</div>'
    else:
        right = ('<h2>Describe a loan application</h2><p class="sub">Fill in the loan and the applicant&#x27;s finances on the left. The demo estimates the chance '
                 'the loan defaults, says whether to approve, review or decline it, gives the main reasons and shows how the estimate would change.</p>'
                 f'<p class="sub">Model: {e(meta["model"])} · ROC AUC {test["roc_auc"]:.3f} on unseen loans · '
                 f'decline from {meta["thresholds"]["approve_below"] * 100:.1f}% default chance</p>')
    summary = e(errors["_"]) if "_" in errors else ""
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Credit Risk Prediction</title><style>{CSS}</style></head><body><main>"
            "<h1>Credit Risk Prediction<span class='badge'>demo · synthetic data</span></h1>"
            f"<div class='note'>{e(S.DISCLAIMER)}</div>"
            f"<div class='layout'><div class='card'>{('<div class=error>' + summary + '</div>') if summary else ''}"
            f"{form_html(values or {}, errors)}</div><div class='card' aria-live='polite'>{right}</div></div>"
            "<footer>Developed by Sneha S · Trained on generated data to learn the machine-learning workflow. The model uses no personal "
            "attributes such as gender, religion or age.</footer></main></body></html>")


def make_server(predictor, host="127.0.0.1", port=8050):
    meta = predictor.meta

    class Handler(BaseHTTPRequestHandler):
        server_version = "CreditRisk/1.0"

        def log_message(self, fmt, *args):   # keep the terminal quiet; real problems go to the log file
            log.info("web: " + fmt, *args)

        # ----- helpers
        def send(self, status, body, content_type="text/html; charset=utf-8"):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            for key, value in SECURITY_HEADERS.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, status, payload):
            self.send(status, json.dumps(payload, ensure_ascii=False), "application/json; charset=utf-8")

        def host_ok(self):
            host = (self.headers.get("Host") or "").strip().lower()
            name = host[:host.index("]") + 1] if host.startswith("[") else host.rsplit(":", 1)[0]
            return name in LOCAL_HOSTS

        def read_body(self):
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                raise ValidationError("Content-Length header is required.") from None
            if length < 0 or length > MAX_BODY:
                self.discard_body(length)
                raise OverflowError
            return self.rfile.read(length)

        def discard_body(self, length):
            """Read and throw away (a bounded amount of) a rejected body before replying.

            If the server answers and closes while the client is still sending, the operating system may
            reset the connection and the client never sees the 413 message. Huge bodies are not drained
            (that would let one request tie the server up); for those the connection is simply closed.
            """
            remaining = min(max(length, 0), DRAIN_LIMIT)
            while remaining > 0:
                chunk = self.rfile.read(min(65536, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
            self.close_connection = True

        def guard(self):
            if not self.host_ok():
                self.send(403, "Forbidden: this server only answers to localhost.", "text/plain; charset=utf-8")
                return False
            return True

        # ----- routes
        def do_GET(self):
            if not self.guard():
                return
            path = urlsplit(self.path).path
            if path == "/":
                self.send(200, page(meta))
            elif path == "/api/health":
                self.send_json(200, {**predictor.health(), "features": S.RAW_FEATURES, "required": S.REQUIRED_INPUT})
            else:
                self.send(404, page(meta, message="Page not found."))

        def do_POST(self):
            if not self.guard():
                return
            path = urlsplit(self.path).path
            if path not in ("/", "/api/predict"):
                return self.send(404, page(meta, message="Page not found."))
            api = path == "/api/predict"
            try:
                body = self.read_body()
            except OverflowError:
                return self.send_json(413, {"error": "Request too large."}) if api else \
                    self.send(413, page(meta, message="That request is too large."))
            except ValidationError as exc:
                return self.send_json(411, {"error": str(exc)}) if api else self.send(411, page(meta, message=str(exc)))
            try:
                if api:
                    if "application/json" not in (self.headers.get("Content-Type") or ""):
                        return self.send_json(415, {"error": "Send JSON with Content-Type: application/json."})
                    try:
                        raw = json.loads(body.decode("utf-8"))
                    except (ValueError, UnicodeDecodeError):
                        return self.send_json(400, {"error": "The request body is not valid JSON."})
                    return self.send_json(200, predictor.predict(raw).to_dict())
                form = parse_qs(body.decode("utf-8", "replace"), keep_blank_values=True, max_num_fields=30)
                raw = {k: v[0] for k, v in form.items()}
                try:
                    prediction = predictor.predict(raw)
                except ValidationError as exc:
                    return self.send(422, page(meta, raw, exc.errors or {"_": str(exc)}))
                self.send(200, page(meta, raw, prediction=prediction))
            except ValidationError as exc:
                self.send_json(422, {"error": str(exc), "fields": exc.errors})
            except CreditError as exc:
                self.send_json(500, {"error": str(exc)}) if api else self.send(500, page(meta, message=str(exc)))
            except Exception:  # noqa: BLE001 - never leak internals to the browser
                log.exception("Unexpected error handling %s", path)
                self.send_json(500, {"error": "Something went wrong."}) if api else \
                    self.send(500, page(meta, message="Something went wrong. See the log file."))

    return ThreadingHTTPServer((host, port), Handler)


def serve(predictor, port=8050, host="127.0.0.1"):
    server = make_server(predictor, host, port)
    print(f"  Credit Risk Prediction is running at http://{host}:{server.server_address[1]}/  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Web interface stopped.")
    finally:
        server.server_close()
