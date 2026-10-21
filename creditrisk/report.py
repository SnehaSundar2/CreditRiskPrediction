"""Model reports: a text summary for the terminal and a self-contained HTML report with charts.

Chart conventions: one validated colour per chart, thin bars with a rounded data end, solid hairline
gridlines, text in text colours (never the series colour), a tooltip on every mark, a "show as table"
view for every chart, and light/dark themes.
"""

import html
import math
from datetime import datetime
from pathlib import Path

from . import schema as S
from .exceptions import ModelError

WIDTH = 78


def bar(value, top, width=24):
    return "█" * max(1 if value > 0 else 0, round(width * value / top)) if top else ""


CSS = """
.rep{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--axis:#c3c2b7;--series:#2a78d6;--border:rgba(11,11,11,.10);--other:#c3c2b7}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .rep{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--series:#3987e5;--border:rgba(255,255,255,.10);--other:#44443f}}
:root[data-theme="dark"] .rep{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--series:#3987e5;--border:rgba(255,255,255,.10);--other:#44443f}
*{box-sizing:border-box}html,body{margin:0}
body.rep{background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 56px}h1{font-size:24px;margin:0}h2{font-size:17px;margin:0 0 4px}
h3{font-size:15px;margin:0 0 2px}.sub{color:var(--ink2);font-size:13px}section{margin-top:28px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:16px}
.tile .label{color:var(--ink2);font-size:13px}.tile .value{font-size:26px;font-weight:600;line-height:1.2}
.tile.hero .value{font-size:48px}.tile .note{color:var(--muted);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:12px;margin-top:12px}
svg.chart{width:100%;height:auto;display:block;overflow:visible}svg.chart text{fill:var(--muted);font-size:13px;font-family:inherit}
svg.chart .label{fill:var(--ink2)}svg.chart .value{fill:var(--ink);font-weight:600}
svg.chart .gridline{stroke:var(--grid)}svg.chart .baseline{stroke:var(--axis)}svg.chart .ref{stroke:var(--axis);stroke-width:1.5;stroke-dasharray:5 4}
svg.chart .line{fill:none;stroke:var(--series);stroke-width:2.5;stroke-linejoin:round;stroke-linecap:round}
.mark{fill:var(--series);outline:none}.mark.other{fill:var(--other)}.mark:hover,.mark:focus-visible{fill-opacity:.75}
.mark:focus-visible{stroke:var(--ink);stroke-width:1.5}circle.dot{fill:var(--series);stroke:var(--surface);stroke-width:2}
ul.notes{margin:8px 0 0;padding-left:20px}ul.notes li{margin:4px 0}
details.tv{margin-top:8px}details.tv summary{cursor:pointer;color:var(--ink2);font-size:13px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid)}
th{color:var(--ink2);font-weight:600}.num{text-align:right;font-variant-numeric:tabular-nums}.scroll{overflow-x:auto}
#tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--border);border-radius:8px;
padding:6px 10px;font-size:13px;box-shadow:0 4px 16px rgba(0,0,0,.12);display:none;z-index:10;max-width:280px}#tip b{display:block;font-size:15px}
footer{margin-top:32px;color:var(--muted);font-size:12px}@media (max-width:520px){.tile.hero .value{font-size:40px}.grid{grid-template-columns:1fr}}
"""
TOOLTIP_JS = """(function(){var tip=document.getElementById('tip');
function show(el,x,y){tip.textContent='';var b=document.createElement('b');b.textContent=el.getAttribute('data-value');
var s=document.createElement('span');s.textContent=el.getAttribute('data-label');tip.appendChild(b);tip.appendChild(s);
tip.style.display='block';var w=tip.offsetWidth;tip.style.left=Math.min(x+14,window.innerWidth-w-8)+'px';tip.style.top=(y+14)+'px';}
function hide(){tip.style.display='none';}
document.querySelectorAll('[data-value]').forEach(function(el){
el.addEventListener('pointermove',function(e){show(el,e.clientX,e.clientY);});el.addEventListener('pointerleave',hide);
el.addEventListener('focus',function(){var r=el.getBoundingClientRect();show(el,r.right,r.top);});el.addEventListener('blur',hide);});})();"""


def _e(value):
    return html.escape(str(value), quote=True)


def _tick_ceiling(value, intervals=4):
    """Smallest axis maximum with a 'nice' step (1, 2, 2.5, 5, 10 × a power of ten) that covers `value`."""
    raw = max(value, 1e-9) / intervals
    magnitude = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * magnitude:
            return m * magnitude * intervals
    return 10 * magnitude * intervals


def _nice_axis(lo, hi, intervals=5):
    """(low, high, step) with round numbers that cover [lo, hi]."""
    span = (hi - lo) or 1.0
    raw = span / intervals
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if raw <= m * magnitude)
    return math.floor(lo / step) * step, math.ceil(hi / step) * step, step


def _end_rounded(x, y, w, h, r=4):
    r = min(r, h / 2, w)
    return (f"M{x:.1f},{y:.1f}H{x + w - r:.1f}Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f}V{y + h - r:.1f}"
            f"Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f}H{x:.1f}Z")


def _top_rounded(x, y, w, h, r=4):
    r = min(r, w / 2, h)
    return (f"M{x:.1f},{y + h:.1f}V{y + r:.1f}Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f}H{x + w - r:.1f}"
            f"Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f}V{y + h:.1f}Z")


def hbar_chart(points, aria, highlight=None, fmt="{:g}"):
    """points [(label, value)]; with `highlight`, that bar is the series colour and the rest recede."""
    w, row, label_w, right = 420, 30, 190, 58
    h = row * len(points) + 8
    vmax = max(v for _, v in points) or 1
    pw = w - label_w - right
    parts = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{_e(aria)}">',
             f'<line class="baseline" x1="{label_w}" x2="{label_w}" y1="0" y2="{h - 4}"/>']
    for i, (label, value) in enumerate(points):
        y = 4 + i * row
        bw = max(pw * value / vmax, 2)
        cls = "mark" + ("" if highlight in (None, label) else " other")
        shown = fmt.format(value)
        parts.append(f'<text class="label" x="{label_w - 8}" y="{y + 19}" text-anchor="end">{_e(label[:28])}</text>'
                     f'<path class="{cls}" tabindex="0" d="{_end_rounded(label_w, y + 6, bw, 18)}" data-value="{_e(shown)}" '
                     f'data-label="{_e(label)}" aria-label="{_e(label)}: {_e(shown)}"/>'
                     f'<text class="value" x="{label_w + bw + 6:.1f}" y="{y + 19}">{_e(shown)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def column_chart(points, aria, x_title="", unit="applicants", fmt="{:g}"):
    """points [(label, value)] as thin columns with round ticks."""
    w, h, left, right, top, bottom = 420, 236, 40, 6, 20, 52
    pw, ph = w - left - right, h - top - bottom
    vmax = max(5, _tick_ceiling(max((v for _, v in points), default=0), 5))
    band = pw / max(len(points), 1)
    bar_w = min(24, band - 2)
    tallest = max(range(len(points)), key=lambda i: points[i][1]) if points else -1
    parts = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{_e(aria)}">']
    for k in range(6):
        y = top + ph - ph * k / 5
        parts.append(f'<line class="{"baseline" if k == 0 else "gridline"}" x1="{left}" x2="{w - right}" y1="{y:.1f}" y2="{y:.1f}"/>'
                     f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end">{vmax * k / 5:g}</text>')
    for i, (label, value) in enumerate(points):
        x = left + band * i + (band - bar_w) / 2
        bh = ph * value / vmax if vmax else 0
        shown = fmt.format(value)
        if value:
            parts.append(f'<path class="mark" tabindex="0" d="{_top_rounded(x, top + ph - bh, bar_w, bh)}" data-value="{_e(shown)} {unit}" '
                         f'data-label="{_e(label)}" aria-label="{_e(label)}: {_e(shown)}"/>')
        parts.append(f'<text class="label" x="{x + bar_w / 2:.1f}" y="{h - bottom + 18}" text-anchor="middle">{_e(label)}</text>')
        if i == tallest and value:
            parts.append(f'<text class="value" x="{x + bar_w / 2:.1f}" y="{top + ph - bh - 5:.1f}" text-anchor="middle">{_e(shown)}</text>')
    parts.append(f'<text class="label" x="{left + pw / 2}" y="{h - 6}" text-anchor="middle">{_e(x_title)}</text></svg>')
    return "".join(parts)


def xy_chart(points, aria, x_title, y_title, x_range=(0, 1), y_range=None, x_fmt="{:.0%}", y_fmt="{:.0%}",
             marker=None, marker_label="", diagonal=False, dots=False):
    """A line (or dots) chart. `points` are (x, y) or (x, y, tooltip_label); `marker` = (x, y) is highlighted."""
    w, h, left, bottom, top, right = 420, 340, 74, 40, 10, 14
    pw, ph = w - left - right, h - top - bottom
    ys = [p[1] for p in points] + ([marker[1]] if marker else [])
    y_lo, y_hi, y_step = (y_range + (None,))[:3] if y_range else _nice_axis(min(0, min(ys)), max(ys))
    if y_step is None:
        y_step = (y_hi - y_lo) / 5
    x_lo, x_hi = x_range
    sx = lambda v: left + pw * (v - x_lo) / ((x_hi - x_lo) or 1)
    sy = lambda v: top + ph - ph * (v - y_lo) / ((y_hi - y_lo) or 1)
    parts = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{_e(aria)}">']
    ticks = int(round((y_hi - y_lo) / y_step))
    for k in range(ticks + 1):
        v = y_lo + k * y_step
        parts.append(f'<line class="{"baseline" if abs(v) < 1e-9 else "gridline"}" x1="{left}" x2="{w - right}" y1="{sy(v):.1f}" y2="{sy(v):.1f}"/>'
                     f'<text x="{left - 6}" y="{sy(v) + 4:.1f}" text-anchor="end">{_e(y_fmt.format(v))}</text>')
    for k in range(6):
        v = x_lo + (x_hi - x_lo) * k / 5
        parts.append(f'<text x="{sx(v):.1f}" y="{h - bottom + 16}" text-anchor="middle">{_e(x_fmt.format(v))}</text>')
    if y_lo < 0 < y_hi:
        parts.append(f'<line class="baseline" x1="{left}" x2="{w - right}" y1="{sy(0):.1f}" y2="{sy(0):.1f}"/>')
    if diagonal:
        parts.append(f'<line class="ref" x1="{sx(x_lo):.1f}" y1="{sy(y_lo):.1f}" x2="{sx(x_hi):.1f}" y2="{sy(y_hi):.1f}"/>')
    if not dots and len(points) > 1:
        path = " ".join(f"{'M' if i == 0 else 'L'}{sx(p[0]):.1f},{sy(p[1]):.1f}" for i, p in enumerate(points))
        parts.append(f'<path class="line" d="{path}"/>')
    if dots:
        for p in points:
            label = p[2] if len(p) > 2 else f"{x_fmt.format(p[0])}, {y_fmt.format(p[1])}"
            parts.append(f'<circle class="dot" cx="{sx(p[0]):.1f}" cy="{sy(p[1]):.1f}" r="5" tabindex="0" '
                         f'data-value="{_e(y_fmt.format(p[1]))}" data-label="{_e(label)}" aria-label="{_e(label)}"/>')
    if marker:
        parts.append(f'<circle class="dot" cx="{sx(marker[0]):.1f}" cy="{sy(marker[1]):.1f}" r="6" tabindex="0" '
                     f'data-value="Chosen threshold" data-label="{_e(marker_label)}" aria-label="{_e(marker_label)}"/>')
    parts.append(f'<text class="label" x="{left + pw / 2}" y="{h - 6}" text-anchor="middle">{_e(x_title)}</text>'
                 f'<text class="label" x="12" y="{top + ph / 2}" text-anchor="middle" transform="rotate(-90 12 {top + ph / 2})">{_e(y_title)}</text></svg>')
    return "".join(parts)


def _table(headers, rows, numeric=()):
    head = "".join(f'<th class="{"num" if i in numeric else ""}">{_e(h)}</th>' for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f'<td class="{"num" if i in numeric else ""}">{_e(c)}</td>' for i, c in enumerate(r)) + "</tr>"
                   for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _table_view(headers, rows, numeric=(1,)):
    return f'<details class="tv"><summary>Show as table</summary><div class="scroll">{_table(headers, rows, numeric)}</div></details>'


def _tile(label, value, note="", hero=False):
    return (f'<div class="card tile{" hero" if hero else ""}"><div class="label">{_e(label)}</div><div class="value">{_e(value)}</div>'
            + (f'<div class="note">{_e(note)}</div>' if note else "") + "</div>")


def lakh(amount):
    return S.format_lakh(amount)


def _policy(meta, name):
    return next(p for p in meta["policies"] if p["policy"].startswith(name))


def _verdict(meta):
    """Plain-language findings, each backed by a number from the test results."""
    test, train, cal, eco = meta["metrics"]["test"], meta["metrics"]["train"], meta["calibration"], meta["economics"]
    model, everyone = _policy(meta, "Use the model"), _policy(meta, "Approve everyone")
    lines = [f"Ranking: the model puts a borrower who defaults above one who repays {test['roc_auc'] * 100:.0f}% of the time "
             f"(ROC AUC {test['roc_auc']:.3f}; a coin flip scores 0.5)."]
    lines.append(f"Lending rule: a loan earns {eco['margin'] * 100:.0f}% of its amount when repaid and loses {eco['loss'] * 100:.0f}% when it defaults, "
                 f"so it is worth making only when the default chance is below {eco['break_even'] * 100:.1f}%. That is the decline threshold.")
    lines.append(f"On the untouched test loans the rule approves {test['approval_rate']:.0f}% of applicants; {test['bad_rate_approved']:.1f}% of the approved "
                 f"loans defaulted, against {everyone['default_rate']:.1f}% if everybody had been approved.")
    outcome = "made" if everyone["profit"] >= 0 else "lost"
    lines.append(f"Money: approving everyone would have {outcome} {lakh(abs(everyone['profit']))} on {lakh(everyone['lent'])} lent; the model's rule makes "
                 f"{lakh(model['profit'])} on {lakh(model['lent'])}. A crystal ball would make {lakh(_policy(meta, 'Perfect')['profit'])}.")
    lines.append(f"Declines: {test['declined']} applicants were declined; {test['precision']:.0f}% of them really defaulted and the rule caught "
                 f"{test['recall']:.0f}% of all defaulters. Many declined people would have repaid, which is the price of avoiding losses.")
    lines.append(f"Accuracy {test['accuracy']:.0f}% is not the headline: approving everybody scores {test['majority_accuracy']:.0f}% while losing money.")
    if cal.get("used"):
        lines.append(f"Probabilities were calibrated ({cal['method']}): the Brier score improved from {cal['brier_before']:.3f} to {cal['brier_after']:.3f}.")
    else:
        lines.append(f"The model's probabilities were already honest (Brier {cal['brier_before']:.3f}); calibration was tried but not needed. "
                     "Honest probabilities matter because the threshold is a probability.")
    gap = train["roc_auc"] - test["roc_auc"]
    lines.append(f"Overfitting check: AUC on training loans {train['roc_auc']:.3f} against {test['roc_auc']:.3f} on test loans"
                 + (" - a gap this size suggests some memorising." if gap > 0.05 else " - close, so the model generalises."))
    if meta.get("ceiling"):
        c = meta["ceiling"]
        lines.append(f"Because the data is generated, the best possible AUC is known: {c['roc_auc']:.3f} (even the true rule cannot predict who "
                     f"is unlucky). The model reaches {min(test['roc_auc'] / c['roc_auc'] * 100, 99.9):.1f}% of it.")
    for key, name in (("age", "age group"), ("employment", "employment type")):
        rows = [s for s in meta["segments"].get(key, []) if s["n"] >= 20]
        if len(rows) >= 2:
            low, high = min(rows, key=lambda s: s["approval_rate"]), max(rows, key=lambda s: s["approval_rate"])
            note = " The gap follows real differences in default rates, but it should be watched." if key == "employment" else \
                " Age is not a model input; any gap comes from the financial details that differ between age groups."
            lines.append(f"Fairness check by {name}: approval rates run from {low['approval_rate']:.0f}% ({low['group']}) to "
                         f"{high['approval_rate']:.0f}% ({high['group']}).{note}")
    lines.append("Limitation: a real lender only has data about loans it approved in the past, so a model trained on it knows little about people it "
                 "turned down.")
    lines.append(S.DISCLAIMER)
    return lines


def model_text(meta):
    test, rows, eco = meta["metrics"]["test"], meta["rows"], meta["economics"]
    out = ["=" * WIDTH, "  MODEL REPORT", f"  {meta['model']}" + (f" ({meta['model_note']})" if meta.get("model_note") else "")
           + f" · trained {meta['trained_at'].replace('T', ' ')} in {meta['training_seconds']} s · data: {meta['source']}",
           f"  {rows['raw']} rows read · {rows['clean']} after cleaning · {rows['train']} train / {rows['test']} test", "=" * WIDTH, "",
           "  HOW WELL IT RANKS BORROWERS (test set)",
           f"   ROC AUC            {test['roc_auc']:.3f}     (0.5 = coin flip, 1.0 = perfect)",
           f"   PR AUC             {test['average_precision']:.3f}     (guessing scores {test['default_rate'] / 100:.3f})",
           f"   Brier score        {test['brier']:.3f}     (lower = more honest probabilities)", "",
           f"  LENDING RULE: DECLINE AT {eco['break_even'] * 100:.1f}% DEFAULT CHANCE (margin {eco['margin'] * 100:.0f}%, loss {eco['loss'] * 100:.0f}% of the loan)",
           f"   Approved           {test['approval_rate']:.0f}%      of applicants; {test['bad_rate_approved']:.1f}% of them defaulted",
           f"   Defaulters caught  {test['recall']:.0f}%      of all who defaulted were declined",
           f"   Declines correct   {test['precision']:.0f}%      of declined applicants really defaulted",
           f"   Accuracy           {test['accuracy']:.0f}%      (approving everybody scores {test['majority_accuracy']:.0f}%)", "",
           "                          Defaulted   Repaid",
           f"   Declined             {test['tp']:>9}   {test['fp']:>7}", f"   Approved             {test['fn']:>9}   {test['tn']:>7}", "",
           "  WHAT EACH POLICY EARNS ON THE TEST LOANS (lakh rupees)"]
    for p in meta["policies"]:
        out.append(f"   {p['policy']:<32} approved {p['approved']:>4}   profit {p['profit']:>8.1f}")
    out += ["", "  RISK BANDS (test set)"]
    for b in meta["bands"]:
        out.append(f"   {b['band']:<7} {b['decision']:<14} {b['applicants']:>5} applicants ({b['share']:>3.0f}%)   {b['rate']:>5.1f}% defaulted   {bar(b['rate'], 60, 12)}")
    out += ["", "  WHAT THE MODEL RELIES ON (drop in AUC when scrambled)"]
    top = meta["importance"][:6]
    for i in top:
        out.append(f"   {i['label']:<32} {i['importance']:.3f}  {bar(i['importance'], top[0]['importance'] or 1)}")
    out += ["", "  WHAT THIS MEANS"]
    out += [f"  → {line}" for line in _verdict(meta)]
    return "\n".join(out)


def comparison_text(meta):
    rows = ["  MODEL COMPARISON (cross-validation on training data; higher AUC is better)",
            f"  {'MODEL':<36} {'ROC AUC':>8} {'PR AUC':>7} {'BRIER':>7}", f"  {'-' * 36} {'-' * 8} {'-' * 7} {'-' * 7}"]
    for c in meta["comparison"]:
        mark = "  ← chosen" if c["model"] == meta["metrics"]["cv"]["model"] else ""
        rows.append(f"  {c['model']:<36} {c['roc_auc']:>8.3f} {c['average_precision']:>7.3f} {c['brier']:>7.3f}{mark}")
    return "\n".join(rows)


def render_html(meta):
    if "roc" not in meta or "profit_curve" not in meta:
        raise ModelError("This model was saved without test results; train it again to build a report.")
    test, cv, rows, eco = meta["metrics"]["test"], meta["metrics"]["cv"], meta["rows"], meta["economics"]
    model, everyone = _policy(meta, "Use the model"), _policy(meta, "Approve everyone")
    body = [f'<header><h1>Credit Risk Prediction — model report</h1><div class="sub">{_e(meta["model"])}'
            f'{" (" + _e(meta["model_note"]) + ")" if meta.get("model_note") else ""} · trained {_e(meta["trained_at"].replace("T", " "))} · '
            f'data: {_e(meta["source"])} · {rows["raw"]} rows read, {rows["clean"]} after cleaning, {rows["train"]} train / {rows["test"]} test</div>'
            f'<div class="note">{_e(S.DISCLAIMER)}</div></header>']
    body.append('<section aria-label="Summary"><div class="tiles">' + "".join([
        _tile("ROC AUC", f"{test['roc_auc']:.3f}", "0.5 would be a coin flip", hero=True),
        _tile("Approved", f"{test['approval_rate']:.0f}%", f"of applicants; {test['bad_rate_approved']:.1f}% of those defaulted"),
        _tile("Defaulters declined", f"{test['recall']:.0f}%", f"at the {eco['break_even'] * 100:.1f}% decline threshold"),
        _tile("Declines that were right", f"{test['precision']:.0f}%", f"{test['declined']} applicants declined"),
        _tile("Profit of the rule", lakh(model["profit"]), f"approving everyone: {lakh(everyone['profit'])}"),
        _tile("Brier score", f"{test['brier']:.3f}", "honesty of probabilities, lower is better")]) + "</div></section>")
    body.append('<section class="card"><h2>What this means</h2><ul class="notes">'
                + "".join(f"<li>{_e(line)}</li>" for line in _verdict(meta)) + "</ul></section>")

    comp = [(c["model"], round(c["roc_auc"], 3)) for c in meta["comparison"]]
    comparison_rows = [[c["model"], f"{c['roc_auc']:.3f}", f"{c['average_precision']:.3f}", f"{c['brier']:.3f}"] for c in meta["comparison"]]
    roc = meta["roc"]
    cal_points = [(r["predicted"] / 100, r["observed"] / 100, f"{r['band']}: predicted {r['predicted']:.0f}%, "
                   f"observed {r['observed']:.0f}% ({r['applicants']} applicants)") for r in meta["calibration_table"]]
    bands = [(b["band"], round(b["rate"], 1)) for b in meta["bands"]]
    imp = [(i["label"], round(i["importance"], 3)) for i in meta["importance"][:8]]
    profit = meta["profit_curve"]
    chosen_t = min(eco["break_even"], profit[-1][0])
    chosen_profit = min(profit, key=lambda point: abs(point[0] - chosen_t))[1]
    cal_rows = [[r["band"], r["applicants"], f"{r['predicted']:.0f}%", f"{r['observed']:.0f}%"] for r in meta["calibration_table"]]
    band_rows = [[b["band"], b["decision"], b["applicants"], f"{b['share']:.0f}%", b["defaults"], f"{b['rate']:.1f}%"] for b in meta["bands"]]
    profit_rows = [[f"{t:.0%}", f"{v:,.1f}"] for t, v in profit[::4]]
    roc_label = f"Chosen threshold: declines {test['recall']:.0f}% of defaulters and {100 * test['fp'] / max(test['fp'] + test['tn'], 1):.0f}% of good borrowers"
    roc_chart = xy_chart(roc, "ROC curve: share of defaulters declined against share of good borrowers declined", "Good borrowers declined",
                         "Defaulters declined", marker=(test["fp"] / max(test["fp"] + test["tn"], 1), test["recall"] / 100),
                         marker_label=roc_label, diagonal=True, y_range=(0, 1, 0.2))
    cal_chart = xy_chart(cal_points, "Predicted against observed default rate", "Predicted default chance", "Observed default rate",
                         dots=True, diagonal=True, y_range=(0, 1, 0.2))
    profit_chart = xy_chart(profit, "Profit on the test loans against the decline threshold", "Decline applicants whose default chance is at least",
                            "Profit (lakh rupees)", x_range=(0, profit[-1][0]), y_fmt="{:,.0f}", marker=(chosen_t, chosen_profit),
                            marker_label=f"Break-even threshold {eco['break_even'] * 100:.1f}%")
    cards = [
        f'<div class="card"><h3>Model comparison</h3><div class="sub">Cross-validated ROC AUC — higher is better</div>'
        f'{hbar_chart(comp, "Cross-validated ROC AUC of each model", cv["model"], "{:.3f}")}'
        f'{_table_view(["Model", "ROC AUC", "PR AUC", "Brier"], comparison_rows, (1, 2, 3))}</div>',
        f'<div class="card"><h3>Catching defaulters against declining good borrowers</h3><div class="sub">The dashed line is guessing; the dot is the chosen threshold</div>'
        f'{roc_chart}{_table_view(["Good borrowers declined", "Defaulters declined"], [[f"{x:.0%}", f"{y:.0%}"] for x, y in roc[::4]], (0, 1))}</div>',
        f'<div class="card"><h3>What the threshold is worth</h3><div class="sub">Profit on the test loans for each decline threshold; the dot is the break-even rule</div>'
        f'{profit_chart}{_table_view(["Threshold", "Profit (lakh)"], profit_rows, (0, 1))}</div>',
        f'<div class="card"><h3>Are the probabilities honest?</h3><div class="sub">Predicted default chance against what really happened; dots on the dashed line are perfect</div>'
        f'{cal_chart}{_table_view(["Band", "Applicants", "Predicted", "Observed"], cal_rows, (1, 2, 3))}</div>',
        f'<div class="card"><h3>Default rate by risk band</h3><div class="sub">Share of applicants in each band who really defaulted</div>'
        f'{column_chart(bands, "Default rate by risk band", "Risk band", "% defaulted", "{:g}")}'
        f'{_table_view(["Band", "Decision", "Applicants", "Share", "Defaults", "Rate"], band_rows, (2, 3, 4, 5))}</div>',
        f'<div class="card"><h3>What the model relies on</h3><div class="sub">Drop in ROC AUC when a feature is scrambled</div>'
        f'{hbar_chart(imp, "Feature importance", None, "{:.3f}")}{_table_view(["Feature", "AUC drop"], imp)}</div>',
    ]
    body.append(f'<section><h2>Charts</h2><div class="grid">{"".join(cards)}</div></section>')

    matrix = _table(["", "Declined", "Approved"],
                    [["Defaulted", test["tp"], f"{test['fn']} (loss)"],
                     ["Repaid", f"{test['fp']} (lost business)", test["tn"]]], (1, 2))
    policy_rows = [[p["policy"], p["approved"], f"{p['default_rate']:.1f}%", lakh(p["lent"]), lakh(p["profit"])] for p in meta["policies"]]
    policies = _table(["Policy", "Approved", "Defaults among them", "Lent", "Profit"], policy_rows, (1, 2, 3, 4))
    seg_head = ["Applicants", "Really defaulted", "Predicted", "Approved", "Defaults among approved"]

    def seg(key):
        return [[s["group"], s["n"], f"{s['actual']:.1f}%", f"{s['predicted']:.1f}%", f"{s['approval_rate']:.0f}%",
                 "n/a" if s["bad_rate_approved"] is None else f"{s['bad_rate_approved']:.1f}%"] for s in meta["segments"][key]]
    fairness = [f'<div class="card"><h3>By employment type</h3>{_table(["Employment"] + seg_head, seg("employment"), (1, 2, 3, 4, 5))}</div>',
                f'<div class="card"><h3>By loan purpose</h3>{_table(["Purpose"] + seg_head, seg("purpose"), (1, 2, 3, 4, 5))}</div>']
    if "age" in meta["segments"]:
        fairness.append(f'<div class="card"><h3>By age group (age is not a model input)</h3>{_table(["Age"] + seg_head, seg("age"), (1, 2, 3, 4, 5))}</div>')
    body.append('<section><h2>Money and fairness checks</h2><div class="grid">'
                f'<div class="card"><h3>What each lending policy earns</h3><div class="sub">Test loans. A repaid loan earns {eco["margin"] * 100:.0f}% of its amount, '
                f'a default loses {eco["loss"] * 100:.0f}%; the break-even default chance is {eco["break_even"] * 100:.1f}%.</div>{policies}</div>'
                f'<div class="card"><h3>Confusion matrix</h3><div class="sub">Test loans, declined from {eco["break_even"] * 100:.1f}% default chance</div>{matrix}</div>'
                + "".join(fairness) + '</div></section>')

    steps = meta["cleaning"]["steps"]
    body.append('<section class="card"><h2>Data cleaning log</h2><div class="sub">'
                f'{meta["cleaning"]["rows_in"]} rows read → {meta["cleaning"]["rows_out"]} kept</div>'
                + _table(["Problem found", "Rows", "Action"], [[s["label"], s["count"], s["outcome"]] for s in steps], (1,))
                + "</section>")
    params = ", ".join(f"{k} = {v}" for k, v in meta.get("model_params", {}).items()) or "defaults"
    env = meta["environment"]
    body.append(f'<footer>Generated {datetime.now():%d %b %Y, %H:%M} · settings: {_e(params)} · seed {meta["seed"]} · '
                f'Python {_e(env["python"])}, scikit-learn {_e(env["sklearn"])} · developed by Sneha S · {_e(S.DISCLAIMER)}</footer>')
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Credit risk model report</title><style>{CSS}</style></head><body class='rep'><main>" + "\n".join(body)
            + f"</main><div id='tip' role='status' aria-live='polite'></div><script>{TOOLTIP_JS}</script></body></html>")


def save_html(meta, path):
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_html(meta), encoding="utf-8")
    except OSError as exc:
        raise ModelError(f"Could not write {path}: {exc.strerror or exc}") from exc
    return path
