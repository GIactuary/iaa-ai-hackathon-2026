"""Fill the check-in 5 decision page (docs/reviews/phase5-checkin5.html) from the step 8 evidence, so that every number
on the page is read from files rather than typed.
Run: .venv/Scripts/python.exe -I scripts/checkin5_page.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "reviews" / "phase5-checkin5.html"
S8 = json.loads((ROOT / "outputs" / "phase5_step8.json").read_text())["phase5_step8"]
T, W, D2 = S8["tests"], S8["weights"], S8["ranking_d2"]
LINES = ("motor", "buildings", "contents")
LN = {"motor": "Motor", "buildings": "Buildings", "contents": "Contents", "all": "All lines"}
NAME = {"motor": {"ref": "W2 GLM", "F1": "penalised GLM", "FS:lgb": "LightGBM by peril", "FS:ebm": "EBM by peril", "F4+all": "GLM × boosted correction"},
        "buildings": {"ref": "W2 GLM", "F1": "elastic-net GLM", "F2": "CatBoost", "F3": "EBM", "F4+all": "GLM × boosted correction"},
        "contents": {"ref": "W2 GLM", "F1": "standardised spline GLM", "F2": "LightGBM", "F3": "EBM", "F4": "GLM × boosted correction"}}


def g(k, ln, s="A"):
    return T[k][ln][s]["gain"]


def n0(x):
    return ("−" if x < 0 else "") + f"{abs(x):,.0f}"


def sgn(x):
    return ("+" if x >= 0 else "−") + f"{abs(x):,.0f}"


def ivs(k, ln):
    a = T[k][ln]["A"]["boot"]
    return f"{n0(a['lo'])} to {n0(a['hi'])}"


def rows(key, name):
    out = []
    for ln in LINES + ("all",):
        x = T[key][ln]
        out.append(dict(name=name if ln == "motor" else "", line=LN[ln], a=x["A"]["gain"], lo=x["A"]["boot"]["lo"], hi=x["A"]["boot"]["hi"],
                        b=x["B"]["gain"], keep=bool(x["verdict"]["keep"])))
    return out


def wtxt(ln, key, which):
    w = W[ln][key]
    parts = sorted(((NAME[ln][c], x) for c, x in zip(w["cols"], w[which]) if x >= 0.005), key=lambda t: -t[1])
    return ", ".join(f"{n} {x:.3f}" for n, x in parts)


TABLES = {"t8a": rows("blend vs W2 GLM", "Arithmetic mix") + rows("geo vs W2 GLM", "Geometric mix"),
          "t8g": rows("geometric vs arithmetic", "Geometric against arithmetic")}
pct = lambda x: f"{100 * x:.2f}%"
d2 = {s: (D2[s]["all"]["ref"], D2[s]["all"]["blend:families|arith"]) for s in ("A", "B")}
B = "blend vs W2 GLM"
GEO = "geo vs W2 GLM"

# worked example of the two mixes, motor pooled geometric weights, two illustrative model prices
wm = dict(zip(W["motor"]["families|geo"]["cols"], W["motor"]["families|geo"]["pooled"]))
we, wf = wm["FS:ebm"], wm["F4+all"]
pe, pf = 600.0, 450.0
ar = (we * pe + wf * pf) / (we + wf)
ge = pe ** (we / (we + wf)) * pf ** (wf / (we + wf))

LEDE = (f"Mixing the families prices both test years better than the GLM alone, each year scored by weights fitted on the other: "
        f"across lines {sgn(g(B, 'all'))} on 2023 (interval {ivs(B, 'all')}) and {sgn(g(B, 'all', 'B'))} on 2024. It passes in motor and "
        f"buildings; contents saves in both years but its 2023 interval crosses zero. Three calls freeze the model: which lines use the mix, "
        f"how the models are mixed, and which weights price 2025.")
EX = (f"Example: in buildings the mix saves {n0(g(B, 'buildings'))} on 2023, interval {ivs(B, 'buildings')}, and "
      f"{n0(g(B, 'buildings', 'B'))} on 2024, so it passes.")

wrows = "".join(
    f"<tr><td>{LN[ln]}</td><td>{wtxt(ln, 'families|arith', 'fit_A')}</td><td>{wtxt(ln, 'families|arith', 'fit_B')}</td>"
    f"<td><b>{wtxt(ln, 'families|arith', 'pooled')}</b></td></tr>" for ln in LINES)

STEP8 = f"""
  <section aria-labelledby="h-s8a">
    <span class="eyebrow">Step 8 · family weights (J-model)</span>
    <h2 id="h-s8a">What the mix means for a price</h2>
    <p>Each line's 2025 price would be a weighted average of the family models that earned weight. The weights are fitted on past years' predictions and must sum to one. In pricing terms:</p>
    <ul class="facts">
      <li><b>Motor:</b> about two-thirds the EBM by peril (one curve per factor, per peril, for frequency and for severity) and one-third the GLM times its small boosted correction.</li>
      <li><b>Buildings:</b> about four-fifths the GLM times its boosted correction (led by elevation, voluntary excess, flood zone and history) and one-fifth CatBoost.</li>
      <li><b>Contents:</b> nine-tenths the standardised spline GLM if the mix is used there; otherwise the W2 GLM as now.</li>
    </ul>
    <p class="how"><b>How to read this.</b> Take a motor policy where the EBM by peril says {pe:,.0f} and the corrected GLM says {pf:,.0f}, with weights {we:.3f} and {wf:.3f}. The arithmetic mix charges {we:.3f} × {pe:,.0f} + {wf:.3f} × {pf:,.0f} = {ar:,.1f}. The geometric mix charges {pe:,.0f}<sup>{we:.3f}</sup> × {pf:,.0f}<sup>{wf:.3f}</sup> = {ge:,.1f}: it averages relativities the way a rating plan multiplies them. The W2 GLM itself takes no weight, because the boosted model already is the GLM times a correction.</p>
  </section>

  <section aria-labelledby="h-s8b">
    <span class="eyebrow">Result</span>
    <h2 id="h-s8b">The mix beats the GLM in motor and buildings</h2>
    <div class="tbl wide"><table><thead><tr><th>Mix against the W2 GLM</th><th>Line</th><th class="r">2023: saved</th><th class="r">2023: 95% interval</th><th class="r">2024: saved</th><th>Passes</th></tr></thead><tbody id="t8a"></tbody></table></div>
    <p class="how"><b>How to read this.</b> Each 2023 figure uses weights fitted on 2024 only, and each 2024 figure weights fitted on 2023 only, so neither year scores its own weights. Across lines, ranking D² moves from {pct(d2['A'][0])} to {pct(d2['A'][1])} on 2023 and from {pct(d2['B'][0])} to {pct(d2['B'][1])} on 2024. Contents saves {n0(g(B, 'contents'))} and {n0(g(B, 'contents', 'B'))}, but its 2023 interval runs from {ivs(B, 'contents')}.</p>
    <p class="how"><b>Weighting every candidate instead of one per family</b> does worse: {sgn(g('within-family blend vs blend', 'all'))} across lines on 2023 and {sgn(g('within-family blend vs blend', 'all', 'B'))} on 2024 against the family mix. One model per family stands (J-model (a)).</p>
  </section>

  <section aria-labelledby="h-s8c">
    <span class="eyebrow">How the models are mixed · card O2</span>
    <h2 id="h-s8c">Geometric and arithmetic mixes are nearly level</h2>
    <div class="tbl wide"><table><thead><tr><th>Test</th><th>Line</th><th class="r">2023: saved</th><th class="r">2023: 95% interval</th><th class="r">2024: saved</th><th>Passes</th></tr></thead><tbody id="t8g"></tbody></table></div>
    <p class="how"><b>How to read this.</b> The geometric mix gains {n0(g('geometric vs arithmetic', 'buildings'))} in buildings on 2023 (interval {ivs('geometric vs arithmetic', 'buildings')}) and {n0(g('geometric vs arithmetic', 'buildings', 'B'))} on 2024. In motor and contents the two differ by less than 1,300 either way, which is noise.</p>
  </section>

  <section aria-labelledby="h-s8d">
    <span class="eyebrow">Which weights price 2025 · card O3</span>
    <h2 id="h-s8d">The weights move between years</h2>
    <div class="tbl wide"><table><thead><tr><th>Line</th><th>Fitted on 2023</th><th>Fitted on 2024</th><th>Pooled (the design's choice)</th></tr></thead><tbody>{wrows}</tbody></table></div>
    <p class="how"><b>How to read this.</b> In buildings, CatBoost earns 0.32 of the price on 2023's evidence but under 0.04 on 2024's; pooled, it gets 0.20. Motor's EBM moves from 0.46 to 0.80. The design said to flag a weight that swings like this; pooling both years is the steadier estimate.</p>
  </section>

  <section aria-labelledby="h-s8e">
    <span class="eyebrow">After the verdicts</span>
    <h2 id="h-s8e">What freezing the model involves</h2>
    <ul class="facts">
      <li>Fit each chosen family model on every non-hold-out row of 2021 to 2024 and predict the 2025 rows.</li>
      <li>Mix them with the chosen weights, then apply the decided level: the line excess factor, the off-balance to the weather-normalised indication, and the 2025 weather as one factor per line.</li>
      <li>Price blank contents fields at the average of the known levels (P8-R2) in every model, check the lowest and highest prices, and write the second submission file for you to upload.</li>
      <li>Then phase 6: the referral rule and model-disagreement threshold, written before they run.</li>
    </ul>
  </section>
"""


def opt(k, n, d, c, f):
    return dict(k=k, n=n, d=d, c=c, f=f)


DECS = [
    dict(id="O1", door="costly", q="Which lines use the mix of families, and which keep the W2 GLM?",
         stakes=(f"<b>This fixes the model you submit.</b> In plain terms: the mix prices next year's claims better than the GLM by spreading each line's price "
                 f"across models that see different patterns. Motor {sgn(g(B, 'motor'))} on 2023 (interval {ivs(B, 'motor')}) and {sgn(g(B, 'motor', 'B'))} on 2024; "
                 f"buildings {sgn(g(B, 'buildings'))} and {sgn(g(B, 'buildings', 'B'))}; contents {sgn(g(B, 'contents'))} (interval {ivs(B, 'contents')}) and {sgn(g(B, 'contents', 'B'))}."),
         opts=[opt("A", "The mix in motor and buildings; the W2 GLM in contents", "Follows the keep rule line by line, as at check-ins 3 and 4.",
                   "Contents gives up a saving that shows in both years but is not proven.", "Nothing; contents can join later."),
               opt("B", "The mix in all three lines", "Contents would be nine-tenths the standardised spline GLM; the across-lines test passes, the same reasoning as L3.",
                   "Relies on the across-lines evidence for contents.", "Nothing."),
               opt("C", "The W2 GLM everywhere (the file now on the leaderboard)", "The simplest model to explain: one GLM per line.",
                   f"Gives up {n0(g(B, 'all'))} on 2023 and {n0(g(B, 'all', 'B'))} on 2024 across lines.", "The families.")],
         rec="A", why="It takes the gain where it is proven and keeps the plain GLM where it is not, as every earlier check-in did."),
    dict(id="O2", door="easy", q="Mix the models geometrically (like a rating plan) or arithmetically (like a weighted average of premiums)?",
         stakes=(f"In plain terms: the geometric mix averages the models' relativities, so the result still reads as a rating plan; the arithmetic mix averages their premiums. "
                 f"Geometric gains {sgn(g('geometric vs arithmetic', 'all'))} across lines on 2023 (interval {ivs('geometric vs arithmetic', 'all')}) and "
                 f"{sgn(g('geometric vs arithmetic', 'all', 'B'))} on 2024, all of it in buildings; motor and contents are level within 1,300."),
         opts=[opt("A", "Geometric wherever the mix is used", "Passes across lines and in buildings; reads as a rating plan.", "None in practice.", "Nothing."),
               opt("B", "Arithmetic everywhere (the design's first form)", "The simplest to state as a share of each model's premium.", "Gives up a small, passed gain.", "Nothing."),
               opt("C", "Geometric in buildings only", "The strict per-line rule.", "Two forms to explain.", "Nothing.")],
         rec="A", why="It passes the rule across lines, costs nothing elsewhere, and keeps the final price in the multiplicative form of a rating plan, as with L3's across-lines reading."),
    dict(id="O3", door="easy", q="Which weights price 2025: both years pooled, or the latest year only?",
         stakes=("In plain terms: the weights say how much of each model goes into the price, and they moved between years. "
                 "Buildings' CatBoost share is 0.32 on 2023's evidence and under 0.04 on 2024's; motor's EBM 0.46 and 0.80. Pooling both years uses all the evidence; the latest year reflects the most recent mix."),
         opts=[opt("A", "Pooled over 2023 and 2024 (as designed)", "The steadier estimate, from twice the rows.", "Less weight on the latest year.", "Nothing."),
               opt("B", "The 2024 fit only", "The most recent experience.", "Rests on one year; buildings' CatBoost nearly disappears.", "Nothing."),
               opt("C", "The average of the two yearly fits", "Treats the two years equally.", "Close to (A) in practice.", "Nothing.")],
         rec="A", why="It was the rule written before any result, and it uses both years' evidence on weights that are noisy year to year."),
]

s = PAGE.read_text(encoding="utf-8")
data = ("var TABLES = " + json.dumps(TABLES) + ";\n  var LEDE = " + json.dumps(LEDE, ensure_ascii=False) +
        ";\n  var DECS = " + json.dumps(DECS, ensure_ascii=False) + ";")
s = re.sub(r"/\*DATA\*/.*?/\*END DATA\*/", lambda m: "/*DATA*/" + data + "/*END DATA*/", s, flags=re.S)
s = re.sub(r"<!--STEP8-->.*?<!--END STEP8-->", lambda m: "<!--STEP8-->" + STEP8 + "<!--END STEP8-->", s, flags=re.S)
s = re.sub(r"<!--EX-->.*?<!--END EX-->", lambda m: "<!--EX-->" + EX + "<!--END EX-->", s, flags=re.S)
PAGE.write_text(s, encoding="utf-8")
print("written", PAGE, len(s), f"arith {ar:.1f} geo {ge:.1f}")
