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
NAMES = {"ref": "W2 GLM", "F1": "Penalised GLM", "F2": "Plain boosting", "F3": "EBM", "F4": "GLM + boosted correction", "F4+all": "GLM + boosted correction",
         "FS:lgb": "LightGBM per peril", "FS:ebm": "EBM per peril"}
SPEC = {"motor": {"F2": "FS:lgb", "F3": "FS:ebm"}}
FAM_NAME = {"motor": {"F1": "Spline GLM", "FS:lgb": "LightGBM per peril", "FS:ebm": "EBM per peril", "F4+all": "GLM + boosted correction"},
            "buildings": {"F1": "Elastic-net GLM", "F2": "CatBoost", "F3": "EBM", "F4+all": "GLM + boosted correction"},
            "contents": {"F1": "Spline GLM", "F2": "LightGBM", "F3": "EBM", "F4": "GLM + boosted correction"}}


def g(k, ln, s="A"):
    return T[k][ln][s]["gain"]


def n0(x):
    return ("−" if x < 0 else "") + f"{abs(x):,.0f}"


def sgn(x):
    return ("+" if x >= 0 else "−") + f"{abs(x):,.0f}"


def ivs(k, ln):
    a = T[k][ln]["A"]["boot"]
    return f"{n0(a['lo'])} to {n0(a['hi'])}"


def rows(key, name, lines=LINES + ("all",)):
    out = []
    for i, ln in enumerate(lines):
        x = T[key][ln]
        out.append(dict(name=name if i == 0 else "", line=LN[ln], a=x["A"]["gain"], lo=x["A"]["boot"]["lo"], hi=x["A"]["boot"]["hi"],
                        b=x["B"]["gain"], keep=bool(x["verdict"]["keep"])))
    return out


B = "blend vs W2 GLM"
TABLES = {"t8b": rows(B, "Blend against the W2 GLM"),
          "t8c": rows("within-family blend vs blend", "Every candidate weighted") + rows("geometric vs arithmetic", "Geometric mix")}
pct = lambda x: f"{100 * x:.2f}%"
d2 = lambda s, ln, c: D2[s][ln][c]

wrows = ""
for l in LINES:
    w = W[l]["families|arith"]
    for c, a, b, p in zip(w["cols"], w["fit_A"], w["fit_B"], w["pooled"]):
        if max(a, b, p) < 0.01:
            continue
        wrows += (f"<tr><td>{LN[l]}</td><td>{FAM_NAME[l].get(c, NAMES.get(c, c))}</td><td class=\"r num\">{a:.3f}</td>"
                  f"<td class=\"r num\">{b:.3f}</td><td class=\"r num\">{p:.3f}</td></tr>")

wm = W["motor"]["families|arith"]
pm = dict(zip(wm["cols"], wm["pooled"]))
wb = dict(zip(W["buildings"]["families|arith"]["cols"], W["buildings"]["families|arith"]["pooled"]))
wc = dict(zip(W["contents"]["families|arith"]["cols"], W["contents"]["families|arith"]["pooled"]))

LEDE = (f"Weighting the families beats the W2 GLM: across lines the blend saves {sgn(g(B, 'all'))} on 2023 (interval {ivs(B, 'all')}) and "
        f"{sgn(g(B, 'all', 'B'))} on 2024, scored each time with weights fitted on the other year. It passes in motor and buildings; contents is a near miss. "
        f"Ranking D² across lines moves from {pct(d2('A', 'all', 'ref'))} to {pct(d2('A', 'all', 'blend:families|arith'))} on 2023 and from "
        f"{pct(d2('B', 'all', 'ref'))} to {pct(d2('B', 'all', 'blend:families|arith'))} on 2024. Two calls: freeze the blend, and the form of the mix.")

EX = (f"Example: in buildings the blend saves {n0(g(B, 'buildings'))} on A, interval {ivs(B, 'buildings')}, and {n0(g(B, 'buildings', 'B'))} on B, so it passes.")

STEP8 = f"""
  <section aria-labelledby="h-s8a">
    <span class="eyebrow">Step 8 · family weights (J-model)</span>
    <h2 id="h-s8a">The blend beats the W2 GLM in motor and buildings</h2>
    <div class="tbl wide"><table><thead><tr><th>Test</th><th>Line</th><th class="r">A: saved</th><th class="r">A: 95% interval</th><th class="r">B: saved</th><th>Passes</th></tr></thead><tbody id="t8b"></tbody></table></div>
    <p class="how"><b>How to read this.</b> The blend is a weighted average of the families' prices in each line. In motor it saves {n0(g(B, 'motor'))} on 2023 (interval {ivs(B, 'motor')}) and {n0(g(B, 'motor', 'B'))} on 2024. Motor ranking D² moves from {pct(d2('B', 'motor', 'ref'))} to {pct(d2('B', 'motor', 'blend:families|arith'))} on 2024, buildings from {pct(d2('B', 'buildings', 'ref'))} to {pct(d2('B', 'buildings', 'blend:families|arith'))}. Across lines it also beats every single family, the closest being the GLM with the boosted correction ({sgn(g('blend vs F4', 'all'))} on 2023, interval {ivs('blend vs F4', 'all')}).</p>
  </section>

  <section aria-labelledby="h-s8b">
    <span class="eyebrow">What the 2025 price is made of</span>
    <h2 id="h-s8b">Weights by line</h2>
    <div class="tbl wide"><table><thead><tr><th>Line</th><th>Model</th><th class="r">Fitted on 2023</th><th class="r">Fitted on 2024</th><th class="r">Both years (2025)</th></tr></thead><tbody>{wrows}</tbody></table></div>
    <p class="how"><b>How to read this.</b> A 2025 motor policy is priced {100 * pm['FS:ebm']:.0f}% from the per-peril EBM and {100 * pm['F4+all']:.0f}% from the GLM with its boosted correction; a buildings policy {100 * wb['F4+all']:.0f}% from the GLM with its correction and {100 * wb['F2']:.0f}% from CatBoost; a contents policy {100 * wc['F1']:.0f}% from the spline GLM. The W2 GLM itself has weight zero because the boosted correction starts from it, so it reaches every motor and buildings price through that model. Two weights swing between the years (buildings CatBoost, motor correction); the scores above use the other year's weights and still pass, so the blend does not depend on the exact split.</p>
  </section>

  <section aria-labelledby="h-s8c">
    <span class="eyebrow">Challengers</span>
    <h2 id="h-s8c">Weighting every candidate does worse; a geometric mix is a hair better in buildings</h2>
    <div class="tbl wide"><table><thead><tr><th>Challenger against the blend</th><th>Line</th><th class="r">A: saved</th><th class="r">A: 95% interval</th><th class="r">B: saved</th><th>Passes</th></tr></thead><tbody id="t8c"></tbody></table></div>
    <p class="how"><b>How to read this.</b> Weighting every tuned candidate rather than one per family overfits the weights: {sgn(g('within-family blend vs blend', 'all'))} on 2023 and {sgn(g('within-family blend vs blend', 'all', 'B'))} on 2024 across lines. A geometric mix (averaging log prices) passes in buildings by {n0(g('geometric vs arithmetic', 'buildings'))} on 2023 and {n0(g('geometric vs arithmetic', 'buildings', 'B'))} on 2024, and makes no difference in motor or contents.</p>
  </section>
"""


def opt(k, n, d, c, f):
    return dict(k=k, n=n, d=d, c=c, f=f)


DECS = [
    dict(id="O1", door="one", q="Freeze the blend as the Part 1 model?",
         stakes=(f"<b>In plain terms:</b> this sets the price you submit. The blend prices more accurately than the W2 GLM in both test years ({sgn(g(B, 'all'))} and {sgn(g(B, 'all', 'B'))} across lines), "
                 f"at the cost of explaining each price as a weighted mix: in motor two-thirds per-peril EBM and one-third GLM with its correction; in buildings mostly the GLM with its correction; contents mostly the spline GLM. "
                 f"The freeze is due by the end of 21 October; the final file refits each family on 2021 to 2024 and applies the decided level (excess factor, off-balance to the indication, 2025 weather) and the P8-R2 blank rule to every family."),
         opts=[opt("A", "Freeze the blend", "Each family is refitted on 2021 to 2024 and mixed with the pooled weights; the level chain is applied to the blend line by line.",
                   "A layered explanation: GLM relativities, the correction table, and the weights.", "The pure GLM as the submitted price."),
               opt("B", "Freeze the W2 GLM", "The submitted price is the GLM already uploaded as w2_glm_v1u.", f"Gives up {n0(g(B, 'all'))} on 2023 and {n0(g(B, 'all', 'B'))} on 2024.", "The blend.")],
         rec="A", why="It passes the keep rule against the GLM on weights it never saw, in motor, buildings and across lines, and every part of it has been explained at check-ins 3 and 4."),
    dict(id="O2", door="easy", q="Mix the models by plain weighted average everywhere, or by weighted log average in buildings?",
         stakes=(f"<b>In plain terms:</b> a plain average reads as \"so much of each model's price\"; a log average multiplies the models' prices, each raised to its weight. "
                 f"In buildings the log average ranks a little better ({sgn(g('geometric vs arithmetic', 'buildings'))} on 2023, interval {ivs('geometric vs arithmetic', 'buildings')}; "
                 f"{sgn(g('geometric vs arithmetic', 'buildings', 'B'))} on 2024) and passes the keep rule; in motor and contents it makes no difference."),
         opts=[opt("A", "Plain weighted average everywhere", "One form, the easiest to explain.", "Gives up a small passed gain in buildings.", "Nothing."),
               opt("B", "Log average in buildings, plain elsewhere", "What the keep rule says, line by line.", "Two forms to describe.", "Nothing.")],
         rec="B", why="The rule written before the results says a challenger that passes replaces the default; the weights are almost the same, so the explanation barely changes."),
]

s = PAGE.read_text(encoding="utf-8")
data = ("var TABLES = " + json.dumps(TABLES) + ";\n  var LEDE = " + json.dumps(LEDE, ensure_ascii=False) +
        ";\n  var DECS = " + json.dumps(DECS, ensure_ascii=False) + ";")
s = re.sub(r"/\*DATA\*/.*?/\*END DATA\*/", lambda m: "/*DATA*/" + data + "/*END DATA*/", s, flags=re.S)
s = re.sub(r"<!--STEP8-->.*?<!--END STEP8-->", lambda m: "<!--STEP8-->" + STEP8 + "<!--END STEP8-->", s, flags=re.S)
s = re.sub(r"<!--EX-->.*?<!--END EX-->", lambda m: "<!--EX-->" + EX + "<!--END EX-->", s, flags=re.S)
PAGE.write_text(s, encoding="utf-8")
print("written", PAGE, len(s))
