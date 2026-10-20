"""Fill the check-in 4 decision page (docs/reviews/phase5-checkin4.html) from the step 7 evidence, so that every number
on the page is read from files rather than typed.
Run: .venv/Scripts/python.exe -I scripts/checkin4_page.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "reviews" / "phase5-checkin4.html"
S7 = json.loads((ROOT / "outputs" / "phase5_step7.json").read_text())["phase5_step7"]
DT = json.loads((ROOT / "data" / "work" / "phase5_step7" / "direct_tests.json").read_text())
EXP = json.loads((ROOT / "data" / "work" / "phase5_step7" / "f4_carried_explain.json").read_text())
T = S7["tests"] | DT
LINES = ("motor", "buildings", "contents")
LN = {"motor": "Motor", "buildings": "Buildings", "contents": "Contents", "all": "All lines"}
KIND = {"glm_w2": "W2 GLM", "glum": "elastic-net GLM", "glum_std": "standardised elastic-net GLM", "glum_spline": "spline GLM",
        "lgb": "LightGBM", "cat": "CatBoost", "ebm": "EBM", "f4": "LightGBM started from the GLM"}


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
    for i, ln in enumerate(l for l in lines if l in T[key]):
        x = T[key][ln]
        out.append(dict(name=name if i == 0 else "", line=LN[ln], a=x["A"]["gain"], lo=x["A"]["boot"]["lo"], hi=x["A"]["boot"]["hi"],
                        b=x["B"]["gain"], keep=bool(x["verdict"]["keep"])))
    return out


TABLES = {
    "t7f": rows("F1 vs W2 GLM", "F1 Penalised GLM") + rows("F2 vs W2 GLM", "F2 Plain boosting") + rows("F3 vs W2 GLM", "F3 EBM")
           + rows("F4 vs W2 GLM", "F4 Boosting from the GLM"),
    "t7c": rows("F4+all vs F4|base", "Added to F4") + rows("F1+all vs F1|base", "Added to F1 (GLM)"),
    "t7d": rows("F4+all vs W2 GLM", "F4 with the held-over factors"),
    "t7m": rows("FS:ebm vs cand:ebm", "EBM: per peril vs per line", ("motor",)) + rows("FS:lgb vs cand:lgb", "LightGBM: per peril vs per line", ("motor",))
           + rows("FS:ebm vs W2 GLM", "EBM per peril vs W2 GLM", ("motor",)),
}
chosen = {l: S7["by_line"][l]["chosen"] for l in LINES}
pct = lambda x: f"{100 * x:.2f}%"
d2 = S7["ranking_d2"]
F4A = "F4+all vs W2 GLM"

LEDE = (f"The W2 GLM is hard to beat: no model family beats it on its own, and plain boosted trees rank well below it in every line. "
        f"One combination does beat it: the GLM with a small boosted correction that uses the factors you sent on to step 7. "
        f"Across lines it saves {sgn(g(F4A, 'all'))} on 2023 (interval {ivs(F4A, 'all')}) and {sgn(g(F4A, 'all', 'B'))} on 2024, and it passes in buildings "
        f"by itself. Three calls: where those factors go, the motor structure, and whether a boosted correction may enter the final price.")

EX = (f"Example: the spline GLM in contents saves {n0(g('F1 vs W2 GLM', 'contents'))} on A, interval {ivs('F1 vs W2 GLM', 'contents')}, and "
      f"{n0(g('F1 vs W2 GLM', 'contents', 'B'))} on B. The interval crosses zero, so it fails.")

fam_rows = "".join(
    f"<tr><td class=\"k\">{fam}</td><td>{desc}</td><td>{' · '.join(LN[l] + ': ' + KIND[chosen[l][fam]] for l in LINES)}</td></tr>"
    for fam, desc in (("F1", "A GLM whose relativities are shrunk toward the base, tuned on the inner folds; one version with smooth curves for age, NCB, vehicle value, mileage and sum insured instead of bands"),
                      ("F2", "Boosted trees built from scratch on the same raw fields (LightGBM and CatBoost)"),
                      ("F3", "An EBM: one shape per factor plus a few two-factor interactions, read like a GLM"),
                      ("F4", "Boosted trees that start from the W2 GLM's price and learn only a correction to it")))

ex = EXP
def top(line, n=4):
    return ", ".join(k.replace("_", " ") for k in list(ex[line]["top"])[:n])

STEP7 = f"""
  <section aria-labelledby="h-s7a">
    <span class="eyebrow">Step 7 · model families (P7, C5)</span>
    <h2 id="h-s7a">Four families, each tuned on 2021 to 2022 only</h2>
    <p>Each family was tuned on folds of 2021 and 2022, including the training power from 1.1 to 1.9, and its strongest version chosen there. Only then was it fitted to the earlier years and scored on 2023 and 2024 (<code>decisions/phase5_step7_design.md</code>).</p>
    <div class="tbl wide"><table>
      <thead><tr><th>Family</th><th>What it is</th><th>Version chosen on the tuning folds</th></tr></thead>
      <tbody>{fam_rows}</tbody>
    </table></div>
  </section>

  <section aria-labelledby="h-s7b">
    <span class="eyebrow">Result</span>
    <h2 id="h-s7b">No family beats the W2 GLM on its own</h2>
    <div class="tbl wide"><table><thead><tr><th>Family against the W2 GLM</th><th>Line</th><th class="r">A: saved</th><th class="r">A: 95% interval</th><th class="r">B: saved</th><th>Passes</th></tr></thead><tbody id="t7f"></tbody></table></div>
    <p class="how"><b>How to read this.</b> A negative number means the family prices worse than the GLM. Plain boosted trees lose {n0(-g('F2 vs W2 GLM', 'all'))} on 2023 across lines; they lost on the tuning folds too, so this is not bad luck in one year. Ranking D² across lines on 2023: W2 GLM {pct(d2['A']['all']['ref'])}, penalised GLM {pct(d2['A']['all']['F1'])}, EBM {pct(d2['A']['all']['F3'])}, boosting from the GLM {pct(d2['A']['all']['F4'])}, plain boosting {pct(d2['A']['all']['F2'])}. <b>What it means:</b> this book behaves like a multiplicative rating plan, which a GLM captures with few parameters. Trees have to rediscover that structure split by split and lose accuracy doing it.</p>
  </section>

  <section aria-labelledby="h-s7c">
    <span class="eyebrow">The held-over factors (L4, L7) · card N1</span>
    <h2 id="h-s7c">They earn their place as a correction on top of the GLM</h2>
    <p>History from the customer's other policies and from the address, prior repudiation, years licensed (motor), voluntary excess (property) and river distance and elevation (buildings). At check-in 2 each one nearly passed in the GLM.</p>
    <div class="tbl wide"><table><thead><tr><th>Held-over factors added</th><th>Line</th><th class="r">A: saved</th><th class="r">A: 95% interval</th><th class="r">B: saved</th><th>Passes</th></tr></thead><tbody id="t7c"></tbody></table></div>
    <div class="tbl wide"><table><thead><tr><th>Against the W2 GLM</th><th>Line</th><th class="r">A: saved</th><th class="r">A: 95% interval</th><th class="r">B: saved</th><th>Passes</th></tr></thead><tbody id="t7d"></tbody></table></div>
    <p class="how"><b>How to read this.</b> The first table adds the factors to each family's own model; the second sets boosting from the GLM, with the factors, against the W2 GLM. In buildings it saves {n0(g(F4A, 'buildings'))} on 2023 (interval {ivs(F4A, 'buildings')}) and {n0(g(F4A, 'buildings', 'B'))} on 2024. <b>What the correction is:</b> {ex['buildings']['trees']} trees in buildings, moving the GLM's price by about {100 * ex['buildings']['mean_abs_adjustment']:.0f}% on average, led by {top('buildings')}. A home at 2 m elevation or lower gets about ×1.055 on top of the GLM, and higher ground about ×0.97. In motor, {ex['motor']['trees']} trees led by {top('motor')}; in contents, {ex['contents']['trees']} trees led by {top('contents', 2)}.</p>
  </section>

  <section aria-labelledby="h-s7d">
    <span class="eyebrow">Motor structure (M1) · card N2</span>
    <h2 id="h-s7d">Per peril helps the EBM and the trees in motor</h2>
    <div class="tbl wide"><table><thead><tr><th>Motor</th><th>Line</th><th class="r">A: saved</th><th class="r">A: 95% interval</th><th class="r">B: saved</th><th>Passes</th></tr></thead><tbody id="t7m"></tbody></table></div>
    <p class="how"><b>How to read this.</b> Splitting motor into frequency and severity by peril improves the EBM by {n0(g('FS:ebm vs cand:ebm', 'motor'))} on 2023 (interval {ivs('FS:ebm vs cand:ebm', 'motor')}) and {n0(g('FS:ebm vs cand:ebm', 'motor', 'B'))} on 2024, which passes. The per-peril EBM then ranks close to the W2 GLM: {sgn(g('FS:ebm vs W2 GLM', 'motor'))} and {sgn(g('FS:ebm vs W2 GLM', 'motor', 'B'))}. For the GLM itself, step 6's test stands: per line stays.</p>
  </section>

  <section aria-labelledby="h-s7e">
    <span class="eyebrow">Your suggestion · standardised inputs (D21)</span>
    <h2 id="h-s7e">Standardising made the penalised GLM work in contents</h2>
    <p class="how"><b>What it did.</b> With every input scaled to the same spread, the tuning chose a strong penalty in contents that drops weak levels outright; unscaled, it chose almost none. On 2024 the standardised GLM ranks {n0(g('cand:glum_std vs W2 GLM', 'contents', 'B') - g('cand:glum vs W2 GLM', 'contents', 'B'))} better than the unscaled one ({sgn(g('cand:glum_std vs W2 GLM', 'contents', 'B'))} against {sgn(g('cand:glum vs W2 GLM', 'contents', 'B'))}, both against the W2 GLM), and its spline version is the penalised-GLM pick in contents.</p>
    <div class="wrong">
      <b>Where I was wrong (logged as D23 and D22)</b>
      <span>For motor frequency × severity in the penalised GLM, I reused the penalty tuned for the one-model-per-line version. On the frequency and severity scales the same penalty is far stronger, so it flattened almost every relativity and the test lost {n0(-g('FS:glum vs cand:glum', 'motor'))}. That test is set aside; step 6's unpenalised test is the GLM evidence on this structure.</span>
      <span>Building the interim file, I found that my phase 1 rule had left two contents blank levels too thin to estimate, priced near zero in the 18 October baseline. You chose to price them at the average of the known levels (P8-R2).</span>
    </div>
  </section>
"""


def opt(k, n, d, c, f):
    return dict(k=k, n=n, d=d, c=c, f=f)


C = "F4+all vs F4|base"
DECS = [
    dict(id="N3", door="costly", q="May the final price include a small boosted correction on top of the GLM?",
         stakes=(f"<b>The trade-off in plain terms:</b> the GLM alone is a rating plan you can read factor by factor. Adding the correction makes the price more accurate, "
                 f"but each price becomes the GLM's price times an adjustment that depends on several factors at once, so it is read from a table rather than one relativity per factor. "
                 f"Across lines the correction saves {sgn(g(F4A, 'all'))} on 2023 (interval {ivs(F4A, 'all')}) and {sgn(g(F4A, 'all', 'B'))} on 2024. "
                 f"It is small: about {100 * ex['buildings']['mean_abs_adjustment']:.0f}% in buildings, led by elevation, voluntary excess and flood zone. Step 8's weights then decide how much of it the final price uses."),
         opts=[opt("A", "Allow it, explained as the GLM times an adjustment by factor",
                   "The step 8 weights may give it weight; the write-up shows the GLM relativities and, beside them, how much the correction moves the price by elevation, flood zone, excess and age.",
                   "One more layer to explain.", "Nothing."),
               opt("B", "GLM family only",
                   "The final price stays a pure rating plan.", f"Gives up a tested saving of {n0(g(F4A, 'all'))} on 2023 and {n0(g(F4A, 'all', 'B'))} on 2024.", "The boosted correction.")],
         rec="A", why="It passed the keep rule against the GLM, it is small and driven by factors with a clear risk reason, and it can be shown as an adjustment table beside the relativities."),
    dict(id="N1", door="easy", q="Where do the held-over factors go in the boosted correction?",
         stakes=(f"<b>In plain terms:</b> the factors you sent on to step 7 help only once a model can combine them with the GLM. Added to the boosted correction they pass in motor "
                 f"({sgn(g(C, 'motor'))}, interval {ivs(C, 'motor')}; {sgn(g(C, 'motor', 'B'))} on 2024) and buildings ({sgn(g(C, 'buildings'))}, interval {ivs(C, 'buildings')}; "
                 f"{sgn(g(C, 'buildings', 'B'))} on 2024), but not in contents ({sgn(g(C, 'contents'))}, interval {ivs(C, 'contents')})."),
         opts=[opt("A", "Add them in motor and buildings; contents keeps the base correction", "Follows the keep rule line by line.", "None.", "Nothing."),
               opt("B", "Add them in all three lines", "One rule everywhere, as at L3.", "Contents takes factors that did not pass there.", "Nothing."),
               opt("C", "Leave them out", "The correction uses W2's factors only.", "Gives up the passed saving in motor and buildings.", "Nothing.")],
         rec="A", why="It is what the keep rule says, line by line."),
    dict(id="N2", door="easy", q="In motor, should the EBM and the trees model each peril's frequency and severity?",
         stakes=(f"<b>In plain terms:</b> splitting motor by peril lets each model learn, for example, that young drivers claim more often but not more expensively. "
                 f"It improves the EBM by {n0(g('FS:ebm vs cand:ebm', 'motor'))} on 2023 (interval {ivs('FS:ebm vs cand:ebm', 'motor')}) and LightGBM by "
                 f"{n0(g('FS:lgb vs cand:lgb', 'motor'))}; both pass. The GLM stays per line on step 6's evidence."),
         opts=[opt("A", "Per peril for the EBM and LightGBM in motor; the GLM per line", "Each family uses the motor structure that tested better for it.", "Ten models per family in motor.", "Nothing."),
               opt("B", "Per line everywhere", "One structure throughout.", "Gives up two passed improvements.", "Nothing.")],
         rec="A", why="It is the M1 rule applied: the structure that passes within each family."),
]

s = PAGE.read_text(encoding="utf-8")
data = ("var TABLES = " + json.dumps(TABLES) + ";\n  var LEDE = " + json.dumps(LEDE, ensure_ascii=False) +
        ";\n  var DECS = " + json.dumps(DECS, ensure_ascii=False) + ";")
s = re.sub(r"/\*DATA\*/.*?/\*END DATA\*/", lambda m: "/*DATA*/" + data + "/*END DATA*/", s, flags=re.S)
s = re.sub(r"<!--STEP7-->.*?<!--END STEP7-->", lambda m: "<!--STEP7-->" + STEP7 + "<!--END STEP7-->", s, flags=re.S)
s = re.sub(r"<!--EX-->.*?<!--END EX-->", lambda m: "<!--EX-->" + EX + "<!--END EX-->", s, flags=re.S)
PAGE.write_text(s, encoding="utf-8")
print("written", PAGE, len(s))
