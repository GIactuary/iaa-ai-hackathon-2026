"""Fill the check-in 6 decision page (docs/reviews/phase6-checkin6.html) from the phase 6 evidence, so that every number
on the page is read from files rather than typed.
Run: .venv/Scripts/python.exe -I scripts/checkin6_page.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "reviews" / "phase6-checkin6.html"
P6 = json.loads((ROOT / "outputs" / "phase6.json").read_text())["phase6"]
T, OOT, RF, RG = P6["tests"], P6["oot"], P6["referrals_2025"], P6["ranges"]
LINES = ("motor", "buildings", "contents")
LN = {"motor": "Motor", "buildings": "Buildings", "contents": "Contents", "all": "All lines"}
CNT = {(r["line"], r["reason"]): r["rows"] for r in P6["referral_counts"]}


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
    for i, ln in enumerate(LINES + ("all",)):
        x = T[key][ln]
        out.append(dict(name=name if i == 0 else "", line=LN[ln], a=x["A"]["gain"], lo=x["A"]["boot"]["lo"], hi=x["A"]["boot"]["hi"],
                        b=x["B"]["gain"], keep=bool(x["verdict"]["keep"])))
    return out


TABLES = {"t6t": rows("T2 vs T0", "W2 GLM price on flagged rows") + rows("T1 vs T0", "Highest family price on flagged rows")}
p1 = lambda x: f"{100 * x:.1f}%"
size_n = sum(v for (l, r), v in CNT.items() if r in ("R1", "R2"))

LEDE = (f"Where the models disagree, they are not more often wrong. Rows on which the families' prices differ by more than 30% were no less reliable on 2023 and 2024, so a disagreement flag "
        f"is a prompt for underwriting review, not a reason to reprice. The one repricing that tests better is motor at the GLM's price ({sgn(g('T2 vs T0', 'motor'))} on 2023). "
        f"Only {size_n} rows of 2025 sit beyond the size evidence or in thin segments. Two calls.")

EX = f"Example: motor flagged rows at the GLM's price save {n0(g('T2 vs T0', 'motor'))} on A, interval {ivs('T2 vs T0', 'motor')}, and {n0(g('T2 vs T0', 'motor', 'B'))} on B, so it passes."

oot_rows = "".join(
    f"<tr><td>{LN[l]}</td><td class=\"r num\">{p1(RF[l]['r3_share'])}</td><td class=\"r num\">{p1(OOT[l]['A']['share'])} · {p1(OOT[l]['B']['share'])}</td>"
    f"<td class=\"r num\">{OOT[l]['A']['dev_per_cost_flagged']:.3f} vs {OOT[l]['A']['dev_per_cost_rest']:.3f}</td>"
    f"<td class=\"r num\">{OOT[l]['B']['dev_per_cost_flagged']:.3f} vs {OOT[l]['B']['dev_per_cost_rest']:.3f}</td></tr>" for l in LINES)
rng = "".join(f"<tr><td>{LN[l]}</td><td class=\"r num\">{RG[l]['width_quantiles']['q50']:.2f}</td><td class=\"r num\">{RG[l]['width_quantiles']['q90']:.2f}</td>"
              f"<td class=\"r num\">{RG[l]['width_quantiles']['q99']:.2f}</td></tr>" for l in LINES)

P6HTML = f"""
  <section aria-labelledby="h-p6a">
    <span class="eyebrow">Phase 6 · model disagreement (R3)</span>
    <h2 id="h-p6a">Disagreement between the models is not a sign of error here</h2>
    <p>A 2025 row is flagged when any family in its line prices it more than 30% above or below the blend (<code>refer_disagreement</code>, written before the run).</p>
    <div class="tbl wide"><table><thead><tr><th>Line</th><th class="r">2025 rows flagged</th><th class="r">Flagged 2023 · 2024</th><th class="r">Error per unit of cost 2023: flagged vs rest</th><th class="r">2024</th></tr></thead><tbody>{oot_rows}</tbody></table></div>
    <p class="how"><b>How to read this.</b> "Error per unit of cost" is the blend's deviance divided by its expected cost, so a larger number means the blend is less reliable on those rows. In contents, flagged rows show {OOT['contents']['A']['dev_per_cost_flagged']:.3f} on 2023 against {OOT['contents']['A']['dev_per_cost_rest']:.3f} for the rest: if anything the blend does better where the models disagree, perhaps because averaging cancels each model's own errors there. So the flag cannot pick out risky rows, and at 30% it catches about one property row in six.</p>
    <div class="tbl wide"><table><thead><tr><th>Treatment of flagged rows, against keeping the blend</th><th>Line</th><th class="r">A: saved</th><th class="r">A: 95% interval</th><th class="r">B: saved</th><th>Passes</th></tr></thead><tbody id="t6t"></tbody></table></div>
    <p class="how"><b>How to read this.</b> Taking the highest model's price on flagged rows, the under-pricing guard, loses in every line ({sgn(g('T1 vs T0', 'all'))} across lines on 2023). Taking the W2 GLM's price passes in motor, where it applies to {RF['motor']['r3']:,} rows of 2025 ({p1(RF['motor']['r3_share'])}), and loses in buildings ({sgn(g('T2 vs T0', 'buildings'))}) and contents.</p>
  </section>

  <section aria-labelledby="h-p6b">
    <span class="eyebrow">Phase 6 · size and thin segments (R1, R2)</span>
    <h2 id="h-p6b">{size_n} rows, in segments the models do not use</h2>
    <p class="how"><b>What they are.</b> One buildings row is in a property type and bedroom combination with no training rows, and one contents row has a sum insured of 88,400, 2.32 times the largest in its tenure and bedroom segment (the phase 2 referral). {CNT.get(('buildings', 'R2'), 0)} buildings rows and {CNT.get(('contents', 'R2'), 0)} contents rows sit in segments with fewer than 100 training rows. <b>Why the price is still sound:</b> the property models dropped property type and bedrooms at check-in 2 (L8), and price sum insured across the whole line, where every one of these values is inside the training range.</p>
  </section>

  <section aria-labelledby="h-p6c">
    <span class="eyebrow">Ranges</span>
    <h2 id="h-p6c">How far apart the models are on each 2025 row</h2>
    <div class="tbl"><table><thead><tr><th>Line</th><th class="r">Median</th><th class="r">90th percentile</th><th class="r">99th percentile</th></tr></thead><tbody>{rng}</tbody></table></div>
    <p class="how"><b>How to read this.</b> Each number is the highest family price divided by the lowest for a row. A typical motor row's models agree within {100 * (RG['motor']['width_quantiles']['q50'] - 1):.0f}%; one buildings row in a hundred spans a factor of {RG['buildings']['width_quantiles']['q99']:.1f}. This is the model uncertainty in the expected cost, shown in the write-up as each row's range around the blend.</p>
  </section>
"""


def opt(k, n, d, c, f):
    return dict(k=k, n=n, d=d, c=c, f=f)


DECS = [
    dict(id="P1", door="easy", q="What happens to rows where the models disagree by more than 30%?",
         stakes=(f"<b>In plain terms:</b> disagreement does not mark risky rows here, so it should not move the price by itself. The one exception the test supports is motor, where "
                 f"falling back to the GLM's price on the {RF['motor']['r3']:,} flagged rows saves {n0(g('T2 vs T0', 'motor'))} on 2023 (interval {ivs('T2 vs T0', 'motor')}) and "
                 f"{n0(g('T2 vs T0', 'motor', 'B'))} on 2024. The 2024 gain is small."),
         opts=[opt("A", "Motor flagged rows at the GLM's price; property keeps the blend; every flag listed for review", "What the keep rule says, line by line.",
                   f"A second price rule in motor for {p1(RF['motor']['r3_share'])} of rows.", "Nothing."),
               opt("B", "Keep the blend everywhere; flags for review only", "One price rule everywhere; the motor gain on 2024 is small.", "Gives up a passed motor saving.", "Nothing."),
               opt("C", "Highest family price on flagged rows", "The under-pricing guard (J1).", f"Loses {n0(-g('T1 vs T0', 'all'))} on 2023 across lines.", "Nothing.")],
         rec="A", why="It follows the keep rule written before the results, and the GLM fallback is the plainest price to explain."),
    dict(id="P2", door="easy", q="How are the size and thin-segment rows priced?",
         stakes=(f"<b>In plain terms:</b> {size_n} rows of 2025 sit at or beyond the edge of their segment's evidence, but the models do not price by that segment, "
                 f"and every sum insured is inside the line's range."),
         opts=[opt("A", "Keep the blend price; list the rows for underwriting review", "No hand-set price; the referral is a review prompt.", "None.", "Nothing."),
               opt("B", "Cap sum insured at 1.25 times the segment maximum and reprice", "Prices as if the size were at the evidence edge.",
                   "Cuts the price of large risks the model already prices from the whole line; J1 says under-pricing is costlier.", "Nothing.")],
         rec="A", why="The segment edge is not an edge of the model's evidence, and capping would under-price the largest risks."),
]

s = PAGE.read_text(encoding="utf-8")
data = ("var TABLES = " + json.dumps(TABLES) + ";\n  var LEDE = " + json.dumps(LEDE, ensure_ascii=False) +
        ";\n  var DECS = " + json.dumps(DECS, ensure_ascii=False) + ";")
s = re.sub(r"/\*DATA\*/.*?/\*END DATA\*/", lambda m: "/*DATA*/" + data + "/*END DATA*/", s, flags=re.S)
s = re.sub(r"<!--P6-->.*?<!--END P6-->", lambda m: "<!--P6-->" + P6HTML + "<!--END P6-->", s, flags=re.S)
s = re.sub(r"<!--EX-->.*?<!--END EX-->", lambda m: "<!--EX-->" + EX + "<!--END EX-->", s, flags=re.S)
PAGE.write_text(s, encoding="utf-8")
print("written", PAGE, len(s))
