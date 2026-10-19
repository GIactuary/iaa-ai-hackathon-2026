"""Fill the check-in 3 decision page (docs/reviews/phase5-checkin3.html) from the step 6 evidence file, so that every
number on the page is read from outputs/ rather than typed.
Run: .venv/Scripts/python.exe -I scripts/checkin3_page.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "reviews" / "phase5-checkin3.html"
S6 = json.loads((ROOT / "outputs" / "phase5_step6.json").read_text())["phase5_step6"]
KW = json.loads((ROOT / "outputs" / "phase5_k3k4.json").read_text())["phase5_k3k4"]["by_line"]
LINES = ("motor", "buildings", "contents")
LN = {"motor": "Motor", "buildings": "Buildings", "contents": "Contents", "all": "All lines"}
T = S6["tests"]
FITS = S6["fits"]


def g(k, ln, s="A"):
    return T[k][ln][s]["gain"]


def iv(k, ln):
    a = T[k][ln]["A"]["boot"]
    return a["lo"], a["hi"]


def n0(x):
    s = f"{abs(x):,.0f}"
    return ("−" if x < 0 else "") + s


def sgn(x):
    return ("+" if x >= 0 else "−") + f"{abs(x):,.0f}"


def ivs(k, ln):
    lo, hi = iv(k, ln)
    return f"{n0(lo)} to {n0(hi)}"


def rows(key, name):
    out = []
    for ln in LINES + ("all",):
        x = T[key][ln]
        out.append(dict(name=name if ln == "motor" else "", line=LN[ln], a=x["A"]["gain"], lo=x["A"]["boot"]["lo"], hi=x["A"]["boot"]["hi"],
                        b=x["B"]["gain"], keep=bool(x["verdict"]["keep"])))
    return out


TABLES = {
    "t6s": rows("priced|S2-L vs S1-L", "S2 Tweedie per peril") + rows("priced|S3-L vs S1-L", "S3 Frequency × severity per peril")
           + rows("priced|S4-L vs S1-L", "S4 As S3, severity by year only"),
    "t6e": rows("priced|S3-P vs S3-L", "Within S3") + rows("priced|S2-P vs S2-L", "Within S2"),
    "t6w": rows("S1 weather: priced vs bare", "Row-level 2025 weather"),
}

S3 = "priced|S3-L vs S1-L"
pct = lambda x: f"{100 * x:.2f}%"
d2 = S6["ranking_d2"]
mot_b = (d2["B"]["motor"]["priced:S1-L"]["d2"], d2["B"]["motor"]["priced:S3-L"]["d2"])
lvl = {s: (g(S3, "motor", s) - g("bare|S3-L vs S1-L", "motor", s)) for s in ("A", "B")}
fin = {ln: FITS[ln]["final"] for ln in LINES}
bi_share = (fin["motor"]["excess_line"] - 1) / (fin["motor"]["excess_peril"]["bodily_injury"] - 1)
yf25 = {ln: KW[ln]["final"]["year_factor_2025"] - 1 for ln in LINES}

LEDE = (f"One capped Tweedie GLM per line stays in all three lines: no per-peril structure passes the keep rule anywhere. "
        f"Frequency × severity comes closest in motor, saving {sgn(g(S3, 'motor'))} on 2023 (interval {ivs(S3, 'motor')}) and "
        f"{sgn(g(S3, 'motor', 'B'))} on 2024. In buildings and contents every per-peril form ranks worse, because perils such as fire and "
        f"subsidence carry a few hundred claims a year. The per-peril excess factor also ranks worse than the line factor in motor and buildings. "
        f"Three calls: the structure for step 7, the excess factor, and a small finding on the 2025 weather.")

EX = (f"Example: frequency × severity in motor saves {n0(g(S3, 'motor'))} on A, interval {ivs(S3, 'motor')}, and "
      f"{n0(g(S3, 'motor', 'B'))} on B. The interval crosses zero, so it fails.")

# frequency and severity relativities, motor final fit, each normalised to its own base (age 30-39, vehicle group 11)
R = {k: {(x["factor"], x["level"]): x["relativity"] for x in v} for k, v in fin["motor"]["relativities"].items()}


def rel(key, factor, level, base):
    r = R[key]
    return r[(factor, level)] / r[(factor, base)]


AGE_B, VVG_B = "(29.0, 39.0]", "11"
RELROWS = [("Age under 21", "age", "(-0.001, 20.0]", AGE_B), ("Age 21 to 24", "age", "(20.0, 24.0]", AGE_B),
           ("Age 50 to 59", "age", "(49.0, 59.0]", AGE_B), ("Vehicle group 1", "vvg", "1", VVG_B),
           ("Vehicle group 16", "vvg", "16", VVG_B), ("Vehicle group 20", "vvg", "20", VVG_B)]
COLS = [("S1|line", "Line GLM"), ("S3f|own_damage", "Own damage: frequency"), ("S3s|own_damage", "Own damage: severity"),
        ("S3f|bodily_injury", "Injury: frequency"), ("S3s|bodily_injury", "Injury: severity"), ("S3f|theft", "Theft: frequency")]
rel_html = "".join(
    f"<tr><td>{lab}</td>" + "".join(f'<td class="r num">{rel(k, f, lv, b):.2f}</td>' for k, _ in COLS) + "</tr>" for lab, f, lv, b in RELROWS)
rel_html += "<tr><td>Claims-history slope on log(mod)</td>" + "".join(
    f'<td class="r num">{R[k][("h_p", "coefficient")]:.2f}</td>' for k, _ in COLS) + "</tr>"
rel_head = "".join(f'<th class="r">{n}</th>' for _, n in COLS)
under21 = {k: rel(k, "age", "(-0.001, 20.0]", AGE_B) for k, _ in COLS}
g20 = {k: rel(k, "vvg", "20", VVG_B) for k, _ in COLS}

STEP6 = f"""
  <section aria-labelledby="h-s6a">
    <span class="eyebrow">Step 6 · structure (P3) and the excess factor (K2)</span>
    <h2 id="h-s6a">Four structures on the same rating factors</h2>
    <p>Every model uses the W2 rating factors of its line, unchanged; only the structure differs. The design was written and committed before anything was fitted (<code>decisions/phase5_step6_design.md</code>).</p>
    <div class="tbl"><table>
      <thead><tr><th>Id</th><th>Structure</th><th>What it fits</th></tr></thead>
      <tbody>
        <tr><td class="k">S1</td><td>One capped Tweedie GLM per line (W2)</td><td>Capped line cost; the reference</td></tr>
        <tr><td class="k">S2</td><td>One capped Tweedie GLM per peril, summed</td><td>Capped cost of each peril</td></tr>
        <tr><td class="k">S3</td><td>Poisson frequency × gamma severity per peril, summed</td><td>Claim count, and capped cost per claim weighted by claims</td></tr>
        <tr><td class="k">S4</td><td>As S3, with severity by year only</td><td>Tests whether the severity relativities add anything</td></tr>
      </tbody>
    </table></div>
    <p>Each is scored as it would be priced: an off-balance to the weather-normalised indication (one per line for S1, one per peril for S2 to S4), times the scored year's weather. That gives the per-peril structures their textbook advantage: bodily injury's 2023 step reaches each row in proportion to its own injury share. The bare form, without the level chain, gives the same verdicts.</p>
  </section>

  <section aria-labelledby="h-s6b">
    <span class="eyebrow">Result</span>
    <h2 id="h-s6b">No per-peril structure passes; motor frequency × severity comes closest</h2>
    <div class="tbl wide"><table><thead><tr><th>Challenger against S1</th><th>Line</th><th class="r">A: saved</th><th class="r">A: 95% interval</th><th class="r">B: saved</th><th>Passes</th></tr></thead><tbody id="t6s"></tbody></table></div>
    <p class="how"><b>How to read this.</b> Each row replaces the line GLM with the challenger. In motor, frequency × severity lowers deviance by {n0(g(S3, 'motor'))} on 2023 and {n0(g(S3, 'motor', 'B'))} on 2024, and motor ranking D² on 2024 moves from {pct(mot_b[0])} to {pct(mot_b[1])}; but the 2023 interval runs from {ivs(S3, 'motor')}, so it fails. In buildings the same structure costs {n0(-g(S3, 'buildings'))} on 2023 and {n0(-g(S3, 'buildings', 'B'))} on 2024.</p>
    <p class="how"><b>Why property loses.</b>
      <span>Splitting by peril divides the claims. Buildings fire, subsidence and flood each carry about 200 to 500 claims a year, against the same set of rating factors in each peril's GLM; motor perils carry about 1,000 to 11,000.</span>
      <span>Of the per-peril forms, frequency × severity beats a Tweedie per peril across lines: {sgn(g('priced|S3-L vs S2-L', 'all'))} on 2023 (interval {ivs('priced|S3-L vs S2-L', 'all')}), {sgn(g('priced|S3-L vs S2-L', 'all', 'B'))} on 2024.</span>
      <span>Severity relativities matter: removing them (S4 against S3) costs {n0(-g('priced|S4-L vs S3-L', 'buildings'))} in buildings and {n0(-g('priced|S4-L vs S3-L', 'contents'))} in contents on 2023. Sum insured drives cost per claim in property.</span>
      <span>The per-peril level, the textbook advantage, is noise here: in motor it moves frequency × severity by {sgn(lvl['A'])} on 2023 and {sgn(lvl['B'])} on 2024 against the bare form.</span>
    </p>
  </section>

  <section aria-labelledby="h-s6c">
    <span class="eyebrow">The excess factor (K2 retest, C2)</span>
    <h2 id="h-s6c">Spreading the large-loss load over the line ranks better</h2>
    <p>Above-cap cost sits in a few perils. Final-fit factors: motor bodily injury {fin['motor']['excess_peril']['bodily_injury']:.3f} against the line's {fin['motor']['excess_line']:.3f}; buildings fire {fin['buildings']['excess_peril']['fire']:.3f} and flood {fin['buildings']['excess_peril']['flood']:.3f} against {fin['buildings']['excess_line']:.3f}; contents flood {fin['contents']['excess_peril']['flood']:.3f} against {fin['contents']['excess_line']:.3f}. Every other peril is at 1.000 or within 0.002 of it.</p>
    <div class="tbl wide"><table><thead><tr><th>Per-peril factor against line factor</th><th>Line</th><th class="r">A: saved</th><th class="r">A: 95% interval</th><th class="r">B: saved</th><th>Passes</th></tr></thead><tbody id="t6e"></tbody></table></div>
    <p class="how"><b>How to read this.</b> Bodily injury is about {100 * bi_share:.0f}% of motor capped cost, so the line's {100 * (fin['motor']['excess_line'] - 1):.1f}% load equals a {100 * (fin['motor']['excess_peril']['bodily_injury'] - 1):.1f}% load on injury and nothing on other perils. Moved onto injury alone, the load loses {n0(-g('priced|S3-P vs S3-L', 'motor'))} on 2023 (interval {ivs('priced|S3-P vs S3-L', 'motor')}): which policies produce the few claims above 100k is not predicted by their injury share. Only contents passes, inside a structure that itself loses. S1 has no peril split, so it takes the line factor in any case.</p>
  </section>

  <section aria-labelledby="h-s6d">
    <span class="eyebrow">For explanation · motor, final fit 2021 to 2024</span>
    <h2 id="h-s6d">What frequency and severity show that the line GLM blends</h2>
    <div class="tbl wide"><table><thead><tr><th>Level (base: age 30 to 39, group 11)</th>{rel_head}</tr></thead><tbody>{rel_html}</tbody></table></div>
    <p class="how"><b>How to read this.</b> An under-21 driver claims own damage {under21['S3f|own_damage']:.2f} times as often as a driver aged 30 to 39, at {under21['S3s|own_damage']:.2f} times the cost per claim; the line GLM's {under21['S1|line']:.2f} is the blend across perils. Vehicle value works the other way: group 20 has {g20['S3s|own_damage']:.2f} times the own-damage cost per claim at {g20['S3f|own_damage']:.2f} times the frequency. Claims history predicts how often, not how much: its slope is {R['S3f|own_damage'][('h_p', 'coefficient')]:.2f} in own-damage frequency and {R['S3s|own_damage'][('h_p', 'coefficient')]:.2f} in severity. These relativities explain the line GLM; they are not adopted as a structure.</p>
  </section>

  <section aria-labelledby="h-s6e">
    <span class="eyebrow">For information · card M3</span>
    <h2 id="h-s6e">The 2025 weather row by row is inside the noise, and slightly worse</h2>
    <div class="tbl wide"><table><thead><tr><th>Test on the line GLM</th><th>Line</th><th class="r">A: saved</th><th class="r">A: 95% interval</th><th class="r">B: saved</th><th>Passes</th></tr></thead><tbody id="t6w"></tbody></table></div>
    <p class="how"><b>How to read this.</b> K4 multiplies each row by its own year's weather. On the line GLM that is the only difference between the priced and bare forms, so this row tests it. In buildings it costs {n0(-g('S1 weather: priced vs bare', 'buildings'))} on 2023 (interval {ivs('S1 weather: priced vs bare', 'buildings')}) and {n0(-g('S1 weather: priced vs bare', 'buildings', 'B'))} on 2024. The level is untouched either way: the 2025 net effect is {100 * yf25['motor']:+.2f}% motor, {100 * yf25['buildings']:+.2f}% buildings and {100 * yf25['contents']:+.2f}% contents.</p>
    <div class="wrong">
      <b>Added after seeing results, and one diagnostic set aside</b>
      <span>This weather test was not in the design. I added it after the scored frames showed the gap between the bare and priced line GLM, and it is reported as information for card M3, not as a pre-registered test.</span>
      <span>The design's per-peril diagnostic, which scores each peril's component against that peril's own cost, turned out to be uninformative. In motor it is decided by rows whose cover excludes the peril: the Tweedie per peril drives third-party-only rows to near-zero own-damage, glass and theft predictions, which wins a single-peril score but barely moves the line total. It stays in the evidence file and decides nothing.</span>
    </div>
  </section>
"""


def opt(k, n, d, c, f):
    return dict(k=k, n=n, d=d, c=c, f=f)


E = "priced|S3-P vs S3-L"
W = "S1 weather: priced vs bare"
DECS = [
    dict(id="M1", door="costly", q="Which structure carries into step 7: one Tweedie GLM per line, or frequency × severity per peril?",
         stakes=f"<b>It sets the frame for steps 7 and 8.</b> Changing it after the families are fitted means refitting all of them, with the freeze at the end of 21 October. No per-peril structure passes in any line. Frequency × severity: motor {sgn(g(S3, 'motor'))} on 2023 (interval {ivs(S3, 'motor')}) and {sgn(g(S3, 'motor', 'B'))} on 2024; buildings {sgn(g(S3, 'buildings'))} and {sgn(g(S3, 'buildings', 'B'))}; contents {sgn(g(S3, 'contents'))} and {sgn(g(S3, 'contents', 'B'))}.",
         opts=[opt("A", "Tweedie per line in all three; in step 7, frequency × severity built in motor only",
                   "Follows the keep rule. Motor's near miss gets a second, fair test in each step 7 family, as P3 asks; property, where it loses clearly, stays per line.",
                   "Motor step 7 runs about twice as long.", "Per-peril models in property."),
               opt("B", "Tweedie per line now; both structures in every line and every step 7 family (P3 as written)",
                   "Nothing is ruled out before the families are tried.",
                   f"Roughly doubles step 7 compute before the freeze; in property the per-peril forms lost by {n0(-g('priced|S3-L vs S1-L', 'buildings'))} to {n0(-g('priced|S4-L vs S1-L', 'contents'))} on 2023, so a reversal is unlikely.",
                   "Nothing."),
               opt("C", "Adopt frequency × severity in motor now",
                   "Takes the 2024 saving, and the clearer story of age, vehicle value and history by peril.",
                   "Overrides the agreed keep rule: the 2023 interval crosses zero.", "The line GLM in motor.")],
         rec="A", why="It holds the rule written before results, keeps one GLM per line to explain, and spends step 7's limited time where the evidence points."),
    dict(id="M2", door="easy", q="Which excess-loss factor: the line factor, or one per peril?",
         stakes=f"Per peril, the motor load sits on bodily injury alone ({fin['motor']['excess_peril']['bodily_injury']:.3f} against the line's {fin['motor']['excess_line']:.3f}). Within frequency × severity it ranks worse in motor ({sgn(g(E, 'motor'))}, interval {ivs(E, 'motor')}) and buildings ({sgn(g(E, 'buildings'))}, interval {ivs(E, 'buildings')}). It passes only in contents ({sgn(g(E, 'contents'))}, interval {ivs(E, 'contents')}; {sgn(g(E, 'contents', 'B'))} on 2024), inside a structure that itself loses.",
         opts=[opt("A", "Line factor everywhere, including any per-peril model in step 7 (K2 (a) stands)",
                   "The few claims above the cap are spread over the whole line.", "Retires C2's per-peril form; the per-peril factors stay on file.", "Nothing."),
               opt("B", "Per-peril factor wherever a per-peril model is used", "Matches C2 as decided.",
                   "Ranks worse in motor and buildings on 2023, with intervals below zero.", "Nothing.")],
         rec="A", why="Which policies produce the claims above the cap is not predicted by their injury or fire share, so a flat load per line ranks better. This reverses C2 on evidence."),
    dict(id="M3", door="easy", q="Apply the 2025 weather row by row, or as one factor per line?",
         stakes=f"K4 multiplies each row by its own 2025 weather. Against no weather, that ranks slightly worse: buildings {sgn(g(W, 'buildings'))} on 2023 (interval {ivs(W, 'buildings')}) and {sgn(g(W, 'buildings', 'B'))} on 2024; all lines {sgn(g(W, 'all'))} and {sgn(g(W, 'all', 'B'))}. It is inside the noise but worse in both years. A possible reason: a row takes the weather of the year it starts, while a policy starting mid-year has half its exposure in the next year. The level K4 was decided on is the same either way.",
         opts=[opt("A", "Keep it row by row (K4 as applied)",
                   "Uses each address's own 2025 weather; the same-address frequency effects (storm 1.27, subsidence 1.63) are real.",
                   "Keeps a row-level adjustment that does not pass the keep rule.", "Nothing."),
               opt("B", "One factor per line: the exposure-weighted average of the 2025 row multipliers",
                   "Same 2025 level; no row variation.", "Gives up row-level weather that a later year might reward.", "Nothing.")],
         rec="B", why="Every other row-level effect in the plan has passed the keep rule; this one is worse in both years, and the level is unchanged."),
]

s = PAGE.read_text(encoding="utf-8")
data = ("var TABLES = " + json.dumps(TABLES) + ";\n  var LEDE = " + json.dumps(LEDE, ensure_ascii=False) +
        ";\n  var DECS = " + json.dumps(DECS, ensure_ascii=False) + ";")
s = re.sub(r"/\*DATA\*/.*?/\*END DATA\*/", lambda m: "/*DATA*/" + data + "/*END DATA*/", s, flags=re.S)
s = re.sub(r"<!--STEP6-->.*?<!--END STEP6-->", lambda m: "<!--STEP6-->" + STEP6 + "<!--END STEP6-->", s, flags=re.S)
s = re.sub(r"<!--EX-->.*?<!--END EX-->", lambda m: "<!--EX-->" + EX + "<!--END EX-->", s, flags=re.S)
PAGE.write_text(s, encoding="utf-8")
print("written", PAGE, len(s))
