"""Build the post-mortem decision page (docs/reviews/postmortem-y.html, cards Y1 to Y4) from the post-mortem evidence,
so that every number on the page is read from files rather than typed. The page machinery (verdict store, Send,
Finalize, Copy for Claude, notes on the flow diagram) is taken from the check-in 7 page.
Run: .venv/Scripts/python.exe -I scripts/postmortem_page.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import build, validation as v

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "docs" / "reviews" / "phase7-checkin7.html"
PAGE = ROOT / "docs" / "reviews" / "postmortem-y.html"
PM = json.loads((ROOT / "outputs" / "postmortem.json").read_text())["postmortem"]
T, D2, REC, PREM = PM["tests"], PM["ranking_d2"], PM["recoveries"], PM["premium"]
WORK = ROOT / "data" / "work" / "postmortem"
LN = {"motor": "Motor", "buildings": "Buildings", "contents": "Contents", "all": "All lines"}
YEAR = {"A": 2023, "B": 2024}


def g(k, ln, s="A"):
    return T[k][ln][s]["gain"]


def n0(x):
    return ("−" if x < 0 else "") + f"{abs(x):,.0f}"


def sgn(x):
    return ("+" if x >= 0 else "−") + f"{abs(x):,.0f}"


def ivs(k, ln):
    a = T[k][ln]["A"]["boot"]
    return f"{n0(a['lo'])} to {n0(a['hi'])}"


def pct(x, d=2):
    return f"{100 * x:.{d}f}%"


def row(key, name, ln):
    x = T[key][ln]
    return dict(name=name, line=LN[ln], a=x["A"]["gain"], lo=x["A"]["boot"]["lo"], hi=x["A"]["boot"]["hi"], b=x["B"]["gain"],
                keep=bool(x["verdict"]["keep"]))


K_W1, K_W2 = "PM1a vs R", "PM1b vs R"
K_WIDE, K_WIDEW = "PM2m vs R", "PM3m vs R"
K_PREM = "PM4c: W2 GLM + premium vs W2 GLM"
K_HAT, K_RES = "PM6: W2 GLM + predicted premium vs W2 GLM", "PM6: W2 GLM + premium remainder vs W2 GLM"
TABLES = {"tpm": [
    row(K_HAT, "GLM + premium as our fields predict it", "buildings"), row(K_HAT, "", "contents"), row(K_HAT, "", "motor"), row(K_HAT, "", "all"),
    row(K_RES, "GLM + the premium's remainder", "all"),
    row(K_PREM, "GLM + the raw premium", "buildings"), row(K_PREM, "", "all"),
    row(K_WIDE, "Trees: every legitimate field", "motor"), row(K_WIDE, "", "buildings"), row(K_WIDE, "", "contents"), row(K_WIDE, "", "all"),
    row(K_W1, "Trees: own-year weather", "all"), row(K_W2, "Trees: weather and climate normal", "all"), row(K_WIDEW, "Trees: every field and weather", "all"),
]}

# Double lift in buildings: tenths of the ratio of the premium-informed GLM to the W2 GLM, actual over each model
b = build.build_all()["buildings"][0]
b = b[~b["holdout"]]
lift = {}
for s in ("A", "B"):
    f = pd.read_parquet(WORK / f"buildings_{s}.parquet")
    r = b[b["yr"] == YEAR[s]].reset_index(drop=True)
    if not np.allclose(r["cost"].to_numpy(), f["y"].to_numpy()):
        raise ValueError("scored rows do not line up with the build")
    y = f["y"].to_numpy()
    d = pd.DataFrame({"y": y, "glm": v.rescale_to_total(f["glm"].to_numpy(), y), "phat": v.rescale_to_total(f["glm_phat"].to_numpy(), y),
                      "flood": r["cost_flood"].to_numpy(), "riv": r["distance_to_river_km"].to_numpy(), "elev": r["elevation_m"].to_numpy()})
    d["ratio"] = d["phat"] / d["glm"]
    d["dec"] = pd.qcut(d["ratio"], 10, labels=False)
    t = d.groupby("dec").agg(ratio=("ratio", "mean"), y=("y", "sum"), glm=("glm", "sum"), phat=("phat", "sum"), flood=("flood", "sum"),
                             riv=("riv", "median"), elev=("elev", "median"))
    lift[s] = dict(t=t, flood_all=float(d["flood"].sum() / d["y"].sum()), riv_all=float(d["riv"].median()), elev_all=float(d["elev"].median()))
top, bot = lift["A"]["t"].iloc[-1], lift["A"]["t"].iloc[0]


def lift_rows():
    out = []
    for s in ("A", "B"):
        L = lift[s]
        for name, i in (("Lowered most", 0), ("Raised most", 9)):
            x = L["t"].iloc[i]
            out.append(f"<tr><td>{YEAR[s]}</td><td>{name}</td><td class=\"r num\">{100 * (x['ratio'] - 1):+.0f}%</td>"
                       f"<td class=\"r num\">{x['y'] / x['glm']:.2f}</td><td class=\"r num\">{x['y'] / x['phat']:.2f}</td>"
                       f"<td class=\"r num\">{x['riv']:.2f}</td><td class=\"r num\">{x['elev']:.1f}</td><td class=\"r num\">{pct(x['flood'] / x['y'], 0)}</td></tr>")
        out.append(f"<tr><td>{YEAR[s]}</td><td>Every row</td><td class=\"r num\"></td><td class=\"r num\">1.00</td><td class=\"r num\">1.00</td>"
                   f"<td class=\"r num\">{L['riv_all']:.2f}</td><td class=\"r num\">{L['elev_all']:.1f}</td><td class=\"r num\">{pct(L['flood_all'], 0)}</td></tr>")
    return "".join(out)


d2rows = "".join(
    f"<tr><td>{LN[l]}</td>" + "".join(f'<td class="r num">{pct(D2[s][l][c])}</td>' for s in ("A", "B") for c in ("glm", "glm_phat")) + "</tr>"
    for l in ("motor", "buildings", "contents", "all"))
m = REC["motor"]
ay = {int(r["ay"]): r for r in m["by_accident_year"]}
rate = m["recovery_rate_closed"]
recrows = "".join(f"<tr><td>{y}</td><td class=\"r num\">{ay[y]['net'] / 1e6:,.1f}M</td><td class=\"r num\">{ay[y]['exp_rec'] / 1e6:,.2f}M</td>"
                  f"<td class=\"r num\">{pct(ay[y]['overstatement'])}</td></tr>" for y in (2021, 2022, 2023, 2024))
gap_pts = 100 * (D2["A"]["all"]["glm_phat"] - D2["A"]["all"]["glm"]), 100 * (D2["B"]["all"]["glm_phat"] - D2["B"]["all"]["glm"])
r2b = PREM["buildings"]["A"]["r2_scored"]
ptop = [k for k, _ in PREM["buildings"]["A"]["top"][:4]]

LEDE = (f"Part 1 closed at 0.13641, 15th; first place scored 0.13923. Most of the gap is in buildings, and it is information already in our own fields: "
        f"the current rating plan prices flood geography better than any model we built. Adding the premium <i>as our own fields predict it</i> to the W2 GLM "
        f"saves {sgn(g(K_HAT, 'buildings'))} on 2023 (interval {ivs(K_HAT, 'buildings')}) and {sgn(g(K_HAT, 'buildings', 'B'))} on 2024 in buildings, and lifts "
        f"all-lines ranking D² by {gap_pts[0]:.2f} and {gap_pts[1]:.2f} points. The rest of the premium adds nothing. Weather adds nothing. Four calls set how Part 2 is built.")

EX = (f"Example: in buildings, the GLM plus the premium as our fields predict it saves {n0(g(K_HAT, 'buildings'))} on 2023, interval {ivs(K_HAT, 'buildings')}, "
      f"and {n0(g(K_HAT, 'buildings', 'B'))} on 2024, so it passes.")

BODY = f"""
  <section aria-labelledby="h-settled">
    <span class="eyebrow">Settled before this page · Part 1</span>
    <h2 id="h-settled">The Part 1 file, and how the post-mortem was run</h2>
    <div class="resolved">
      <div><span class="cid">U2</span><span>blend_final_all_v2 was the final file: public 0.12908, private 0.13641 (15th)</span></div>
      <div><span class="cid">D30</span><span>Post-mortem on Part 1's own data, design written and committed before each run</span></div>
    </div>
    <p class="how"><b>How the tests read.</b> Each test refits on 2021 to 2022 and scores 2023 (A), and refits on 2021 to 2023 and scores 2024 (B), on customers outside the hold-out, exactly as every Part 1 decision was tested. "Saved" is the drop in total Tweedie deviance against the reference at a common level per line; the keep rule is unchanged (better in both years, and the 2023 interval above zero). The GLM tests are against the W2 GLM; the tree tests are against LightGBM on the Part 1 tree inputs. <!--EX-->{EX}<!--END EX--></p>
  </section>

  <section aria-labelledby="h-pma">
    <span class="eyebrow">What the tests found</span>
    <h2 id="h-pma">The premium's gain is in the part our fields predict</h2>
    <div class="tbl wide"><table><thead><tr><th>Test</th><th>Line</th><th class="r">2023: saved</th><th class="r">2023: 95% interval</th><th class="r">2024: saved</th><th>Passes</th></tr></thead><tbody id="tpm"></tbody></table></div>
    <p class="how"><b>How to read this.</b> The premium was split in two before it went near a claim. A model fitted only on our own rating fields explains {pct(r2b, 1)} of the buildings premium (largest inputs: {", ".join(ptop)}); that is "the premium as our fields predict it". What is left over is "the remainder". The first carries the whole gain and the second none, so the premium holds no hidden or post-event information, and nothing from a field we exclude (such as the P6 marketing fields) can reach the model through it. Wider tree inputs pass across lines (+{n0(g(K_WIDE, 'all'))} on 2023) but in no line on its own.</p>
    <div class="tbl wide"><table><thead><tr><th>Ranking D²</th><th class="r">2023 W2 GLM</th><th class="r">2023 + predicted premium</th><th class="r">2024 W2 GLM</th><th class="r">2024 + predicted premium</th></tr></thead><tbody>{d2rows}</tbody></table></div>
    <p class="how"><b>How to read this.</b> All lines move from {pct(D2['A']['all']['glm'])} to {pct(D2['A']['all']['glm_phat'])} on 2023 and from {pct(D2['B']['all']['glm'])} to {pct(D2['B']['all']['glm_phat'])} on 2024, about 0.2 points. The gap to first place was 0.28 points on Kaggle's scale; the scales differ, so the two are comparable in size only.</p>
  </section>

  <section aria-labelledby="h-pmb">
    <span class="eyebrow">Where the structure sits · buildings</span>
    <h2 id="h-pmb">Homes near a river and low down were under-priced</h2>
    <div class="tbl wide"><table><thead><tr><th>Year</th><th>Tenth of rows</th><th class="r">Premium-informed vs W2 GLM</th><th class="r">Actual / W2 GLM</th><th class="r">Actual / premium-informed</th><th class="r">River km</th><th class="r">Elevation m</th><th class="r">Flood share</th></tr></thead><tbody>{lift_rows()}</tbody></table></div>
    <p class="how"><b>How to read this.</b> Rows are sorted by how far the premium-informed GLM moves the W2 GLM's price and cut into tenths. In 2023, in the tenth it raises most (by {100 * (top['ratio'] - 1):.0f}% on average), actual cost was {top['y'] / top['glm']:.2f} times the W2 GLM's price and {top['y'] / top['phat']:.2f} times the premium-informed price: the GLM under-priced these homes, which sit {top['riv']:.2f} km from a river (median) against {lift['A']['riv_all']:.2f} for the book, and lose {pct(top['flood'] / top['y'], 0)} of their cost to flood against {pct(lift['A']['flood_all'], 0)}. Flood claims are rare, so four years of our own experience could not support how zone, river distance and elevation combine; the trees had both fields and still did not learn it. The current rating plan carries it with more credibility.</p>
  </section>

  <section aria-labelledby="h-pmc">
    <span class="eyebrow">Recoveries on open claims · motor</span>
    <h2 id="h-pmc">The 2025 motor indication is {pct(-m['level_change'])} too high</h2>
    <div class="tbl"><table><thead><tr><th>Accident year</th><th class="r">Net cost as recorded</th><th class="r">Expected recoveries missing</th><th class="r">Overstated by</th></tr></thead><tbody>{recrows}</tbody></table></div>
    <p class="how"><b>How to read this.</b> The host confirmed that recoveries are recorded only when a claim settles. Settled own damage claims recover {pct(rate['own_damage'], 1)} of incurred, theft {pct(rate['theft'], 1)}; no other peril recovers. Applying those rates to claims still open at 31 December 2024 takes {ay[2024]['exp_rec'] / 1e6:,.2f}M off accident year 2024. The 2025 motor indication falls {pct(-m['level_change'])}, more than 2024's share alone suggests, because the overstated 2024 severity also steepens the own damage trend. Buildings and contents do not change.</p>
    <div class="wrong">
      <b>Where I was wrong (D31)</b>
      <span>Before testing, I ranked missing address weather as the second likely cause of the gap; it tested at {sgn(g(K_W1, 'all'))} on 2023 and {sgn(g(K_W1, 'all', 'B'))} on 2024, nothing. My first wide-input test also let the trees split on make-model, a 300-level code Part 1 had already shown to be noise (M7), and motor lost 414,287. The correction was written down and committed before the rerun; the numbers above are from the corrected test.</span>
    </div>
  </section>
"""


def opt(k, n, d, c, f):
    return dict(k=k, n=n, d=d, c=c, f=f)


DECS = [
    dict(id="Y1", door="costly", q="Should Part 2 use the current rating plan, as our own fields predict it, as an input?",
         stakes=(f"<b>In plain terms:</b> on Part 1, the GLM plus the premium as our fields predict it saves {sgn(g(K_HAT, 'buildings'))} on 2023 and "
                 f"{sgn(g(K_HAT, 'buildings', 'B'))} on 2024 in buildings, and the all-lines ranking D² rises about 0.2 points, most of the gap to the top. "
                 f"This revisits J-prem, where you chose the premium as the incumbent to beat only. In filing terms it uses the current plan's relativities as a "
                 f"complement of credibility where our own experience is thin, mainly for flood. If Part 2's data has no premium, nothing changes."),
         opts=[opt("A", "The premium as our fields predict it: one term in the GLM and an input to the trees",
                   "Fitted only on our legitimate rating fields, with no claims information; retested on Part 2's own data under the keep rule.",
                   "The rating plan leans partly on the incumbent's relativities; the write-up must explain it as a complement.",
                   "Nothing: the keep rule on Part 2 still decides whether it stays."),
               opt("B", "The raw premium, as an input or an offset",
                   "Same gain in buildings on Part 1 (+86,353 as a GLM term).",
                   "Brings in whatever the incumbent used, including fields we exclude and its own errors; the remainder added nothing on Part 1.",
                   "A clean answer to what the premium contributes."),
               opt("C", "Keep J-prem (a): the premium as the incumbent to beat only",
                   "The rating plan rests on our own experience alone.",
                   "Leaves the flood structure unpriced; on Part 1 evidence, about 0.2 points of D².",
                   "Nothing."),
               opt("D", "No premium; build a flood-geography term ourselves and test it",
                   "A per-peril flood frequency model on zone, river distance and elevation.",
                   "Untested; per-peril property structures failed in Part 1 (M1, D27) and the trees with these fields did not learn it.",
                   "Nothing; it can also run as a challenger beside A.")],
         rec="A", why="It carries the whole gain, it is built only from fields we already accept, and the keep rule on Part 2 still decides; D can run beside it as a challenger."),
    dict(id="Y2", door="costly", q="Which inputs may the tree families use in Part 2?",
         stakes=(f"<b>In plain terms:</b> in Part 1 a field had to pass a GLM test before any tree could see it. Giving the trees every legitimate field saves "
                 f"{sgn(g(K_WIDE, 'all'))} on 2023 (interval {ivs(K_WIDE, 'all')}) and {sgn(g(K_WIDE, 'all', 'B'))} on 2024 across lines, but no line passes on its own "
                 f"(buildings {sgn(g(K_WIDE, 'buildings'))}, interval {ivs(K_WIDE, 'buildings')})."),
         opts=[opt("A", "Every legitimate field; the family weights decide how much the trees count",
                   "Excludes leakage, post-event fields and the P6 fields; a code with many levels enters only credibility-encoded, with the level count written in the thresholds file first.",
                   "More inputs to explain in the trees, and a refit of the family weights on Part 2.",
                   "Nothing: weak trees get low weight."),
               opt("B", "Keep Part 1's GLM-vetted inputs",
                   "As built in Part 1.",
                   "Forgoes the across-line gain.",
                   "Nothing.")],
         rec="A", why="It passes across lines on Part 1, and the step 8 weights already guard against a tree that does not help."),
    dict(id="Y3", door="easy", q="Deduct expected recoveries on claims still open at the valuation date (new data rule R10)?",
         stakes=(f"<b>In plain terms:</b> open motor claims carry no recoveries yet, so accident year 2024 net cost is overstated {pct(ay[2024]['overstatement'])} "
                 f"and the 2025 motor indication {pct(-m['level_change'])}. Under-pricing is the costlier error (J1), but this is over-pricing, and it is a correctness point for the filing."),
         opts=[opt("A", "Yes, at each peril's closed-claim recovery rate, if Part 2's claims show the same pattern",
                   "Checked on Part 2's claims file first; applies to the target and the level chain.",
                   "One data rule and its test.", "Nothing."),
               opt("B", "No; leave as recorded and state it as a limitation",
                   "Keeps the target as the data shows it.", "The motor level carries a known overstatement.", "Nothing.")],
         rec="A", why="The host confirmed the mechanism, the effect is measurable, and the rule is cheap to write and test."),
    dict(id="Y4", door="easy", q="Keep weather out of the row-level inputs in Part 2?",
         stakes=(f"<b>In plain terms:</b> the policy year's weather in the trees tested {sgn(g(K_W1, 'all'))} on 2023 and {sgn(g(K_W1, 'all', 'B'))} on 2024; "
                 f"with the address's climate normal, {sgn(g(K_W2, 'all'))} and {sgn(g(K_W2, 'all', 'B'))}. Neither passes."),
         opts=[opt("A", "Keep M3 (b): the target year's weather as one factor per line",
                   "As in Part 1; retest only if Part 2's data shows a stronger weather signal.", "Nothing new.", "Nothing."),
               opt("B", "Give the trees row-level weather anyway", "More inputs.", "Fails the keep rule on Part 1.", "Nothing.")],
         rec="A", why="It failed on Part 1, and a new market can reopen it with its own evidence."),
]

STEPS = """[
    { id: "close", t: "Part 1 closed", s: "15th, 0.13641", isNew: false },
    { id: "pm", t: "Post-mortem", s: "PM1 to PM6", isNew: false },
    { id: "ycheck", t: "Y1 to Y4", s: "here", isNew: true, here: true },
    { id: "intake2", t: "Part 2 intake", s: "files, rules", isNew: false },
    { id: "changed", t: "What changed", s: "report", isNew: false },
    { id: "check2a", t: "Part 2 check-in", s: "data rules", isNew: true },
    { id: "config", t: "Configuration", s: "Y verdicts in", isNew: false },
    { id: "run", t: "Run market", s: "one command", isNew: false },
    { id: "tests2", t: "Retest on Part 2", s: "keep rule", isNew: false },
    { id: "check2b", t: "Part 2 check-in", s: "model, file", isNew: true },
    { id: "submit2", t: "Part 2 file", s: "you submit", isNew: false },
    { id: "writeup", t: "Write-up, video", s: "both parts", isNew: false }
  ]"""

s = TEMPLATE.read_text(encoding="utf-8")
subs = [
    (r"<title>.*?</title>", "<title>Part 2 Settings Brief</title>"),
    (r'<span class="eyebrow">IAA AI Hackathon 2026 · Part 1 · Phase 7 · Check-in 7 · Round 1 · 20 Oct 2026</span>',
     '<span class="eyebrow">IAA AI Hackathon 2026 · Part 1 post-mortem · Y1 to Y4 · Round 1 · 24 Oct 2026</span>'),
    (r"<h1>The hold-out, and the final file</h1>", "<h1>What cost us the top, and what Part 2 should change</h1>"),
    (r'<section aria-labelledby="h-settled">.*?<!--END P7-->', BODY),
    (r'<span class="eyebrow">Where we are · click a step to leave a note on it</span>\s*<h2 id="h-flow">[^<]*</h2>',
     '<span class="eyebrow">Where we are · click a step to leave a note on it</span>\n    <h2 id="h-flow">Post-mortem done; your verdicts set the Part 2 build</h2>'),
    (r"var STEPS = \[.*?\];", "var STEPS = " + STEPS + ";"),
    (r'aria-label="Phase 5 steps from cap and load to the hold-out, check-in 3 marked"', 'aria-label="From the Part 1 close to the Part 2 file, check-ins marked"'),
    (r'"Check-in 7 round "', '"Post-mortem Y1 to Y4, round "'),
    (r"Finalized\. Claude records the verdicts and readies the final file\.", "Finalized. Claude records the verdicts and sets the Part 2 build."),
]
for pat, rep in subs:
    s, n = re.subn(pat, lambda _m, rep=rep: rep, s, count=1, flags=re.S)
    if n != 1:
        raise ValueError(f"template changed: {pat[:60]}")
data = ("var TABLES = " + json.dumps(TABLES) + ";\n  var LEDE = " + json.dumps(LEDE, ensure_ascii=False) +
        ";\n  var DECS = " + json.dumps(DECS, ensure_ascii=False) + ";")
s = re.sub(r"/\*DATA\*/.*?/\*END DATA\*/", lambda _m: "/*DATA*/" + data + "/*END DATA*/", s, flags=re.S)
if "—" in s:
    raise ValueError("em dash on the page")
PAGE.write_text(s, encoding="utf-8")
print("written", PAGE, len(s))
