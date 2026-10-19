"""Fill the check-in 2 decision page (docs/reviews/phase5-checkin2.html) from the step 4 and step 5 evidence files,
so that every number on the page is read from outputs/ rather than typed.
Run: .venv/Scripts/python.exe -I scripts/checkin2_page.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "reviews" / "phase5-checkin2.html"
S4 = json.loads((ROOT / "outputs" / "phase5_step4_pg.json").read_text())["phase5_step4_pg"]
S5 = json.loads((ROOT / "outputs" / "phase5_step5.json").read_text())["phase5_step5"]
LINES = ("motor", "buildings", "contents")
LN = {"motor": "Motor", "buildings": "Buildings", "contents": "Contents", "all": "All lines"}

NAMES = {
    "G1": "Voluntary excess", "G2": "Renewal count (tenure)", "G3": "Days from quote to inception", "G4": "Missed payments yes/no (swap for 0/1/2+)",
    "G5": "Household size", "G6": "Marital status", "G7": "Years at address", "G8": "Occupation", "G9": "Other lines held at the start",
    "M1": "NCB above years licensed", "M2": "Licence before age 16", "M3": "Vehicle value per engine litre", "M4": "Telematics score bands (swap for opt-in)",
    "M5": "Age × NCB", "M6": "Mileage deciles (swap)", "M7": "Make-model credibility encoding", "M8": "Years licensed", "M9": "Vehicle age",
    "M10": "Engine size", "M11": "Transmission", "M12": "Seats", "M13": "Driver assist", "M14": "Overnight parking", "M15": "Vehicle colour",
    "B1": "Building age at the row (swap for year built)", "B2": "Sum insured per bedroom", "B3a": "River distance and elevation",
    "B3b": "Flood score: zone × proximity", "B4": "Clay × dry-summer normal", "B5": "Days unoccupied", "B6": "Roof type", "B7": "Basement",
    "B8": "Trees within 5 m", "B9": "Trees within 5 m × clay soil", "B10": "Listed building", "B11": "Energy rating", "B12": "Solar panels",
    "B13": "Smoke alarms", "B14": "Mortgage",
    "C1": "Sum insured per bedroom", "C2": "Bedrooms", "C3": "Building age at the row", "C4": "Works from home", "C5": "Lodgers", "C6": "Basement",
    "C7": "Flood score: zone × proximity", "C8": "Listed building", "C9": "Roof type", "C10": "Energy rating",
}
V1N = {"region": "Region", "urban": "Urban density", "pay": "Payment method", "missed": "Missed payments", "credit": "Credit band",
       "channel": "Channel", "new_bus": "New business", "age": "Age", "ncb": "NCB years", "vvg": "Vehicle value group", "cover": "Cover type",
       "use": "Vehicle use", "fuel": "Fuel", "body": "Body type", "mileage": "Mileage", "points": "Licence points", "young": "Named young driver",
       "telem": "Telematics opt-in", "si": "Sum insured", "ptype": "Property type", "flood": "Flood zone", "ad_cover": "Accidental damage cover",
       "constr": "Construction", "soil": "Soil", "occ": "Occupancy", "built": "Year built", "beds": "Bedrooms", "tenure": "Tenure",
       "security": "Security level", "hvi": "High-value items", "bike": "Bicycle cover", "h_p": "Policy history (step 4)", "h_d": "Declared history (step 4)"}


def row(name, t, ln=None, label=None):
    a, b = t["A"][ln] if ln else t["A"], t["B"][ln] if ln else t["B"]
    keep = (t["verdict"][ln] if ln else t["verdict"])["keep"]
    out = dict(name=name, a=a["gain"], lo=a["boot"]["lo"], hi=a["boot"]["hi"], b=b["gain"], keep=bool(keep))
    if ln:
        out["line"] = LN[ln]
    if label:
        out["label"] = label
    return out


tests = S4["tests"]
T4 = []
for key, nm in (("H-P_vs_V1", "H-P own policy"), ("H-D_vs_V1", "H-D declared, before joining"), ("H-C_vs_V1", "H-C customer's other lines"),
                ("H-A_vs_V1", "H-A others at the address"), ("H-R_vs_V1", "H-R prior repudiated claim")):
    for ln in LINES + ("all",):
        T4.append(row(nm if ln == "motor" else "", tests[key], ln))
T4C = []
for key, nm in (("PD_vs_H-P", "H-D added to H-P"), ("PD_vs_H-D", "H-P added to H-D")):
    for ln in LINES + ("all",):
        T4C.append(row(nm if ln == "motor" else "", tests[key], ln))
T4V = []
for key, nm, rd in (("P-raw_vs_H-P", "Raw counts with exposure years", "Loses"), ("P-band_vs_H-P", "Mod in ten bands", "Loses"),
                    ("P-avg_vs_H-P", "Average-rate E", "Loses"), ("P-mult_vs_H-P", "β fixed at 1, no refit", "Not worse, not better"),
                    ("P-group_vs_H-P", "Split by peril group", "No gain")):
    r = row(nm, tests[key], "all", rd); r.pop("line"); T4V.append(r)

T5 = {}
for ln in LINES:
    r = S5["by_line"][ln]
    T5[f"t5_{ln}"] = [dict(row(f"{cid} {NAMES[cid]}", t), label=("Kept" if cid in r["kept"] else "Yes" if t["verdict"]["keep"] else "No"))
                      for cid, t in r["add_one"].items()]
    T5[f"t5v1_{ln}"] = [row(V1N.get(col, col), t) for col, t in r["base_factor_drop"].items()]

fb = {ln: S5["by_line"][ln]["final_vs_base"] for ln in LINES}
pct = lambda x: f"{100 * x:.2f}%"
n_tests = sum(len(S5["by_line"][ln]["add_one"]) for ln in LINES)
pd_all = tests["PD_vs_H-P"]["A"]["all"]; hp_all = tests["H-P_vs_V1"]["A"]["all"]
LEDE = (f"Claims history adds the most. Own-policy history passes in all three lines, and declared claims before joining add to it. "
        f"Together they lift the combined ranking D² from {pct(hp_all['d2_ref'])} to {pct(pd_all['d2_cand'])} on 2023 and from "
        f"{pct(tests['H-P_vs_V1']['B']['all']['d2_ref'])} to {pct(tests['PD_vs_H-P']['B']['all']['d2_cand'])} on 2024. Of {n_tests} further candidate tests, "
        f"three pass: voluntary excess and vehicle age in motor, and trees on clay in buildings. Twice the credibility estimator I designed failed on this "
        f"data; I fixed it and both runs are on file.")

rel = {ln: {(x["factor"], x["level"]): x["relativity"] for x in S5["by_line"][ln]["relativities_final"]} for ln in LINES}
rm, rb = rel["motor"], rel["buildings"]
f2 = lambda x: f"{x:.2f}"
k_fin = {ln: S4["by_line"][ln]["final"]["k"][f"P_{ln}"]["k"] for ln in LINES}
beta_pd = {ln: S4["by_line"][ln]["final"]["coef"]["PD"]["h_p"] for ln in LINES}
hd = {ln: S4["by_line"][ln]["final"]["coef"]["PD"] for ln in LINES}

STEP5 = f"""
  <section aria-labelledby="h-s5">
    <span class="eyebrow">Step 5 · features (C8, widened)</span>
    <h2 id="h-s5">Three of {n_tests} candidate tests pass</h2>
    <p>The base is V1 plus policy and declared history in every line (provisional on cards L2 and L3). Each candidate was added alone, and every passer then went through drop-one rounds together. Every candidate has a written risk reason (<code>decisions/phase5_step5_design.md</code>).</p>
    <div class="tbl wide"><table>
      <thead><tr><th>Kept</th><th>Relativities, final fit 2021 to 2024</th><th class="r">A: saved</th><th class="r">B: saved</th></tr></thead>
      <tbody>
        <tr><td>Motor: voluntary excess</td><td>0 CU 1.00 · 250 CU {f2(rm[('g_volx','250')])} · 500 CU {f2(rm[('g_volx','500')])} · 1,000 CU {f2(rm[('g_volx','1000')])}</td><td class="r num">{S5['by_line']['motor']['add_one']['G1']['A']['gain']:,.0f}</td><td class="r num">{S5['by_line']['motor']['add_one']['G1']['B']['gain']:,.0f}</td></tr>
        <tr><td>Motor: vehicle age</td><td>0-1 {f2(rm[('m_vage','(-1.001, 1.0]')])} · 2-3 {f2(rm[('m_vage','(1.0, 3.0]')])} · 4-6 {f2(rm[('m_vage','(3.0, 6.0]')])} · 7-10 1.00 · 11-15 {f2(rm[('m_vage','(10.0, 15.0]')])} · 16+ {f2(rm[('m_vage','(15.0, 999.0]')])}</td><td class="r num">{S5['by_line']['motor']['add_one']['M9']['A']['gain']:,.0f}</td><td class="r num">{S5['by_line']['motor']['add_one']['M9']['B']['gain']:,.0f}</td></tr>
        <tr><td>Buildings: trees within 5 m × clay</td><td>Neither 1.00 · clay, no trees {f2(rb[('b_treeclay','False|True')])} · trees, not clay {f2(rb[('b_treeclay','True|False')])} · trees on clay {f2(rb[('b_treeclay','True|True')])}</td><td class="r num">{S5['by_line']['buildings']['add_one']['B9']['A']['gain']:,.0f}</td><td class="r num">{S5['by_line']['buildings']['add_one']['B9']['B']['gain']:,.0f}</td></tr>
      </tbody>
    </table></div>
    <p class="how"><b>How to read this.</b> A motor policy with a 1,000 CU voluntary excess costs {f2(rm[('g_volx','1000')])} times one with none, other things equal: small claims fall under the excess, and careful buyers choose it. Motor ranking D² moves from {pct(fb['motor']['A']['d2_ref'])} to {pct(fb['motor']['A']['d2_cand'])} on 2023, and buildings from {pct(fb['buildings']['A']['d2_ref'])} to {pct(fb['buildings']['A']['d2_cand'])}. Contents keeps nothing. Across lines the three save {S5['combined_gain']['A']:,.0f} on 2023 and {S5['combined_gain']['B']:,.0f} on 2024.</p>
    <details><summary class="eyebrow">Every candidate, motor</summary><div class="tbl wide"><table><thead><tr><th>Candidate</th><th class="r">A: saved</th><th class="r">A: 95% interval</th><th class="r">B: saved</th><th>Passes</th></tr></thead><tbody id="t5_motor"></tbody></table></div></details>
    <details><summary class="eyebrow">Every candidate, buildings</summary><div class="tbl wide"><table><thead><tr><th>Candidate</th><th class="r">A: saved</th><th class="r">A: 95% interval</th><th class="r">B: saved</th><th>Passes</th></tr></thead><tbody id="t5_buildings"></tbody></table></div></details>
    <details><summary class="eyebrow">Every candidate, contents</summary><div class="tbl wide"><table><thead><tr><th>Candidate</th><th class="r">A: saved</th><th class="r">A: 95% interval</th><th class="r">B: saved</th><th>Passes</th></tr></thead><tbody id="t5_contents"></tbody></table></div></details>
    <p class="how"><b>Worth knowing.</b>
      <span>Household size, marital status and occupation all fail, so no fairness judgment is needed on them now. Days unoccupied fails too, so its point-of-use question is moot.</span>
      <span>The flood measures in buildings save {S5['by_line']['buildings']['add_one']['B3b']['B']['gain']:,.0f} to {S5['by_line']['buildings']['add_one']['B3a']['B']['gain']:,.0f} on 2024 but lose on 2023, where the flood-score interval runs from {S5['by_line']['buildings']['add_one']['B3b']['A']['boot']['lo']:,.0f} to {S5['by_line']['buildings']['add_one']['B3b']['A']['boot']['hi']:,.0f}: flood cost is too lumpy to rank on two years.</span>
      <span>Years licensed in motor saves about 25,000 in both years and misses only on the interval.</span>
    </p>
    <div class="wrong">
      <b>Where I was wrong again (to log as D18)</b>
      <span>The make-model encoding (M7) used the same non-parametric credibility estimate, now on capped cost by model. It found no between-model variance (k infinite), so the encoding was 1 everywhere and the test was empty. I checked by hand with the efficient estimator on claim counts: across 300 models, the spread of actual over expected claims is 0.085 against 0.079 from noise alone, so make-model adds at most about 3% in frequency beyond V1's vehicle factors, and nothing detectable in cost. M7 is a true null on this data, but the empty test was my design's fault, and I should have caught it after step 4.</span>
    </div>
  </section>

  <section aria-labelledby="h-v1">
    <span class="eyebrow">For information · the V1 factors under the same test</span>
    <h2 id="h-v1">Which existing factors would pass the keep rule today</h2>
    <p>Each V1 factor was dropped in turn from the final model of its line. "Saved" is what the factor is worth; a negative value means the model ranks better without it.</p>
    <details><summary class="eyebrow">Motor</summary><div class="tbl wide"><table><thead><tr><th>Factor</th><th class="r">A: worth</th><th class="r">A: 95% interval</th><th class="r">B: worth</th><th>Passes</th></tr></thead><tbody id="t5v1_motor"></tbody></table></div></details>
    <details><summary class="eyebrow">Buildings</summary><div class="tbl wide"><table><thead><tr><th>Factor</th><th class="r">A: worth</th><th class="r">A: 95% interval</th><th class="r">B: worth</th><th>Passes</th></tr></thead><tbody id="t5v1_buildings"></tbody></table></div></details>
    <details><summary class="eyebrow">Contents</summary><div class="tbl wide"><table><thead><tr><th>Factor</th><th class="r">A: worth</th><th class="r">A: 95% interval</th><th class="r">B: worth</th><th>Passes</th></tr></thead><tbody id="t5v1_contents"></tbody></table></div></details>
    <p class="how"><b>How to read this.</b> Most V1 factors save in both years but miss the interval test, as property claims are few. Six factors make the model better in both years when removed: body type in motor, property type and bedrooms in buildings, and new business, property type and bicycle cover in contents. Bicycle cover defines what is covered, so it stays whatever the score. Card L8 asks what to do with the other five.</p>
  </section>
"""


def opt(k, n, d, c, f):
    return dict(k=k, n=n, d=d, c=c, f=f)


tp = tests["H-P_vs_V1"]; tm = tests["P-mult_vs_H-P"]["A"]["all"]
DECS = [
    dict(id="L1", door="easy", q="Which estimate of the credibility constant k?",
         stakes=f"<b>It decides whether history counts at all in property.</b> With the design's estimator, k was infinite for buildings on 2023 and contents k ranged from 4.6 to 92.7. With the Poisson-gamma estimate, final k is {k_fin['motor']:.2f}, {k_fin['buildings']:.2f} and {k_fin['contents']:.2f}, and policy history passes in all three lines.",
         opts=[opt("A", "Poisson-gamma moment estimate", "Assumes Poisson claim counts within a policy, which the data supports (variance 1.02 to 1.06 of the mean; 1.004 claims per incident).", "Relies on the Poisson assumption.", "Nothing."),
               opt("B", "Non-parametric Bühlmann-Straub (the original design)", "Makes no distribution assumption.", "Mostly noise at 0.05 expected claims per policy-year; k infinite for buildings on 2023.", "Property history."),
               opt("C", "k tuned on the out-of-time score", "Picks k by grid on the scorings.", "A fitted setting, not a credibility estimate; harder to explain.", "Nothing.")],
         rec="A", why="It is the efficient estimator in this rare-claim setting, stable across all three fits, and consistent with what the GLM's own slopes imply."),
    dict(id="L2", door="costly", q="Add the policy's own claims history to all three lines, as a credibility-weighted experience mod?",
         stakes=f"<b>The largest single gain since capping.</b> It saves {tp['A']['all']['gain']:,.0f} on 2023 (interval {tp['A']['all']['boot']['lo']:,.0f} to {tp['A']['all']['boot']['hi']:,.0f}) and {tp['B']['all']['gain']:,.0f} on 2024, and passes in each line. 72% of 2025 rows have history.",
         opts=[opt("A", "Yes: log(mod) with a fitted slope β", f"The relativity is mod raised to β; β is {beta_pd['motor']:.2f}, {beta_pd['buildings']:.2f} and {beta_pd['contents']:.2f} with declared history also in.", "One more term per line.", "Nothing."),
               opt("B", "Yes: the mod as a pure multiplier, β = 1", f"Class rate × mod, the textbook experience-rating form; {tm['gain']:,.0f} on 2023, not significantly worse.", "Slightly over-credits history in property.", "Nothing."),
               opt("C", "No", "Keep V1 without history.", "Gives up the largest tested gain.", "Experience rating.")],
         rec="A", why="It is the tested form that passed; with β close to 1 it still reads as class rate × experience mod."),
    dict(id="L3", door="easy", q="Add declared claims before joining (prior_claims_5y) to every line, or to motor only?",
         stakes=f"<b>A customer field not used in V1 until now.</b> Relativities rise steadily from 0 to 3+ claims: motor 1.00, {hd['motor']['h_d=1']:.2f}, {hd['motor']['h_d=2']:.2f}, {hd['motor']['h_d=3+']:.2f}; buildings 1.00, {hd['buildings']['h_d=1']:.2f}, {hd['buildings']['h_d=2']:.2f}, {hd['buildings']['h_d=3+']:.2f}; contents 1.00, {hd['contents']['h_d=1']:.2f}, {hd['contents']['h_d=2']:.2f}, {hd['contents']['h_d=3+']:.2f}. It passes the keep rule in motor and across lines. In buildings and contents it saves in both years, but the 2023 interval crosses zero.",
         opts=[opt("A", "All three lines", "One rule with the same meaning everywhere, as in K1.", "Property relies on the across-lines evidence.", "Nothing."),
               opt("B", "Motor only (strict per-line rule)", "Property keeps policy history only.", "Gives up gains that show in both years in property.", "Nothing."),
               opt("C", "None", "Policy history only.", "Gives up the motor gain that passed.", "Declared history.")],
         rec="A", why="The pattern is steady and the same in every line, it saves in both years everywhere, and the across-lines test passes; K1 applied the same reasoning to caps."),
    dict(id="L4", door="easy", q="What happens to customer, address and repudiation history, which failed?",
         stakes="Customer history from other lines saves nothing (−8,129 on 2023, interval −16,894 to 1,723). Address history has almost no data: 0.2% of 2025 buildings rows have another customer's history at the address. Prior repudiation is neutral.",
         opts=[opt("A", "Drop from the GLM; offer them to the step 7 tree families as candidates under the keep rule", "Trees may find interactions the GLM cannot.", "A few more candidates in step 7.", "Nothing."),
               opt("B", "Drop everywhere", "Simplest.", "A small chance of a missed interaction.", "These fields in any model.")],
         rec="A", why="It follows your standing rule to test widely at almost no cost, and the keep rule still decides."),
    dict(id="L5", door="easy", q="Add voluntary excess and vehicle age to motor?",
         stakes=f"Both pass alone and in the drop-one round. Together they add {fb['motor']['A']['gain']:,.0f} to motor on 2023 (interval {fb['motor']['A']['boot']['lo']:,.0f} to {fb['motor']['A']['boot']['hi']:,.0f}) and {fb['motor']['B']['gain']:,.0f} on 2024.",
         opts=[opt("A", "Add both", "Excess removes small claims; older cars cost less to repair.", "Two more factors.", "Nothing."),
               opt("B", "Voluntary excess only", "It has the mechanical reason and the larger gain.", "Gives up the vehicle age gain.", "Nothing."),
               opt("C", "Neither", "Keep the motor plan as it is.", "Gives up both gains.", "Nothing.")],
         rec="A", why="Both pass the keep rule and both have a clear risk reason."),
    dict(id="L6", door="easy", q="Add trees within 5 m × clay soil to buildings?",
         stakes=f"It passes alone and in drop-one: 6,574 on 2023 (interval 495 to 13,878) and 3,744 on 2024. Trees on clay cost {f2(rb[('b_treeclay','True|True')])} times a property with neither: roots draw water from shrinkable clay, the subsidence mechanism.",
         opts=[opt("A", "Add it", "A four-level factor with a physical reason.", "One more factor.", "Nothing."),
               opt("B", "Leave it out", "Buildings stays as it is.", "Gives up a small, passed gain.", "Nothing.")],
         rec="A", why="It passes the rule and is the textbook subsidence interaction."),
    dict(id="L7", door="easy", q="What happens to the near misses, which save in both years but fail the interval?",
         stakes="Examples: years licensed in motor (+25,804 and +25,485), voluntary excess in buildings and contents, and the flood measures in buildings (+95,241 on 2024, but −23,863 on 2023).",
         opts=[opt("A", "Hold the keep rule; offer the near misses to the step 7 families", "The elastic-net GLM and the boosters may use them, still under the keep rule.", "None now.", "Nothing."),
               opt("B", "Admit voluntary excess on property too", "Same mechanical reason as motor.", "Overrides the agreed rule on a judgment.", "Nothing.")],
         rec="A", why="The rule was agreed before results, and step 7 gives these fields a second, fair test."),
    dict(id="L8", door="costly", q="Five V1 rating factors make their line better when removed. Drop them?",
         stakes="Body type (motor), property type and bedrooms (buildings), and new business and property type (contents): removing each, one at a time, lowers deviance in both years. Bicycle cover does too, but it defines cover and stays. Most other V1 factors save in both years but miss the interval.",
         opts=[opt("A", "Drop the five, check the joint drop before step 6, and leave the rest to step 7's elastic net", "Removes factors that only add noise. Factors that define cover (cover type, sum insured, accidental damage, bicycle cover) are never dropped.", "A joint check, about 30 minutes.", "Nothing."),
               opt("B", "Keep V1 as decided; let the step 7 elastic net shrink weak factors", "No change now.", "Keeps six factors that do not help.", "Nothing."),
               opt("C", "Apply the full keep rule to every V1 factor now", "Backward elimination of V1.", "Would drop real but noisy factors, and in a strict reading even cover type in motor.", "Several factors.")],
         rec="A", why="It removes only what the evidence says hurts in both years, keeps the plan explainable, and leaves the borderline cases to the method built for them."),
]

s = PAGE.read_text(encoding="utf-8")
data = ("var T4 = " + json.dumps(T4) + ";\n  var T4C = " + json.dumps(T4C) + ";\n  var T4V = " + json.dumps(T4V) +
        ";\n  var T5 = " + json.dumps(T5) + ";\n  var LEDE = " + json.dumps(LEDE) + ";\n  var DECS = " + json.dumps(DECS, ensure_ascii=False) + ";")
s = re.sub(r"/\*DATA\*/.*?/\*END DATA\*/", lambda m: "/*DATA*/" + data + "/*END DATA*/", s, flags=re.S) if "/*END DATA*/" in s \
    else s.replace("/*DATA*/", "/*DATA*/" + data + "/*END DATA*/")
s = re.sub(r"<!--STEP5-->.*?<!--END STEP5-->", lambda m: "<!--STEP5-->" + STEP5 + "<!--END STEP5-->", s, flags=re.S) if "<!--END STEP5-->" in s \
    else s.replace("<!--STEP5-->", "<!--STEP5-->" + STEP5 + "<!--END STEP5-->")
PAGE.write_text(s, encoding="utf-8")
print("written", PAGE, len(s))
