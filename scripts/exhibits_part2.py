"""Part 2 exhibits for the write-up and the video, built only from committed evidence files (written 29 October).

Every number in `docs/writeup/exhibits/` comes from one of:
  - outputs/markets/part2/*.json and outputs/part2_decision_evidence*.json (committed evidence files);
  - a committed decision note, where a figure was only ever recorded there: the script checks the quoted text is in
    the file and fails if it is not (`quoted`);
  - for Appendix A, the 20 reference policies and the expense table (competition files, not committed) and each
    policy's prediction from the submission file (not committed). Those 20 rows are written to
    outputs/markets/part2/part2_exhibits.json, which is committed, so the build-ups trace to a file.

The final file: motor priced by peril (K16, H-MP (a)), buildings at the check file's shape (H-F (b)), the
new-business factor at the claim-count credibility standard (H-NS (a), exchange D49).

Run: MARKET=part2 .venv/Scripts/python.exe -I scripts/exhibits_part2.py
Writes docs/writeup/exhibits/part2_final.md, docs/writeup/exhibits/*.png and outputs/markets/part2/part2_exhibits.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.evidence import write_evidence

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "outputs" / "markets" / "part2"
RAW = ROOT / "data" / "raw_part2" / "emerging"
DOCS = ROOT / "docs" / "writeup" / "exhibits"
FILES = {"final": "part2_v2_nb-cred_wx-row_mprem_mperil_bchk_all"}             # H-MP (a), H-NS (a)
PP_FILE = "part2_v2_nb-pp_wx-row_mprem_mperil_bchk_all"                          # the same shape, pure-premium standard
LABEL = {"final": "the final file (motor by peril, claim-count new-business standard)"}
LINES = ("motor", "buildings", "contents")
# the expense table as the actuary read it (OP-T (a), OP-B (a), P2-E (d)); checked against the competition file below
EXPENSES = {"fixed_expense_per_policy": 85.0, "variable_expense": 0.15, "commission_direct": 0.0, "commission_broker": 0.20,
            "premium_tax": 0.08, "cat_reinsurance": 0.12, "target_profit_margin": 0.08}
PROPERTY = ("buildings", "contents")


# ------------------------------------------------------------------------------------------------ reading evidence
def ev(name: str) -> dict:
    d = json.loads((EV / f"{name}.json").read_text(encoding="utf-8"))
    return d[name] if name in d else d[next(k for k in d if k != "meta")]


def quoted(path: str, text: str) -> str:
    """A figure recorded only in a committed note: fail unless the note holds it verbatim."""
    body = (ROOT / path).read_text(encoding="utf-8")
    assert text in body, f"{path} does not contain {text!r}"
    return text


def gain(x: float) -> str:
    return ("+" if x >= 0 else "−") + f"{abs(x):,.0f}"


def ci(d: dict) -> str:
    return f"[{gain(d['lo']).lstrip('+')}, {gain(d['hi']).lstrip('+')}]"


def signed(x: float, fmt: str) -> str:
    """A signed number with a true minus sign, e.g. signed(-0.116, ".1%") -> "−11.6%"."""
    return ("+" if x >= 0 else "−") + format(abs(x), fmt)


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}%"


def table(head: list, rows: list) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ the exhibits
def ladder() -> dict:
    """Panel 1: validation on the non-hold-out customers (rounds 1 to 3). Panel 2: every customer, after the hold-out."""
    r2, t, mp, bo = ev("part2_candidates_round2"), ev("part2_tuning"), ev("part2_motor_peril"), ev("part2_boosted")
    prem_v1 = quoted("decisions/part2_m2_design.md", "+2,844 [44, 5,469]")
    prem_v2 = quoted("decisions/part2_m2_design.md", "+5,981 [1,547, 10,441]")
    p1 = [
        dict(step="K0: the check file's shape (Part 1 relativities, emerging level)", v1="", v2="",
             d2=(r2["d2"]["V1"]["K0"], r2["d2"]["V2"]["K0"]), kept="incumbent"),
        dict(step="K9 + K7b: gender layer in motor, buildings by peril with address hazards, Part 1's tree in contents, "
                  "each row's own-year weather",
             v1=gain(r2["K9+K7b"]["V1"]["gain"]), v2=f"{gain(r2['K9+K7b']['V2']['gain'])} {ci(r2['K9+K7b']['V2'])}",
             d2=(r2["d2"]["V1"]["K9+K7b"], r2["d2"]["V2"]["K9+K7b"]), kept="yes (M2-W)"),
        dict(step="I: + Part 1's premium model applied to emerging fields, motor only (re-tested on motor after K11 failed pooled)", v1=prem_v1, v2=prem_v2,
             d2=(t["d2"]["V1"]["I"], t["d2"]["V2"]["I"]), kept="yes (Y1, motor)"),
    ]
    p2 = [
        dict(step="K0: the check file's shape", v1="", v2="", d2=(bo["d2"]["V1"]["K0"], bo["d2"]["V2"]["K0"]), kept="reference"),
        dict(step="F: I with buildings back at K0 (H-F (b), after the hold-out)", v1="", v2="",
             d2=(bo["d2"]["V1"]["F"], bo["d2"]["V2"]["F"]), kept="yes (H-F (b))"),
        dict(step="K16: F with motor priced by peril (after four gap reviews)",
             v1=f"{gain(mp['p2v']['motor']['V1']['gain'])} {ci(mp['p2v']['motor']['V1'])}",
             v2=f"{gain(mp['p2v']['motor']['V2']['gain'])} {ci(mp['p2v']['motor']['V2'])}",
             d2=(mp["d2"]["V1"]["K16"], mp["d2"]["V2"]["K16"]), kept="final (H-MP (a))"),
    ]
    return dict(panel1=p1, panel2=p2)


def candidates() -> list:
    """Everything tested against the incumbent, deviance saved, pooled and by line."""
    c1, c2, c3 = ev("part2_candidates"), ev("part2_candidates_round2"), ev("part2_candidates_round3")
    t, rr, bo, mp = ev("part2_tuning"), ev("part2_motor_rerate"), ev("part2_boosted"), ev("part2_motor_peril")
    rows = [
        ("Motor by peril, no premium term (K1)", c1["K1"]),
        ("Every peril separately, large-loss perils as frequency × severity, address hazards for fire and flood (K3)", c1["K3"]),
        ("Part 1's tree in contents (K4)", c1["K4_tree_contents"]),
        ("Gender in motor (K5)", c1["K5_gender"]),
        ("Prior-year claims, customer across lines (K8)", c2["K8_history"]),
        ("K9 + K7b: K3 in buildings, K4 in contents, K5 in motor, each row's own-year weather", c2["K9+K7b"]),
        ("Part 1's premium model applied to emerging fields, every line (K11)", c3["K11_premium"]),
        ("Credibility re-rate of every factor, penalised GLM (K12)", c3["K12_penalised"]),
        ("Region and occupation, credibility-encoded (K13)", c3["K13_cred_codes"]),
        ("Settings tuned on Part 1 rows", t["T_against_I"]),
        ("Credibility re-rate per motor factor (K14)", rr["K14_cred"]),
        ("Boosted correction learned on emerging rows (K15)", {"V1": bo["pooled"]["V1"], "V2": bo["pooled"]["V2"],
                                                                "passes": bo["pooled"]["passes"]}),
        ("Motor by peril with the premium term (K16; motor interval)", {"V1": {**mp["p2v"]["motor"]["V1"], "per_line": {"motor": mp["p2v"]["motor"]["V1"]["gain"]}},
                                                       "V2": {**mp["p2v"]["motor"]["V2"], "per_line": {"motor": mp["p2v"]["motor"]["V2"]["gain"]}},
                                                       "passes": mp["p2v"]["motor"]["passes"]}),
    ]
    out = []
    for name, d in rows:
        pl = d["V2"].get("per_line") or {}
        if name.startswith("Boosted"):
            pl = {k: bo["per_line"][k]["V2"]["gain"] for k in LINES}
        out.append(dict(candidate=name, v1=d["V1"]["gain"], v2=d["V2"]["gain"], v2_lo=d["V2"]["lo"], v2_hi=d["V2"]["hi"],
                        v2_motor=pl.get("motor"), v2_buildings=pl.get("buildings"), v2_contents=pl.get("contents"),
                        passes=bool(d.get("passes"))))
    return out


def holdout() -> dict:
    h, mp = ev("part2_holdout"), ev("part2_motor_peril")
    reg = "decisions/judgment_register.md"
    every = dict(motor=(quoted(reg, "motor +10,773 and +19,797"), "+10,773", "+19,797"),
                 buildings=(quoted(reg, "buildings −12,834 and −2,754"), "−12,834", "−2,754"),
                 contents=(quoted(reg, "contents +11,415 and +14,818"), "+11,415", "+14,818"))
    out = {}
    for k in ("H1", "H2"):
        c = h[k]["compare"]["I_vs_K0"]
        out[k] = dict(per_line={l: c["per_line"][l] for l in LINES}, pooled=c["gain"], lo=c["lo"], hi=c["hi"],
                      claims={l: h[k]["scored"][l]["claims"] for l in LINES},
                      d2={m: h[k]["d2"][m]["pooled"] for m in ("K0", "I")},
                      k16_motor=mp["holdout"][k]["motor"])
    out["every_customer"] = {l: dict(out_of_time=v[1], cross_validation=v[2]) for l, v in every.items()}
    return out


def level(variant: str) -> dict:
    f = ev(FILES[variant])["by_line"]
    return {l: dict(rows=f[l]["rows"], indication=f[l]["indication_capped_lc"], excess=f[l]["excess_factor"],
                    off_balance=f[l]["off_balance"], nb_ae=f[l]["new_business"]["actual_over_model"],
                    nb_claims=f[l]["new_business"]["claims"], nb_z=f[l]["new_business"]["z"],
                    nb_factor=f[l]["new_business"]["factor"], renewal=f[l]["new_business"]["renewal_rebalance"],
                    weather=f[l]["weather"]["factor_2025_mean"], weather_p05_p95=f[l]["weather"]["factor_2025_p05_p95"],
                    weather_mode="one factor per line" if l == "buildings" else "each row's own year",
                    predicted=f[l]["predicted"], written=f[l]["written"], ratio=f[l]["predicted_over_written"],
                    per_year=f[l]["mean_per_exposure"]) for l in LINES}


def nb_credibility(variant: str) -> dict:
    """The factor in the file (claim-count standard) beside the same shape at the pure-premium standard."""
    n, c, f = ev("part2_nb_standard"), ev(FILES[variant])["by_line"], ev(PP_FILE)["by_line"]
    shape = "F16"
    rows = {}
    for l in LINES:
        a, b = f[l]["new_business"], c[l]["new_business"]
        rows[l] = dict(claims=a["claims"], ae=a["actual_over_model"], cv2=a["severity_cv2"], z_count=b["z"], z_pp=a["z"],
                       f_count=b["factor"], f_pp=a["factor"], full_standard_pp=1082 * (1 + a["severity_cv2"]))
    vs = n["vs_count"][f"{shape}|pp"]["motor"]
    d2 = {k: (n["d2"]["V1"][f"{shape}|{k}"], n["d2"]["V2"][f"{shape}|{k}"]) for k in ("none", "pp", "count", "full")}
    log = "ai/ai_log.json"
    board = dict(count=quoted(log, "Credibility-weighted 0.15312"), none=quoted(log, "no adjustment 0.15080"),
                 full=quoted(log, "full emerging figure 0.14646"),
                 final_pp=quoted("decisions/part2_m2_design.md", "scored 0.15657"),
                 final_count=quoted("decisions/part2_m2_design.md", "public against 0.15712"))
    return dict(rows=rows, pp_vs_count_motor=vs, d2=d2, board={k: v.split()[-1] for k, v in board.items()},
                by_year_motor_ae=quoted("decisions/part2_m2_design.md", "(1.49 in 2023, 0.85 in 2024)"))


def expense_table() -> dict:
    e = pd.read_parquet(RAW / "expenses.parquet").set_index("item")["emerging"]
    for k, v in EXPENSES.items():
        assert abs(float(e[k]) - v) < 1e-12, (k, e[k], v)
    assert pd.isna(e["commission_comparison_site"]), "the table now gives a comparison-site rate: P2-E needs a look"
    return dict(EXPENSES)


def buildups(variant: str) -> list:
    """Appendix A (OP-T (a), OP-B (a)): annual risk premium, cat load on property, fixed expense, divided by what is
    left of each premium after variable expense, commission, premium tax and profit."""
    x = expense_table()
    ref = pd.read_parquet(RAW / "reference_policies.parquet")
    sub = pd.read_csv(EV / "submissions" / f"{FILES[variant]}.csv", dtype={"row_id": str}).set_index("row_id")["prediction"]
    out = []
    for r in ref.itertuples():
        assert r.channel in ("broker", "direct"), r.channel          # P2-E (d): no comparison-site policy among the 20
        pred = float(sub[str(r.row_id)])
        risk = pred / r.exposure
        cat = x["cat_reinsurance"] * risk if r.line in PROPERTY else 0.0
        comm = x[f"commission_{r.channel}"]
        div = 1 - x["variable_expense"] - comm - x["premium_tax"] - x["target_profit_margin"]
        office = (risk + cat + x["fixed_expense_per_policy"]) / div
        desc = (f"{r.driver_age_band}, {r.cover_type.replace('_', ' ')}, {r.fuel_type}" if r.line == "motor" else
                f"{r.construction_type}, sum insured {r.sum_insured:,.0f}" if r.line == "buildings" else
                f"{r.tenure}, sum insured {r.sum_insured:,.0f}")
        out.append(dict(row_id=str(r.row_id), line=r.line, channel=r.channel, new_business=bool(r.new_business),
                        description=desc, exposure=float(r.exposure), prediction=pred, risk_premium=risk, cat_load=cat,
                        fixed=x["fixed_expense_per_policy"], divisor=div, office_premium=office,
                        written_premium=float(r.written_premium), office_over_written=office / r.written_premium,
                        risk_over_written=risk / r.written_premium))
    return out


def gender(variant: str) -> dict:
    return ev("part2_doc_evidence")["gender"][variant]


def weather() -> dict:
    return ev("part2_doc_evidence")["weather"]


# ------------------------------------------------------------------------------------------------ rendering
def render(variant: str, X: dict) -> str:
    L, nb, B, G, W, H = X["level"], X["nb"], X["buildups"], X["gender"], X["weather"], X["holdout"]
    s = [f"# Part 2 exhibits: {LABEL[variant]}",
         "",
         f"*Generated by `scripts/exhibits_part2.py` from committed evidence files; do not edit by hand. File: `{FILES[variant]}.csv`.*", ""]

    # ---- ladder
    lad = X["ladder"]
    s += ["## Exhibit P2-1. The evidence ladder", "",
          "Panel 1: validation on the 85% of customers outside the hold-out (rounds 1 to 3). Deviance saved against the step above; positive is better.", "",
          table(["Step", "Out of time (fit 2023, score 2024)", "Cross-validation [95%]", "Pooled D², out of time / cross-validation", "Kept"],
                [[r["step"], r["v1"] or "", r["v2"] or "", f"{r['d2'][0]:.4f} / {r['d2'][1]:.4f}", r["kept"]] for r in lad["panel1"]]),
          "", "Panel 2: every customer, after the hold-out was opened once. K16's gains are in motor, against F.", "",
          table(["Step", "Out of time [95%]", "Cross-validation [95%]", "Pooled D², out of time / cross-validation", "Status"],
                [[r["step"], r["v1"] or "", r["v2"] or "", f"{r['d2'][0]:.4f} / {r['d2'][1]:.4f}", r["kept"]] for r in lad["panel2"]]), ""]
    k9, k16 = lad["panel1"][1], lad["panel2"][2]
    s += [f"> **How to read this.** D² is the share of a flat rate's deviance the model removes, each line at its actual level. "
          f"The combination that passed saves {k9['v2'].split(' [')[0].lstrip('+')} of deviance in cross-validation and moves pooled D² from "
          f"{lad['panel1'][0]['d2'][1]:.4f} to {k9['d2'][1]:.4f}; its interval only just clears zero. Panel 2 is on a different set of rows "
          f"(every customer), so its D² is not comparable with panel 1's. There, motor by peril moves cross-validated D² from "
          f"{lad['panel2'][1]['d2'][1]:.4f} to {k16['d2'][1]:.4f}.", ""]

    # ---- candidates
    C = X["candidates"]
    s += ["## Exhibit P2-2. Every candidate change, deviance saved against the incumbent", "",
          table(["Candidate", "Out of time", "Cross-validation [95%]", "Motor / buildings / contents (cross-validation)", "Passes P2-V"],
                [[c["candidate"], "0" if abs(c["v1"]) < 0.5 else gain(c["v1"]), f"{gain(c['v2'])} {ci({'lo': c['v2_lo'], 'hi': c['v2_hi']})}",
                  " / ".join("·" if c[f"v2_{l}"] is None else "0" if abs(c[f"v2_{l}"]) < 0.5 else gain(c[f"v2_{l}"]) for l in LINES),
                  "yes" if c["passes"] else "no"] for c in C]), ""]
    k1 = next(c for c in C if c["candidate"].startswith("Motor by peril, no"))
    s += [f"> **How to read this.** P2-V keeps a change only if it saves deviance out of time and in cross-validation, with the "
          f"cross-validation interval above zero. Rounds 1 to 3 judged the pooled interval; K15 to K18 were judged line by line, "
          f"as the actuary judged buildings at H-F. Motor by peril without the premium term (K1) saved {gain(k1['v2_motor'])} in motor in "
          f"cross-validation but {gain(k1['v2'])} pooled, because buildings swung the pooled figure; it was rejected on the pooled rule. "
          f"The rule was not applied the same way throughout: the premium term (K11) failed pooled and was then re-tested in motor alone, "
          f"where it passed, a line chosen after its per-line gains were seen. From the hold-out on, every line was judged on its own.", ""]

    # ---- hold-out
    s += ["## Exhibit P2-3. The hold-out, opened once, by line", "",
          "The frozen model (I) against the check file's shape (K0), deviance saved; positive means I is better.", "",
          table(["Reading", "Motor", "Buildings", "Contents", "Pooled [95%]"],
                [["H1: hold-out customers' 2024 rows, fitted on 2023", *[gain(H["H1"]["per_line"][l]) for l in LINES],
                  f"{gain(H['H1']['pooled'])} {ci(H['H1'])}"],
                 ["H2: every hold-out row, fitted on all other customers", *[gain(H["H2"]["per_line"][l]) for l in LINES],
                  f"{gain(H['H2']['pooled'])} {ci(H['H2'])}"],
                 ["Every customer, fit one year and score the next", *[H["every_customer"][l]["out_of_time"] for l in LINES], ""],
                 ["Every customer, cross-validated", *[H["every_customer"][l]["cross_validation"] for l in LINES], ""],
                 ["Claims scored in H1 / H2", *[f"{H['H1']['claims'][l]:,} / {H['H2']['claims'][l]:,}" for l in LINES], ""]]), "",
          f"Motor by peril (K16) against the final file's motor on the same hold-out: H1 {gain(H['H1']['k16_motor']['gain'])} "
          f"{ci(H['H1']['k16_motor'])}, H2 {gain(H['H2']['k16_motor']['gain'])} {ci(H['H2']['k16_motor'])}.", "",
          f"> **How to read this.** Pooled, the frozen model fell short of K0 on the hold-out ({gain(H['H1']['pooled'])} and "
          f"{gain(H['H2']['pooled'])}), inside noise. By line, motor gains in every reading and contents in three of four; buildings "
          f"loses in three of four and, across every customer, shows no gain; only {H['H1']['claims']['buildings']} buildings claims sit in H1. "
          f"So the actuary kept motor and contents and returned buildings to K0 (H-F (b)), a decision made after the hold-out was opened.", ""]

    # ---- level chain
    s += ["## Exhibit P2-4. The level chain by line (2025)", "",
          table(["", *[l.capitalize() for l in LINES]],
                [["Indication, capped cost per policy year at normal weather", *[f"{L[l]['indication']:.1f}" for l in LINES]],
                 ["Excess factor (cost above the cap)", *[f"{L[l]['excess']:.3f}" for l in LINES]],
                 ["Off-balance, shape to indication", *[f"{L[l]['off_balance']:.3f}" for l in LINES]],
                 ["New-business factor (Z); rebalance on every row", *[f"{L[l]['nb_factor']:.3f} ({L[l]['nb_z']:.2f}); {L[l]['renewal']:.3f}" for l in LINES]],
                 ["2025 weather, mean (5th to 95th percentile)", *[f"{L[l]['weather']:.3f} ({L[l]['weather_p05_p95'][0]:.3f} to {L[l]['weather_p05_p95'][1]:.3f})" for l in LINES]],
                 ["2025 expected cost per policy year", *[f"{L[l]['per_year']:.1f}" for l in LINES]],
                 ["2025 expected cost, M CU; over written premium", *[f"{L[l]['predicted'] / 1e6:.2f}; {L[l]['ratio']:.2f}" for l in LINES]]]), ""]
    m = L["motor"]
    s += [f"> **How to read this.** The shape sets relativities; the chain sets each line's total. The indication is built peril by peril "
          f"from weather-normalised emerging experience with Part 1's trends, then loaded for cost above the cap. A motor renewal priced at "
          f"500 CU by the shape at the 2024 level becomes 500 × {m['off_balance']:.3f} × {m['renewal']:.3f} × {m['weather']:.3f} = "
          f"{500 * m['off_balance'] * m['renewal'] * m['weather']:.1f} CU; a new-business policy with the same shape takes the "
          f"new-business factor too: × {m['nb_factor']:.3f} = {500 * m['off_balance'] * m['renewal'] * m['weather'] * m['nb_factor']:.1f} CU "
          f"(the excess factor is inside the off-balance). The last row reads as a loss ratio before expenses: expected buildings "
          f"losses are {L['buildings']['ratio']:.2f} times the premium charged today.", ""]

    # ---- new-business credibility
    R = nb["rows"]
    s += ["## Exhibit P2-5. The new-business factor and its credibility", "",
          table(["", *[l.capitalize() for l in LINES]],
                [["New-business claims, 2023 to 2024", *[f"{R[l]['claims']:.0f}" for l in LINES]],
                 ["New against renewal, actual over model (apparent: fitting rows)", *[f"{R[l]['ae']:.3f}" for l in LINES]],
                 ["Z, claim-count standard √(n ÷ 1,082): factor used (H-NS (a))", *[f"{R[l]['z_count']:.2f}: {R[l]['f_count']:.3f}" for l in LINES]],
                 ["Severity CV²; pure-premium standard 1,082 × (1 + CV²) claims", *[f"{R[l]['cv2']:.2f}; {R[l]['full_standard_pp']:,.0f}" for l in LINES]],
                 ["Z, pure-premium standard: factor (not used)", *[f"{R[l]['z_pp']:.2f}: {R[l]['f_pp']:.3f}" for l in LINES]]]), "",
          f"Pooled D² on this shape (motor by peril), out of time / cross-validation, by standard: none {nb['d2']['none'][0]:.4f} / {nb['d2']['none'][1]:.4f}; "
          f"pure premium {nb['d2']['pp'][0]:.4f} / {nb['d2']['pp'][1]:.4f}; count {nb['d2']['count'][0]:.4f} / {nb['d2']['count'][1]:.4f}; "
          f"full {nb['d2']['full'][0]:.4f} / {nb['d2']['full'][1]:.4f}. Public board: paired check decided before upload (shape before motor by peril), "
          f"count {nb['board']['count']}, none {nb['board']['none']}, full {nb['board']['full']}; final shape, count {nb['board']['final_count']} "
          f"against pure premium {nb['board']['final_pp']}.", "",
          f"> **How to read this.** The 2025 book is 76% new business against 14% in training, so this factor carries more of the 2025 "
          f"level than any other judgment. In motor, new business ran {R['motor']['ae']:.3f} times what the model expects on the training rows. "
          f"With {R['motor']['claims']:.0f} new-business claims the claim-count standard gives Z = √({R['motor']['claims']:.0f} ÷ 1,082) = "
          f"{R['motor']['z_count']:.2f} and a factor of 1 + {R['motor']['z_count']:.2f} × ({R['motor']['ae']:.3f} − 1) = {R['motor']['f_count']:.3f}. "
          f"The factor weights a ratio of cost, for which the textbook standard is the pure-premium one, {R['motor']['full_standard_pp']:,.0f} claims: "
          f"Z {R['motor']['z_pp']:.2f}, factor {R['motor']['f_pp']:.3f}. Our validation cannot separate the two: on this shape, in motor, the "
          f"pure-premium standard wins out of time by {gain(nb['pp_vs_count_motor']['V1']['gain'])} {ci(nb['pp_vs_count_motor']['V1'])} and loses "
          f"in cross-validation by {gain(nb['pp_vs_count_motor']['V2']['gain']).lstrip('−')} {ci(nb['pp_vs_count_motor']['V2'])}, and the motor "
          f"excess comes mostly from one year {nb['by_year_motor_ae']}. The actuary first chose the pure-premium standard (H-NS (b)), then "
          f"returned to the count standard after both readings on the 2025 book favoured the larger factor (H-NS (a), decided after a board "
          f"reading).", ""]

    # ---- Appendix A
    x = X["expenses"]
    s += ["## Appendix A. Office premium build-up for the 20 reference policies", "",
          f"Office premium = (risk premium + catastrophe load + fixed expense) ÷ (1 − variable expense − commission − premium tax − profit). "
          f"Risk premium is annual: the row's expected cost ÷ its exposure. Catastrophe load {pct(x['cat_reinsurance'], 0)} of the risk premium on "
          f"buildings and contents; fixed expense {x['fixed_expense_per_policy']:.0f} CU; variable expense {pct(x['variable_expense'], 0)}; "
          f"commission broker {pct(x['commission_broker'], 0)}, direct {pct(x['commission_direct'], 0)}; premium tax {pct(x['premium_tax'], 0)}; "
          f"profit {pct(x['target_profit_margin'], 0)}. Divisor: broker {1 - .15 - .20 - .08 - .08:.2f}, direct {1 - .15 - 0 - .08 - .08:.2f}. "
          f"No reference policy is sold through a comparison site, whose commission the table does not give (P2-E (d)).", "",
          table(["Policy", "Line", "Risk", "Channel", "Exposure", "Risk premium", "Cat load", "Office premium", "Written premium", "Office ÷ written"],
                [[b["row_id"], b["line"], b["description"] + (", new" if b["new_business"] else ", renewal"), b["channel"], f"{b['exposure']:.2f}",
                  f"{b['risk_premium']:,.1f}", f"{b['cat_load']:.1f}", f"{b['office_premium']:,.1f}", f"{b['written_premium']:,.1f}",
                  f"{b['office_over_written']:.2f}"] for b in B]), ""]
    med = {l: dict(office=float(np.median([b["office_over_written"] for b in B if b["line"] == l])),
                   risk=float(np.median([b["risk_over_written"] for b in B if b["line"] == l]))) for l in LINES}
    over = [b for b in B if b["risk_premium"] > b["written_premium"]]
    ex = next(b for b in B if b["row_id"] == "B00229821")
    s += [f"Median office ÷ written premium: motor {med['motor']['office']:.2f}, buildings {med['buildings']['office']:.2f}, "
          f"contents {med['contents']['office']:.2f}. Median risk ÷ written: {med['motor']['risk']:.2f}, {med['buildings']['risk']:.2f}, "
          f"{med['contents']['risk']:.2f}. On risk premium alone, before any expense, {len(over)} of the 20 cost more than they are charged "
          f"({sum(b['line'] == 'buildings' for b in over)} of 7 buildings, {sum(b['line'] == 'motor' for b in over)} of 7 motor, "
          f"{sum(b['line'] == 'contents' for b in over)} of 6 contents).", "",
          f"> **How to read this.** Take buildings policy {ex['row_id']} (broker): expected claims cost {ex['risk_premium']:.1f} CU a year, plus "
          f"{pct(x['cat_reinsurance'], 0)} for catastrophe reinsurance ({ex['cat_load']:.1f}), plus {ex['fixed']:.0f} CU fixed expense, is "
          f"{ex['risk_premium'] + ex['cat_load'] + ex['fixed']:.1f} CU (from the unrounded parts). Divided by {ex['divisor']:.2f}, the share left after 15% variable expense, "
          f"20% commission, 8% tax and 8% profit, it gives {ex['office_premium']:,.1f} CU against {ex['written_premium']:.1f} CU charged today. "
          f"Most of the distance is the host's expense table: at the broker divisor a motor policy needs {1 / .49:.2f} times its risk premium "
          f"plus {85 / .49:.0f} CU, and a property policy {1.12 / .49:.2f} times plus {85 / .49:.0f} CU. The acquired broker book is priced well below the technical premium.", ""]

    # ---- Appendix B
    mf, lo, hi = G["men_over_women"], *G["men_over_women_ci"]
    bk = G["book_2025"]
    ae = G["actual_over_expected"]
    s += ["## Appendix B. Gender in motor: fairness exhibit", "",
          "Gender is allowed in this market's pricing (data page). It enters motor only (P2-S (a)), as one factor fitted on emerging rows on "
          "top of Part 1's shape, which carries no gender because Part 1's data has none. No other factor was refitted with gender present, "
          "so no other relativity absorbs or offsets it.", "",
          table(["", "Women", "Men"],
                [["Fitting rows (2023 to 2024)", f"{G['fitting_rows']['by_gender'].get('F', 0):,}", f"{G['fitting_rows']['by_gender'].get('M', 0):,}"],
                 ["Actual over expected, Part 1 shape without the gender factor", f"{ae['without_layer']['F']:.3f}", f"{ae['without_layer']['M']:.3f}"],
                 ["Actual over expected, with the gender factor (apparent: fitting rows)", f"{ae['with_layer']['F']:.3f}", f"{ae['with_layer']['M']:.3f}"],
                 ["2025 rows", f"{bk['F']['rows']:,}", f"{bk['M']['rows']:,}"],
                 ["2025 average price per policy year, as filed", f"{bk['F']['annual_price']:.1f}", f"{bk['M']['annual_price']:.1f}"],
                 ["2025 average price per policy year, without gender (same motor total)", f"{bk['F']['annual_price_without_gender']:.1f}", f"{bk['M']['annual_price_without_gender']:.1f}"]]), "",
          f"Men against women, everything else the same: {mf:.3f} [95%: {lo:.3f}, {hi:.3f}] ({G['n_boot']:,} customer resamples).", "",
          table(["Reference policy", "Gender", "Price as filed", "Same driver, other gender", "Without gender"],
                [[r["row_id"], r["gender"], f"{r['annual_price']:,.1f}", f"{r['annual_price_other_gender']:,.1f}",
                  f"{r['annual_price_without_gender']:,.1f}"] for r in G["reference_policies"]]), "",
          "*Prices are annual risk premiums (expected cost ÷ exposure), CU.*", ""]
    r0 = G["reference_policies"][0]
    s += [f"> **How to read this.** Without the gender factor, the Part 1 shape under-prices men's claims cost ({ae['without_layer']['M']:.3f} "
          f"of expected) and over-prices women's ({ae['without_layer']['F']:.3f}); the factor closes that gap. Two drivers identical in every "
          f"rated field pay prices {mf:.2f} to 1, men higher. Reference policy {r0['row_id']} ({r0['gender']}) is priced at "
          f"{r0['annual_price']:,.1f} CU a year; the same driver of the other gender would be priced at {r0['annual_price_other_gender']:,.1f}, "
          f"and with gender removed and the motor total held, at {r0['annual_price_without_gender']:,.1f}. Removing gender would move every "
          f"motor price by the same two ratios, about {signed(bk['M']['annual_price_without_gender'] / bk['M']['annual_price'] - 1, '.1%')} for men and "
          f"{signed(bk['F']['annual_price_without_gender'] / bk['F']['annual_price'] - 1, '.1%')} for women.", ""]

    # ---- Appendix C
    s += ["## Appendix C. Weather responses", "",
          "Each peril's claim frequency responds to its weather measure's deviation from the address's normal (the same address in different "
          "years), fitted on emerging rows. The 2025 price multiplies each row by 1 + Σ (peril's share of the line's cost) × (exp(b × deviation) − 1).", "",
          table(["Line", "Peril", "Weather measure", "b per unit", "Deviation, 1 sd", "Frequency at +1 sd", "Peril share of line cost", "2025 mean deviation"],
                [[l, w["peril"].replace("_", " "), w["measure"].replace("_", " "), f"{w['coefficient']:.4f}", f"{w['dev_sd_fitting']:.3g}",
                  signed(np.exp(w['coefficient'] * w['dev_sd_fitting']) - 1, ".1%"), pct(w["share_of_line_cost"]), signed(w['dev_mean_2025'], ".3g")]
                 for l in LINES for w in W[l]["responses"]]), "",
          table(["", *[l.capitalize() for l in LINES]],
                [["2025 weather factor, mean", *[f"{W[l]['factor_2025_mean']:.3f}" for l in LINES]],
                 ["5th to 95th percentile of rows", *[f"{W[l]['factor_2025_p05_p95'][0]:.3f} to {W[l]['factor_2025_p05_p95'][1]:.3f}" for l in LINES]],
                 ["Applied", *[W[l]["mode"] for l in LINES]]]), ""]
    fl = next(w for w in W["contents"]["responses"] if w["peril"] == "flood")
    up = np.exp(fl["coefficient"] * fl["dev_sd_fitting"]) - 1
    s += [f"> **How to read this.** A contents policy whose address gets one standard deviation ({fl['dev_sd_fitting']:,.0f} mm) more rain than "
          f"its normal expects {up:.1%} more flood claims; flood is {pct(fl['share_of_line_cost'])} of contents cost, so that policy's price "
          f"moves by about {fl['share_of_line_cost'] * up:.1%}. Realised 2025 weather is used, which a price set before 2025 would not have; "
          f"it moves each line by at most {max(abs(W[l]['factor_2025_mean'] - 1) for l in LINES):.1%} on average. Buildings takes the "
          f"line's average factor ({W['buildings']['factor_2025_mean']:.3f}) on every row, because its shape is the check file's (H-F (b)); "
          f"its responses are shown because they set that average.", ""]
    return "\n".join(s) + "\n"


def charts(variant: str, X: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    ink, accent, mute = "#1f2a44", "#c0392b", "#9aa3b2"

    # reference-policy build-up for the worked example
    b = next(r for r in X["buildups"] if r["row_id"] == "B00229821")
    parts = [("Risk premium", b["risk_premium"]), ("Cat load", b["cat_load"]), ("Fixed expense", b["fixed"]),
             ("Variable, commission,\ntax and profit", b["office_premium"] - b["risk_premium"] - b["cat_load"] - b["fixed"])]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    left = 0.0
    small = 0
    for (name, v), c in zip(parts, [ink, "#4a6fa5", mute, "#d9a441"]):
        ax.barh(["Office premium"], [v], left=left, color=c)
        if v > 120:
            ax.text(left + v / 2, 0, f"{name}\n{v:,.1f}", ha="center", va="center", color="white", fontsize=9)
        else:                                   # narrow segment: label below, then above, the bar
            ax.text(left + v / 2, -0.55 if small == 0 else 0.5, f"{name} {v:,.1f}", ha="center", va="center", color=ink, fontsize=9)
            small += 1
        left += v
    ax.barh(["Written today"], [b["written_premium"]], color=accent)
    ax.text(b["written_premium"] + 10, 1, f"{b['written_premium']:,.1f}", va="center")
    ax.text(left + 10, 0, f"{left:,.1f}", va="center", fontweight="bold")
    ax.set_ylim(-0.8, 1.5)
    ax.set_xlabel("CU per year")
    ax.set_title(f"Buildings {b['row_id']}, broker: office premium build-up", loc="left")
    fig.tight_layout()
    fig.savefig(DOCS / f"part2_{variant}_buildup.png", dpi=150)
    plt.close(fig)

    # hold-out by line
    H = X["holdout"]
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(3)
    ax.bar(xs - 0.2, [H["H1"]["per_line"][l] for l in LINES], 0.4, label="H1: 2024 rows, fitted on 2023", color=ink)
    ax.bar(xs + 0.2, [H["H2"]["per_line"][l] for l in LINES], 0.4, label="H2: every hold-out row", color=mute)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs, [l.capitalize() for l in LINES])
    ax.set_ylabel("Deviance saved, frozen model against K0")
    ax.legend(frameon=False)
    ax.set_title("The hold-out, opened once: motor and contents hold, buildings does not", loc="left")
    fig.tight_layout()
    fig.savefig(DOCS / "part2_holdout.png", dpi=150)
    plt.close(fig)


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    variants = list(FILES)
    common = dict(ladder=ladder(), candidates=candidates(), holdout=holdout(), weather=weather(), expenses=expense_table())
    store = {"common": common}
    for v in variants:
        X = dict(common, level=level(v), nb=nb_credibility(v), buildups=buildups(v), gender=gender(v))
        store[v] = {k: X[k] for k in ("level", "nb", "buildups", "gender")}
        (DOCS / f"part2_{v}.md").write_text(render(v, X), encoding="utf-8", newline="\n")
        charts(v, X)
        print("written", DOCS / f"part2_{v}.md")
    write_evidence({"part2_exhibits": store}, EV / "part2_exhibits.json")


if __name__ == "__main__":
    main()
