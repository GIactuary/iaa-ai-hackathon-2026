"""Notebook exhibits drawn from the evidence dictionary (src/evidence.py) and the AI log (ai/ai_log.json).

Every function here only reads; nothing fits a model. A missing evidence key prints a short notice instead of failing,
so the notebook skeleton runs end to end before the data-specific cells are written.
"""
from __future__ import annotations

import html
import math

import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA, RED, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#0f141a", "#7a838f", "#e3e6ea"


def _display(obj):
    try:
        from IPython.display import display
        display(obj)
    except ImportError:
        print(obj)


def _html(s):
    try:
        from IPython.display import HTML
        return HTML(s)
    except ImportError:
        return s


def style_plots():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": GRID, "axes.edgecolor": "#c3c8cf", "axes.labelcolor": "#48515d", "xtick.color": MUTED,
                         "ytick.color": MUTED, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold"})


def get(ev: dict | None, path: str):
    """Value at a dotted path ("summary.proposed_cv", "compare.0.gain"), or None."""
    cur = ev
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return None
    return cur


def _need(ev, key):
    v = get(ev, key)
    if v is None:
        print(f"[not yet in the evidence file: {key}]")
    return v


def table(records, columns: dict | None = None, formats: dict | None = None):
    """Records as a left-aligned table without the index. columns: {key: header}."""
    df = pd.DataFrame(records)
    if columns:
        df = df[[c for c in columns if c in df.columns]].rename(columns=columns)
    fm = {(columns or {}).get(k, k): v for k, v in (formats or {}).items()}
    fm = {k: v for k, v in fm.items() if k in df.columns}
    return df.style.hide(axis="index").format(fm, na_rep="").set_properties(**{"text-align": "left"})


# ------------------------------------------------------------------------------------------------ AI log
def _esc(s):
    return html.escape(str(s))


def card(entry: dict):
    rows = [("Question", "question"), ("AI answer", "ai_answer"), ("Challenge", "challenge"), ("Test", "test"),
            ("Result", "result"), ("Decision", "decision"), ("Effect", "effect")]
    body = "".join(f'<tr><td style="color:{MUTED};padding:3px 10px 3px 0;vertical-align:top;white-space:nowrap;text-align:left">{k}</td>'
                   f'<td style="padding:3px 0;text-align:left">{_esc(entry.get(f, ""))}</td></tr>' for k, f in rows if entry.get(f))
    head = " · ".join(_esc(x) for x in [f"EXCHANGE {entry.get('id', '')}", entry.get("stage", ""), entry.get("role", ""),
                                        f"raised by {entry.get('proposed_by', '')}"] if x)
    return _html(f'<div style="border:1px solid #d8dde3;border-radius:8px;padding:10px 14px;margin:10px 0;max-width:900px">'
                 f'<div style="font-family:monospace;font-size:12px;color:#1c5cab">{head}</div>'
                 f'<table style="font-size:13.5px;line-height:1.45;margin-top:6px">{body}</table></div>')


def _section_of(entry) -> str:
    return str(entry.get("section", "")).strip().split(" ")[0]


def cards_for(log: list[dict], section: str):
    """Show every exchange card whose notebook section is `section` (for example "3.2")."""
    hits = [e for e in log if _section_of(e) == section]
    for e in hits:
        _display(card(e))
    return len(hits)


def ledger(log: list[dict]):
    if not log:
        print("No exchanges logged yet.")
        return None
    df = pd.DataFrame(log)
    cols = {"id": "Ref", "stage": "Stage", "role": "AI role", "proposed_by": "Raised by", "decision": "Decision", "section": "Shown in"}
    return table(df.to_dict("records"), cols)


def plot_loop():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 3.0))
    ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0.3, 3.0)
    steps = [("Question", "actuary or AI"), ("Proposal", "AI role"), ("Challenge", "actuary"), ("Test", "AI runs it"),
             ("Evidence", "numbers, exhibits"), ("Decision", "actuary decides")]
    for k, (t, s) in enumerate(steps):
        x = 0.3 + k * 1.8
        last = t == "Decision"
        ax.add_patch(plt.Rectangle((x, 1.6), 1.45, 0.9, color=ORANGE if last else BLUE, alpha=.22 if last else .12, lw=0))
        ax.text(x + .72, 2.13, t, ha="center", fontsize=11, weight="bold", color=INK)
        ax.text(x + .72, 1.82, s, ha="center", fontsize=8.5, color=MUTED)
        if k < len(steps) - 1:
            ax.annotate("", (x + 1.75, 2.05), (x + 1.5, 2.05), arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.text(5.5, 0.9, "each exchange that changed a decision is logged as a card; a failed test sends the question back", ha="center", fontsize=9, color=MUTED)
    plt.show()


# ------------------------------------------------------------------------------------------------ development and validation
def show_seed_scores(ev):
    s = _need(ev, "seed_scores")
    if s is None:
        return
    import matplotlib.pyplot as plt
    rows = sorted(s["rows"], key=lambda r: r["mean"])
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(rows) + 0.9))
    for i, r in enumerate(rows[::-1]):
        v = list(r["by_seed"].values())
        ax.plot([min(v), max(v)], [i, i], color=BLUE, alpha=.4, lw=2)
        ax.scatter(v, [i] * len(v), color=BLUE, s=36, zorder=3)
        ax.text(max(v), i + 0.25, f" mean {r['mean']:.4f}", fontsize=9, color=MUTED)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r["model"] for r in rows[::-1]]); ax.grid(axis="y")
    ax.set_title(f"Out-of-fold {s['metric']} by split seed (each dot one seed; lower is better)")
    plt.tight_layout(); plt.show()


def show_compare(ev, key="compare"):
    rows = _need(ev, key)
    if rows is None:
        return
    _display(table(rows, {"split_seed": "Split seed", "score_a": "Current", "score_b": "Proposed", "gain": "Gain",
                          "lo": "95% interval, low", "hi": "95% interval, high", "share_b_better": "Resamples favouring proposed"},
                   {"score_a": "{:.4f}", "score_b": "{:.4f}", "gain": "{:.4f}", "lo": "{:.4f}", "hi": "{:.4f}", "share_b_better": "{:.0%}"}))


def show_score_table(ev):
    rows = _need(ev, "score_table")
    if rows is None:
        return
    _display(table(rows, {"model": "Model", "apparent": "Apparent (fitted rows)", "cv": "Cross-validated", "holdout": "Locked hold-out"},
                   {"apparent": "{:.4f}", "cv": "{:.4f}", "holdout": "{:.4f}"}))


def show_glm_path(ev):
    path = _need(ev, "glm_path")
    if path is None:
        return
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(range(len(path)), [p["cv"] for p in path], "-o", color=BLUE)
    ax.set_xticks(range(len(path))); ax.set_xticklabels([p["var"] for p in path], rotation=60, ha="right", fontsize=8)
    ax.set_title("Forward GLM ladder: cross-validated error after each variable enters"); ax.grid(axis="x")
    plt.tight_layout(); plt.show()


def show_lift(ev, key="lift.proposed", title="Lift, out of fold: actual against predicted by predicted-value decile"):
    rows = _need(ev, key)
    if rows is None:
        return
    import matplotlib.pyplot as plt
    b = [r["bin"] for r in rows]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(b, [r["predicted_mean"] for r in rows], "-o", color=ORANGE, label="predicted")
    ax.plot(b, [r["actual_mean"] for r in rows], "-o", color=BLUE, label="actual")
    if "ci_lo" in rows[0]:
        ax.fill_between(b, [r["ci_lo"] for r in rows], [r["ci_hi"] for r in rows], color=BLUE, alpha=.12, lw=0, label="95% band on actual")
    ax.set_xticks(b); ax.set_xlabel("decile of prediction (equal exposure)"); ax.legend(frameon=False); ax.set_title(title)
    plt.tight_layout(); plt.show()


def show_double_lift(ev, key="double_lift"):
    rows = _need(ev, key)
    if rows is None:
        return
    import matplotlib.pyplot as plt
    b = [r["bin"] for r in rows]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axhline(1, color=INK, lw=1)
    ax.plot(b, [r["ae_proposed"] for r in rows], "-o", color=BLUE, label="proposed")
    ax.plot(b, [r["ae_current"] for r in rows], "-o", color=ORANGE, label="current")
    ax.set_xticks(b); ax.set_xlabel("decile of proposed / current (left: proposed lower; right: proposed higher)")
    ax.set_ylabel("actual / predicted"); ax.legend(frameon=False)
    ax.set_title("Double lift, out of fold: where the models disagree, which tracks the actuals?")
    plt.tight_layout(); plt.show()


def show_ranges(ev, key="ranges"):
    r = _need(ev, key)
    if r is None:
        return
    _display(table(r["by_group"], {"group": "Predicted-value fifth", "n": "Records", "median_pred": "Median prediction",
                                   "lo_factor": "Low end (x prediction)", "hi_factor": "High end (x prediction)",
                                   "coverage": "Inside range", "above": "Above range", "below": "Below range"},
                   {"median_pred": "{:,.4g}", "lo_factor": "{:.3f}", "hi_factor": "{:.3f}", "coverage": "{:.1%}", "above": "{:.1%}", "below": "{:.1%}"}))
    t = r["total"]
    print(f"Overall: {t['coverage']:.1%} inside a nominal {t['nominal']:.0%} range; {t['above']:.1%} above, {t['below']:.1%} below "
          f"(cross-fitted, so each record is checked against percentiles from other folds).")


# ------------------------------------------------------------------------------------------------ rating plan, impact, segments
def show_glm_coef(ev, key="glm_coef"):
    rows = _need(ev, key)
    if rows is None:
        return
    _display(table(rows, {"term": "Term", "coef": "Coefficient", "se": "Std error", "relativity": "Relativity",
                          "rel_lo95": "95% low", "rel_hi95": "95% high"},
                   {"coef": "{:.4f}", "se": "{:.4f}", "relativity": "{:.3f}", "rel_lo95": "{:.3f}", "rel_hi95": "{:.3f}"}))


def show_adjustment(ev, key="adjustment"):
    a = _need(ev, key)
    if a is None:
        return
    p = a["percentiles"]
    print(f"Tree adjustment to the GLM price: median x{p['50']:.3f}; 90% of records between x{p['5']:.3f} and x{p['95']:.3f}; "
          f"{a['within5']:.0%} within 5%; extremes x{a['max_down']:.3f} and x{a['max_up']:.3f}.")


def show_dislocation(ev, key="dislocation"):
    d = _need(ev, key)
    if d is None:
        return
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 3))
    labels = [b["band"] for b in d["bands"]]
    ax.bar(labels, [b["n"] for b in d["bands"]], color=[ORANGE if ("below" in l or "above" in l) else BLUE for l in labels], width=.5)
    ax.tick_params(axis="x", labelsize=8); ax.grid(axis="x")
    ax.set_title(f"Change against the current model, {d['n']:,} records: {d['within5']:.0%} move 5% or less, {d['over15']} move more than 15%")
    plt.tight_layout(); plt.show()


def show_referrals(ev, key="referral"):
    r = _need(ev, key)
    if r is None:
        return
    print(f"Referred: {r.get('n_referred', 0)} of {r.get('n', 0)} "
          f"(out of evidence {r.get('n_out_of_evidence', 0)}, model disagreement {r.get('n_gap', 0)})")
    if r.get("rows"):
        _display(table(r["rows"]))


def show_ae(ev, key, top: int = 15):
    rows = _need(ev, key)
    if rows is None:
        return
    _display(table(rows[:top], {"segment": "Segment", "n": "Records", "exposure": "Exposure", "ae": "A/E"},
                   {"exposure": "{:,.1f}", "ae": "{:.3f}"}))


# ------------------------------------------------------------------------------------------------ reproducibility
def compare_evidence(a, b, rtol: float = 1e-6, path: str = "") -> list[str]:
    """Numeric leaves that differ between two evidence dictionaries (meta is ignored). Empty list = they match."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k == "meta":
                continue
            if k not in a or k not in b:
                out.append(f"{path}{k}: present in only one")
            else:
                out += compare_evidence(a[k], b[k], rtol, f"{path}{k}.")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path[:-1]}: lengths {len(a)} and {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out += compare_evidence(x, y, rtol, f"{path}{i}.")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not math.isclose(a, b, rel_tol=rtol, abs_tol=1e-12):
            out.append(f"{path[:-1]}: {a} against {b}")
    elif a != b:
        out.append(f"{path[:-1]}: {a!r} against {b!r}")
    return out
