"""Notebook packaging: exchange cards and the decision ledger from the AI log (ai/ai_log.json), tables, the working-loop
diagram, and the check that a rerun reproduces the numbers quoted in the text.

Every function here only reads. Exhibits of the data, the models and the evaluations are written once those exist.
"""
from __future__ import annotations

import html
import math

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
