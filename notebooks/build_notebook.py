"""Build the competition notebook from its parts.

    python notebooks/build_notebook.py [--evidence outputs/evidence.json] [--out notebooks/iaa_notebook.ipynb]
                                       [--execute] [--strict]

Reads
  ai/ai_log.json                   the "entries" list: exchange cards by notebook section, and the decision ledger
  ai/prompt_library.md, ai/transcript.md
  decisions/judgment_register.md, decisions/thresholds.md
  src/*.py                         embedded with %%writefile, so the Kaggle notebook runs the same tested code
  the evidence JSON                (src/evidence.write_evidence) every number in the text, and the fallback the
                                   exhibits draw from until the data-specific cells are written

Text numbers are tokens such as {{summary.proposed_cv|.4f}}: a dotted path into the evidence file and an optional
Python format. A token with no value in the evidence file renders as "[pending: path]".

Every data-specific cell is marked PLACEHOLDER. --strict refuses to write the notebook while any placeholder or
pending token remains; use it for the submission build. --execute runs the notebook top to bottom with nbclient in
notebooks/run/ and saves the executed copy.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
MODULES = ["__init__", "thresholds", "validation", "evidence", "exhibits"]
TOKEN = re.compile(r"\{\{\s*([\w.]+)\s*(?:\|\s*([^}]*?))?\s*\}\}")
MARK = "PLACEHOLDER"

# The AI log numbers its "section" field in working order (3.6 is phase 5 steps 1 to 3, 4.1 is phase 6, ...), which
# does not match the notebook's filing sections. Each log section is shown in the notebook section that discusses it;
# a log section not listed here is shown under its own number.
PLACEMENT = {
    "3.4": "3.5",    # D11 validation design and keep rule -> Validation
    "3.5": "3.4",    # D12 baselines -> Model development
    "3.6": "3.4",    # D14 to D16 thresholds, cap, trend, level, weather -> Model development
    "3.7": "3.3",    # D17 to D19 claims history and features -> Rating variables
    "3.8": "3.4",    # D20 model structure
    "3.9": "3.4",    # D21 standardised inputs
    "3.10": "3.2",   # D22 contents blanks -> Data (missing values)
    "3.11": "3.4",   # D23 penalised per-peril GLM
    "3.12": "3.4",   # D24 model families
    "3.13": "3.6",   # D25 the frozen blend -> Rating plan
    "3.14": "3.10",  # D27 runway to the leader -> Competitive comparison
    "4.1": "3.8",    # D26 referral -> Impact and referrals
    "4.2": "3.5",    # D28 nil claim at the final refit -> Validation
    "4.3": "3.5",    # D29 hold-out and final file -> Validation
}


def section_key(s: str) -> tuple:
    return tuple(int(p) if p.isdigit() else p for p in str(s).split("."))


def log_section(entry: dict) -> str:
    return str(entry.get("section", "")).strip().split(" ")[0]


class BuildError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------ helpers
def lookup(ev: dict, path: str):
    cur = ev
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return None
    return cur


def fill_tokens(text: str, ev: dict, pending: list[str]) -> str:
    def sub(m):
        v = lookup(ev, m.group(1))
        if v is None:
            pending.append(m.group(1))
            return f"[pending: {m.group(1)}]"
        return format(v, m.group(2)) if m.group(2) else str(v)
    return TOKEN.sub(sub, text)


def literal(obj) -> str:
    """A JSON value as a Python expression for a code cell."""
    s = json.dumps(obj, ensure_ascii=False, indent=1)
    if "'''" in s:
        raise BuildError("embedded JSON contains ''' and cannot be quoted")
    return f"json.loads(r'''{s}''')"


def reviewer(text: str) -> str:
    return (f'<div style="border:1px dashed #c3c8cf;border-radius:8px;padding:8px 14px;color:#48515d">'
            f"<b>What a reviewer asks.</b> {text}</div>")


def reading(text: str) -> str:
    return f'<div style="background:#eef6f1;border-left:3px solid #0f7a52;padding:10px 14px;margin:8px 0"><b>How to read this.</b> {text}</div>'


def ph(text: str) -> str:
    return (f'<div style="border:2px dashed #eb6834;border-radius:8px;padding:8px 14px;color:#9a3c12">'
            f"<b>{MARK} (data-specific).</b> {text}</div>")


# ------------------------------------------------------------------------------------------------ the notebook
def build(evidence_path: Path | None, out_path: Path, root: Path = ROOT, log_path: Path | None = None,
          strict: bool = False) -> dict:
    ev = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path and evidence_path.exists() else {}
    log_file = log_path or root / "ai" / "ai_log.json"
    log = json.loads(log_file.read_text(encoding="utf-8"))
    if "entries" not in log:
        raise BuildError(f"{log_file} has no 'entries' list (expected the 2026 format with 'schema' and 'entries')")
    entries = log["entries"]
    read = lambda p: (root / p).read_text(encoding="utf-8")
    prompts, transcript = read("ai/prompt_library.md"), read("ai/transcript.md")
    judgments, thresholds = read("decisions/judgment_register.md"), read("decisions/thresholds.md")

    cells, pending = [], []

    def md(s):
        cells.append(nbf.v4.new_markdown_cell(fill_tokens(s.strip("\n"), ev, pending)))

    def code(s, hidden=False):
        c = nbf.v4.new_code_cell(s.strip("\n"))
        if hidden:
            c.metadata.update({"_kg_hide-input": True, "jupyter": {"source_hidden": True}})
        cells.append(c)

    log_sections = sorted({log_section(e) for e in entries}, key=section_key)

    def cards(section):
        here = [s for s in log_sections if PLACEMENT.get(s, s) == section]
        if here:
            code("\n".join(f'cards_for(AI_LOG, "{s}");' for s in here))

    # ============================================================================ title and summary
    md(f"""
# {{{{summary.title}}}}
### IAA AI for Actuaries Hackathon 2026 · model documented in the structure of a Canadian rate filing

{ph("Title in summary.title: name the business use decided in J1, not the method.")}

**How to read this notebook.** It follows the IAA template (sections 1 to 5). Inside section 3 the model is documented the way a Canadian rate filing justifies a rating model: what the model is for, what data went in, why each choice was made, how it was validated, what changes against the current model, and which judgments the actuary made. Grey boxes state what a reviewer would ask; green boxes explain the technical terms with a worked example; exchange cards show where AI changed a decision and the test that settled it.
""")
    md(f"""
## 1. Summary

| | |
|---|---|
| **Participant** | {{{{summary.team}}}} |
| **Public leaderboard ({{{{summary.metric}}}})** | {{{{summary.public_lb}}}} |

| Measure | Current model | Proposed model |
|---|---|---|
| Cross-validated {{{{summary.metric}}}}, mean of {{{{summary.n_seeds}}}} split seeds | {{{{summary.current_cv|.4f}}}} | {{{{summary.proposed_cv|.4f}}}} |
| Locked hold-out ({{{{summary.n_holdout}}}} records, opened once) | {{{{summary.current_holdout|.4f}}}} | {{{{summary.proposed_holdout|.4f}}}} |
| Test records whose prediction moves by 5% or less | | {{{{summary.within5|.0%}}}} |
| Test records referred for review | | {{{{summary.n_referred}}}} of {{{{summary.n_test}}}} |

{ph("Proposal in two or three sentences: what is replaced, by what, and the largest individual effects. Then a green box reading the error in plain terms with one worked example in real units (see docs/FILING_TEMPLATE.md, plain-language conventions).")}
""")

    # ============================================================================ set-up (hidden)
    md("""
<small>Set-up: the hidden cells below write the project's code to `src/`. It is the same code, with the same tests, as the project repository.</small>
""")
    code("import os, json\nos.makedirs('src', exist_ok=True)", hidden=True)
    for m in MODULES:
        src = (root / "src" / f"{m}.py").read_text(encoding="utf-8")
        code(f"%%writefile src/{m}.py\n{src}", hidden=True)

    # ============================================================================ 2. generative AI integration
    md("""
## 2. Generative AI integration

AI was used as a team of defined roles, each set up once with a standing prompt (section 5), working inside a loop the actuary controls: a question, an AI proposal, a challenge, a test, the evidence, and a decision the actuary makes. Every exchange that changed a decision is logged as a card beside the analysis it changed. The ledger below lists them all.
""")
    code("""
# ---------- the AI decision log (ai/ai_log.json, "entries"); input hidden on Kaggle ----------
import json
AI_LOG = __AI_LOG__
print(f"{len(AI_LOG)} logged exchanges")
""".replace("__AI_LOG__", literal(entries)), hidden=True)
    md("""
| Role | What it does here | Judging criterion it serves |
|---|---|---|
| Data steward | Reads the data dictionary, classifies every blank, tests where the data came from | Data understanding |
| Modeller | Reproduces the baseline, proposes challengers that make different errors | Modelling strategy |
| Filing writer | Drafts the filing-style documentation and the judgment register | Interpretability, documentation |
| Plain-language explainer | Writes the green "how to read this" boxes | Explaining outputs |
| Independent reviewer | A fresh session that did not build the model, asked to break it | Validation |
| Actuary (human) | Challenges every claim and makes every judgment | |
""")
    code("""
from src.exhibits import cards_for, ledger, plot_loop, style_plots
style_plots(); plot_loop()
lg = ledger(AI_LOG)
if lg is not None: display(lg)
""")
    cards("2")

    md("""
## 3. Step-by-step analysis
""")
    code("""
# ---------- PLACEHOLDER (data-specific): configuration, from competition/brief.md after /intake ----------
COMPETITION = None    # Kaggle slug; data read from /kaggle/input/<slug>/
TARGET, ID = None, None

import os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from src import exhibits as X
T0 = time.time()
""")
    code(f"""
# ---------- evidence embedded at build time: the numbers quoted in the text ----------
EVIDENCE_EMBEDDED = {literal(ev)}
""", hidden=True)
    code("""
# ---------- PLACEHOLDER (data-specific): load, clean, model, write the evidence ----------
# Built after the data arrives, from the principles in docs/PLAYBOOK.md and the decided judgments.
# Until then the exhibits use the evidence copy embedded at build time.
EVIDENCE = None
""")
    code("""
if EVIDENCE is None:
    EVIDENCE = EVIDENCE_EMBEDDED
    print("Dry run: numbers drawn from the evidence file embedded at build time.")
""")

    # ============================================================================ 3.1 to 3.10, filing sections
    def section(num, title, asks, placeholder, exhibit=None, read=None):
        md(f"""
### {num} {title}

{reviewer(asks)}

{ph(placeholder)}
""")
        if exhibit:
            code(f"# ---------- PLACEHOLDER (data-specific): {exhibit} ----------")
        if read:
            md(reading(read + " " + ph("Worked example with real numbers from the exhibit above.")))
        cards(num)

    section("3.1", "Purpose and use",
            "Model type and why it suits its role; who uses the output. <i>(Filing section 1; FSRA rating and underwriting guidance)</i>",
            "The intended use decided in J1: candidate uses, the costlier error for each, and what each choice changes (inputs allowed, target scale, range shape).")
    section("3.2", "Data",
            "Sources, adjustments, missing values, exclusions with reasons, limitations. <i>(Filing section 2; NAIC model white paper A.1 to A.4)</i>",
            "Data origin and the external-data judgment; the reading of every blank with the check behind it; the outlier screen with a reason for each decision; the train-versus-test evidence check; limitations.",
            exhibit="data exhibits")
    section("3.3", "Rating variables",
            "Data dictionary, rationale linking each variable to the outcome, rejected variables, point-of-use and proxy concerns. <i>(Filing section 3; NAIC B.3, C.2, C.5)</i>",
            "Variables used and their business rationale; engineered variables and the evidence that each earns its place; variables rejected and why; the point-of-use test.")
    section("3.4", "Model development",
            "Model type, target, link and distribution, how the data was split, how variables were selected. <i>(Filing section 4; NAIC B.1, B.2)</i>",
            "Candidate models tried in each family, the strongest in each family, and how the families are weighted (J-model). Every score out of fold.",
            exhibit="model development exhibits",
            read="Cross-validation is a back-test: each record is predicted by a model fitted without it, like testing a reserving method on accident years it was not fitted to. Repeating it on different shuffles of the records shows how much of a difference between two models is luck.")
    section("3.5", "Validation",
            "Fit on validation data, stability, overfitting, and why the new model beats the current one. <i>(Filing section 5; NAIC B.4, B.5; FSRA standard filing requirements)</i>",
            "The evaluations, designed with the actuary once the data and its use are known. Apparent error is shown only beside the honest out-of-fold and hold-out numbers.",
            exhibit="validation exhibits",
            read="Apparent error is measured on the records a model was fitted to, so it flatters the model, like a student marked on the exam they memorised. The hold-out is a sealed envelope opened once, after the model was frozen.")
    section("3.6", "Rating plan",
            "How the model produces predictions; relativities against one-way views; interpretability tools. <i>(Filing section 6; NAIC 2024 guidance for non-GLM models)</i>",
            "How the filed prediction is built from its parts, and the tools that explain each part.",
            exhibit="rating plan exhibits",
            read="A relativity of 1.09 means priced 9% above an otherwise identical record at the base level.")
    section("3.7", "Rating example",
            "The algorithm applied to a real record. <i>(Filing section 7)</i>",
            "One record from start to finish: its inputs, each model's prediction, the filed prediction and its range, and the breakdown by variable.")
    section("3.8", "Impact and referrals",
            "Distribution of changes, the largest swings, and how they are handled. <i>(Filing section 8)</i>",
            "How far predictions move against the current model; the records referred for review, with the trigger and treatment for each.",
            exhibit="impact and referral exhibits")
    section("3.9", "Segment testing",
            "Calibration by segment; proxy and disparate-impact concerns. <i>(Filing section 9; FSRA guidance on outcomes testing)</i>",
            "Actual against expected by the segments that matter for this data, each with its count.",
            exhibit="segment exhibits",
            read="A/E of 1.07 means actuals ran 7% above prediction, so the group is under-priced. Read each A/E beside its count, as you would credibility-weight a territory.")

    md(f"""
### 3.10 Competitive comparison

{reviewer("How the proposal compares with competitors. <i>(Filing section 10)</i>")}

{ph("Leaderboard context; other methods rerun on the same folds; predictions for the records that decide the metric.")}
""")
    cards("3.10")

    # ============================================================================ 4. findings
    md(f"""
## 4. Findings and insights

{ph("Three to five sentences for a stakeholder, written by the actuary after the final run: what drives the outcome, how accurate the model is in real units, where it should not be trusted, what the business should do with it.")}

**Judgment register.** Every decision a number cannot settle. Only the actuary marks a judgment decided.
""")
    md(judgments.split("\n", 1)[1] if judgments.startswith("# ") else judgments)
    cards("4")

    # ============================================================================ 5. reproducibility
    md("""
## 5. Reproducibility

- Every number is produced by the code in this notebook from the competition files, with fixed seeds. No AI model is called while the notebook runs; AI exchanges are recorded as data in the log above.
- Every threshold was written down, with its date, before the test that uses it:
""")
    md(thresholds.split("\n", 1)[1] if thresholds.startswith("# ") else thresholds)
    cards("5")
    md(f"<details><summary><b>Prompt library: AI role set-ups and challenge prompts</b></summary>\n\n{prompts}\n</details>")
    md(f"<details><summary><b>Session transcript (condensed)</b></summary>\n\n{transcript}\n</details>")
    code("""
# ---------- PLACEHOLDER (data-specific): write submission.csv in the competition's format and check it ----------
""")
    code("""
# ---------- versions, and a check that the numbers in the text match this run ----------
import platform, sklearn, lightgbm, scipy
print("python", platform.python_version(), "| numpy", np.__version__, "| pandas", pd.__version__, "| scipy", scipy.__version__,
      "| scikit-learn", sklearn.__version__, "| lightgbm", lightgbm.__version__, f"| run time {time.time() - T0:.0f}s")
diff = X.compare_evidence(EVIDENCE, EVIDENCE_EMBEDDED)
print("Numbers in the text match this run." if not diff else f"{len(diff)} numbers differ from the text:\\n" + "\\n".join(diff[:20]))
""")

    # ============================================================================ write
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata.update({"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                        "language_info": {"name": "python"}})
    placeholders = sum(MARK in c.source and not c.source.startswith("%%writefile") for c in cells)
    shown = {"2", "3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8", "3.9", "3.10", "4", "5"}
    orphans = [s for s in log_sections if PLACEMENT.get(s, s) not in shown]
    report = dict(cells=len(cells), placeholders=placeholders, pending=sorted(set(pending)), entries=len(entries),
                  orphan_sections=orphans, evidence=str(evidence_path) if ev else None)
    if strict and (placeholders or pending or orphans):
        raise BuildError(f"not ready for submission: {placeholders} placeholder cells, pending tokens {sorted(set(pending))}, "
                         f"AI-log sections with no place in the notebook {orphans}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(out_path))
    return report


def execute(nb_path: Path, run_dir: Path, timeout: int = 3600) -> None:
    from nbclient import NotebookClient
    nb = nbf.read(str(nb_path), as_version=4)
    run_dir.mkdir(parents=True, exist_ok=True)
    NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(run_dir)}}).execute()
    nbf.write(nb, str(nb_path))


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--evidence", type=Path, default=ROOT / "outputs" / "evidence.json")
    a.add_argument("--out", type=Path, default=ROOT / "notebooks" / "iaa_notebook.ipynb")
    a.add_argument("--execute", action="store_true")
    a.add_argument("--strict", action="store_true")
    args = a.parse_args(argv)
    rep = build(args.evidence, args.out, strict=args.strict)
    print(f"wrote {args.out}: {rep['cells']} cells, {rep['entries']} logged exchanges, {rep['placeholders']} placeholder cells, "
          f"{len(rep['pending'])} pending tokens" + ("" if rep["evidence"] else " (no evidence file yet)"))
    if rep["orphan_sections"]:
        print("AI-log entries whose section has no place in the notebook:", rep["orphan_sections"])
    if args.execute:
        execute(args.out, ROOT / "notebooks" / "run")
        print("executed", args.out)


if __name__ == "__main__":
    sys.exit(main())
