"""The notebook builder: reads the 2026 AI-log format, fills text numbers from an evidence file, marks placeholders,
and the built notebook runs top to bottom on the evidence alone."""
import json
import shutil
from pathlib import Path

import nbformat
import pytest

from build_notebook import BuildError, ROOT, build, execute
from src.exhibits import compare_evidence

ENTRY = dict(id="D1", stage="3.2 Data", role="Data steward", proposed_by="AI", question="What does a blank mean?",
             ai_answer="Feature absent.", challenge="Prove it.", test="Cross-checked against the matching area column.",
             result="Every blank has area 0.", decision="Adopt the rule (actuary).", effect="Rule table in 3.2.", section="3.2")


@pytest.fixture()
def project(tmp_path):
    """A copy of the files the builder reads, so tests never touch the real ai/ or decisions/."""
    for p in ["ai/prompt_library.md", "ai/transcript.md", "decisions/judgment_register.md", "decisions/thresholds.md"]:
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / p, tmp_path / p)
    shutil.copytree(ROOT / "src", tmp_path / "src", ignore=shutil.ignore_patterns("__pycache__"))
    log = json.loads((ROOT / "ai/ai_log.json").read_text(encoding="utf-8"))
    log["entries"] = [ENTRY]
    (tmp_path / "ai/ai_log.json").write_text(json.dumps(log), encoding="utf-8")
    return tmp_path


def test_build_without_evidence_marks_pending(project):
    rep = build(None, project / "nb.ipynb", root=project)
    assert rep["placeholders"] > 5 and "summary.proposed_cv" in rep["pending"]
    nb = nbformat.read(str(project / "nb.ipynb"), as_version=4)
    nbformat.validate(nb)
    text = "\n".join(c.source for c in nb.cells)
    assert "[pending: summary.proposed_cv]" in text
    assert '"D1"' in text and "%%writefile src/validation.py" in text
    assert "—" not in text                                           # house style: no em dashes
    with pytest.raises(BuildError, match="not ready"):
        build(None, project / "nb2.ipynb", root=project, strict=True)


def test_old_log_format_refused(project):
    (project / "ai/ai_log.json").write_text(json.dumps([ENTRY]), encoding="utf-8")
    with pytest.raises(BuildError, match="entries"):
        build(None, project / "nb.ipynb", root=project)


def test_build_and_execute_with_evidence(project):
    from src.evidence import write_evidence
    ev = dict(summary=dict(title="Test run", team="test", metric="RMSE", public_lb="(not submitted)", n_seeds=3,
                           current_cv=0.1234, proposed_cv=0.1111, n_holdout=100, current_holdout=0.125,
                           proposed_holdout=0.112, within5=0.8, n_referred=3, n_test=500))
    ev_path = write_evidence(ev, project / "evidence.json")
    out = project / "nb.ipynb"
    rep = build(ev_path, out, root=project)
    assert [p for p in rep["pending"] if p.startswith("summary.")] == []
    nb = nbformat.read(str(out), as_version=4)
    assert "0.1111" in chr(10).join(c.source for c in nb.cells if c.cell_type == "markdown")
    execute(out, project / "run", timeout=600)
    nb = nbformat.read(str(out), as_version=4)
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert errors == []
    last = "".join(o.get("text", "") for o in nb.cells[-1].outputs)
    assert "Numbers in the text match this run." in last
    cards = [c for c in nb.cells if c.cell_type == "code" and 'cards_for(AI_LOG, "3.2")' in c.source][0]
    assert any("EXCHANGE D1" in str(o.get("data", {})) for o in cards.outputs)


def test_compare_evidence_reports_differences():
    a = dict(x=1.0, rows=[dict(v=0.5)], meta=dict(created="now"))
    b = dict(x=1.0, rows=[dict(v=0.6)], meta=dict(created="later"))
    assert compare_evidence(a, a) == []
    assert compare_evidence(a, b) == ["rows.0.v: 0.5 against 0.6"]
