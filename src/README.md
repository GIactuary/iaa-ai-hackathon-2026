# 2026 code

What is here is process code, kept from the preparation: it holds the rules of the playbook and packages our evidence into the notebook. No model code exists yet. Models, evaluations and data rules are built after the data arrives, one decision at a time with the actuary.

| Module | Contents |
|---|---|
| `thresholds.py` | Reads `decisions/thresholds.md`; refuses a threshold that has no value or no "Written on" date (rule 2) |
| `validation.py` | Folds on the split seeds we choose (optionally grouped or stratified); a hold-out drawn once, written to disk and opened once, with each opening recorded; an out-of-fold runner that checks every row is predicted once per seed; a paired bootstrap for any metric; the drop-one verdict |
| `evidence.py` | Writes the evidence file (results plus run metadata) that the notebook's numbers come from |
| `exhibits.py` | Exchange cards and the decision ledger from `ai/ai_log.json`, tables, the working-loop diagram, and the rerun check |

The 2025 code is in `reference/2025/` for context. Run the tests with `python -m pytest` from the repo root.
