# IAA hackathon notebook kit

A notebook skeleton that runs end to end today on the 2025 data (3 minutes in fast mode). On October 16 we point it at the new data, rewrite the data-specific rules, and log the exchanges as we work.

## Files

| File | What it is |
|---|---|
| `iaa_notebook.ipynb` | The notebook, executed on the 2025 data. Follows the IAA template headings 1 to 5, with the rate-filing structure inside. |
| `build_notebook.py` | Builds the notebook from its parts. `python3 build_notebook.py --execute --data <folder>` runs it and fills the summary numbers. |
| `ai_log.json` | The AI decision log. Each entry becomes an exchange card and a ledger row. Pre-filled with eight real 2025 exchanges, marked "2025 example". |
| `prompt_library.md` | Set-up prompt for each AI role, the challenge prompts, and the logging rule. Rendered in section 5. |
| `ai_transcript.md` | Condensed transcript of the preparation sessions. Attach to the Kaggle notebook as a dataset or paste into section 5. |

## How the notebook is laid out

1. **Summary**: filing-style summary, numbers filled automatically from the run.
2. **Generative AI integration**: the working loop, the decision ledger, the AI roles table.
3. **Step-by-step analysis**: 3.1 data, 3.2 cleaning, 3.3 modelling, 3.4 results. Each opens with what a reviewer asks (grey box), shows exchange cards where AI changed a decision, and explains terms in plain language (green boxes).
4. **Findings and insights**: plain-language findings (written last) and the judgment register.
5. **Reproducibility**: seeds, versions, prompt library, transcript, and the planned independent review.

## On October 16: what changes

| Where | Change |
|---|---|
| Configuration cell | Competition slug, `TARGET`, `ID`, team name |
| Section 3.2 rules | `NA_NONE`, `NA_ZERO`, estimation rules, ordinal maps, `exclusion_rule`, `engineer`, `SALE_TIME_VARS`, `SEGMENTS` (data steward drafts, actuary checks) |
| Section 4 | `JUDGMENTS`, decided by the actuary |
| `ai_log.json` | Replace the 2025 examples with the live exchanges, logged as we work |
| Final run | Set `FAST = False` (three split seeds, full tree settings), then submit `submission.csv` and record the public score in the summary |

If the new task is not a regression on a log-scale target, the model zoo and metric in 3.3 change too.

## On Kaggle

Upload `iaa_notebook.ipynb`, attach the competition data, and run the install line at the top of the rating-plan cell (`!pip install -q midlearn` and the R packages) if the glass-box step is wanted. The AI log cell is marked "hide input" so the page shows the ledger and cards, not the raw JSON. If R cannot be installed, the notebook skips the glass box and still runs.

## Before packaging: decisions still open

J1 intended use, J3 non-market lookalikes, J4 blend or rating plan as the filed model, J5 shape of the 90% range. The skeleton uses the recommended defaults until these are signed.

Fast-mode scores are slightly worse than the full run (blend 0.0900 against 0.0893 on seed 42) because the tree models use fewer trees.
