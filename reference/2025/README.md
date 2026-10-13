# 2025 reference code and results

Working code from the 2025 preparation. It ran against a work folder laid out as `data/train.csv`, `data/test.csv`, `data/data_description.txt` and `out/`, with `WORK=<folder>` set. The 2025 data is not included.

| Folder | Contents |
|---|---|
| `decided_model/` | The decided model. `v2lib.py`: cleaning (via `mine_models.clean`), forward-built GLM, LightGBM wrapper, 50/50 pipeline, referral rule with size cap and comparable-sale rating. `v2_oof.py`: out-of-fold runs on three seeds and the drop-one test. `v2_full.py`: locked hold-out, full fit, test prices, explanation curves, Shapley adjustment drivers, permutation importance. `v2_evidence.py`: every number the filing shows |
| `winner_reproduction/` | Faithful rerun of the 1st-prize notebook, its hold-out run and the paired comparison |
| `other_winners/` | Reruns of the 2nd and 3rd prizes on our folds, and the comparison assembler |
| `notebook_kit/` | The notebook builder (IAA template headings, exchange cards, ledger, hidden AI-log cell) with the 2025 example log |
| `results/` | Cross-validated scores of every method on identical folds, the leaderboard, the decided model's results, and the 2025 filing page (an earlier version, before the decisions) as a format example |

What to reuse: the structure, the validation code, the forward GLM, the referral logic, the evidence assembly, the notebook builder. What not to reuse: any column name, cleaning rule, threshold value or exclusion. Those come from the 2026 data.
