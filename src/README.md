# 2026 pipeline

Data-agnostic modules, built and tested on synthetic data before the competition opened. Nothing here names a 2026 column, cleaning rule or threshold value; thresholds are read from `decisions/thresholds.md`, and a threshold without a "Written on" date is refused.

All predictions are carried on the link scale (log, logit, or log of 1 + y for a price); `families.py` converts and scores them.

| Module | Contents |
|---|---|
| `families.py` | Log-Gaussian (RMSLE scale), Poisson, gamma, Tweedie, binomial: link, variance, per-row loss, score, LightGBM objective |
| `thresholds.py` | Reads `decisions/thresholds.md`; refuses missing, open or undated thresholds |
| `validation.py` | K-fold on seeds 42, 1, 2 (optionally grouped or stratified); hold-out drawn once, written to disk, opened once; out-of-fold runner; paired bootstrap; drop-one verdict |
| `glm.py` | `ForwardGLM`: forward selection on cross-validated error, level pooling, IRLS with offsets and weights, relativities with standard errors, centred terms |
| `trees.py` | `TreeModel`: LightGBM with the matching objective, starting from the same base and offset as the GLM; exact Shapley contributions |
| `pipeline.py` | `ModelPair` (fixed 50/50 blend on the link scale), `ReferralRule` (out of evidence, model disagreement), `PredictionRanges` (by predicted-value fifth, cross-fitted coverage) |
| `evidence.py` | Exhibit calculations for the filing template; `write_evidence` writes the JSON the notebook reads |
| `exhibits.py` | Notebook tables, charts, exchange cards and ledger, all drawn from the evidence and the AI log |

Still to write once the data arrives: `clean.py` (data-specific, from the data dictionary), `explain.py` (relativities against ALE curves, adjustment drivers, permutation importance), and the run script that ties them together. `tests/synthetic_run.py` shows the shape of that run script.

`reference/2025/decided_model/v2lib.py` is the 2025 version of most of this. Its data-specific rules are not carried over.

Run the tests with `python -m pytest` from the repo root (about 30 seconds).
