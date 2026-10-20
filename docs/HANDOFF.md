# Hand-off: state of the 2026 entry at 2026-10-19

This note is for a fresh session. Before any work, read it together with `CLAUDE.md`, `docs/PLAYBOOK.md`, `docs/specs/SPEC_pricing_plan.md` and `decisions/judgment_register.md`. Everything below is committed on branch `intake-and-plan`, which has not been merged to `main`. Last commit before this note: `665592d`.

## Where we are

- **Phases 0 to 4 are done.** That covers intake, data rules, screens, the validation design with the hold-out locked, and the baselines.
- **Phase 5 steps 1 to 3 are done and decided:** cap and load, trend and level, weather. Check-in 1 was settled (K1 to K4).
- **Next is phase 5 step 4, claims history**, then steps 5 to 8 (see "Next steps").
- **Freeze the model by the end of Wednesday 21 October.** The actuary submits the final Part 1 file by **Friday 23 October, 12:00 UTC**. The exact close is unpublished; it is assumed to be 23 October.

## How the actuary wants to work

1. **Decisions.** One decision at a time, presented as a decision-brief artifact with cards: question, stakes, evidence, the options considered, a recommendation and its reason.
   - Load the `anthropic-skills:decision-brief` and `artifact-design` skills first.
   - Verdicts come back through the page's `db` store, in the `verdicts` collection; read them with `ArtifactData`.
   - The actuary also likes HTML artifacts for explanations, and asked for a data overview.
2. **Standing rule (actuary, 2026-10-19).** Test as widely as possible. Compute and coding effort are not constraints; explainability is. This is saved in memory as well.
3. **Thresholds** go in `decisions/thresholds.md`, dated, before the test that uses them. `src/thresholds.py` refuses undated ones.
4. **Log every exchange that changes a decision** in `ai/ai_log.json`. The last entry is D16; the next is **D17**. Where the AI was wrong, log it plainly. These cards are the strongest evidence for the AI-use criterion.
5. **No em dashes, plain prose, conclusion first.** Every exhibit gets a "how to read this" note with real numbers.
6. **Before quoting a number to the actuary, recompute it.** Three AI arithmetic errors have reached documents so far: the 2025 premium move, two loss-ratio sets, and peril shares on the overview.

## Validation design (decided: V1, P2)

- **Hold-out.** 15% of customers, seed 2026, locked in `data/work/holdout_lock.json` (gitignored; `scripts/lock_holdout.py` reproduces it). It is never read before phase 7. `src/build.py` flags hold-out rows; exclude them everywhere.
- **Scoring A:** fit 2021 to 2022, score 2023.
- **Scoring B:** fit 2021 to 2023, score 2024, with each prediction scaled by the row's reporting completion.
- **Inner folds** (tuning only): 5 folds grouped by customer, on seeds 11, 22 and 33.
- **Metric.** Row-level Tweedie deviance at power 1.5. Ranking D² is computed at a common level (each line rescaled to its actual total). The level is judged on its own, as the level ratio.
- **Keep rule.** A change must be better in both scorings, and the 95% customer-bootstrap interval on A (2,000 draws) must lie above zero. Use `validation.paired_bootstrap_groups` and `keep_verdict`.

## The current model (working base, decided)

1. **Target.** Net cost per row, matched by loss date. Repudiated claims are excluded. Each claim is capped (C1, K1) at 100k for motor, 200k for buildings and 30k for contents.
2. **Exposure offset.** Exposure × reporting completion (P10). Line completion is weighted by **each year's own peril mix** (the fix in D16, `build.line_completion_by_year`).
3. **Relativities.** One capped Tweedie GLM per line (`src/glm.TweedieGLM`, main effects in `glm.features`, year factor). This is variant V1. Multiply by the line excess factor, total over capped cost: motor 1.031, buildings 1.022, contents 1.018 (K2). Per-peril excess factors come back in step 6.
4. **Level** (P11, C3, C4, K3, K4).
   - The indication is built peril by peril from **weather-normalised** experience. Frequency is normalised with the same-address weather elasticities; trends are bounded at −5% to +12% a year, with bodily injury severity treated as a step from 2023.
   - The off-balance moves the model's latest-year rows to the indication.
   - Multiply by the 2025 weather year factor. Its net effect is −0.04% motor, +0.06% buildings, −0.28% contents.
   - The C4 band is ±3% of the indication. After the fix all lines pass: motor −0.7%, buildings −1.5%, contents −2.0%.
5. **Weather.** No climate normals in the plan; they failed twice (K4).
6. **Not used as inputs:**
   - renewal_decision and customer_service_contacts (leakage)
   - repairer_id and third_party_id (post-event)
   - marketing and identity fields (P6)
   - written_premium, which is the incumbent only (J-prem)

Evidence: `decisions/phase5_checkin1.md`, `outputs/phase5_steps123*.json` and `outputs/phase5_k3k4.json`. The step 1 to 3 runs used the old completion weights, so ranking comparisons from them stand but their level numbers predate the fix.

## Next steps (agreed plan; brief: https://claude.ai/artifact/H8SnbQNmq2kqjh5uFiWosU)

1. **Step 4: claims history (C7).** Build counts of prior claims for the policy, the customer across lines and the address (property), using only claims reported before the row starts. Weight them with Bühlmann-Straub, Z = n/(n+k), with k estimated from the data. Test each piece with the keep rule against V1.
2. **Step 5: engineered features (C8, widened under the standing rule).**
   - The base list: building age, sum insured per bedroom, a flood score (zone × river distance × elevation, keeping the blank-zone level), dry summer × clay, vehicle value against engine size, and the NCB-above-licence and licence-before-16 flags.
   - Plus every other candidate with a risk rationale, for example telematics score bands with the opt-in flag, missed payments as yes/no against 0/1/2+ (D9), age × NCB, mileage bands, days unoccupied bands, listed building, basement, trees, roof type, energy rating, lodgers, works from home, bicycle cover, make-model by credibility encoding, licence points and years licensed.
   - Each one is drop-one tested.
3. **Check-in 2 with the actuary:** features kept and dropped.
4. **Step 6 (P3).** Tweedie per line against frequency × severity per peril. Retest the per-peril excess factor (K2, C2) here.
5. **Step 7 (P7, C5).** Model families: spline/elastic-net GLM, EBM, LightGBM and CatBoost, and boosting started from the GLM prediction. Tune on inner folds. Grid the training Tweedie power from 1.1 to 1.9 in steps of 0.1. Keep the best model per family.
6. **Step 8 (J-model).** Fit non-negative family weights out of time and score them on rows not used to fit them. The within-family blend is the challenger.
7. **Check-in 3:** freeze the model; build a second submission file for the actuary to upload, and log it in `decisions/submissions.md`.
8. **Phase 6: tail and referral.** `refer_disagreement` is still open and needs a written value. Add a minimum segment size to the referral rule (X3); write the threshold first.
9. **Phase 7.** Open the hold-out once with a recorded reason, as `decisions/validation_design.md` section 6 describes. At the final refit on all data, show X1: the 11.9M bodily injury claim is in the hold-out, and it lifts the motor load from about 2.8% to about 5.5%. The actuary confirms it then.
10. **Phase 8.** Write-up (5 pages), video script, and the notebook or report via `notebooks/build_notebook.py`. Run the independent review (`/review`).

## Code map

| File | What it does |
|---|---|
| `src/data_rules.py` | Repairs R1 to R9 and claim matching |
| `src/screens.py` | Phase 2 screens |
| `src/targets.py` | Delay curves, completion, row targets (capped) |
| `src/build.py` | Cached model-ready rows: `build.build_all(rebuild=True)` after any data-rule or threshold change. The cache is in `data/work/lines/` |
| `src/glm.py` | Features and the Tweedie GLM with an exact offset |
| `src/level.py` | Excess factors, trends, indication, off-balance, `_ec` with weather multipliers |
| `src/weather.py` | Normals, deviations, elasticities (standardised), year factor |
| `src/validation.py` | Folds, hold-out lock, deviance, D², common level, customer bootstrap, keep verdict |
| `scripts/` | `phase4_baselines.py`, `phase5_steps123.py`, `phase5_steps123_run2.py`, `phase5_k3k4.py`, `data_overview.py`, `lock_holdout.py` |

- **Tests:** `.venv/Scripts/python.exe -m pytest -q`, 53 passing.
- **Run Python as** `.venv/Scripts/python.exe -I`. Long runs go in the background; a full line-by-line run takes about 15 to 30 minutes.

## Artifacts (private to the actuary)

| Artifact | Link |
|---|---|
| Plan brief, finalized | https://claude.ai/artifact/PhDnuDVjiDGdT332jEPHz5 |
| Phase 5 modelling calls (C1 to C8) | https://claude.ai/artifact/H8SnbQNmq2kqjh5uFiWosU |
| Data overview | https://claude.ai/artifact/2kx18DjTnTneRfVobfUJPT |
| Check-in 1 (K1 to K4) | https://claude.ai/artifact/JJmj4pzgSoUcnT81N1tSPr |

## Known facts and caveats

- **Public leaderboard.** The baseline plain GLM scored 0.11710 (16th; the leader was 0.12996). The public board uses about 33% of test rows. Kaggle's D² baseline seems to differ from ours, so compare by order, not size.
- **The 2025 premium rate** is up 2.6% to 5.6%, not 36% to 44% (D13).
- **Data.** UK-like but stylised. No external data (J-ext).
- **Customers table.** It is a 2025 snapshot; J-cust says use it as given.
- **Claims-handling finding.** Three repairers (RP0733, RP0901, RP0417) show a fraud or repair-inflation pattern. Their claims stay in the target (X2).

## Update after check-in 2 (2026-10-19, commit 80e4a3a)

- **Steps 4 and 5 are done and decided** (L1 to L8, all (a); D17 to D19).
  - Working base W2: `decisions/phase5_checkin2.md`, last section.
  - Check-in 2 page: https://claude.ai/artifact/8tRL6guGbPhmu7dM4CQ3ZG.
- **Claims history.**
  - `src/history.py` builds point-in-time A and E per scope.
  - k is the Poisson-gamma moment estimate (`k_poisson_gamma` in `data/work/phase5_step4/k.json`; run with `P5_K=pg`). The non-parametric estimate failed twice (D17, D18): do not use it at these claim frequencies.
  - The GLM takes numeric terms now (`glm.TweedieGLM`, float columns).
- **Caveat for the final refit.** History for hold-out customers' 2025 rows is empty until phase 7. Rebuild the history on all customers at the final refit.
- **Next: step 6.**
  - Tweedie per line against frequency × severity per peril, on W2, with the per-peril excess factor retested.
  - Step 7 carries these extra candidates: customer, address and repudiation history; years licensed; property voluntary excess; the flood measures.
  - Motor step 5 fits were slow, about 2 hours for 24 candidates. Budget for it.

## Update after check-in 3 (2026-10-19)

- **Step 6 is done and decided** (M1 to M3; D20). Evidence: `decisions/phase5_checkin3.md`, `outputs/phase5_step6.json`. Page: https://claude.ai/artifact/5wm9GbRV98cZ9zxKPy1wiF (generator `scripts/checkin3_page.py`).
- **Structure (M1 a).** One capped Tweedie GLM per line stays in all three lines; no per-peril structure passed anywhere. In step 7, each family also builds frequency × severity per peril for motor only (the near miss: +12,541 on A [−12,076, 34,902], +35,947 on B). Property stays per line in every family.
- **Excess (M2 a).** The line excess factor everywhere, including any per-peril motor model. C2's per-peril form is retired on evidence.
- **Weather (M3 b).** The 2025 weather enters as one exposure-weighted factor per line, not row by row. Apply it when the final pricing chain is assembled.
- **Code.** `scripts/phase5_step6.py` fits the structures (`fit_plan`); `TweedieGLM.fit` takes a `weight` for gamma severity (ec = n, weight = n). The full run took about 35 minutes with 4 workers.
- **Next: step 7 (families, P7, C5)**, then step 8 (weights), then the freeze by the end of 21 October.

## Update after check-ins 4 and 5 (2026-10-20): the model is frozen

- **Step 7** (families) and **step 8** (weights) are done; check-ins 4 and 5 decided N1 to N3 (a), O1 (a) and O2 (b) (D24, D25). Pages: https://claude.ai/artifact/Fa2ytmQAp5sZwCHycqR4oQ and https://claude.ai/artifact/LHMvjWt3Pzr7bbVX8CaUgK.
- **Frozen model:** `scripts/freeze_build.py` writes `outputs/submissions/blend_frozen_v1.csv`. Per line, the families are refitted on 2021-2024 and mixed with the pooled step 8 weights; line totals come from the decided level chain (`scripts/submit_w2.py`, file `w2_glm_v1u.csv`).
- **Interim submission:** w2_glm_v1u scored 0.12695 public (10th; leader 0.12996).
- **P8-R2** (D22): contents blanks left after recovery are priced at the known-level average in every family.
- **Errors logged:** D22 (the thin blank levels) and D23 (penalised per-peril GLM penalty).
- A forked session drafted a different check-in 5; it was reverted (0bfc720). Only this session's check-in 5 stands.
- **Next: phase 6 (tail and referral, `refer_disagreement` still open), phase 7 (open the hold-out once), phase 8 (write-up, notebook, review).** At the final refit, rebuild the claims history on all customers (hold-out customers' 2025 history is empty until phase 7).

## Update after check-in 6 (2026-10-20): phase 6 done

- Referral rule run under thresholds written first (`decisions/phase6_design.md`; refer_disagreement 0.3, refer_min_segment 100). Q1 (a), Q2 (a) decided (D26; cards P1 and P2 on the page). Page: https://claude.ai/artifact/AqYBZKnd3DSWkWRnwKf9LB.
- The candidate final file is `blend_frozen_v2.csv` (`scripts/freeze_build.py`, then `scripts/phase6_referral.py`, then `scripts/apply_referral.py`). blend_frozen_v1 scored 0.12821 public (8th).
- Next: phase 7, opening the hold-out once with a recorded reason (`decisions/validation_design.md` section 6), and X1 at the final refit. Then phase 8: the write-up, the notebook via `notebooks/build_notebook.py`, and `/review`.
