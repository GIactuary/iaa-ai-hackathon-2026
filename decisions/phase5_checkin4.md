# Phase 5 check-in 4: model families (2026-10-20)

Run: `scripts/phase5_step7.py` (one line at a time, then `tests`), evidence in `outputs/phase5_step7.json`; direct comparisons against the W2 GLM in `data/work/phase5_step7/direct_tests.json`; explanation of boosting from the GLM in `data/work/phase5_step7/f4_carried_explain.json`. Design written and committed first: `decisions/phase5_step7_design.md` (66d283f), with the section 7 amendment (standardised GLM, D21) made before any out-of-time score. Every test is out of time on non-hold-out customers, at a common level per line, under the keep rule. The decisions are register items N1 onward.

## Result in one line

No family beats the W2 GLM on its own. Boosting from the GLM, given the held-over factors, does: across lines +57,963 on 2023 [23,812, 89,927] and +77,222 on 2024, and it passes in buildings by itself. Plain boosting (LightGBM, CatBoost) ranks far worse than the GLM in every line.

## Each family's model against the W2 GLM

Deviance saved, A with its 95% customer-bootstrap interval, then B.

| Family (chosen on inner folds) | Motor | Buildings | Contents | All lines |
|---|---|---|---|---|
| F1 penalised GLM (spline in motor and contents; elastic net in buildings) | +8,000 [−22,858, 39,722], −906 | −6,019 [−20,097, 7,423], −6,039 | +13,776 [−939, 27,773], +15,576 | +15,757 [−22,148, 52,440], +8,631 |
| F2 plain boosting (CatBoost in motor and buildings; LightGBM in contents) | −123,476, −80,698 | −42,945, −84,103 | −75,907, −40,448 | −242,328 [−365,672, −118,968], −205,248 |
| F3 EBM | −15,962, +6,990 | −15,600, −21,966 | −25,146, −11,200 | −56,707 [−128,410, 19,328], −26,176 |
| F4 boosting from the GLM | −6,690, −3,782 | −14,229, −13 | −2,976, −337 | −23,895 [−53,008, 4,391], −4,132 |

Ranking D², all lines: W2 GLM 10.51% on A and 10.12% on B; F1 10.53% and 10.14%; F2 10.09% and 9.76%; F3 10.41% and 10.08%; F4 10.47% and 10.12%.

**Reading.** The W2 GLM, built feature by feature in steps 4 and 5, is hard to beat. Trees on the same raw inputs rank worse; on the inner folds too, so this is not an out-of-time accident. The data behave like a multiplicative rating structure, which a GLM captures with few parameters.

## Held-over factors (L4, L7)

Within each family, the group added to the family's model:

| Group added to | Motor | Buildings | Contents | All lines |
|---|---|---|---|---|
| F4 boosting from the GLM | +25,332 [9,714, 41,089], +821: passes | +50,235 [20,218, 78,192], +66,280: passes | +6,290 [−7,796, 20,516], +14,252 | +81,858 [42,577, 118,529], +81,354: passes |
| F1 GLM | +21,813, +24,495 | −22,619, +117,045 | +7,962, +15,244 | +7,156, +156,784 |
| F2 plain boosting | +14,671, +25,302 | +59,436, +174,999 | +2,885, +23,321 | +76,992, +223,622 |
| F3 EBM | −3,974, +7,966 | −425, +93,311 | −11,160, −36,545 | −15,560, +64,732 |

One at a time in the GLM, none passes: years licensed in motor +22,223 [−13,718, 55,314], +25,818; voluntary excess in contents +11,052, +14,741; flood measures in buildings −23,156 on 2023, +103,436 on 2024.

**Boosting from the GLM with the held-over factors, against the W2 GLM:** motor +18,642 [3,002, 34,324], −2,961; buildings +36,006 [10,603, 58,867], +66,267 (passes); contents +3,314 [−13,845, 18,805], +13,915; all lines +57,963 [23,812, 89,927], +77,222 (passes).

**What the correction is.** A small adjustment to the GLM's rate, from few trees:
- buildings: 8 trees, a mean absolute adjustment of 0.054 to the log rate (about 5%), led by elevation, voluntary excess, flood zone and claims history; homes at 2 m or lower ×1.055, the rest about ×0.97;
- motor: 23 trees, 0.039 (about 4%), led by driver age, years licensed, policy history and region;
- contents: 20 trees, 0.067 (about 7%), led by voluntary excess and customer history.

## Frequency × severity in motor (M1 (a))

- **EBM:** per peril against the per-line EBM +40,237 [9,211, 69,872], +43,761: passes. Against the W2 GLM +24,275 [−17,227, 68,486], +50,751.
- **LightGBM:** per peril against per-line LightGBM +101,833 [51,574, 156,912], +67,975: passes. Against the W2 GLM −32,917, −36,213.
- **Penalised GLM:** invalid, set aside (D23). The per-line penalty, reused on Poisson and gamma fits, flattened the relativities. The GLM evidence on this structure is step 6's: it failed the keep rule.

## Standardised inputs (D21)

Standardised, the penalty tuning chose a strong lasso in contents (α 0.01, L1 only) where the unscaled GLM chose almost none. Out of time the standardised contents GLM is level with the W2 GLM on 2023 and 50,180 better than the unscaled one on 2024. Its spline version is F1's pick in contents.

## Where the AI was wrong

- **D23.** The per-peril penalised GLM reused the per-line penalty strength, which is far too strong on the Poisson and gamma scales.
- **D22** (found while building the interim file). The P8-R rule left two contents blank levels too thin to estimate. They were priced near zero in the baseline submission; fixed under P8-R2 (a).

## Cards for the actuary (round 1)

- **N1.** Boosting from the GLM: the held-over factors join its model in motor and buildings, where they pass; contents keeps the base.
- **N2.** Motor: the EBM and LightGBM use frequency × severity per peril; the GLM stays per line.
- **N3.** Whether the final price may include a boosted correction on top of the GLM, if the step 8 weights favour it.

## Verdicts (actuary, 2026-10-20, check-in 4 round 1)

N1 (a), N2 (a), N3 (a): the recommendations, with no notes. Exchange D24.

What changes for step 8:
- **F4 (boosting from the GLM)** uses the held-over factors in motor and buildings; contents keeps the F4 base.
- **Motor:** the EBM (F3) and LightGBM (F2) use frequency × severity per peril; the GLM stays per line.
- **The boosted correction may enter the final price**, explained as the GLM times an adjustment by factor.
