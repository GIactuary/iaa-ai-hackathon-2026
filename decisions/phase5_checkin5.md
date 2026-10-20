# Phase 5 check-in 5: family weights and the frozen model (2026-10-20)

Run: `scripts/phase5_step8.py`, evidence in `outputs/phase5_step8.json`. Design written and committed before any step 7 result was seen: `decisions/phase5_step8_design.md` (4334a9c). Family columns follow the check-in 4 verdicts (N1, N2): motor uses the EBM and LightGBM by peril and boosting from the GLM with the held-over factors; buildings uses boosting from the GLM with the held-over factors; contents uses each family's tuned model.

## Result in one line

A weighted mix of the families beats the W2 GLM: across lines +92,357 on 2023 [46,571, 139,010] and +114,384 on 2024, passing in motor and buildings; contents saves in both years but its 2023 interval crosses zero. Each year is scored by weights fitted on the other year only.

## How the weights were tested

Weights fitted on the 2023 predictions score 2024, and weights fitted on 2024 score 2023, so no year is scored by weights fitted on it. Weights are non-negative and sum to one per line; every family's prediction is first rescaled to the actual total, so the weights decide ranking only.

## The blend against the W2 GLM (cross-fitted)

Deviance saved, A with its 95% customer-bootstrap interval, then B.

| Blend | Motor | Buildings | Contents | All lines |
|---|---|---|---|---|
| Arithmetic mix of families | +37,467 [3,827, 73,728], +45,313: passes | +42,006 [18,728, 62,383], +52,783: passes | +12,884 [−1,734, 26,978], +16,288 | +92,357 [46,571, 139,010], +114,384: passes |
| Geometric mix of families | +37,496 [3,509, 74,095], +44,015: passes | +45,329 [22,372, 66,041], +55,648: passes | +12,890 [−1,689, 27,112], +15,920 | +95,715 [49,310, 142,446], +115,583: passes |

Geometric against arithmetic: motor +29 [−993, 1,029], −1,298; buildings +3,323 [424, 6,188], +2,865 (passes); contents +6 [−877, 751], −368; all lines +3,358 [284, 6,566], +1,199 (passes).

**Within-family blend (J-model (b)), every candidate weighted:** worse than the family blend, −36,174 across lines on 2023 [−80,645, 8,566] and −74,487 on 2024. J-model (a) stands: one model per family.

Ranking D², all lines: W2 GLM 10.51% (A) and 10.12% (B); family blend 10.67% and 10.33%. Per line, A then B: motor 5.68% to 5.83% and 6.35% to 6.54%; buildings 6.90% to 7.11% and 6.63% to 6.90%; contents 6.51% to 6.63% and 5.83% to 5.99%.

## The weights

Arithmetic mix; weights below 0.005 not shown.

| Line | Fitted on 2023 | Fitted on 2024 | Pooled, for 2025 |
|---|---|---|---|
| Motor | EBM by peril 0.455, boosted GLM 0.454, LightGBM by peril 0.091 | EBM by peril 0.800, boosted GLM 0.200 | EBM by peril 0.666, boosted GLM 0.334 |
| Buildings | boosted GLM 0.679, CatBoost 0.321 | boosted GLM 0.962, CatBoost 0.038 | boosted GLM 0.798, CatBoost 0.202 |
| Contents | penalised GLM 0.974, LightGBM 0.017, EBM 0.009 | penalised GLM 0.830, LightGBM 0.094, EBM 0.076 | penalised GLM 0.910, LightGBM 0.058, EBM 0.032 |

The W2 GLM itself takes no weight once boosting from the GLM is available: that model is the GLM times a small correction. Flagged under the design: buildings' CatBoost weight moves from 0.321 to 0.038 between the two years, and motor's EBM from 0.455 to 0.800.

## What each line's price would be

- **Motor:** two-thirds the EBM by peril (frequency and severity curves per peril, read factor by factor), one-third the GLM times its boosted correction.
- **Buildings:** four-fifths the GLM times its boosted correction (led by elevation, voluntary excess, flood zone and history), one-fifth CatBoost.
- **Contents:** under O1 (a), the W2 GLM as now; under O1 (b), nine-tenths the standardised spline GLM.

## Cards for the actuary (round 1)

- **O1.** The frozen model per line: the blend in motor and buildings, the W2 GLM in contents.
- **O2.** The form of the mix: geometric in every line where a blend is used.
- **O3.** The weights for 2025: pooled over 2023 and 2024, as designed.
