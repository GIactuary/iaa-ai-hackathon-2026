# Phase 5 check-in 5: family weights and the freeze (2026-10-20)

Run: `scripts/phase5_step8.py`, evidence in `outputs/phase5_step8.json`. Design written and committed before any step 7 result was seen: `decisions/phase5_step8_design.md` (4334a9c). Family columns set from the check-in 4 verdicts (D24). Weights are fitted on one scoring and scored on the other (cross-fitted); keep rule as before.

## Result in one line

The weighted blend of the families beats the W2 GLM: across lines +92,357 on 2023 [46,571, 139,010] and +114,384 on 2024. It passes in motor and buildings; contents is a near miss.

## The blend against the W2 GLM (cross-fitted, arithmetic mix)

| Line | A saved [95%] | B saved | Keep |
|---|---|---|---|
| Motor | +37,467 [3,827, 73,728] | +45,313 | Yes |
| Buildings | +42,006 [18,729, 62,383] | +52,783 | Yes |
| Contents | +12,884 [−1,734, 26,978] | +16,288 | No (interval) |
| All lines | +92,357 [46,571, 139,010] | +114,384 | Yes |

Ranking D², all lines: W2 GLM 10.51% (A) and 10.12% (B); blend 10.67% and 10.33%.

Against each single family, across lines, the blend passes against all four (F4, the closest: +40,684 [497, 82,611], +51,414).

## Weights

| Line | Family | Fitted on 2023 | Fitted on 2024 | Pooled (for 2025) |
|---|---|---|---|---|
| Motor | EBM per peril (F3) | 0.455 | 0.800 | 0.666 |
| Motor | GLM + boosted correction (F4) | 0.454 | 0.200 | 0.334 |
| Motor | LightGBM per peril (F2) | 0.091 | 0.000 | 0.000 |
| Buildings | GLM + boosted correction (F4) | 0.679 | 0.962 | 0.798 |
| Buildings | CatBoost (F2) | 0.321 | 0.038 | 0.202 |
| Contents | Spline GLM (F1) | 0.974 | 0.830 | 0.910 |
| Contents | LightGBM (F2) | 0.017 | 0.094 | 0.058 |
| Contents | EBM (F3) | 0.009 | 0.076 | 0.032 |

Every other weight is zero in all three fits, including the W2 GLM's in every line: its information reaches the blend through F4, which starts from it.

**Flagged swings (design section 4).** Buildings CatBoost (0.321 to 0.038) and motor F4 (0.454 to 0.200). The cross-fitted scores, which use the other year's weights, still pass, so the blend is not fragile to the swing.

## Challengers

- **Within-family blend** (every candidate weighted): worse than the family blend, −36,174 on 2023 [−80,645, 8,566] and −74,487 on 2024 across lines. Not kept.
- **Geometric mix**: +3,358 [284, 6,566] and +1,200 across lines; passes in buildings (+3,323 [424, 6,188], +2,865), not in motor (+29, −1,298) or contents (+7, −368). Weights almost the same as the arithmetic mix.

## Cards for the actuary (round 1)

- **O1.** Freeze the blend as the Part 1 model (the final price), or keep the W2 GLM.
- **O2.** Arithmetic mix everywhere, or geometric in buildings where it passes.

## Verdicts (actuary, 2026-10-20, check-in 5 round 1)

O1 (a), O2 (b), with no notes. Exchange D25.

- **The frozen Part 1 model is the blend** in all three lines, each family refitted on 2021 to 2024 and mixed with the pooled weights. The decided level chain applies per line: the line excess factor, the off-balance to the weather-normalised 2025 indication, the 2025 weather as one factor per line. P8-R2 applies to every family.
- **Mix:** a weighted log average in buildings, a plain weighted average in motor and contents.
- A forked session drafted a different check-in 5 (O1 to O3, a second page) at the same time; the actuary decided on this page and had that draft reverted (0bfc720).

## Per-peril EBM in property (step 7 design section 8; tested 2026-10-20 after the freeze, at the actuary's request)

Stage 1 fails in both lines, so stages 2 and 3 do not run and the frozen model stands. Against the per-line EBM: contents −16,027 on 2023 [−43,505, 9,334], +5,411 on 2024; buildings −40,535 [−99,080, 11,802], +7,257. Against the W2 GLM: contents −41,173 [−72,132, −12,784], −5,789; buildings −56,135 [−138,176, 20,454], −14,709. Splitting property by peril divides too few claims, as the GLM showed in step 6. Evidence: `outputs/phase5_step7b.json`; exchange D27.
