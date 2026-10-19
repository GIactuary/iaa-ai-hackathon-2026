# Phase 5 step 6: model structure (P3) and the per-peril excess factor (K2, C2), test design

Written 2026-10-19, before any step 6 model was fitted or scored. Decision P3 (c): build both structures and let the test decide per line. K2 (a) deferred the per-peril excess-loss factor to this step; C2 (a) set its form.

## 1. Base and what changes

- **Columns.** Every model in this step uses the working base W2 columns of its line (`decisions/phase5_checkin2.md`, last section), unchanged. Only the structure changes.
- **Data.** Non-hold-out customers; scorings A (fit 2021-2022, score 2023) and B (fit 2021-2023, score 2024 with completion). Every net cost is positive (no nil claims after repudiated claims are removed), so a claim count and an average cost per claim are defined for every claim.

## 2. Structures

| Id | Structure | Fitted on | Offset or weight |
|---|---|---|---|
| S1 | One capped Tweedie GLM per line (W2, the reference) | Capped line cost | exposure × line completion |
| S2 | One capped Tweedie GLM per peril, summed | Capped peril cost | exposure × the peril's completion |
| S3 | Poisson frequency × gamma severity per peril, summed; both parts on the W2 columns | Frequency: the peril's claim count. Severity: capped cost per claim on rows with a claim | Frequency: exposure × the peril's completion. Severity: weight = number of claims, no completion (late claims are not larger here, `src/targets.py`) |
| S4 | As S3, with severity by year only (no rating factors) | As S3 | As S3 |

- S2 separates "one model per peril" from "frequency and severity apart". S4 tells whether the severity relativities add anything beyond the frequency ones.
- All GLMs carry the year factor and are read at the latest fitting year. Training power: 1.5 for Tweedie, 1 for Poisson, 2 for gamma. The power grid is step 7.
- `TweedieGLM` gains an optional row weight for the severity fit (sum of n claims, average cost weighted by n).

## 3. Excess factor

- **L (line factor).** Every peril, and S1, times the line factor: total over capped cost on the fitting rows (K2 (a)).
- **P (per-peril factor, C2 (a)).** Each peril times its own factor on the fitting rows. Only S2, S3 and S4 can take it.
- Seven models per scoring: S1-L, S2-L, S2-P, S3-L, S3-P, S4-L, S4-P.

## 4. Two scoring forms

1. **Bare.** The model's rate at the latest fitting year × exposure × excess factor (× completion on B: the line's for S1, each peril's own for S2 to S4). This is the form of steps 1 to 5 and measures the relativities alone.
2. **Priced.** The bare form with the level chain the 2025 price uses (K3, K4):
   - an off-balance to the weather-normalised, trended indication for the scored year, built from the fitting years only: one factor per line for S1, one per peril for S2 to S4;
   - the model's latest-year level is read at normal weather (each row's rate divided by its weather multiplier for that year) before the off-balance, so a heavy-weather latest year does not distort it;
   - times the scored year's weather: the line year factor for S1, each adjusted peril's own multiplier for S2 to S4.
- A per-line factor changes nothing at a common level, so for S1 the priced form differs from the bare form only through the weather. For S2 to S4 the per-peril off-balance moves the peril mix within each row. This is the classic case for a per-peril structure: each peril's own trend and step (bodily injury from 2023) reaches each row in proportion to its own mix.

## 5. Tests and decision rule

- Every comparison: per line at a common level (each line rescaled to its actual total), deviance saved on A with the 95% customer-bootstrap interval (2,000 draws) and on B. Keep rule: better on A and B, and the A interval above zero. The combined figure (all lines, customers resampled across lines) is reported beside each line.
- **Tests run.**
  1. Each of S2-L, S3-L, S4-L against S1-L: the structure question.
  2. Each of S2-P, S3-P, S4-P against the same structure with L: the K2 retest.
  3. Each of S2-P, S3-P, S4-P against S1-L: structure and excess together.
  4. For information: S3-L against S2-L (does the frequency-severity split help) and S4-L against S3-L (do severity relativities help).
- **Decision rule, per line, on the priced form.**
  - A challenger replaces S1-L only if it passes the keep rule against S1-L. Among those that pass, the recommendation is the one with the largest saving on A; a simpler structure (S2 before S4 before S3, and L before P) is preferred when its A saving is within the other's 95% interval of the paired test between them.
  - The per-peril excess factor is recommended only if Sx-P passes the keep rule against Sx-L for the recommended structure.
  - If nothing passes, S1-L stays and K2 (a) stands: the line factor.
  - The bare form is reported beside the priced form. If the two disagree on the sign of a line's verdict, that line goes to the actuary as its own item, with both readings.
- **Diagnostics (information, not decisions).**
  - Per peril: each structure's peril component scored against that peril's actual cost (common level per peril), to show which perils gain from the frequency-severity split.
  - Level ratio of the priced form (prediction over actual, B with completion), per line.
  - Per-peril excess factors on each plan, and per-peril frequency and severity relativities of the final fit (2021-2024) for the exhibits.

## 6. Thresholds

No new threshold. The step uses `keep_years_better` and `keep_ci`, already written, and the caps, trend bounds and weather elasticities as decided (C1, C3, K4).
