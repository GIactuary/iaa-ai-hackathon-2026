# Phase 5 step 8: family weights (J-model), test design

Written 2026-10-19 while step 7 was running, before any step 7 result was seen. Decision J-model (a): non-negative weights across families, fitted on out-of-time predictions and scored on rows not used to fit them; a within-family blend is the challenger (b).

## 1. Inputs

The step 7 out-of-time predictions saved by row (`data/work/phase5_step7/<line>_<A|B>.parquet`): the W2 GLM (`ref`), each family's model (F1 to F4), each candidate (`cand:*`) and, in motor, frequency × severity per family. Each prediction already carries the line excess factor and, on B, completion. Carried candidates enter a family's prediction only where they passed the keep rule in step 7; the actuary's check-in 4 verdicts settle that before weights are fitted.

## 2. The blend

- **Form.** Per line, an arithmetic mix of the family predictions, ŷ = Σ wₖ ŷₖ with wₖ ≥ 0 and Σ wₖ = 1. An arithmetic mix keeps the blend's total equal to the weighted totals and reads as "so much GLM, so much boosting". A geometric (log-scale) mix is the challenger.
- **Fit.** The weights minimise total Tweedie deviance at power 1.5 on the fitting rows, at a common level (each family's prediction rescaled to the actual total of the rows first, so the weights decide ranking only). Solved on the simplex by projected gradient from equal weights; the result does not depend on the start because the objective is convex in w.
- **Honest score (cross-fitting).** Weights fitted on scoring A's rows (2023) are scored on scoring B's rows (2024), and weights fitted on B are scored on A. No row is scored by weights fitted on it. The keep rule then reads: better than the reference on both cross-fitted scorings, with the 95% customer-bootstrap interval on A above zero (A is scored by weights fitted on B).
- **Candidates for the reference.** The W2 GLM alone, and the single best family on its own (by the same cross-fitted score).

## 3. Challengers

1. **Within-family blend (J-model (b)).** Every candidate of every family enters the weights, not only each family's strongest.
2. **Geometric mix.** As section 2, on log predictions.
3. **Motor structure.** Frequency × severity versions take their family's place where they passed in step 7.

A challenger replaces the section 2 blend only if it passes the keep rule against it on the cross-fitted scores.

## 4. The weights for 2025

The final weights are fitted on A's and B's rows pooled (2023 and 2024), the most recent experience. Both cross-fitted weight sets are reported beside them, so the actuary can see how stable they are. A weight that swings from near zero to large between the two years is flagged.

## 5. Thresholds

None new. The keep rule (`keep_years_better`, `keep_ci`) applies. Weights below 0.01 are reported as zero and the remaining weights rescaled to sum to 1; this is a display rule, not a test, and the scored blend uses the unrounded weights.
