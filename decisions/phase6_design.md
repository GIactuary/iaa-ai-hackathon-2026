# Phase 6: tail, referral and ranges, test design

Written 2026-10-20, before any phase 6 rule was run on the frozen model or tested. Spec section 8 and X3: a referral rule covers records outside the evidence and model disagreement, with thresholds written first. Treatment is a coded rule, never a hand-set price (Foundational 4.b). Under-pricing is the costlier error (J1).

## 1. Referral reasons

| Id | Reason | Rule | Threshold |
|---|---|---|---|
| R1 | Size beyond the evidence | A 2025 row whose size measure exceeds `refer_size_ratio` times the largest training value in its segment (the phase 2 check: buildings sum insured by property type and bedrooms; contents sum insured by tenure and bedrooms; motor vehicle value group, engine size and mileage) | `refer_size_ratio` = 1.25 (written 2026-10-18) |
| R2 | Thin segment (X3) | The R1 segment has fewer than `refer_min_segment` training rows, so its maximum is a weak edge of the evidence | `refer_min_segment` (new) |
| R3 | Model disagreement | Among the families with a positive weight in the row's line, the largest gap between one family's price and the blend's, as a ratio, exceeds 1 + `refer_disagreement` | `refer_disagreement` (new, open since phase 0) |

R3 is measured on the frozen model's own 2025 family predictions (`scripts/freeze_build.py` saves them by row).

## 2. Is disagreement a warning sign? (tested before any treatment is chosen)

On the step 8 out-of-time frames (2023 scored by weights fitted on 2024, and the reverse), with the same rule and threshold:
- the share of rows flagged, per line;
- the cross-fitted blend's Tweedie deviance per unit of expected cost on flagged rows against the rest (at a common level per line), so a higher number means the blend is less reliable there.

## 3. Treatments, tested out of time

Each is a coded rule applied to flagged rows only, compared with the blend on the keep rule (better on A and B, 95% customer-bootstrap interval on A above zero), per line and combined.

| Id | Treatment of R3 rows | Reason to try it |
|---|---|---|
| T0 | Keep the blend price; flag for underwriting review | The default in principle 10: a disagreement alone is not evidence the blend is wrong |
| T1 | The highest family price | Under-pricing is the costlier error (J1) |
| T2 | The W2 GLM's price | The plainest explanation when the models disagree |

R1 and R2 rows are few. Their treatment is chosen by the actuary on the evidence for each row (price within the model's own range of sum insured, or the size capped at the segment edge), since a row-by-row test is not possible.

## 4. Prediction ranges

Per 2025 row, the range from the lowest to the highest price among the families with positive weight, beside the blend: model uncertainty about the expected cost, not an interval for the claim outcome. Reported as the distribution of the range's width per line.

## 5. Thresholds

New rows in `decisions/thresholds.md`, written 2026-10-20 before any phase 6 run: `refer_disagreement`, `refer_min_segment`. `refer_size_ratio` is already written.
