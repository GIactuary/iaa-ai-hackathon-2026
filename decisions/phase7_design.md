# Phase 7: opening the hold-out, and the final refit (design)

Written 2026-10-20, before the hold-out was opened. Validation design section 6 (V1): the frozen pipeline is fitted on non-hold-out rows up to 2022 and scores the hold-out customers' 2023 rows, then fitted up to 2023 and scores their 2024 rows. The hold-out is opened once, with the reason recorded in the lock file.

## 1. Reason for opening (recorded in the lock)

"Phase 7: the model was frozen on 2026-10-20 (D25, D26). Score the frozen pipeline once on the locked customers' 2023 and 2024 rows, then refit on all customers for 2025 so their claims history enters the price."

## 2. Hold-out scoring

- **Fits.** Exactly the step 7 out-of-time fits: each family in the frozen blend, with its tuned setting and tree count, fitted on the non-hold-out rows of scoring A's (2021-2022) and B's (2021-2023) fitting years, with their frozen claims history and k.
- **Scored rows.** The hold-out customers' rows of 2023 (A) and 2024 (B; completion applied, as in every earlier test). Their claims history is built point in time on all customers (`scripts/phase5_step4.py` with `P5_HOLDOUT=include`), so it holds only claims reported before each row's cutoff.
- **Prices.** The frozen blend: the pooled step 8 weights, a log average in buildings and a plain average elsewhere, and the referral rule (motor rows where the families disagree by more than 30% at the W2 GLM's price).
- **Compared with:** the W2 GLM, and the plain baseline GLM submitted on 18 October (phase 4).
- **Reported:** ranking D² per line and combined; deviance saved by the blend against the W2 GLM with the 95% customer-bootstrap interval; the level ratio of the W2 GLM priced by the decided chain (excess factor, off-balance to the weather-normalised indication for the scored year, the year's weather as one factor per line) against actual; and the same figures on the non-hold-out rows (steps 7 and 8) beside them.
- **Stop rule, written first.** If the blend ranks worse than the W2 GLM on the hold-out rows in both years, the frozen model goes back to the actuary as a judgment before any further file is built. Otherwise the hold-out result is reported and the model stays as frozen.

## 3. Final refit for 2025

- Every family of the frozen blend refitted on all customers' rows 2021 to 2024, with claims history and k rebuilt on all customers, the same settings, tree counts and weights.
- The level chain recomputed on all rows. The 11.9M bodily injury claim (X1 (a): a genuine large loss, capped, its excess spread through the motor load over all four years) now enters the motor excess factor. The change in the motor load is shown to the actuary to confirm, as X1 requires.
- P8-R2 and the referral rule applied as decided. The new file is the candidate for the final submission.

## 4. Gate

The independent review (`/review`) runs on the notebook in a fresh context once phase 8 has written it.

## 5. Thresholds

None new.
