# Phase 7 check-in 7: the hold-out, and the final refit (2026-10-20)

Design written and committed before the hold-out was opened: `decisions/phase7_design.md` (04f81df). The hold-out was opened once on 2026-10-20 at 11:46, with the reason recorded in `data/work/holdout_lock.json`. Runs: `scripts/phase7_holdout.py` (`outputs/phase7_holdout.json`); the final refit by `scripts/submit_w2.py`, `scripts/freeze_build.py` and `scripts/apply_referral.py` with `P7_ALL=1`.

## Result in one line

On customers no model had seen, the frozen blend beats the W2 GLM in every line in both years: across lines +41,250 on 2023 [2,450, 103,420] and +18,403 on 2024, which passes the keep rule. The stop rule is not triggered.

## Hold-out scoring (73,227 customers; 2023 and 2024 rows)

Blend against the W2 GLM, deviance saved, A with its 95% customer-bootstrap interval, then B:
- motor +24,250 [−11,753, 83,955], +3,567;
- buildings +9,286 [−1,610, 19,385], +6,923;
- contents +7,714 [1,770, 14,838], +7,913 (passes);
- all lines +41,250 [2,450, 103,420], +18,403 (passes).

Ranking D², hold-out, all lines: blend 12.49% (A) and 10.03% (B); W2 GLM 12.15% and 9.84%; the plain baseline GLM of phase 4 12.66% and 8.98%.

**The 11.9M claim decides motor 2023.** It is 43.8% of the hold-out's 2023 motor cost. With it, motor ranking D² is 5.61% for the plain GLM (fitted on uncapped cost), 3.94% for the W2 GLM and 4.33% for the blend. Without that one row they are 4.30%, 4.76% and 4.77%. The plain GLM wins only because one claim dominates the score.

**Level of the decided chain against actual** (B is the reading that matters: A has no completion adjustment): motor 1.01, buildings 0.90, contents 1.13 on 2024; 0.66, 1.07 and 0.95 on 2023 (motor 2023 carries the 11.9M claim).

Beside the non-hold-out results (step 8, cross-fitted): blend against W2 GLM +92,357 [46,571, 139,010] and +114,384 across lines. The hold-out gain is smaller in size, on about 15% of the rows, and points the same way in every line and year.

## Final refit on all customers

- **Claims history restored** to the 2025 rows of hold-out customers: rows with a past claim rise by a median 14% (motor), 36% (buildings) and 27% (contents); claim-free rows fall by 8%, 0.2% and 5%.
- **X1, the 11.9M bodily injury claim,** now enters the motor load. It is 11.8M of the 23.7M of motor cost above the 100k cap over 2021 to 2024. The motor excess factor rises from 1.031 (non-hold-out) to 1.058 with it; without it the factor would be 1.029.
- **2025 totals against the frozen file:** motor 113.7M (+2.3%, mostly X1), buildings 79.1M (+2.2%), contents 19.3M (−1.7%). The C4 band holds in every line: motor −1.40%, buildings −0.13%, contents −1.50%.
- **Referral rule:** 1,516 motor rows at the W2 GLM's price (2,321 in the frozen file).
- The file is `outputs/submissions/blend_final_all_v2.csv`.

## Where the AI was wrong (D28)

The step 6 design stated that every net cost is positive. That was checked on non-hold-out customers only. The hold-out holds one closed nil claim (CL45729785: 0.02 incurred, 0.02 recovered), which stopped the per-peril severity fit in the refit. Nil claims are now left out of the per-peril frequency and severity, which changes nothing in the frozen model; the frozen file was rebuilt and checked byte-identical.

## Cards for the actuary (round 1)

- **U1.** Confirm X1 (a) now that its effect is known: the motor load with the 11.9M claim.
- **U2.** The final submission: the refit on all customers, or the frozen file.

## Verdicts (actuary, 2026-10-20, check-in 7 round 1)

U1 (a), U2 (a), with no notes. Exchange D29.

- **X1 confirmed:** the 11.9M bodily injury claim stays in the motor large-loss load (excess factor 1.058).
- **The final Part 1 submission is `blend_final_all_v2.csv`:** the frozen pipeline refitted on all customers, with P8-R2 and the referral rule.
