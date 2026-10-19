# Validation design (phase 3)

The design was agreed with the actuary on 2026-10-18 (exchange D11), before any model was fitted. The thresholds are in `decisions/thresholds.md`. The code is in `src/validation.py` and its tests are in `tests/test_validation.py` and `tests/test_validation_2026.py`.

## The design in one paragraph

Every choice is judged on two out-of-time scorings:
- **A:** fit 2021 to 2022, score 2023.
- **B:** fit 2021 to 2023, score 2024, with each row's prediction scaled by its reporting completion.

Both leave out a locked 15% of customers, which is opened once in phase 7. A change is kept only if it lowers Tweedie deviance in both scorings, and if on scoring A the 95% paired bootstrap interval, resampling customers, lies above zero. Ranking is judged at a common level, and the level is judged on its own.

## 1. Hold-out

| Item | Value |
|---|---|
| Unit | Customer: all their lines and years. 44% of customers hold two or three lines (217,140 of 488,180) |
| Share and seed | 0.15 of customers, seed 2026, drawn at random from the sorted list of customer ids |
| Drawn | 73,227 customers (15.00%) and 361,770 rows (15.02%) |
| Rows held out by year | Motor 32,388 / 38,196 / 38,094 / 38,243; buildings 23,551 / 27,794 / 27,623 / 27,620; contents 23,989 / 28,288 / 28,021 / 27,963 (2021 / 2022 / 2023 / 2024) |
| Lock file | `data/work/holdout_lock.json`, kept out of git with the data. `scripts/lock_holdout.py` reproduces the same draw |
| Fingerprints | Customer list sha256 `41613f4ec3e06456519161d9d91c3802362004346ba663420345b24cc80f9cc6`; hold-out list sha256 `b0ac36c7dd590ef9e572420b6a1aae4bb5e057b8187d5943b88ed39a7cf9c16c` |
| Locked | 2026-10-18 22:05. Only row counts were read: no claims, costs or outcomes of held-out customers |

How it is kept closed:
- `fitting_mask` gives the rows models may use.
- `open_holdout` refuses a second opening unless a reason is given, and records each opening.
- `lock_holdout` refuses any draw that differs from the one on disk.

## 2. Out-of-time scorings

| Scoring | Fit on | Score on | Target scored | Role |
|---|---|---|---|---|
| A | 2021, 2022 (non-hold-out) | 2023 (non-hold-out) | Reported net cost (2023 is close to mature) | Primary; the bootstrap interval is taken here |
| B | 2021 to 2023 (non-hold-out) | 2024 (non-hold-out) | Reported net cost, against predictions times the row's reporting completion (P10) | Confirms A on the latest year and tests the completion offset |

## 3. Inner folds (tuning and early stopping only)

- 5 folds grouped by customer, inside the fitting years of each scoring.
- Seeds 11, 22 and 33, so the spread of tuning results is visible.
- The spec's extra grouping by event week for property perils was dropped. The decisions are scored out of time, where one storm cannot fall on both sides. Inside the tuning folds the effect is mild optimism in early stopping, and the out-of-time score would show it.

## 4. Metric

- **Row-level Tweedie deviance at power 1.5.** At 1.5 it equals 4 (√y − √μ)² / √μ. Predictions must be strictly positive.
- **D².** 1 minus the model's deviance over that of a constant rate times exposure.
- **Reported per line and for all three lines combined.** The combined figure mirrors Kaggle, which scores the three test files as one.
- **Ranking D².** Each model's predictions are rescaled per line to the scored year's actual total (the common level of P11). Model and feature choices are judged on this.
- **Level ratio.** Predicted total over actual total, per line. It is judged apart, as the pricing judgment of P1, P9 and P11.

## 5. Keep rule

| Key | Value | Meaning |
|---|---|---|
| `keep_years_better` | 2 | The candidate lowers deviance in both scorings, A and B |
| `keep_ci` | 0.95 | On scoring A, the paired bootstrap interval of the deviance saved lies above zero |

The bootstrap draws customers with replacement 2,000 times. Bootstrap seed 0 is fixed in code.

## 6. Opening the hold-out (phase 7)

The frozen pipeline is fitted on non-hold-out rows up to 2022 and scores hold-out 2023, then fitted up to 2023 and scores hold-out 2024. The hold-out is opened once, with the reason recorded in the lock file.

## 7. Kaggle leaderboard

The public leaderboard is a sample of the test rows, so it is a level check, not a tuning target. Make one submission per settled idea, at most five a day.
