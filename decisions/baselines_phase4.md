# Phase 4 baselines (run 2026-10-18)

Run with `scripts/phase4_baselines.py`. The numbers are in `outputs/phase4_baselines.json`, with the run metadata. Every score is out of time, on non-hold-out customers, under the agreed design (`decisions/validation_design.md`). The intervals are 95% bootstrap intervals that resample customers (1,000 draws).

## Conclusion first

A plain Tweedie GLM already ranks risk better than the insurer's current price in every line, in both scorings. Combined ranking D² is 9.98% against 8.01% on scoring A, and 9.42% against 7.28% on scoring B. The current price is furthest out on level:
- It under-prices buildings by about a quarter (level ratio 0.76 and 0.73).
- It over-prices contents by about a fifth (1.18 and 1.19).

These are the floors every later model must beat.

## The three baselines

| Baseline | What it is |
|---|---|
| Constant | One rate per line per unit of completed exposure |
| Incumbent | Written premium times exposure: the insurer's current price, used as a risk ranking |
| Plain GLM | A Tweedie GLM per line at power 1.5 with log link. Main effects only (age, NCB, vehicle group, cover, mileage and so on for motor; sum insured decile, property type, flood zone and so on for property), with an accident-year factor and the row completion offset. No capping, no engineered features, no weather, no claims history |

## Ranking D² at a common level (each line rescaled to its actual total)

| Line | Scoring A (2023): Constant | Incumbent | Plain GLM | Scoring B (2024): Constant | Incumbent | Plain GLM |
|---|---|---|---|---|---|---|
| Motor | 0.00% | 2.70% (2.22 to 3.19) | 4.81% (4.16 to 5.46) | 0.07% | 2.72% (2.10 to 3.42) | 5.34% (4.48 to 6.46) |
| Buildings | 0.00% | 5.71% (4.38 to 7.05) | 6.58% (5.34 to 7.78) | 0.05% | 4.95% (3.54 to 6.43) | 6.08% (4.83 to 7.33) |
| Contents | 0.00% | 2.14% (1.53 to 2.75) | 6.30% (5.03 to 7.74) | 0.07% | 2.00% (1.38 to 2.61) | 5.41% (4.32 to 6.48) |
| All three | 4.53% | 8.01% (7.40 to 8.58) | 9.98% (9.40 to 10.57) | 4.09% | 7.28% (6.53 to 8.05) | 9.42% (8.68 to 10.24) |

**How to read this.**
- D² is the share of a constant rate's Tweedie error that a set of prices removes.
- A constant rate scores zero within a line by construction. Combined, it still scores 4.5%, because getting the level of each line right is itself worth that much.
- Insurance D² values are small because most rows have no claim. The French motor benchmark in the research scores 5.1% for a GLM.
- Example: in motor on 2023, the current price removes 2.7% of the error and the plain GLM removes 4.8%. The GLM is 1.8 times as discriminating, and the two intervals do not overlap.

## Level ratio (predicted total over actual total, before rescaling)

| Line | A: Incumbent | A: Plain GLM | B: Incumbent | B: Plain GLM |
|---|---|---|---|---|
| Motor | 1.06 | 1.08 | 1.00 | 0.92 |
| Buildings | 0.76 | 0.96 | 0.73 | 0.92 |
| Contents | 1.18 | 0.93 | 1.19 | 0.98 |

Two points on level:
- The GLM's level carries the last fitted year forward with no trend. That is why it runs under in B (0.93 overall). Trend is P9's job in phase 5.
- Motor on scoring A sits above 1. The 2023 actuals are scored without completion, so they still miss about 3% of motor cost, and the 2022 level is carried forward.

## Early submission file

- **File.** `outputs/submissions/baseline_glm_v0.csv`, kept out of git because it holds test row ids. It has 626,639 rows in sample-submission order, all predictions positive.
- **How it was built.** The plain GLM was refitted on all non-hold-out rows from 2021 to 2024 and priced at the 2024 year level, with no trend and no cap.
- **Predicted total against the 2025 written premium:**
  - Motor 0.97 (115.6M against 119.2M)
  - Buildings 1.30 (72.0M against 55.3M)
  - Contents 0.82 (18.9M against 23.2M)
- **Its purpose.** It is a valid Part 1 entry (Foundational rule 4.c) and a first leaderboard reading of level. It is not a candidate final submission.

## What the baselines say for phase 5

1. **The current premium ranks buildings fairly well (5.7%) and contents poorly (2.1%).** The largest gains over the incumbent are in contents and motor.
2. **The plain GLM's intervals are about 1.2 to 2.5 points wide by line.** Feature gains of a few tenths of a point need the paired test, not a comparison of intervals.
3. **Level matters as much as ranking for the combined score.** The constant baseline's 4.5% combined comes from level alone. Phase 5 has to get the trend (P9) and the off-balance (P11) right.
4. **The 11.9M claim and all other claims are uncapped here.** Capping (P4) needs its thresholds agreed before it is tested.
