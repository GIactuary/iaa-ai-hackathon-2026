# Kaggle submissions log

Each row is one file the actuary uploaded. At most 5 submissions a day (rule 2.2.a). The final submission is selected by the actuary on our out-of-time scores, not on the public score, and is selected explicitly (rule 2.2.b). The public leaderboard uses about 33% of the test rows; the final result uses the other 67%.

The internal scores are ranking D² across all three lines on scorings A and B, with the reference a constant rate times exposure. Kaggle's baseline appears to be defined differently: its average-cost file scores 0.0717 rather than zero. So public and internal numbers are compared by their ordering, not their size.

| Date | File | What it is | Internal A | Internal B | Public | Rank at the time | Selected as final |
|---|---|---|---|---|---|---|---|
| 2026-10-18 | baseline_glm_v0.csv | Plain Tweedie GLM, main effects, year factor at 2024 level, completion offset, no trend, no cap, no history or weather (scripts/phase4_baselines.py) | 9.98% | 9.42% | 0.11710 | 16th; leader 0.12996; host's basic GLM 0.10563 | No |
| 2026-10-20 (prepared) | w2_glm_v1u.csv | Working base W2 (one capped Tweedie GLM per line, D19 and D20), fitted on every non-hold-out row 2021-2024, priced at the 2024 year level, times the line excess factor, the off-balance to the weather-normalised 2025 indication (motor 1.0129, buildings 1.0650, contents 1.0441) and the 2025 weather as one factor per line (M3). Contents rows whose security level or high-value items stay blank after recovery are priced at the exposure-weighted average of the known levels (register P8-R2, recommended). `scripts/submit_w2.py` | 10.51% | 10.12% | | | |
| 2026-10-20 (prepared) | w2_glm_v1.csv | As above, with the remaining blanks at their own level as P8-R decided: 674 contents rows priced at about 0.6 a year (`scripts/submit_w2.py --own-level`). Kept for comparison; not recommended | 10.51% | 10.12% | | | |
