# Kaggle submissions log

Each row is one file the actuary uploaded. At most 5 submissions a day (rule 2.2.a). The final submission is selected by the actuary on our out-of-time scores, not on the public score, and is selected explicitly (rule 2.2.b). The public leaderboard uses about 33% of the test rows; the final result uses the other 67%.

The internal scores are ranking D² across all three lines on scorings A and B, with the reference a constant rate times exposure. Kaggle's baseline appears to be defined differently: its average-cost file scores 0.0717 rather than zero. So public and internal numbers are compared by their ordering, not their size.

| Date | File | What it is | Internal A | Internal B | Public | Rank at the time | Selected as final |
|---|---|---|---|---|---|---|---|
| 2026-10-18 | baseline_glm_v0.csv | Plain Tweedie GLM, main effects, year factor at 2024 level, completion offset, no trend, no cap, no history or weather (scripts/phase4_baselines.py) | 9.98% | 9.42% | 0.11710 | 16th; leader 0.12996; host's basic GLM 0.10563 | No |
