# Kaggle submissions log

## Part 1 (developed market)

Each row is one file the actuary uploaded. At most 10 submissions a day (rule 2.2.a said 5; the host confirmed 10 on the forum). The final submission is selected by the actuary on our out-of-time scores, not on the public score, and is selected explicitly (rule 2.2.b). The public leaderboard uses about 33% of the test rows; the final result uses the other 67%.

The internal scores are ranking D² across all three lines on scorings A and B, with the reference a constant rate times exposure. Kaggle's baseline appears to be defined differently: its average-cost file scores 0.0717 rather than zero. So public and internal numbers are compared by their ordering, not their size.

| Date | File | What it is | Internal A | Internal B | Public | Rank at the time | Selected as final |
|---|---|---|---|---|---|---|---|
| 2026-10-18 | baseline_glm_v0.csv | Plain Tweedie GLM, main effects, year factor at 2024 level, completion offset, no trend, no cap, no history or weather (scripts/phase4_baselines.py) | 9.98% | 9.42% | 0.11710 | 16th; leader 0.12996; host's basic GLM 0.10563 | No |
| 2026-10-20 | w2_glm_v1u.csv | Working base W2 (one capped Tweedie GLM per line, D19 and D20), fitted on every non-hold-out row 2021-2024, priced at the 2024 year level, times the line excess factor, the off-balance to the weather-normalised 2025 indication (motor 1.0129, buildings 1.0650, contents 1.0441) and the 2025 weather as one factor per line (M3). Contents rows whose security level or high-value items stay blank after recovery are priced at the exposure-weighted average of the known levels (register P8-R2 (a), decided). `scripts/submit_w2.py` | 10.51% | 10.12% | 0.12695 | 10th; leader 0.12996 | No |
| 2026-10-19 (prepared) | w2_glm_v1.csv | As above, with the remaining blanks at their own level as P8-R decided: 674 contents rows priced at about 0.6 a year (`scripts/submit_w2.py --own-level`). Kept for comparison; not recommended | 10.51% | 10.12% | | | |
| 2026-10-20 | blend_frozen_v1.csv | The frozen Part 1 model (O1 (a), O2 (b); D25): per line, the families refitted on every non-hold-out row 2021-2024 and mixed with the pooled step 8 weights. Motor 0.666 EBM per peril and 0.334 GLM with boosted correction (plain average); buildings 0.788 GLM with boosted correction and 0.212 CatBoost (log average); contents 0.910 standardised spline GLM, 0.058 LightGBM, 0.032 EBM (plain average). Line totals set to the decided 2025 level of w2_glm_v1u; P8-R2 in every family. `scripts/freeze_build.py`. Internal scores are the cross-fitted blend's. The actuary uploaded it on 2026-10-20 | 10.67% | 10.33% | 0.12821 | 8th; leader 0.12996; up from 0.12695 (w2_glm_v1u) | |
| 2026-10-20 | blend_frozen_v2.csv | blend_frozen_v1 with the phase 6 referral rule (Q1 (a), Q2 (a); D26): 2,321 motor rows where the families disagree by more than 30% at the W2 GLM's price, the motor line rescaled to its decided total (other motor rows +0.13%), property unchanged. `scripts/apply_referral.py`. The motor change passed the keep rule out of time: +18,782 on 2023 [7,077, 31,902], +1,478 on 2024. The actuary uploaded it on 2026-10-20 | | | 0.12830 | 9th (another entrant moved ahead); leader 0.12996; up from 0.12821 (blend_frozen_v1) | |
| 2026-10-20 (chosen as final, U2) | blend_final_all_v2.csv | Phase 7 final refit: the frozen pipeline (blend, P8-R2, referral rule) refitted on all customers 2021-2024 after the hold-out was opened, with claims history and k rebuilt on all customers; X1's claim in the motor load (excess factor 1.058). `P7_ALL=1` runs of `scripts/submit_w2.py`, `scripts/freeze_build.py`, `scripts/apply_referral.py`. Hold-out scores: blend against W2 GLM +41,250 (2023), +18,403 (2024) across lines | 12.49% (hold-out) | 10.03% (hold-out) | 0.12908 | 5th; leader 0.12996; up from 0.12830 (blend_frozen_v2) | Yes (U2) |

## Part 2 (emerging market)

One final submission per part, selected explicitly by the actuary. Internal evidence for every file is in `decisions/part2_m2_design.md` (P2-V: fit 2023 score 2024, and customer-grouped cross-validation, Tweedie D² at a common level per line). Build command for every `part2_v2_*` file: `scripts/part2_model_v2.py` with the switches its name lists. All uploads 2026-10-28 (EDT).

| File | What it is | Public | Rank at the time | Selected as final |
|---|---|---|---|---|
| part2_check_v1.csv | Check file: Part 1's W2 GLM per line with emerging levels and the Part 1 level chain (milestone 1) | 0.14266 | 24th; leader 0.17303 | No |
| part2_v2_nb-none_wx-row.csv | Milestone 2 shape (K9 plus own-year weather, M2-W (a)), no new-business factor: one of the paired M2-B check, decided before upload | 0.15080 | | No |
| part2_v2_nb-cred_wx-row.csv | As above, new-business factor credibility-weighted on the claim-count standard (M2-N (b)) | 0.15312 | | No |
| part2_v2_nb-full_wx-row.csv | As above, the full emerging new-business figure; uploaded by the actuary outside the planned pair | 0.14646 | | No |
| part2_v2_nb-cred_wx-row_mprem.csv | Frozen model I: motor with the premium term (Y1), fitted without the hold-out customers | 0.15353 | | No |
| part2_v2_nb-cred_wx-row_mprem_bchk_all.csv | Milestone 2 final: buildings at the check shape (H-F (b)), refitted on every customer | 0.15455 | 21st; leader 0.17353 | No |
| part2_v2_nb-cred_wx-row_mprem_mperil_bchk_all.csv | Motor priced by peril (K16, H-MP (a), decided after the hold-out was opened), count-standard new-business factor (motor 1.114, buildings 0.950, contents 1.094) | 0.15712 | 19th | **Yes** (H-NS (a), exchange D49) |
| part2_v2_nb-pp_wx-row_mprem_mperil_bchk_all.csv | As above with the pure-premium credibility standard (H-NS (b), exchange D48; factors 1.050, 0.983, 1.050) | 0.15657 | | No; reverted after this reading (D49) |
