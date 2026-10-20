# Part 1: a technical price for an established book

*Draft for the appointed actuary, 2026-10-20, sized for about 2.5 of the 5 graded pages (it measures 2.8 at 10.5 pt on Letter). Every number comes from an evidence file in `outputs/` or a decision note in `decisions/`, named under each exhibit.*

## Summary

We priced 2025 motor, buildings and contents with a blend of model families per line, set to a weather-normalised 2025 level. On customers locked away before any model was fitted, the blend beat our working GLM in every line and year and passed the keep rule across lines. The final file scored 0.12908 on the public leaderboard (5th; leader 0.12996), up from 0.11710 for our first baseline. The appointed actuary decided every judgment; of 29 logged exchanges, 11 record an error of the AI's own, each found by a check and corrected.

## 1. Purpose and data

**Use (J1).** A technical risk premium, the expected ultimate cost of each 2025 row, to set base rates and relativities, measure rate adequacy against the current price and refer risks the evidence cannot support. Buildings ran at a loss ratio of 1.29 to 1.36 in every year from 2021 to 2024, while the 2025 premium rate rose only 2.6% to 5.6%. Under-pricing is the costlier error: 35% of motor is sold through comparison sites, where a price below cost attracts more of the same risk.

**Data.** Synthetic: 2.4 million policy rows for 2021 to 2024, 163,150 claims valued at 31 December 2024, 626,639 rows to price, and no public source to add (J-ext). The target had to be built: net cost by loss date, repudiated claims out, each claim capped (100k motor, 200k buildings, 30k contents), and each row's exposure scaled by its expected reporting completion, which acts as IBNR at row level. Nine stated rules repair the planted errors on training and test rows alike (for example, sum insured entered 1,000 times too large on 2,797 training and 730 test rows), and two columns that record the outcome are never used. No record was excluded: the 11.9M bodily injury claim was investigated and kept as a genuine large loss (X1).

## 2. Validation and the keep rule

Every choice was judged out of time, as a 2025 price is set from earlier years: fit 2021 to 2022 and score 2023 (A); fit 2021 to 2023 and score 2024 (B). A random 15% of customers (73,227) was locked away and opened once, after the freeze. The metric is the competition's Tweedie deviance at power 1.5, judged after each line is rescaled to its actual total, like comparing rating plans after an off-balance; the level is a separate pricing judgment. **Keep rule**, written before anything was fitted: lower deviance in both scorings, and on A a 95% interval from 2,000 customer resamples above zero.

## 3. How the model was built

More failed than passed: 67 engineered features gave three keepers, five original factors were removed because their line priced better without them, and climate normals, per-peril GLMs and each model family on its own lost to the GLM.

**Exhibit 1. Each kept step, out of time, all lines**

| Step (register item) | Deviance saved on 2023 [95% interval], on 2024 | Ranking D² 2023, 2024 |
|---|---|---|
| Current premium | | 8.01%, 7.28% |
| Plain Tweedie GLM with completion offset | Beats the current premium in every line | 9.98%, 9.42% |
| Cap and line excess factor (K1, K2) | +32,825 [6,688, 58,429], +53,044 | 10.02%, 9.49% |
| Claims history with credibility (L1 to L3) | +114,842 [68,647, 164,656], +143,043 | 10.34%, 9.95% |
| Three features in, five factors out: GLM "W2" (L5, L6, L8) | Motor +79,907 [45,218, 112,951], +61,068 | 10.51%, 10.12% |
| Blend of model families (O1, O2) | +92,357 [46,571, 139,010], +114,384 | 10.67%, 10.33% |

*Sources: `outputs/phase4_baselines.json`, `phase5_*.json`. The history test is policy history; its D² adds declared claims.*

> **How to read this.** Ranking D² is the share of a flat rate's deviance the model removes: how well risks are sorted. The blend's +92,357 is 0.18% of the W2 GLM's 2023 deviance, small because most deviance is the chance of who claims; its interval sits above zero and it won in all 2,000 resamples, so the gain is not luck of the draw.

**Claims history** is experience rating with credibility: each policy's mod is (A + k) / (E + k), with A its prior claims and E its class's expected claims. In motor, k = 2.76, so a policy expecting 0.37 claims over three years gets a relativity of 1.17 after one claim and 0.90 claim-free (fitted slope 0.86 on the log mod).

**Model families** were tuned on folds inside the fitting years, the best of each kept, and the families weighted per line on one scoring and tested on the other (J-model). Plain gradient boosting ranked far below the GLM (−242,328 on 2023), since the data behaves like a multiplicative rating plan; what helped was a small boosted correction to the GLM rate (about 4% to 7% on average) and, in motor, an additive model per peril.

## 4. The final model

**Exhibit 2. The Part 1 price by line (final refit on all customers)**

| | Motor | Buildings | Contents |
|---|---|---|---|
| Families and weights | Additive model (EBM) per peril 0.666; GLM with boosted correction 0.334 | GLM with boosted correction 0.788; CatBoost 0.212 (log average) | Spline GLM 0.910; LightGBM 0.058; EBM 0.032 |
| Excess factor; off-balance; 2025 weather | 1.058; 1.009; 0.9996 | 1.022; 1.076; 1.0003 | 1.019; 1.036; 0.9973 |
| Gap to the indication (band ±3%) | −1.40% | −0.13% | −1.50% |
| 2025 expected cost; over current earned premium | 113.7M; 0.95 | 79.1M; 1.43 | 19.3M; 0.83 |

*Sources: `outputs/submission_blend_final_all_v1.json`, `submission_w2_glm_v1u_all.json`.*

> **How to read this.** The families set the relativities; the three factors set each line's total. A motor row the GLM rates at 500 CU at the 2024 level is priced at 500 × 1.058 × 1.009 × 0.9996 = 533.5 CU: the cost above the cap, the move to the trended 2025 indication, and 2025's realised weather. The last row reads as an expected loss ratio before expenses: buildings losses are expected to exceed the current premium by 43%.

**Level.** The 2025 indication is built peril by peril from weather-normalised experience, the appointed actuary's method (K4): each year's frequency is restated to normal weather, trends are fitted on the restated years (bounded at −5% to +12%), and the result is multiplied by the 2025 weather.

**Referral.** Rows are flagged when the sum insured is beyond 1.25 times its segment's training maximum, the segment has under 100 training rows, or a weighted family prices more than 30% from the blend. Out of time, flagged rows were no less accurate, so a flag alone does not change the price. Only in motor did a treatment pass the keep rule: 1,516 rows take the W2 GLM's price. Every flag is listed for underwriting review.

## 5. Hold-out result

**Exhibit 3. The frozen pipeline on 73,227 unseen customers**

| Line | Blend against W2 GLM: saved on 2023 [95%], on 2024 | Ranking D² 2023: plain GLM, W2, blend |
|---|---|---|
| Motor | +24,250 [−11,753, 83,955], +3,567 | 5.61%, 3.94%, 4.33% |
| Buildings | +9,286 [−1,610, 19,385], +6,923 | 4.41%, 5.44%, 5.71% |
| Contents | +7,714 [1,770, 14,838], +7,913 | 7.51%, 7.85%, 8.28% |
| All lines | +41,250 [2,450, 103,420], +18,403 | 12.66%, 12.15%, 12.49% |

*Source: `outputs/phase7_holdout.json`; figures without the 11.9M claim from `decisions/phase7_checkin7.md`.*

> **How to read this.** The blend saves deviance in every line and year. Across lines the 2023 interval sits above zero and the blend won in 98.7% of resamples, so it passes the keep rule on unseen customers; motor and buildings alone do not, on 15% of the data. The plain GLM leads motor 2023 only because the 11.9M claim is 43.8% of that year's hold-out motor cost and only the plain GLM was fitted on uncapped cost; without that row motor reads 4.30%, 4.76%, 4.77%.

The final file refits the frozen pipeline on all customers, restoring claims history to 12.7% of 2025 rows and bringing the 11.9M claim into the motor load (excess factor 1.029 to 1.058; U1, U2).

## 6. Limitations

- One claim moves the motor level by about 3% (113.7M against about 110.6M without it).
- Realised 2025 weather is used, which a real 2025 price would not have; it moves each line by at most 0.3%.
- The hold-out level is noisy: 2024 ran at 1.01, 0.90 and 1.13 of actual by line on 825 to 3,502 claims; on the full book every line sits within 1.5% of its indication.

## 7. How AI was used and challenged

AI worked in roles set by standing prompts (`ai/prompt_library.md`): a data steward run as a separate subagent, a modeller, a filing writer and an independent reviewer. Each decision went to the appointed actuary on a page with options, evidence and a recommendation; he challenged or decided. At his direction no model, setting or threshold was carried over from the 2025 competition (D4), and compute was never a reason to test less (D14).

**Exhibit 4. Exchanges that changed the outcome**

| | Exchange | Effect |
|---|---|---|
| Actuary | D16: asked why before choosing the motor level, and proposed weather-normalised experience | The AI found a bug in its own completion weights (motor gap −5.3% to −0.7%); his method cut the 2024 level error from 2.0% to 0.3% |
| Actuary | D21: asked whether to standardise inputs for the penalised GLM | Contents GLM 50,180 better on 2024; its spline form carries 0.910 of the contents price |
| AI wrong | D17, D18: a credibility estimator that is mostly noise at these claim frequencies | k infinite for buildings; Poisson-gamma estimate used instead |
| AI wrong | D22: thin blank levels given their own relativity | 674 contents rows priced at about 0.6 CU a year, also in the first submission; fixed |
| AI wrong | D23: a penalty tuned on one scale reused for frequency and severity | Relativities flattened; the invalid test set aside |
| AI wrong | D28: "every claim cost is positive", checked without the hold-out | A nil claim stopped the final refit; fixed, frozen file rebuilt byte-identical |

*Source: `ai/ai_log.json`, D1 to D29; 11 exchanges were raised by the actuary.*

> **How to read this.** Each row is a logged card: question, AI answer, challenge, test, result, decision. In D16 the AI recommended the model's own motor projection; asked to explain it, it found that 2024 motor completion was understated by 4.8% (0.785 against 0.823) and withdrew the recommendation. The errors were caught by tests and implausible numbers, and only D22 reached a submitted file, the first baseline.

The data rules, screens, keep rule, level chain and referral rule are code that reruns on new data, so Part 2 starts from the same pipeline and the same decision loop.
