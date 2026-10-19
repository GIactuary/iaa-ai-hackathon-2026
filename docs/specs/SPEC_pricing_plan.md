# Build spec: pricing plan, Part 1

Frozen from the plan brief on 2026-10-18. The actuary finalized round 2 on the brief (https://claude.ai/artifact/PhDnuDVjiDGdT332jEPHz5); the round 2 page is kept at `docs/reviews/pricing-plan-r2.html`. The decisions are in `decisions/judgment_register.md`, the exchanges in `ai/ai_log.json` (D5 to D7), and the evidence in `docs/research/2026-10-18_round2_research.md`.

Every item below is a constraint the build must meet, in the order the build needs them. A settled decision is not reopened mid-build. If implementation shows one was wrong, work stops and the question goes back to the actuary as a new decision.

## Purpose

**J1.** The model gives a technical risk premium per 2025 row: base rates and relativities by line, rate adequacy against the current price, and referral of risks outside the evidence. Under-pricing is the costlier error. The metric agrees: halving a price saves less on a row without a claim than it costs on a row with one.

## 1. Data rules (phase 1)

- **P8.** Planted errors are repaired by stated rules. Every repaired row is flagged and the same rules apply to test. Each rule is written in `decisions/` with its evidence before use.
  - Exposure given in days becomes years (exposure above 1 that matches the day count).
  - Sum insured is divided by 1,000 when it sits above a written multiple of the line's 99th percentile.
  - Construction codes are normalised to one spelling each. W and T are checked against the documented timber level first.
  - Motor `customer_id` is padded to 8 digits.
- **Never inputs.**
  - `renewal_decision`: an outcome.
  - `customer_service_contacts` as recorded: it counts contacts during the period. Test contacts appear to be measured differently.
  - `days_unoccupied` stays out until a point-of-use check clears it.
- **P6.** `surname`, `email_domain`, `marketing_segment`, `campaign_code`, `branch_id`, `marketing_consent` and `document_preference` stay out of the rating plan. They are used only in a proxy test on final prices. `credit_band` and `occupation` are a separate, later judgment.
- **J-ext.** No external data. The market search is recorded as a check.
- **J-prem.** `written_premium` is the incumbent to beat only. It is never an input or an offset. An offset may be tested later as a challenger.
- Every blank is classified as absent, zero, estimable or unknown, with a cross-check (phase 1 gate).

## 2. Screens (phase 2)

- **P4.** The 11,894,277 CU open bodily injury reserve is investigated before any cap is set.
- The outlier screen and the train-versus-test evidence check run with thresholds written first. Exclusions and referrals are decided by the actuary.

## 3. Validation (phase 3)

- **P2.** Validation is out of time: fit on years up to Y−1, then score Y. 2023 is the main scoring year. 2024 is scored second, on targets adjusted by the completion offset.
- Inner tuning folds are grouped by policy. For property perils they are also grouped by event week within region.
- A locked hold-out of policies, with all their rows, is opened once in phase 7. Comparisons use a paired bootstrap.
- Models are compared only at a common level (P11). Calibration is checked by decile out of time. Lift, double lift and Gini are supporting exhibits.

## 4. Target

- **Row target.** Claims net of excess and recoveries, repudiated claims excluded, matched to rows by policy and loss date. Losses before a row's start (414 unmatched claims, mostly December 2020) are outside the target.
- **P1.** Shape and level are set apart. Relativities are fitted with an accident-year factor. As the Monograph 5 check, the GLM is refitted on developed, trended targets with the year factor kept, and any year effect left over is investigated.
- **P10.** Each row's expected reported share by peril, from its own dates and the peril's delay curve, enters the exposure offset. This is pure IBNR at row level, with amounts handled in aggregate. The curves are re-estimated whenever data changes and backtested at a pseudo-valuation.
- No claim-level IBNER factor is used, because there is no case-estimate history. Development is checked in aggregate by peril; bodily injury and subsidence are watched most closely.
- **P4.** Claims are capped at per-line levels written before results. The expected excess over the cap is added back as a load per unit of exposure.

## 5. Level

- **P9.** Frequency and severity trends are fitted per peril on developed 2021 to 2024 data, with stated breaks. Bodily injury uses its 2023 to 2024 level without extrapolating the drop. Trends are projected from the average accident date of the experience to mid-2025. Each selection is written with a band before results.
- **P11.** After the relativities, each line and peril is rescaled so the prediction on the 2025 test mix equals the selected base loss cost: an explicit off-balance. The total is compared with the triangles and with the insurer's own 2025 premium move, inside a band the actuary sets.

## 6. Features

- **J-weather.** The rating plan uses address climate normals (2021 to 2024 averages). The policy year's weather enters only as a separate, disclosed year adjustment for storm (wind), subsidence (dry summer), escape of water (freezing days) and motor glass (hail days). Each has its own scored number.
- **P5.** Claims history for the policy, customer and address uses only claims reported before the row starts, weighted by credibility, and is kept only on the drop-one test.
- The engineered features in E8 are kept only if removing them worsens error on every out-of-time split.

## 7. Models

- **P3.** Tweedie per line and frequency × severity per peril are both built in each family. The winner is kept per line on the paired test.
- **P7.** Candidate families, in order of expected accuracy:
  1. Boosting started from the GLM prediction.
  2. Plain boosting (LightGBM, CatBoost).
  3. EBM with pairwise terms.
  4. Spline GLM with elastic-net selection (the anchor and filing baseline).
- **J-model.** The best candidate per family is kept. Non-negative weights across families are fitted on out-of-time predictions and scored on rows not used to fit them. A within-family blend is the challenger.
- Technical defaults (challengeable):
  - Row cost with a log(exposure × completion) offset.
  - Training Tweedie power tuned on inner folds and scored at 1.5.
  - Monotone constraints only where the direction is fixed in advance.
  - Make-model encoded by credibility weighting inside each fold.
  - Early stopping on deviance.
- Every prediction is strictly positive.

## 8. Tail, ranges, submission

- A referral rule covers records outside the evidence and model disagreement, with thresholds written first. The treatment is a coded rule, never a hand-set price (Foundational 4.b).
- A valid baseline file is submitted early (Foundational 4.c). The model is frozen at the end of 21 Oct. The actuary submits the final Part 1 file by 23 Oct, 12:00 UTC.
- Every module is rebuilt as code that reruns on new data, ready for Part 2.

## Open items the decisions did not close

1. **Thresholds still to write with the actuary before the tests that use them:** `screen_z`, `refer_size_ratio`, `refer_disagreement`, `drop_one_seeds_worse` (now "out-of-time splits"), `holdout_share` and the hold-out seed, the cap levels by line, the trend bands by peril, the level band against the triangles, and the training-power grid.
2. **The 11.9M bodily injury reserve:** keying error or true large loss.
3. **Construction codes W and T:** both timber, or one of them something else.
4. **`days_unoccupied`:** whether it can be used at the point of pricing.
5. **The form of the weather-year adjustment.** It could be a frequency multiplier per peril from within-address elasticities, or the year's weather as a feature with the year factor held at the experience average. This is a design choice to put to the actuary when built.
6. **The 2025 level against the insurer's premium move.** Corrected in exchange D13: the move is +2.6% to +5.6%, not +36% to +44%, and is in line with our trends. The comparison stays as a reasonableness check.
7. **The deadline.** The exact time and the public and private split are not published.
