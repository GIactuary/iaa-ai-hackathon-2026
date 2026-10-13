# 2025 lessons in detail

Competition: IAA AI Assisted Hackathon, Kaggle, 28 to 30 November 2025. Predict house sale price; metric RMSLE. Data: synthetic, generated from the Ames, Iowa housing data (1,465 train, 1,465 test, 80 inputs, `data_description.txt`). No training row matches the public Ames file exactly. Public leaderboard on about 60% of test rows, private (final) on the other 40%. Invitation only, no late submissions.

## Leaderboard (read 12 October 2026)

| Entrant | Prize | Public | Private (rank of 49) |
|---|---|---|---|
| Tajmahal Ghaza Antoni | 3rd | 0.0761 | 0.0772 (1) |
| SberIns Team | none | 0.0799 | 0.0838 (2) |
| Ryo Asashi | 1st | 0.0680 | 0.1137 (3) |
| Funky Turtles | 2nd | 0.1077 | 0.1156 (10) |
| Kohei Kudo | special mention | 0.1122 | 0.1179 (18) |
| Median team | | | 0.1197 |

## Each entry, rebuilt on our folds

**1st prize (negoto, Ryoichi Asashiba).** LightGBM, XGBoost and CatBoost with Optuna-tuned settings, a voting ensemble with weights 0.04 / 0.51 / 0.45 (effectively two trees). Excluded two training sales with an eyeball rule (Edwards neighbourhood and over 4,000 sq ft) and priced the test lookalikes with a separate straight line through those two points. Explained the ensemble with MID, a surrogate from his own R package (midr), fitted on test rows. Reproduced to 0.0934 against his reported 0.0932. Cross-validated 0.0925 on three seeds. His Edwards treatment explains his strong public score; three tree models with correlated errors explain why averaging gained little.

**2nd prize (Funky Turtles).** Careful cleaning with row-specific fixes, one-hot encoding, six engineered features, Yeo-Johnson on skewed inputs, robust scaling. A StackingRegressor over kernel ridge, SVR, Lasso, Elastic Net, gradient boosting, XGBoost, LightGBM and CatBoost with an XGBoost meta-model, then blended 50% stack and 10% each of five single models. No cross-validation score reported. Rerun on our folds: 0.0920, 0.0894, 0.0901 (mean 0.0905), level with our pair; its SVR alone scored 0.090. Priced the two large Edwards new builds at about $700k and $650k.

**3rd prize (Tajmahal Ghaza Antoni).** Stacked the public Ames file (2,930 real sales) onto the training data, used most-frequent and mean imputation, an XGBoost model for lot frontage, four engineered features, dropped about 40 columns, rounded years into 5-year categories, CatBoost target encoding, one XGBoost tuned with Optuna (1,303 trees, depth 6, rate 0.083). His saved notebook is a cleaned version; the leaderboard file came from an unsaved draft. Rerun on our folds with the encoder fitted inside each fold: 0.081 with the public sales, 0.105 without. Priced the Edwards new builds at about $190k, learned from the large partial sales in the public file.

**Special mention (wideflat, Kohei Kudo).** A Gemini-driven feature agent (20 features tried, 4 kept, a 1.9% gain), a seven-model base layer, weighted average and stacking layers, a dashboard and a public repo. The headline cross-validated 0.058 came from a stacking layer scored on in-sample predictions; the honest weighted layer scores 0.092. Fourteen submission runs on the final evening. Priced the Edwards new builds at about $720k and $700k.

## The houses that decided the leaderboard

Six test houses exceed 4,000 sq ft. The two that matter most are Edwards new builds of 5,520 and 5,130 sq ft, quality 10, sold before completion, in a neighbourhood whose largest remaining training sale is 2,520 sq ft. Their only comparable is a training sale of the same profile that sold for $189k.

| House | Ours | 1st | 2nd | 3rd | Special |
|---|---|---|---|---|---|
| Edwards new build, 5,520 sq ft | $169k | $224k | $702k | $192k | $724k |
| Edwards new build, 5,130 sq ft | $157k | $248k | $649k | $184k | $696k |
| Edwards, built 1935, 4,060 sq ft | $270k | $330k | $358k | $398k | $347k |
| NoRidge, 4,930 sq ft | $712k | $517k | $691k | $723k | $614k |
| OldTown, built 1892, 4,060 sq ft | $385k | $399k | $400k | $501k | $480k |
| NoRidge, 4,020 sq ft | $593k | $556k | $669k | $597k | $616k |

If those two new builds really sold near the comparable, pricing them at $700k costs about 0.01 on the whole test set and more on whichever leaderboard part holds them. That matches the gap between the pack and the top three.

## Our decided model in brief

- Use: insured-value check at homeowners underwriting. Sale-time variables dropped (cost about 0.2%).
- GLM built forward on the log target: about 28 variables entered at the 0.2% gain threshold, total floor area and quality first. Rare levels pooled below 5 records. QualSF (quality × area) excluded from the GLM on purpose, left to the trees.
- LightGBM: 6 leaves, rate 0.01, 4,000 trees, row and column sampling. Hand-set.
- Blend 50/50 on the log scale, fixed. Cross-validated 0.0900 against 0.0925 for the 1st prize (paired 95% intervals exclude zero on all three seeds). Locked 293-house hold-out: level (0.0901 against 0.0900).
- Referral: 19 of 1,465 test houses (size over 1.25 times the neighbourhood maximum, or GLM and tree more than 30% apart). Size-flagged houses capped at the largest evidenced house; the two Edwards new builds rated from the comparable sale at about $30.7 per sq ft.
- 90% range: by predicted-price fifth, 8th to 98th percentile of out-of-fold misses (tilted up for the insured-value use).
- Judgment register J1 to J10, all decided by the actuary.

## What the AI got wrong in 2025 preparation, and how it was caught

These are the best material for the creative-AI section: they show the loop working.

- Claimed the public Ames file leaked the test answers; matching the rows showed the data is synthetic. Withdrawn.
- Proposed a size cap for the two Edwards new builds; testing showed the capped prices still at twice the only comparable. Replaced with comparable-sale rating.
- Proposed a forward GLM as cost-free; testing showed it costs about 1% in the pair. Disclosed and accepted.
- Recommended the winner's two-row exclusion; the screen showed one of the two sales prices normally.
- Built a disagreement flag that missed one problem house because both models extrapolated the same way; added the size check.
