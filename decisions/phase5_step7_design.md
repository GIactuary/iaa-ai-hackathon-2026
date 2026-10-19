# Phase 5 step 7: model families (P7, C5, J-model), test design

Written 2026-10-19, before any step 7 model was tuned or scored. Decisions applied: P7 (a) the four families, C5 (a) training power tuned on a grid from 1.1 to 1.9 and scored at 1.5, J-model (a) the strongest candidate per family, M1 (a) one Tweedie model per line plus frequency × severity in motor, M2 (a) the line excess factor, M3 (b) the 2025 weather as one factor per line.

## 1. What each model sees

- **Rows.** Non-hold-out customers. Tuning uses only the fitting years of scoring A (2021 and 2022). Out-of-time scoring is A (fit 2021-2022, score 2023) and B (fit 2021-2023, score 2024 with completion), as in every earlier step.
- **Target.** Capped cost, with exposure × line completion as the exposure measure. Every family fits the rate y / (e c) with weight (e c)^(2 − p), which equals a log(e c) offset under Tweedie deviance (the identity `TweedieGLM` already uses), so the families differ only in the shape of the rate.
- **Inputs (W2).** The same information as the W2 GLM of the line:
  - GLM families use the W2 columns as they are (banded factors, log(mod) of policy history, declared history).
  - Tree families and the EBM use the raw fields behind them: continuous fields as numbers (age, NCB years, vehicle value group, mileage, licence points, voluntary excess, vehicle age, sum insured, year built, missed payments, prior claims, log(mod)) and the rest as categories. Trees within 5 m and soil type enter separately; the tree can find their interaction.
  - The accident year enters every model; scored rows are read at the latest fitting year, as the GLM is.
- **Carried candidates (L4, L7).** Offered to every family as one group, then one at a time for the families where a fit is cheap (LightGBM, the elastic-net GLM): customer history, address history and prior repudiation (all lines), years licensed (motor), voluntary excess (buildings and contents), river distance and elevation (buildings).

## 2. Families and candidates

| Family | Candidates | Tuned on inner folds |
|---|---|---|
| F1 Penalised GLM (the anchor) | (a) the W2 GLM as decided; (b) elastic-net Tweedie GLM on the W2 columns (`glum`); (c) as (b) with natural cubic splines in place of the bands of age, NCB, vehicle value, mileage, sum insured and year built | power; penalty strength α ∈ {1e-6, 1e-5, 1e-4, 1e-3, 1e-2} × L1 share ∈ {0, 0.5, 1}; spline knots `spline_knots` |
| F2 Plain boosting | (a) LightGBM; (b) CatBoost | power; LightGBM leaves ∈ {15, 31, 63} × minimum rows per leaf ∈ {100, 400, 1600}; CatBoost depth ∈ {4, 6, 8}; trees by early stopping |
| F3 EBM with pairwise terms | `interpret` ExplainableBoostingRegressor | power; pairwise terms ∈ {0, 5, 15} |
| F4 Boosting from the GLM | LightGBM started from the W2 GLM's log rate (the GLM refitted inside each inner fold) | power; leaves ∈ {7, 15, 31} × minimum rows per leaf ∈ {400, 1600}; trees by early stopping |

- **New packages.** `interpret-core` 0.7.8 (MIT) and `glum` 3.4.1 (BSD) are added to `requirements.txt`, with `wrapt` pinned at 2.4.1 so the resolver does not pick a pre-release. Both are OSI-licensed (Foundational 6.c).
- **Search, in two stages per candidate, on the five customer-grouped folds of seed 11:**
  1. Training power over the grid 1.1 to 1.9 (`power_min`, `power_max`, `power_step`), other settings at their middle value.
  2. The other settings in the table, at the best power.
- **Selection.** The candidate setting with the lowest mean inner-fold deviance wins: Tweedie deviance at power 1.5 on capped cost, each fold rescaled to its actual total. Within a family, the candidate with the lowest inner-fold deviance is the family's model (J-model (a)). The out-of-time years play no part in tuning or in choosing within a family.
- **Spread.** The winning setting is refitted on the folds of seeds 22 and 33, and its deviance is reported against the W2 GLM on the same folds for all three seeds.
- **Trees.** Learning rate `gbm_learning_rate` (LightGBM) and `cat_learning_rate` (CatBoost); early stopping after `early_stop_rounds` rounds without improvement on the held-out fold, at most `max_trees`. The out-of-time refits use the mean best iteration over the five folds of seed 11.
- **EBM.** `ebm_outer_bags` outer bags; its own internal early stopping; other settings at the package defaults.

## 3. Frequency × severity in motor (M1 (a))

- Per peril, a Poisson frequency model (exposure × the peril's completion) and a gamma severity model (capped cost per claim, weighted by claims), summed over perils, times the line excess factor.
- Built for F1 (elastic-net GLM, the F1 setting), F2 (LightGBM, the per-line F2 tree settings with the Poisson and gamma objectives; trees by early stopping on fold 1 of seed 11) and F3 (EBM, the F3 setting).
- Not built for CatBoost (each fit takes about two minutes on motor; ten models per fit would not finish before the freeze) or F4 (it would need a per-peril GLM anchor refitted inside every fold). Both omissions are stated in the results.
- Each family's frequency × severity model is compared with the same family's per-line model out of time on the keep rule, as in step 6.

## 4. Out-of-time tests

Each comparison: per line at a common level, deviance saved on A with the 95% customer-bootstrap interval (2,000 draws) and on B; keep rule as before; combined figure beside each line. With M3 (b) the 2025 weather is one factor per line, so for per-line models the priced and bare forms rank identically; every comparison uses the bare form (rate at the latest fitting year × exposure × the line excess factor, × completion on B).

1. **Each family's model against the W2 GLM.** This is evidence for the weights in step 8, which decide how much each family counts. A family that ranks worse than the GLM on both scorings is reported as such; step 8's weights, fitted out of time, settle whether it carries any weight.
2. **Carried candidates.** Within each family, the group added to the family's model, on the keep rule; one at a time for LightGBM and the elastic-net GLM. A candidate enters a family only if it passes there.
3. **Frequency × severity in motor** against the same family's per-line model, on the keep rule (section 3).
4. **For step 8.** Each family's final out-of-time predictions on A and B are saved by row, so the weights can be fitted on one scoring and scored on the other.

## 5. Explanation

For each family model of each line: the five inputs with the largest effect (mean absolute change in log rate, SHAP for the trees, term importance for the EBM, coefficients for the GLMs), and the shape of the age and vehicle-value (motor) or sum-insured (property) effect, so the actuary can read each family against the GLM's relativities.

## 6. Thresholds

New rows in `decisions/thresholds.md`, written 2026-10-19 before any step 7 run: `gbm_learning_rate`, `cat_learning_rate`, `early_stop_rounds`, `max_trees`, `ebm_outer_bags`, `spline_knots`. The grids in section 2 are written here. The keep rule (`keep_years_better`, `keep_ci`) and the power grid (`power_min`, `power_max`, `power_step`) are already written.

## 7. Amendment (2026-10-19, during the run, before any out-of-time score)

**Standardised predictors in the elastic-net GLM (the actuary's suggestion).** The penalty in candidate F1 (b) acts on raw coefficients. A rare factor level and a common one, or a numeric term on a small scale, are then penalised unevenly. The actuary pointed out that the optimisation works better when every input is standardised (z-score: divided by its standard deviation).

- **Change.** F1 gains candidate (d): the elastic-net GLM with every predictor scaled to standard deviation one before the penalty applies (`glum`'s `scale_predictors`; coefficients are reported on the original scale). It is tuned exactly like (b): power, then α × L1 share, on the same folds.
- **Knock-on.** The spline variant (c), the carried-candidate tests and the motor frequency × severity GLM use whichever of (b) and (d) has the lower inner-fold deviance.
- **Where it does not apply.** The tree families and the EBM split on order, so scaling an input changes nothing. The W2 GLM has an almost negligible penalty (α = 1e-5), so scaling changes only how fast it converges.
- **What had already run.** Contents had finished tuning LightGBM, F4, the unscaled elastic-net GLM and the EBM, and was part way through CatBoost. No out-of-time score had been computed. Finished tuning results are kept; tuning now checkpoints every trial.
