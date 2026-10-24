# Part 1 post-mortem: design, written before any test runs

Written 2026-10-24, after the Part 1 private leaderboard closed and before any of the tests below was run. Asked for by the actuary on 2026-10-24 (exchange D30) to prepare Part 2.

## Why

The final Part 1 file scored 0.13641 on the private leaderboard, 15th. First place scored 0.13923 and 4th 0.13864. Every modelling step after the W2 GLM moved our public score from 0.12695 to 0.12908 (+0.0021). The gap to first is larger than that, so a missing input is a more likely cause than modelling. Places 5 to 20 sit within 0.0019 of each other and moved up to 24 places between the public and private boards, so ranks inside that pack are mostly noise.

The Part 1 record shows three fences, and all three carry into Part 2 through `scripts/run_market.py`:

1. **The trees saw only GLM-vetted inputs** (`decisions/phase5_step7_design.md` section 1). A field had to pass a main-effects GLM test before any tree could use it, so a signal that works only through interactions could not reach the model able to find it.
2. **No address weather in any model.** Climate normals failed twice in the GLM (K4). Row-level 2025 weather entered only as a GLM elasticity multiplier, which tested inside the noise (M3), so it became one factor per line.
3. **The current premium was never tested** as an input or an offset (J-prem (a)). Option (c), the premium as an offset, was meant to be tested later as a challenger and never was.

A fourth item is a data point, not a fence: the host confirmed on the forum that recoveries are recorded only when a claim settles, so open claims at 31 December 2024 carry none.

## What this can and cannot show

- It shows whether each fenced group carries out-of-time signal on Part 1 data, under the same keep rule as every Part 1 decision.
- It cannot change the Part 1 score, so there is nothing to overfit.
- It does not show what the top teams did. No winner has published a write-up.
- A Part 1 result is evidence for Part 2, not proof. Part 2 is a new market. Anything that passes here goes into the Part 2 run from day one and is tested again on Part 2's own data.

## Tests

**Vehicle.** LightGBM, the tree family, on the same scorings as Part 1: A (fit 2021-2022, score 2023) and B (fit 2021-2023, score 2024 with row completion), hold-out customers excluded. Each line separately. Fitting target, exposure × completion offset and line excess factor as in step 7 (`scripts/phase5_step7.py`).

**Reference (R).** LightGBM on the step 7 tree inputs: the raw fields behind the W2 GLM plus the carried candidates (customer, address and repudiation history, years licensed, voluntary excess, river distance and elevation).

**Candidates.** Each is R plus one group of inputs:

| Id | Adds | Question |
|---|---|---|
| PM1a | The row's realised weather for its own year (address × year): wind exposure index, dry summer index, freezing days, hail days, rainfall, wildfire index, mean temperature | Does the policy year's weather add signal in a tree? The 2025 values are supplied for the test rows, so this is usable in pricing the test file |
| PM1b | PM1a plus the address's climate normal: the mean of each variable over the fitting years only | Does the place's climate add to the year's weather? |
| PM2 | Every legitimate field the trees did not see (list below) | Was the GLM gate too tight? |
| PM3 | PM2 plus PM1b | The widest legitimate model: an upper reference for what the fences cost |
| PM4a | log written premium and discount applied, as inputs | Does the incumbent's price carry information our inputs lack? |
| PM4b | R started from the log written premium (an offset; boosting models cost relative to the current price) | J-prem option (c) in tree form |
| PM4c | The W2 GLM with log written premium as one numeric term, against the W2 GLM | J-prem option (c) in GLM form |

**PM2 fields.** Every column of the model-ready rows except:
- identifiers and dates: row, policy, customer and address ids, period start and end, date of birth (age is already in R);
- the target and its pieces: cost, capped cost, claim counts, completion factors; the exposure itself (it is the offset);
- leakage and post-event fields: renewal decision, customer service contacts;
- the incumbent's price: written premium, discount applied (tested on their own in PM4);
- data-quality flags (`dq_*`), which mark repairs on training rows and say nothing about risk;
- the P6 fields (surname, email domain, marketing segment, campaign code, branch, marketing consent, document preference) are not in the model-ready rows at all.

`days_unoccupied` is included: phase 1 read it as a declared pre-period figure (`decisions/missing_values.md`).

**Settings, chosen the same way for R and every candidate.** Training power at the step 7 tuned value for the line (motor 1.3, buildings 1.3, contents 1.4). Leaves in {15, 31} × minimum rows per leaf in {400, 1,600}, with the tree count set by early stopping, all chosen on one customer-grouped inner fold (fold 1 of 5, seed 11) of scoring A's fitting rows. The chosen setting and tree count are then used on both scorings. Learning rate 0.05, early stopping after 100 rounds, at most 3,000 trees (thresholds `gbm_learning_rate`, `early_stop_rounds`, `max_trees`). R gets the same small grid, so no candidate is compared against a less-tuned reference.

**Keep rule.** Unchanged and already written (`keep_years_better` 2, `keep_ci` 0.95): better on both A and B, and the 95% customer-bootstrap interval on A (2,000 draws) above zero. Each candidate against R per line and across lines at a common level per line. PM4c against the W2 GLM. No new threshold is needed.

**Explanation.** For PM3 on scoring A, the largest inputs by mean absolute SHAP contribution on 20,000 scored rows, so any gain can be traced to named fields.

## Recoveries on open claims (PM5): a measurement, not a keep test

The scored actuals have the same flaw as the fitting target (open claims without recoveries), so a keep test of a corrected target would be scored against flawed labels. PM5 is therefore measured, not tested:

- Expected recovery on an open claim = its incurred × the recovery rate of its peril on closed claims (accident years 2021-2024).
- Reported: the change in net cost by line and accident year. Evidence so far (2026-10-24, before this design): only motor own damage (23.6% of incurred) and theft (about 4%) recover; the motor 2024 accident year is overstated by about 1.5% (1.36M), 2023 by about 0.01%.
- Read as a level question for Part 2's data rules. Whether it becomes a data rule is the actuary's call.

## What happens with the results

- A group that passes on all lines combined, and in the lines where it passes, is offered to the actuary as a change to the Part 2 tree inputs, with the step 8 weights refitted.
- PM4 is J-prem. Even if it passes, using the premium remains the actuary's judgment, because it makes the rating plan partly a function of the incumbent's price.
- A group that fails stays out, and the write-up says so with the number. That is still useful: it rules out an explanation for the gap.

Script: `scripts/postmortem.py`. Evidence: `outputs/postmortem.json`.

## Amendment, written 2026-10-24 after the first results and before these runs

**What the first results showed that the design did not foresee.** PM2 and PM3 lost heavily in motor (PM2 −414,287 on A, −368,362 on B), and the largest input in the PM3 motor tree was `make_model`, 300 levels entered as a native LightGBM category. Step 5 found make-model a true null (M7, D18: about 3% spread in frequency, nothing in cost), so the tree was splitting on noise. The motor PM2 and PM3 results measure the overfitting of a 300-level category, not the value of the fenced fields. That is a flaw in this design, not evidence.

**PM2m and PM3m.** PM2 and PM3 without `make_model`, settings chosen as before. Same keep rule.

**PM6, what the premium knows.** PM4 passed in buildings. Before the premium is put to the actuary as an input (J-prem), we need to know whether its information is already in our fields, which would mean a modelling gap we can close without the premium, or is information our fields lack. Per line and scoring:
- A LightGBM regression (squared error) of log written premium on the PM3m inputs, fitted on the fitting rows only (no claims information enters it). Its R² on the scored rows is reported.
- The log premium splits into the part our fields predict (`lwp_hat`) and the remainder (`lwp_res`).
- The W2 GLM plus each part as one numeric term, against the W2 GLM, under the keep rule.
- Reading: if `lwp_hat` carries the gain, the information is in our fields and our models miss it; if `lwp_res` carries it, the premium holds information we cannot get any other way. The largest inputs of the premium model (mean absolute SHAP) are reported.
