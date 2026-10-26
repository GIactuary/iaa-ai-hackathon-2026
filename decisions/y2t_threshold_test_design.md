# Y2-T threshold test: design, written before any run

26 October 2026. The actuary's note on the Y2-T brief (round 1), recorded as no verdict yet:

> "I believe here we need to make a decision based on what is the level that will keep the prediction power as much at the same time dont give us diminishing returns- model computations should not be considered - if there a way to optimize the level statistically we may do that exercise too. come back to me i will make my decision"

This test answers that question. It reports evidence and recommends a level; the actuary decides Y2-T. Nothing here is run before this file is committed.

## The question

`code_levels_max` (N) decides which tree inputs enter as raw categories and which enter as one credibility-weighted number (`src/fields.py`). The question is which N predicts best out of sample, and past which N the gains stop: where letting more levels in raw adds complexity without a gain the data can tell apart from noise.

N matters only where it crosses a code's level count, so each candidate is a regime, not a free number. The script lists, for each candidate and line, exactly which codes are encoded. Candidates that encode the same codes in a line are the same model and are scored once.

## Candidates

- **Emerging market:** N = 4, 5, 6, 7, 20, 100, 300. Their level counts (blanks not counted) are make_model 300, occupation 80 as typed, vehicle colour 8, energy rating 7, region 6, and body type, age band and soil type 5. So the candidates span "encode every code with more than 4 levels" to "every code raw". At N = 300, make_model enters raw, because the rule encodes only codes with more than N levels.
- **Occupation cleaned (exploratory).** The same candidates with occupation trimmed and case-folded (40 levels). The cleaning rule is not yet decided (check-in item 5), so this is reported beside the main candidates and does not enter the selection.
- **Reference R0:** the same inputs with every code above 20 levels left out altogether. Every candidate's gain is shown against it, so the value of having make_model and occupation at all is visible.
- **Part 1, as support:** N = 4, 5, 7, 11, 20, 300 (region has 12 levels, occupation 20, make_model 300).

## Data, target and model

- **Emerging rows:** the training files after rules R1 (exposure) and R3 (construction codes), with addresses and the customer fields `src/build.py` merges. Gender and previous insurer are excluded, as in the build. Sum insured is left unrepaired: R2's cutoff is not yet re-confirmed for this market, and a tree places a value 1,000 times too large in its top bin either way. Blanks stay blank (LightGBM's native handling).
- **Emerging target: claim count (Poisson).** The large-loss caps for this market are not re-confirmed, so cost would mean uncapped cost and a few claims would decide the comparison. Frequency is the steadier signal and needs no cap. Repudiated claims are excluded (R6).
- **Part 1 target:** claim count (Poisson) and capped cost (Tweedie 1.5, Part 1's confirmed caps), on non-hold-out customers from the build cache, with the reporting completion Part 1 uses as exposure.
- **Model:** LightGBM (`src/families.py`), learning rate 0.05, early stopping after 100 rounds without gain, at most 3,000 trees (Part 1's step 7 values). For every candidate the setting is picked the same way: leaves {15, 31, 63} × minimum rows per leaf {100, 400, 1600}, on fold 1 of five customer-grouped inner folds of the fitting rows (seed 11), with early stopping on that fold. The model is then refitted on all fitting rows with that setting and tree count.
- **Encoding (`fields.credibility_encode`):** expected claims (or capped cost) per row are exposure times the fitting rows' rate in that year. The encoded value is then the level's credibility-weighted log relativity. In the real build the base is the W2 GLM, so these encodings are marginal where the build's are relative to the GLM. Fitting rows get out-of-fold values, as in the build.

## Validation

- **Emerging V1 (primary): out of time, fit 2023, score 2024.** This mirrors pricing a later year. Bühlmann-Straub needs at least two periods per level within the fitting data, so in V1 the encoding's k uses the calendar quarter of the row's start as its period.
- **Emerging V2: customer-grouped five-fold cross-validation over 2023-2024**, seeds 11, 22 and 33, with k periods by year. It uses both years in every fit and tests the same question with more fitting data.
- **Part 1: scorings A (fit 2021-2022, score 2023) and B (fit 2021-2023, score 2024).**
- **Scoring:** deviance at a common level per line (each candidate's predictions rescaled to the scored rows' actual total, as in step 8): Poisson deviance for counts, Tweedie 1.5 for cost. Gains are against R0 and against the best candidate. 95% intervals come from a paired bootstrap resampling customers (2,000 draws, `validation.paired_bootstrap_groups`).

## Selection rule (pre-registered)

1. On emerging V1, across the three lines together (summed deviance, customers resampled), find the candidate with the lowest deviance, B.
2. A candidate is within reach of B if the 95% interval of its extra deviance over B includes zero.
3. **The recommended regime is the lowest N within reach of B.** A lower N encodes more codes, so it is the more economical model. Any higher N adds raw levels for a gain the data cannot tell apart from zero: that is the point of diminishing returns. The same rule is reported per line.
4. **Consistency check:** the rule is applied to emerging V2 and to Part 1 (scoring A, both targets) as well. If V2 picks a different regime from V1, the test reports both and recommends neither. The actuary decides with both in view.
5. The threshold value is any N inside the recommended regime's range. Where Part 1's value of 20 falls inside that range, it is kept.

Threshold row: `y2t_select_ci` (0.95) in `decisions/thresholds_part2.md`, dated before the run.

## Outputs

`scripts/y2t_threshold_test.py`; evidence in `outputs/y2t_threshold_test.json`; logs in `data/work/y2t/`. The results go to the Y2-T brief as round 2.
