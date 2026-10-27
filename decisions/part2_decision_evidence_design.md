# Part 2 decision evidence: design, written before any run

27 October 2026. The actuary asked for every open Part 2 decision in one brief, with the information needed to decide each. This file fixes, before any figure is computed, what `scripts/part2_decision_evidence.py` measures. The figures inform decisions; none of them decides one. Nothing here is a model choice for the submission.

## Data

Emerging training rows 2023-2024 after rules R1 (exposure) and R3 (construction codes); R2 left off, as in the Y2-T test. Developed (Part 1) rows 2021-2024 the same way, from `data/raw_part2/developed`. Claims matched to rows by rule R6, repudiated claims excluded. Cost is net of recoveries to date, capped at Part 1's caps (100,000 motor, 200,000 buildings, 30,000 contents). The caps are not yet confirmed for the emerging market, so every cost figure here is provisional. R10 is not applied. No hold-out is drawn: no hold-out exists for the emerging market yet, and none of these figures is a validation result.

## Descriptive figures (no thresholds)

1. Rows, exposure and claims by line and year; new-business share in training and test.
2. New business against renewal: claim frequency and cost per exposure year by line, accident year 2023, with 95% Poisson intervals on the frequency ratio.
3. Blanks: frequency of rows with at least one blank rating field against rows with none, by line; the same per field for the fields blank on more than 5% of rows.
4. Gender: frequency, mean cost and cost per exposure year by gender and line, with the 95% interval on the frequency ratio.
5. Large losses: claims above each Part 1 cap, their share of net cost, and the 99th, 99.9th percentile and largest claim, by line, in both markets.
6. Reporting: the share of accident-year claims reported within 30, 90, 180 and 365 days of loss, emerging 2023 against developed 2021, by line.
7. Year-on-year change: emerging frequency and severity, 2024 against 2023, against developed 2022 against 2021 at the same valuation age (claims reported within 365 days of the year end). This is one year of change, not a trend.

## Carry-over test (exploratory)

The question: does the developed market's experience help predict the emerging market?

- **E (emerging only):** LightGBM on emerging rows.
- **D (developed only):** LightGBM fitted on all developed rows 2021-2024, applied to emerging rows. Developed ages are mapped to the emerging bands (17-24, 25-34, 35-49, 50-64, 65+).
- **DE (developed start, emerging decides):** LightGBM on emerging rows, started from D's prediction (its log as the initial score), so the emerging data moves the developed shape only where it has the evidence.

Inputs: every field both markets carry, by `src/fields.tree_fields` with `code_levels_max` 11. Region, make_model and occupation are left out of D, because the regions differ between markets and the two codes need the encoding step. E and DE take region as a raw input. Gender is held out everywhere (D33). Settings: 31 leaves, 400 minimum rows per leaf (100 for the emerging fits, which are about 20 times smaller), learning rate 0.05, early stopping after 100 rounds on fold 1 of five customer-grouped inner folds (seed 11).

Targets: claim count (Poisson deviance) and capped cost (Tweedie 1.5 deviance, provisional caps).

Validation:
- **V1, out of time:** fit 2023, score 2024. D needs no emerging fitting rows.
- **V2, cross-validation:** five customer-grouped folds over 2023-2024, seeds 11, 22 and 33.

Scoring: deviance at a common level per line (each prediction rescaled to the scored rows' actual total), summed over the lines, and D² against the scored rows' mean. Differences between E, D and DE come with 95% paired bootstrap intervals resampling customers (2,000 draws). An interval clear of zero is a difference the data can tell apart from noise.

Outputs: `outputs/part2_decision_evidence.json`.
