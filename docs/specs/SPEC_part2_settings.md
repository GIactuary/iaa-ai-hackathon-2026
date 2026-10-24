# Part 2 build settings from the Part 1 post-mortem

Decided by the actuary on 2026-10-24: Y1 to Y4, option (a) on each, with no notes, on the post-mortem brief (https://claude.ai/artifact/1Fjpei88G1pBuSXFXE2a7R, finalized round 1; exchange D32). Evidence: `decisions/postmortem_results.md`. These are constraints on the Part 2 build. None is in `scripts/run_market.py` yet.

Every setting below is still tested on Part 2's own data under the keep rule before it enters the final file. A Part 1 result is evidence for Part 2, not proof.

## In the order the build needs them

### 1. Y3: recoveries on claims open at the valuation date (data rule R10, `build` stage)

- **Check first, on Part 2's claims file.** Do settled claims show recoveries while open claims show none, and in which perils? Report the closed-claim recovery rate by peril, and the share of open incurred by accident year.
- **If the pattern holds:** deduct each open claim's expected recovery, at its peril's closed-claim rate over the experience years, from its net cost before capping. The adjusted cost feeds both the target and the level chain.
- **Where:** `src/data_rules.py` (or `src/targets.py`), written up in `decisions/data_rules.md` with its evidence.
- **Measure it the same way as on Part 1** (`scripts/postmortem.py recoveries` as the pattern): the change by accident year, and the change in each line's target-year indication.
- **If Part 2 has no recoveries column, or no such pattern,** the rule does nothing, and the write-up says why.

### 2. Y2: inputs the tree families may use (`weights` and `blend` stages)

- **The trees get every legitimate field.** Excluded: identifiers and dates, the target and its pieces, exposure, leakage and post-event fields (as found at Part 2 intake), the incumbent's price on its own (it enters through Y1 only), data-quality flags, and the P6 marketing and identity fields.
- **Codes with many levels** enter only credibility-encoded, never as raw categories. On Part 1, `make_model`'s 300 levels lost 414,287 in motor.
- **Threshold to write first:** the level count above which a code is credibility-encoded goes in the market's thresholds file, dated, before the run that uses it.
- **The family weights decide** how much the trees count. Refit the step 8 weights with the wider trees in place.
- **Where:** the tree frame in `scripts/freeze_build.py` (`frames`) and in `scripts/fast_weights.py`, which follow `scripts/phase5_step7.py` (`RAW_NUM`, `RAW_CAT`, `CARRIED`). The field list in `scripts/postmortem.py` (`wide_fields`, `EXCLUDE`) is the pattern.

### 3. Y1: the premium as our fields predict it (`weights`, `level` and `blend` stages)

- **If Part 2's data has no current premium,** skip this step and say so in the write-up.
- **The premium model.** Fit a LightGBM regression of log written premium on the legitimate rating fields (the Y2 field list, weather excluded), using the fitting rows only and no claims information. The settings are in `scripts/postmortem.py` (`premium_parts`): 63 leaves, 100 minimum rows per leaf, early stopping on fold 1 of the customer-grouped inner folds.
- **What the model takes.** Its prediction, `lwp_hat`, enters the W2 GLM as one numeric term and the tree families as one input. The remainder is never used.
- **Report** the premium model's R² on the scored rows and its largest inputs, as on Part 1. A low R² means the premium carries something our fields lack. That changes the reading, so stop and bring it to the actuary.
- **Keep rule on Part 2's own scorings:** the W2 GLM plus `lwp_hat` against the W2 GLM, per line and across lines. It stays only where it passes.
- **Explanation for the write-up.** The current rating plan's relativities, read through our own fields, serve as a complement of credibility where our experience is thin. On Part 1 that was flood geography in buildings. No field we exclude can enter through it, because only our fields predict it.

### 4. Y4: weather (`level` stage)

- **Keep M3 (b):** the target year's weather as one exposure-weighted factor per line, with no row-level weather input.
- **Reopen it only if** Part 2's what-changed report or screens show a stronger weather signal than Part 1 had. Bring that to the actuary as a new card.

## Open items the verdicts did not close

- **The flood-geography challenger (Y1 option D).** Not chosen. The recommendation suggested running it beside (a). It runs only if the actuary asks.
- **Two thresholds still to be written and dated** before the runs that use them: the Y2 level count, and any Part 2 recovery-pattern criterion for Y3.
- **Fast mode.** With wider trees, Part 1's tuned tree settings may no longer fit. Whether to run the full step 7 tuning on Part 2 is the actuary's call; the runbook lists its cost.
