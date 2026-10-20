# Part 2 adaptation kit: design

Written 2026-10-20, before Part 2 opens (brief section 2: a new market with different characteristics and data problems, about one week, judged on how we detect and respond to change). Part 1 is closed for us: the final file is selected (U2) and the hold-out is spent.

## 1. What the kit must do

1. **Show what changed**, on the day the Part 2 files arrive, before any model runs: a report comparing the new market with Part 1 and its own training rows with its test rows. This is the evidence Part 2 is judged on, and it tells the actuary which Part 1 decisions still hold.
2. **Rerun the whole pipeline with one command** on the new files, from raw data to a submission, with every market-specific value read from one configuration file rather than the code.
3. **Run fast.** Part 1's step 7 tuning took about 6 hours. A fast mode reuses the tuned settings and stops trees by early stopping, so the full pipeline finishes in about an hour; the full tuning stays available.

The kit decides nothing. Every threshold and judgment for the new market still goes to the actuary first (CLAUDE.md rules 2 and 4); the kit makes the evidence and the rerun quick.

## 2. The what-changed report (`src/drift.py`, `scripts/what_changed.py`)

For each file (policies by line, test by line, claims, customers, addresses, weather), against a reference (Part 1's files) and between the new training and test rows:
- **Schema:** columns added, removed or retyped; files missing.
- **Blanks:** share blank per column, and its change.
- **Distributions:** for numbers, quantiles and the population stability index over the reference's deciles; for categories, level shares, new levels and levels gone.
- **Planted-error detectors**, the Part 1 traps as tests: exposure that equals the day count; sum insured far above the line's 99th percentile (unit errors); code spellings outside the documented levels; identifiers of the wrong length; duplicated row ids; outcome or post-event columns present (renewal decision, contacts, repairer, third party).
- **Claims:** peril mix, frequency and severity by accident year, report-delay quantiles, repudiation rate, open share, and the largest claims against the screen (`screen_z`).
- **Train against test:** the phase 2 evidence check (sizes beyond the evidence, unseen levels, impossible combinations).

Output: `outputs/what_changed_<market>.md` (plain prose, conclusion first, the largest changes at the top) and a JSON evidence file.

## 3. Configuration (`config/<market>.json`, read by `src/market.py`)

One file per market holds every value the Part 1 code fixed: lines and perils; the valuation date; the delay-curve and peril-share years; the fitting and scoring years; caps and large-loss perils per line; the step perils; the weather-adjusted perils and their variables; the size-check segments; the raw fields per line for the tree families and the EBM; and the paths to the raw files. `config/part1.json` reproduces Part 1 exactly; the module constants read from it, so the Part 1 code paths do not change.

Check: after the change, `scripts/freeze_build.py` must reproduce `blend_frozen_v1.csv` byte for byte.

## 4. The runner (`scripts/run_market.py`)

`run_market.py --market part2 [--fast] [--from STAGE] [--to STAGE]`, with stages run in order and each writing its evidence:
data rules → screens and what-changed → build → claims history → model families (fast: Part 1's tuned settings with early stopping; full: the step 7 tuning) → weights (step 8) → level → referral → submission file. A stage stops the run if a check fails (missing column, a threshold not written, a level outside the C4 band).

## 5. Dry run

The kit is tested on Part 1's own files before Part 2 opens: the report on Part 1 train against Part 1 test, and the runner end to end in fast mode, which must give a file close to `blend_final_all_v2` (the fast mode refits with fixed tree counts, so it is a check of the plumbing, not a new model).

## 6. Judgments that will go to the actuary when Part 2 opens

- Which Part 1 decisions carry over (data rules, caps, features, families, weights), on the what-changed evidence.
- Every threshold for the new market (caps, trend bounds, level band, referral), written before the tests that use them.
- Whether to spend part of the week on full tuning or stay with the fast mode.
