# Part 2 check-in 1: what changed in the emerging market

24 October 2026. Source: `outputs/what_changed_part2.md` (regenerate with the runbook step 2 command), with every figure below rechecked by a separate script written apart from `src/drift.py`. Claim figures are reported to date, not ultimate, and use row exposure capped at one year: they show direction and rough size only. Nothing has been repaired or decided.

## The conclusion

The emerging data has the same structure and the same planted traps as Part 1, at the same rates. Four things differ in ways that change the model:

1. **The test book is mostly new business.** 76% of test rows are new business, against 12% (2023) and 15% (2024) in training. The share is the same in every channel (75% to 77%), so the broker portfolio is not a broker-channel effect. The channel mix itself does not move: broker 70%, direct 25%, comparison site 5% in both training and test.
2. **Blanks are five times as common.** About 20 fields are blank on 12% of rows, against 2.5% in Part 1: occupation, marital status, household size, years at address, vehicle age, engine size, overnight parking, smoke alarms, days unoccupied, security level, high-value items, soil, roof, energy rating, distance to river. Our Part 1 rules recovered most blanks from the same policy's other rows. With two years of history and a test book of new customers, far fewer will be recoverable.
3. **Rating fields lost and gained.** Lost: exact ages (bands only), credit band, annual mileage, telematics. Gained: gender, which the data page says may be used for pricing here.
4. **The loss experience is different, line by line.**

| Line | Accident year | Frequency (dev / emg) | Mean cost (dev / emg) | Cost per exposure year (dev / emg) |
|---|---|---|---|---|
| Motor | 2023 | 0.140 / 0.189 | 4,191 / 2,294 | 588 / 433 |
| Buildings | 2023 | 0.046 / 0.046 | 9,519 / 11,269 | 433 / 514 |
| Contents | 2023 | 0.041 / 0.035 | 2,848 / 2,618 | 117 / 91 |

How to read this: 2023 is the more developed year (under 5% of its claims still open). Example: emerging motor has a third more claims per policy year (0.189 against 0.140) at a little over half the cost per claim (2,294 against 4,191), so its cost per policy year is about a quarter lower (433 against 588). 2024 figures are in the scratch check and point the same way, but a quarter of 2024 buildings claims are still open.

The peril mix fits the weather files: buildings storm is 44% of claims against 30% (wind exposure is higher), and motor glass and hail 19% against 14% (hail days doubled, median 6 to 12). The wildfire index median moves from 0.15 to 0.51; no peril in the data is named wildfire.

## What carries over unchanged (checked)

- **Construction codes are a relabel.** After BRK, STN, WOD and MIX are mapped to brick, stone, timber and non-standard, the mix matches Part 1 to a tenth of a percent (brick 69.9% against 70.0%, timber 12.0% against 12.1%). All four codes are clean, with no spelling variants.
- **The Part 1 traps are present at Part 1 rates:**
  - exposure entered in days: 0.10% to 0.12% of training rows
  - exposure above one year: about 0.4%
  - sum insured about ten times too large: 0.16% to 0.20%, in the test files as well (104 buildings and 135 contents test rows above ten times the emerging market's own 99th percentile)
- **Identifiers are clean:** one length per identifier, so no lost padding. The motor customer_id padding loss seen in Part 1 is absent.
- **Report delays are similar:** median 6 to 9 days by line.
- **The files agree:** the sample submission holds all 209,063 test rows once each, and the 20 reference policies are all in the test file of their own line.

## New in this market

- **Occupation is free text.** 80 raw values fall to 40 after trimming spaces and ignoring case. 11,975 customers (9%) carry a "Self-employed" prefix, including values such as "Self-employed student".
- **Test rows are shorter.** 30% of test rows run a full year, against 67% of training rows, and 22% start on 1 January.
- **Expenses.** Fixed expense is 85 CU per policy (developed 45), broker commission 20% (15%), premium tax 8% (12%), catastrophe reinsurance 12% of property risk premium (6%), and target profit 8% (5%). The comparison-site commission is blank for the emerging market, yet 10,458 test rows (5%) are sold through comparison sites.

## Against the Part 2 build settings already decided (Y1 to Y4, D32)

The actuary decided Y1 to Y4 (a) on 24 October from the Part 1 post-mortem (`docs/specs/SPEC_part2_settings.md`, `decisions/postmortem_results.md`). Each one's precondition, measured on the emerging data:

- **Y3, recoveries on open claims (rule R10): the pattern holds.** As in Part 1, only motor recovers. Closed-claim recovery rates are own damage 24.0% of incurred (Part 1 23.6%) and theft 4.05% (Part 1 4.01%). None of the 1,056 open emerging claims carries a recovery. Open claims are 35.8% of motor 2024 incurred and 16.8% of 2023, so the overstatement should be at least as large as Part 1's 1.28%. **Built on 25 October (D33):** rule R10 in `decisions/data_rules.md`, with its two pattern thresholds proposed and dated in `decisions/thresholds_part2.md` (register Y3-T, confirmed by the actuary on 25 October). On the emerging market it lowers motor 2024 net cost by 1.28% and leaves property unchanged.
- **Y1, the premium as our fields predict it: the premium is there.** `written_premium` is present and positive on every training and test row. Test medians: motor 540, buildings 257, contents 127. New-business test rows carry a premium like renewals (motor 550 against 514). The premium model's R² is the first check in the build. If it is low, the build stops for the actuary, as the spec says.
- **Y2, every legitimate field to the trees: two fields need a ruling first.**
  - **Gender** is new and is not on the P6 exclusion list, because Part 1 had no such field. Read literally, Y2 would put it into the trees by default. It stays out until the actuary rules on it (item 4 below).
  - **The code-level threshold** (to be written before use) decides which codes are credibility-encoded: make_model has 300 levels, as in Part 1; occupation 80 raw (40 once cleaned); previous_insurer 40. Every other code has 15 levels or fewer.
- **Y4, weather unchanged: a possible reason to reopen, not yet tested.** The weather files differ a lot: hail days double (median 6 to 12), the wildfire index triples (0.15 to 0.51), and freezing days fall. The claim mix moves the same way (storm 44% of buildings claims against 30%, glass and hail 19% of motor against 14%). The spec reopens Y4 only on a stronger weather *signal* (weather against claims), and a level shift alone is not one. That test has not been run. If it shows a stronger signal, it comes to the actuary as a new card.
- **Weather years.** Both markets carry 2025 weather rows, which the data page does not mention (emerging 2023 to 2025, developed 2021 to 2025). Y4's target-year factor can therefore use the emerging market's own 2025 weather.
- **Claims before the training years.** 34 emerging claims (all closed, 89,919 net) have 2021 and 2022 loss dates on policies whose training rows start in 2023. Part 1 had the same pattern (accident-year 2020 claims). The claim-to-row rule (a loss date inside the row's period) leaves them out, as before.

## Decisions to put to the actuary, one at a time

None of these is decided. Each goes to the judgment register with options and evidence when we reach it. Recoveries, on the first list, is settled by Y3.

1. **How much to carry over from Part 1.** Options: (a) fit the emerging market on its own data; (b) use the Part 1 model as a starting point, with the emerging data deciding how far to move from it, by line and by peril (a credibility blend, as with a new territory); (c) fit both markets together, with a market effect. This is the main adaptation story the judges asked for. Y1 already gives one complement inside the market: the emerging insurer's own rating plan, read through our fields. This decision is about the other one, the developed market's experience.
2. **New business against renewal.** Training is 86% renewals and the test book is 76% new business. The question is whether new-business loss experience in training (6,535 motor, 3,786 buildings and 4,716 contents rows) is credible enough to set a loading, or whether the broker book needs its own assumption.
3. **The blanks.** Whether blankness predicts loss here (Part 1 found it did not), and the treatment of the extra 10% of blanks that cannot be recovered from history.
4. **Gender as a rating factor.** The data page allows it here. A professional and fairness judgment, as well as a test result. Needed before the Y2 tree build, which would otherwise include it.
5. **Occupation cleaning.** The mapping rule (trim, ignore case, read "Self-employed" as a separate flag) is a data rule to write down before use. It also sets occupation's level count for the Y2 threshold.
6. **Comparison-site commission.** It is blank for 5% of test rows. This affects only the office premium build-up for the reference policies, not the Kaggle score.

Thresholds to write and date before their runs: the Y2 code-level count, the Y3 pattern criterion, and the Part 2 thresholds file (runbook step 3).

## Changes to the kit made for this run

- `scripts/what_changed.py`: checks each market against its own documented codes (`--codes emerging`). It now skips identifier-length lines when the lengths match and only the row counts differ, shows trap counts as a share of rows, adds a blank-share section, and checks the sample submission, reference policies and expense table (`drift.market_files`, tested in `tests/test_drift.py`).
- `scripts/run_market.py`: reads `codes` from the market config.
- `docs/PART2_RUNBOOK.md`: commands updated for the `developed/` and `emerging/` folders.
