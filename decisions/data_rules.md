# Data rules (phase 1)

Each rule states what it does, the evidence, and the counts. Code: `src/data_rules.py`. Tests: `tests/test_data_rules.py`, including checks on the real files. No rule drops a row. Every repaired row carries a `dq_` flag, so the screens and models can see it. The same rules run on training and test files.

Status: proposed by the AI on 2026-10-18 under spec P8 ("repair by stated rule and flag every row"). The rules need the actuary's confirmation (register P8-R).

## R1. Exposure is restated from the dates when it was entered on the wrong basis

Every training row fits exactly one of three bases:

| Line | days / 365 (correct) | Day count (e.g. 365 for a year) | days / 360 (a full year reads 1.0139) |
|---|---|---|---|
| Motor | 974,315 | 934 | 4,719 |
| Buildings | 703,598 | 656 | 3,472 |
| Contents | 717,364 | 729 | 3,524 |

- **Rule.** Restate the day-count and 360-day rows as days / 365, and flag them (`dq_exposure_days`, `dq_exposure_360`).
- **Test files.** All 626,639 test rows are already on days / 365.
- **Guard.** A row that fits no basis stops the run.
- **Reading.** Left unrepaired, a 365 entered as years would put 365 years of exposure on one row, and the 360-day basis inflates exposure by 1.4%.

## R2. Sum insured entered 1,000 times too large is divided by 1,000

- **Rule.** A sum insured above `si_unit_multiple` (10) times the line's training 99th percentile is divided by 1,000 and flagged (`dq_sum_insured_x1000`). The threshold is in `decisions/thresholds.md`.

| Line | Train rows repaired | Test rows repaired | Largest genuine value | Smallest erroneous value |
|---|---|---|---|---|
| Buildings | 1,395 | 370 | 1,061,000 | 52,000,000 (49 times larger) |
| Contents | 1,402 | 360 | 346,700 | 8,200,000 (24 times larger) |

- **The threshold does not change the result.** The flagged set is identical for any multiple from 5 to 50.
- **Divided by 1,000, the values fit.** They match the same policy's other rows (median ratio 1.00, 90% between 0.86 and 1.15).
- **The errors are spread evenly**, at 0.18% to 0.21% of rows in every year.

## R3. Construction type uses the four documented levels; codes T and W are both timber

The data page documents brick, stone, timber and non-standard. The data holds 11 spellings.

- **Rule.** Case and spelling variants map to their level: Brick, brick and BRICK become brick. T, t, W and w become timber, flagged `dq_construction_timber_code`. Any code outside the map stops the run.
- **Evidence that T and W are one level:**
  - T appears only in 2021 and 2022. W appears only from 2023, including the test file.
  - The share is the same each year (11.4% and 11.5%).
  - All 18,218 T-coded houses that renewed at the same address in 2023 became W. Every other code at the same address kept its level in 100% of cases.
- **Reading.** This is a coding change in 2023. Kept as two levels, a model would read the year effect into construction, and the test file would only ever see W.

## R4. Motor customer ids are padded back to eight digits

- **Rule.** Pad customer ids to 8 digits, flagged `dq_customer_id_padded`.
- **Evidence.** 98,110 motor training rows (10%) and the same share of test rows have customer ids of 2 to 7 digits. The customers table uses 8. Padding with leading zeros matches every one of them, and all buildings and contents ids already match.
- **Reading.** Without it, 10% of motor rows lose their customer fields (credit band, prior claims, licence points).

## R5. Withdrawn: annual mileage of 0 is a real value

**First proposal (wrong).** The AI first read a mileage of exactly 0 as a blank code. Its evidence was that 11,928 training rows show 0, the next value is 1,200, and the zero rows have the same usage mix as the whole book.

**Why it was withdrawn.** The data steward's independent check, rerun here, showed the zero rows claim like the lowest-mileage cars.

| Annual mileage | Motor claim frequency |
|---|---|
| 0 | 0.101 |
| 1 to 3,000 | 0.107 |
| 3,000 to 5,000 | 0.109 |
| 10,000 to 15,000 | 0.157 |
| Over 20,000 | 0.204 |
| Blank | 0.137 (the book average) |

Where the same policy has an earlier row, that row's median mileage is 5,000. A matching usage mix showed only that zero is not tied to usage. It did not show that zero is a blank.

**Rule.** Zero is kept as a value. Exchange D8 records the correction.

## R6. Claims belong to the row whose period contains the loss date

- **Rule.** A claim matches its policy's row with period_start ≤ loss_date ≤ period_end. Net cost is incurred less recoveries. Repudiated claims carry zero cost and are flagged.
- **Evidence:**
  - 99.7% of claims match exactly one row and none match two.
  - The 414 unmatched claims are mostly December 2020 losses on policies starting 1 January 2021. They are outside any row's period, so they are outside the target by its definition.
- **Guard.** A claim matching two rows stops the run.

## R7. Inputs that are never used, and inputs cleared for use

**Never used:**
- `renewal_decision`. It is an outcome: "declined by insurer" rows had a claim 95% to 99% of the time.
- `customer_service_contacts` as recorded.
  - It averages 0.41 on rows without a claim, 2.96 with one and 5.8 with two. It counts contacts made during the row, mostly about the claim.
  - A claim in the previous 12 months does not raise it (0.40 against 0.44), so it carries no prior-period history either.
  - The test mean of 0.40 matches the claim-free training rows. The test values hold no information about test claims.

**Cleared at the point of pricing:**
- `days_unoccupied`.
  - The distribution is the same in train, test and every year (median 7, 99th percentile 102).
  - It exceeds the row's own length on 6,467 rows, so it is not measured inside the row. It is a declared value that changes from year to year.
  - Frequency rises from 0.044 to 0.061 above 30 days, which is a plausible risk effect (unattended escape of water).
- `missed_payments_12m`.
  - The distribution is the same in train and test.
  - It is above zero only for monthly payers, as a count of missed instalments would be.
  - Motor frequency rises from 0.137 at zero to 0.217 at three or more.
- `renewal_count` and `new_business`.
  - Renewal count rises by exactly 1 a year within a policy, and the test value is the 2024 value plus 1 for all 216,108 continuing motor policies.
  - No new-business row has a renewal count above zero.
- `telematics_score` is blank exactly when the policyholder is not opted in, on every train and test row. The blank means absent and is handled with the opt-in flag.

**Kept out of the rating plan** (spec P6): surname, email domain, marketing segment, campaign code, branch, marketing consent, document preference.

## R8. Blanks recovered from the same policy's other rows

Full readings are in `decisions/missing_values.md`.

- **What is recovered.**
  - Motor vehicle age (through the implied model year), engine size and declared mileage come from the same policy and make-model.
  - Motor overnight parking comes from the same policy.
  - Buildings smoke alarms come from the same policy at the same address.
  - Contents security level and high-value items come from the same policy.
  - Each recovered value is flagged `dq_<column>_recovered`.
- **Timing.**
  - Declared mileage changes from year to year, so it comes only from an earlier row.
  - Fixed attributes (model year, engine, parking, security, high-value items, smoke alarms) may come from a later training row.
  - Every recovery on a test row uses an earlier row, so the value was known when the 2025 policy was priced.
- **Never from make_model.** All 300 codes show several engine sizes, body types and fuel types, and the most common engine size per code matches only 12.6% of rows.
- **Evidence.**
  - The implied model year is single-valued in 99.91% of vehicles with two or more known rows.
  - A leave-one-out test reproduces vehicle age 99.92% of the time.
  - Parking, security level and high-value items never change within a policy, and smoke alarms change only with the address.

| Column | Train blanks before | Train blanks after | Test blanks before | Test blanks after |
|---|---|---|---|---|
| Vehicle age | 24,427 | 930 | 6,316 | 1,363 |
| Engine size | 24,694 | 957 | 6,440 | 1,377 |
| Overnight parking | 24,371 | 70 | 6,398 | 958 |
| Annual mileage | 24,711 | 9,728 | 6,271 | 1,362 |
| Smoke alarms | 17,867 | 393 | 4,602 | 815 |
| Security level | 18,087 | 59 | 4,677 | 674 |
| High-value items | 18,088 | 60 | 4,707 | 683 |

## R9. How the remaining blanks enter the models

**Flood zone.** Blank stays its own level, beside distance to river. A zone is never imputed without a blank indicator, because the blank carries risk:
- Blank-zone buildings have 15.1 flood claims per 1,000 years. Zones 1 to 5 have 0.4, 1.7, 6.9, 25.2 and 75.2.
- Blank zones are five times as common within 250 m of a river as beyond 1 km.

**Structural blanks are not imputed.**
- `telematics_score` is used only with the opt-in flag. Opted-in rows claim more (0.153 against 0.139), so the flag stays in any model that uses the score.
- `campaign_code` is blank for every customer who joined before 2021. It is outside the rating plan in any case (P6).

**Blanks missing at random** (about 2.5% of each column, flat by year, region and channel, claim frequency within noise). These are the customer fields, days unoccupied, distance to river, soil, roof, energy rating, and the unrecovered motor and contents fields.
- Trees use their own missing-value handling.
- In the GLM, categorical fields get their own level, and numeric fields get the median plus an indicator.
- Each indicator is kept only if it passes the paired test.

**Post-event fields are never inputs.** `repairer_id` and `third_party_id` exist only on claims.

## Passed to the phase 2 screen and the judgment register

The data cannot say which field is wrong in these, so they are flagged, not repaired.

- **NCB years exceed licensed years** on 127,041 motor training rows (13%) and 34,781 test rows. They are concentrated among drivers licensed under three years: 67% of rows with zero years licensed.
- **A licence before age 16 is implied** on 2,962 training rows and 755 test rows. The implied licensing age goes as low as 7.
- **25,126 customers joined before age 16**, 286 of them at age 3.
- **Three repairers show a non-market pattern.**
  - RP0733, RP0901 and RP0417 carry about 700 claims each, against at most 112 for any other repairer.
  - 34% to 35% of their claims are repudiated, against 0.4% elsewhere.
  - Their roughly 1,400 paid claims are in the target and go to the outlier screen.
- **The customers table is a 2025 snapshot.**
  - `years_at_address` is consistent with ages in 2025 for every customer, but not with ages in 2021 for 46,945 customers.
  - So 2021 to 2024 rows see later customer values. For `licence_points`, `credit_band` and `household_size` a later value can reflect a claim.
  - Whether these fields may be used is a judgment (register J-cust).

## Second check on missed payments (exchange D9, at the actuary's request)

`missed_payments_12m` predicts claims on its own, and does not merely stand in for credit band or payment method. It works like the usual motor surcharge indicator, and in this data it works for property too.

- **Inside every credit band**, monthly payers with a missed payment claim more. In motor the ratio runs from 1.14 to 1.23 across bands A to E. Missed payments are far more common in bands D and E (33% and 55% of monthly payers, against 5% in A), yet the effect holds within each band.
- **After controls.** A Poisson GLM controlled for payment method, credit band, age, channel, new business, licence points, prior claims, renewal count, and NCB, vehicle value, cover and use (motor) or sum insured (property). It was fitted on 2021 to 2022 and scored on 2023 at a common level.

| Line | Relativity: 1 missed | 2 missed | 3 or more | Out-of-time Poisson D², without | D², with |
|---|---|---|---|---|---|
| Motor | 1.17 | 1.18 | 1.24 | 6.68% | 6.72% |
| Buildings | 1.20 | 1.25 | 1.23 | 0.77% | 0.83% |
| Contents | 1.15 | 1.12 | 1.36 | 0.89% | 0.95% |

- **Not an outcome of claims.** Among monthly payers, a claim in the previous year moves the share with a missed payment only from 18.0% to 19.2%.
- **How to read this.** A relativity of 1.17 means a motor policy with one missed instalment claims 17% more often than an otherwise identical policy with none. The D² gains look small in absolute terms. As a share of what the controls already explain, they add 0.7% in motor and about 7% in buildings and contents.

It stays a candidate rating variable and must pass the phase 5 drop-one test like any other.

**Second round (the actuary asked whether the factor could reduce prediction quality).** It does not.

- **The gain holds in every out-of-time test.** Deviance falls in all six tests: three lines, each scored on 2023 (fitted to 2022) and on 2024 (fitted to 2023). The 95% bootstrap interval, resampling policies, lies above zero in five.
  - Motor saves 72 deviance points on 2024, interval 42 to 103.
  - The exception is buildings scored on immature 2024: 13 saved, interval −5 to 30.
- **The relativity for one missed payment is stable.** Fitted on each year alone it runs 1.13 to 1.32 in every line.
- **The three-or-more level is noisy.** It runs from 0.96 to 1.42 and holds only 0.3% of rows. The form will be any missed payment as a yes/no, or 0, 1 and 2 or more, settled in the phase 5 test.
- **Credit band and missed payments share signal; they do not double-count.** Motor credit band E falls from 1.31 to 1.23 once missed payments enters. The joint fit splits the shared signal.
