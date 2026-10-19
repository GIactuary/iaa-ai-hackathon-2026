# Phase 2 screens (run 2026-10-18)

Thresholds were agreed with the actuary and written in `decisions/thresholds.md` before either screen ran: `screen_z` = 4 and `refer_size_ratio` = 1.25. The screens are coded in `src/screens.py` with tests in `tests/test_screens.py`, and the run scripts are in `data/work/screens/`. The screens list records only. The decisions are in the judgment register under X1, X2 and X3.

## Conclusion first

The data is clean apart from one claim and one claims-handling pattern. The claim screen flags a single claim, and the test file sits inside the training evidence almost everywhere. Three decisions remain for the actuary:
- the 11.9M bodily injury reserve
- the three repairers
- the treatment of a short referral list

## 1. Claim screen: one claim above z = 4

- **What was screened.** 161,028 non-repudiated claims with a cost, matched to their rows. Log net cost was standardised within each of the 15 perils.
- **Result.** One claim exceeds z = 4. About 5 would be expected by chance under a log-normal tail.
- **The other perils are lighter-tailed than log-normal.** Outside motor bodily injury, no claim exceeds z = 2.9, and no claim in those perils exceeds z = 3.

| Claim | Peril | Loss / report | Status | Paid | Incurred | z | Peril typical cost |
|---|---|---|---|---|---|---|---|
| CL30973963 | Motor bodily injury | 25 Jun 2023 / 16 Dec 2023 | Open | 738,769 | 11,894,277 | 5.23 | 9,669 |

Evidence that this is a genuine large loss, not a keying error:
- The 739k already paid is the largest paid amount on any open bodily injury claim. It sits below only one closed claim (1.12M).
- Its incurred-to-paid ratio of 16.1 is ordinary for open bodily injury: it ranks 408th of 2,878, against a median of 4.8 and a 90th percentile of 22.5.
- Dividing by ten (1.19M) would assume an error the evidence does not show.

Its weight:
- It is 10.1% of motor accident year 2023 net cost and 2.8% of all motor net cost for 2021 to 2024.
- It sits on a third-party-only policy (vehicle value group 12, driver aged 34 at loss). That row's `renewal_decision` reads "declined_by_insurer" for 2023, yet the policy has a 2024 row.

Spec P4 already caps large claims and loads the excess. The open question is how this claim's excess enters the load (X1).

## 2. Train-versus-test evidence check: one size referral, no unseen levels

| Check | Test rows referred | Note |
|---|---|---|
| Buildings sum insured by property type and bedrooms | 0 | Largest test ratio to the segment's training maximum is 1.17 |
| Contents sum insured by tenure and bedrooms | 1 | C03025218: renter, 7 bedrooms, 88,400 against a segment maximum of 38,100 (2.32 times) |
| Motor vehicle value group, engine size, mileage | 0 | Largest ratios 1.00, 1.03, 0.87 |
| Category levels never seen in training | 0 | Checked on all rating categoricals and address fields |
| Licence implied before age 16 | 755 | Impossible combination |

What each referral means:
- **The contents referral is a thin segment.** Training holds only 38 rows of renters with 7 bedrooms. The referred value sits inside renters overall (maximum 152,600) and owners with 7 bedrooms (maximum 129,600). It is a new-business row.
- **Licence before 16 carries no risk signal.** Those training rows claim at 0.139 against 0.141 for the rest.
- **NCB above licensed years is a flagged segment, not a referral** (34,781 test rows). The NCB value still carries signal. Among drivers licensed for 0 to 3 years, frequency falls from 0.388 at zero NCB to 0.131 at four years, and flagged rows claim less than unflagged ones (0.239 against 0.282). Capping NCB at licensed years would throw that away.
- **Values just outside the training range** are one year of ageing: driver age 92, vehicle age 29, renewal count 19, and engine size 3,700 on one row. They are within any reasonable extrapolation.

The list is in `data/work/screens/referrals_v0.csv`. Treatment is X3.

## 3. The three repairers (RP0733, RP0901, RP0417)

The pattern is steady and runs every year:
- About 500 claims a year, against at most 112 claims in total for any other repairer.
- 32% to 38% repudiated every year from 2021 to 2024, against 0.3% to 0.5% elsewhere.

Paid claims through these repairers cost more:

| Peril | Mean, three repairers | Mean, others | Median, three repairers | Median, others |
|---|---|---|---|---|
| Own damage | 3,322 | 1,838 | 2,815 | 961 |
| Third-party property | 3,949 | 3,673 | 4,262 | 2,525 |

- Recoveries appear on 8.5% of their own-damage claims, against 29% elsewhere.
- They are 2.3% to 2.7% of motor own-damage and third-party property net cost, steady by year.

The pattern belongs to the repairer, not the policyholder:
- 2,115 policies used them, 2,095 of them once.
- The region mix matches the book.
- Policies with a past claim through them have a 2024 frequency of 0.151, below the 0.217 of policies whose past claim went elsewhere.
- Only 4 of 330 claims through them in 2024 came from a returning policy.
- 1,623 of these policies are in the 2025 test file.

Reading: this looks like a claims-handling problem, repair inflation or fraud at three repairers, that is still going on. It is not a policyholder risk the rating plan can see. It is also a finding for the claims function: steering own-damage work away from these three would save roughly 2.5% of that cost.

## Tests

`tests/test_screens.py` covers four things on synthetic data:
- the claim screen flags only the upper tail, within peril
- size referrals work within a segment and on an unseen segment
- size referrals work on the whole book
- unseen levels and the impossible-combination flags are detected
