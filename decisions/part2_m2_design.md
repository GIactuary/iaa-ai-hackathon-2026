# Part 2 milestone 2: test design, written before any run

28 October 2026. The actuary: "Of course, start. I want to get as close as possible, and I want you to give you the best analytical work." The check file scored 0.14266 on the public board. The gap diagnostic (`outputs/markets/part2/part2_gap_diagnostic.json`) ruled out the level, the insurer's premium, the contacts field, region and seasonality as the source of the gap. This design tests the remaining levers on the risk shape. Every choice goes through P2-V; the hold-out stays locked.

## What is scored

- **Rows:** emerging fitting rows (85% of customers), 2023-2024.
- **Target:** net cost to date, uncapped, as the competition scores it; predictions are multiplied by reporting completion, per peril where the candidate prices by peril.
- **Level:** each candidate's predictions are rescaled to the scored rows' actual total in each line (common level per line), so the tests measure the shape within lines, including where the large losses sit. The level is set separately by the level chain.
- **Pooled:** deviance (Tweedie 1.5) summed over the three lines, as the competition pools them.
- **P2-V:** V1 fit 2023, score 2024; V2 customer-grouped five-fold cross-validation over 2023-2024, seeds 11, 22, 33, each row's deviance averaged over the seeds. A candidate replaces the incumbent only if it lowers pooled deviance in both, and its 95% paired bootstrap interval on V2 (customers resampled within line, 2,000 draws) lies above zero.
- Parts fitted on Part 1 rows are fitted once; only the emerging parts (peril levels, layers) are refitted in each split.

## Candidates, tested in this order (each against the incumbent at that point)

1. **K0, incumbent:** the check file's shape: Part 1 W2 GLM on capped cost, one excess factor per line.
2. **K1, by peril:** one Tweedie GLM per peril on Part 1 rows (capped cost, the W2 factors), each peril's level set to the emerging fitting rows' completed capped cost for that peril, plus each peril's own excess factor (emerging own for motor, Part 1's shares for property, as P2-L). Tests whether pricing perils separately moves cost to the right rows.
3. **K2, large-loss perils by property:** K1, with the large-loss perils (buildings fire and flood, contents flood, motor bodily injury) priced as frequency times severity on uncapped cost: a Poisson frequency GLM and a gamma severity GLM, both on Part 1 rows with the W2 factors. Severity then follows sum insured and property type rather than being capped and spread.
4. **K3, address hazard:** K2 with the address's wildfire normal (buildings fire) and distance to river and elevation bands (flood) added to the large-peril frequency GLMs. The emerging wildfire index lies above Part 1's range (0.5 against 0.15), so the term extrapolates; the per-peril level calibration on emerging rows absorbs the average, and the test shows whether the slope holds.
5. **K4, best family per line:** the Part 1 LightGBM (as in the diagnostic) in place of the GLM where it wins, per line, and the 50/50 log blend.
6. **K5, gender in motor:** an emerging layer (gender, offset on the incumbent) in motor only (P2-S).
7. **K6, pooled challenger (P2-C (c)):** one GLM on Part 1 and emerging fitting rows together with a market factor.

Order matters only for what each is compared against; every candidate's result against the original K0 is reported as well.

Outputs: `scripts/part2_candidates.py`, `outputs/markets/part2/part2_candidates.json`.

## Round 1 results and addendum (28 October, written before round 2 runs)

Round 1 (`outputs/markets/part2/part2_candidates.json`): no candidate clears P2-V. Positive in both tests but with the cross-validation interval including zero: gender in motor (V1 +5,942 [2,654, 9,408]; V2 +7,895 [-2,113, 17,898]), the tree in contents (+10,195; +13,264 [-3,419, 30,448]). Address hazards: V2 +30,641 [-8,684, 71,227], V1 -2,049. By peril and fire and flood by property: no gain. The pooled intervals are about ±30,000 wide, set by a few large buildings claims.

Round 2 adds candidates that bring information the incumbent does not use. Each is tested against the incumbent (still K0), under the same rule:

8. **K7, the row's own-year weather.** The incumbent applies the target year's weather as one average per line (Y4). K7 applies each row's own year factor (its address's weather that year against its normal; the normal for a fitting row is the other experience year, so a row's own weather never enters its normal). Variants: (a) Part 1's elasticities on the Y4 perils; (b) elasticities fitted on the emerging fitting rows of each split, on the Y4 perils plus buildings fire with the wildfire index and buildings and contents flood with rainfall. The test-year file carries 2025 weather for every address, so the same factor is available for the 2025 rows. Y4 decided weather as one factor per line; a row-level factor would reopen Y4 and comes to the actuary as a card if it passes.
9. **K8, prior-year claims history.** A layer on the incumbent with the customer's claim count in the previous calendar year across all lines (none, 0, 1, 2 or more; rows with no prior-year row are "none"). For 2025 rows the previous year is 2024. All new-business test rows are "none".
10. **K9, everything positive together:** gender in motor, the tree in contents, address hazards, and whichever of K7 and K8 is positive in both tests.

## Round 2 results and addendum (28 October, written before round 3 runs)

Round 2 (`outputs/markets/part2/part2_candidates_round2.json`): K9+K7b passes P2-V against K0 (V1 +7,498; V2 +37,526 [298, 76,858]); history (K8) fails and hurts in V2. A direct check after round 2: emerging new business against renewal, actual over the Part 1 base's expected, is 1.37 [1.10, 1.66] on contents frequency (Part 1's W2 dropped new business from contents, L8), 0.97 on motor frequency, 0.88 on buildings. The test book is 76% new business against 14% in the scored 2024 rows, so this matters far more to the 2025 level than the 2024 test can show.

11. **K10, new business layer:** a new-business factor fitted on emerging fitting rows on top of K9+K7b, per line, tested against K9+K7b under P2-V. Reported with each line's fitted relativity.

## Paired public-board check (M2-B (a)), rule written before either file is uploaded

28 October. Two files differ only in the new-business factor: `part2_v2_nb-none_wx-row.csv` (no adjustment) and `part2_v2_nb-cred_wx-row.csv` (M2-N (b)). The actuary uploads both. **Rule:** keep the credibility-weighted factor unless its public score is lower than the unadjusted file's. Whatever the result, no further variant is chosen on the public board; the comparison and its outcome are reported in the write-up.

**Result (28 October).** Public scores: credibility-weighted 0.15312, no adjustment 0.15080, and a third file the actuary also uploaded, the full emerging figure, 0.14646 (the check file: 0.14266). Under the rule, the credibility-weighted factor stays. The ordering is the one credibility theory predicts: moving all the way to the thin emerging figure loses, ignoring it loses less, the credibility weight scores best. No further choice is made on the public board.

## Round 3 addendum (28 October, written before round 3 runs)

**Public-board noise, measured first** (2024 scored rows, 1,000 random thirds): one model's D² spans 0.107 to 0.165 (90%). Paired differences between two of our files: K9+K7b against K0 true +0.0016, on a third −0.0085 to +0.0102 (looks worse 39% of the time); gender in motor true +0.0013, on a third +0.0004 to +0.0021 (1%); tree in contents true +0.0022, −0.0017 to +0.0057 (17%). Proposed rule for board use, for the actuary: a variant is judged on the board only if its expected public gain exceeds twice its paired public-board spread measured this way; buildings large-loss changes are judged on our own validation only.

Round 3 candidates, each against the incumbent K9+K7b (the improved file's shape) under P2-V:

12. **K11, Y1 as decided (the premium as our fields predict it):** a LightGBM of log written premium on the fields both markets carry, fitted on Part 1 rows without claims; its prediction (lwp_hat) enters the Part 1 base GLM as one numeric term, in every line, replacing the base where the line's incumbent uses it (motor, buildings attritional perils) and as a tree input in contents. In Part 1 this was the post-mortem's largest gain (PM6 buildings +110,298 [62,978, 164,210]).
13. **K12, every factor credibility-weighted (the penalised form of P2-C (c)), amended before any run:** a layer on the incumbent with every W2 factor and region, fitted on emerging fitting rows with an L2 penalty that shrinks each emerging deviation toward zero (toward Part 1's relativity). The penalty strength is chosen from 1e-5 to 1 on fold 1 of the fitting rows' customer-grouped inner folds. This is pooling with the data choosing how far each factor moves, without refitting a million Part 1 rows in every split.
14. **K13, credibility layers:** region, make_model and occupation each as one credibility-weighted log mod against the incumbent (Bühlmann-Straub, `fields.credibility_encode`, out of fold), in a layer on K9+K7b.

**Round 3 results (28 October)** (`outputs/markets/part2/part2_candidates_round3.json`), against K9+K7b:
- K11, Y1 premium in the Part 1 base: out of time +5,783 [−6,069, 16,992]; cross-validation +2,588 [−10,784, 15,780]. Positive in both tests, interval includes zero: fails. Motor gains in both tests (+2,844; +5,981), buildings and contents are mixed. Premium model R² on Part 1 rows: motor 0.916, buildings 0.957, contents 0.917.
- K12, every factor credibility-weighted: −3,460; −10,984. The inner-fold choice of penalty sits at the strongest value tried (1.0) in most splits: the emerging data cannot support moving the factors away from the incumbent.
- K13, credibility-weighted codes: −43,451; −16,730. Fails.

**Motor-only Y1 premium (28 October, M2-R).** Measured before any upload (`scripts/part2_motor_premium_check.py`): against K9+K7b, out of time +2,844 [44, 5,469], cross-validation +5,981 [1,547, 10,441]. It passes P2-V on its own, so Y1 (decided (a), subject to the keep rule) enters motor. Expected public gain +0.0006 against twice its paired spread on a third (0.0008), so under M2-R the board cannot judge it; the decision rests on our own validation. Buildings and contents keep the K9+K7b shape (Y1 there failed in round 3).

## Tuning round (P2-F (a)), written before it runs, 28 October

Incumbent **I**: K9+K7b with motor's base replaced by the Y1 premium base (`part2_v2_nb-cred_wx-row_mprem`). Settings are chosen where each model is fitted (Part 1 rows), on fold 1 of five customer-grouped inner folds (seed 11), by Tweedie 1.5 deviance on capped cost at common level, as Part 1's step 7 did; the tuned model is then tested against I on emerging rows under P2-V.
- **Motor base GLM and buildings attritional peril GLMs:** training Tweedie power from the dated grid (1.1 to 1.9, step 0.1).
- **Contents tree:** leaves {15, 31, 63} × minimum rows per leaf {100, 400, 1600}, learning rate 0.05, early stopping 100.
- **T, all tuned settings together,** against I. Kept only if it passes P2-V.

**Tuning result (28 October)** (`outputs/markets/part2/part2_tuning.json`). Chosen on Part 1 rows: motor base power 1.4; buildings attritional perils 1.5 (accidental damage), 1.6 (escape of water, subsidence), 1.7 (storm); contents tree 15 leaves, 400 minimum rows, 152 trees. T against I: out of time +317 [−5,449, 6,025]; cross-validation +2,793 [−3,976, 9,604]. Fails P2-V; I keeps Part 1's settings. The gain sits in contents (+506; +3,701); motor and buildings are flat. Pooled D², I: 0.1396 out of time, 0.1448 cross-validation (K0: 0.1374, 0.1402).
