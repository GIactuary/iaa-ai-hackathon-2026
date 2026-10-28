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
