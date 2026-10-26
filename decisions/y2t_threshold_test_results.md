# Y2-T threshold test: results

26 October 2026. Design `decisions/y2t_threshold_test_design.md`, committed before any run (00e5efb). Script `scripts/y2t_threshold_test.py`; evidence `outputs/y2t_threshold_test.json`; run logs in `data/work/y2t/`. All figures are claim-frequency (Poisson) deviance at a common level per line, summed over the three emerging lines; lower is better. Intervals are 95%, from a paired bootstrap resampling customers. The key totals were recomputed by separate code and match.

## Conclusion first

The level matters at one boundary only. A threshold at or above occupation's 80 levels is clearly worse in both designs. From 4 to 20, the designs cannot tell the thresholds apart: the differences are smaller than the spread between repeated fits. The pre-registered rule picks 4. The AI recommends 20 instead, and says why below; the actuary decides.

## Results

| Threshold | Encoded (emerging) | V1, out of time: over best [95%] | V2, cross-validation: over best [95%] |
|---|---|---|---|
| 4 | every code over 4 levels (adds age band, body type, soil, region, energy rating, colour) | +16 [−17, +50] | +21 [−15, +55] |
| 5 | make_model, occupation, colour, region, energy rating | +7 [−28, +43] | +8 [−22, +40] |
| 6 | make_model, occupation, colour, energy rating | +75 [+43, +108] | +11 [−17, +42] |
| 7 | make_model, occupation, colour | best | +8 [−13, +29] |
| 20 | make_model, occupation | +12 [−15, +40] | best |
| R0 | make_model and occupation left out | +47 [+12, +82] | +14 [−14, +40] |
| 100 | make_model only; occupation raw (80 levels) | +261 [+183, +339] | +321 [+240, +401] |
| 300 | nothing; make_model raw too | +369 [+274, +468] | +661 [+551, +766] |

How to read this: in V2, a threshold of 100 lets occupation into the trees raw. Its deviance is 321 above the best threshold, and the interval (240 to 401) is far from zero. The same model is also 307 worse than leaving occupation and make_model out altogether.

- **Repeated fits.** In V2, the same candidate's total moves by about 100 between seeds (threshold 20: 39,831, 39,840 and 39,934). The gaps among thresholds 4 to 20 are 7 to 21. Threshold 6's poor V1 result (+75) did not recur in V2 (+11), which points to fitting noise.
- **Threshold 20 against 4, seed by seed (V2):** 20 is lower in all three seeds, by 44, 16 and 1.
- **Per line (V2).** Motor gains most from encoding: 20 beats leaving the codes out by 17. In buildings and contents, threshold 20 and leaving the codes out are within 1.3 of each other, as expected for a field with no signal there.
- **Cleaned occupation (exploratory).** No measurable effect when encoded: V2 cleaned at 20 is +16.6 over R0, as typed +14.1. Raw, cleaning cuts the damage (−62 against −307 versus R0) but does not remove it.
- **Pre-registered selection.** Both designs pick threshold 4, so the consistency check passes.

## Why the AI recommends 20, not the rule's 4

The rule assumed that a lower threshold is the more economical model, because it encodes more codes. The results do not support that premise:
- **The gain comes from encoding the two codes with many levels.** Encoding the small codes as well (age band, body type, soil, region) buys nothing in either design.
- **Threshold 4 has nothing to show.** In V2 it is the weakest encoded regime and below leaving the codes out (−6.4). It is behind 20 in every seed.
- **It costs interpretability.** Encoding a five-level code such as age band replaces its levels with one smoothed number, so the trees' splits no longer show the bands.

So the point of diminishing returns runs the other way from what the rule assumed:
- Raising the threshold to 80 or above loses heavily.
- Lowering it below 20 gains nothing.
- 20 is the lowest threshold that encodes exactly the codes where raw entry is shown to do harm. It is the best threshold in V2 and within reach of the best in V1. It also falls inside Part 1's evidence: occupation raw at 20 levels passed there.

This departs from the pre-registered rule, and it is stated as such for the actuary's decision.

## Part 1 run

The full Part 1 run (two scorings, two targets, seven candidates, nine-setting search on about 830,000 motor rows) was restarted on 12 threads on the actuary's instruction. Its first two motor candidates took 22 minutes and about 4 hours. At that pace the run needs several days, beyond the Part 2 deadline. It adds no result to this file.
