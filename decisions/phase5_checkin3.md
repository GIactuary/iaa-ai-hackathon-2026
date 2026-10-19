# Phase 5 check-in 3: model structure and the per-peril excess factor (2026-10-19)

Run: `scripts/phase5_step6.py`, evidence in `outputs/phase5_step6.json`. Design written and committed first: `decisions/phase5_step6_design.md` (commit dde2d37). Every test is out of time on non-hold-out customers, on the W2 columns of each line, at a common level per line, under the keep rule. The decisions are register items M1 onward.

## Result in one line

No per-peril structure passes the keep rule in any line, so one capped Tweedie GLM per line stays. Motor frequency × severity is a near miss; property loses clearly. The per-peril excess factor ranks worse than the line factor in motor and buildings.

## Structure against the line GLM (priced form, the decision form)

Deviance saved against S1-L, A with its 95% customer-bootstrap interval, then B.

| Challenger | Motor | Buildings | Contents | All lines |
|---|---|---|---|---|
| S2-L Tweedie per peril | +5,934 [−15,577, 25,981], +31,775 | −76,918 [−138,657, −18,605], −47,721 | −30,869 [−59,978, −1,675], −17,718 | −101,852 [−170,864, −31,249], −33,664 |
| S3-L frequency × severity per peril | +12,541 [−12,076, 34,902], +35,947 | −19,093 [−67,192, 27,406], −33,017 | −32,007 [−60,081, −4,957], −6,706 | −38,559 [−98,249, 21,851], −3,776 |
| S4-L as S3, severity by year only | −6,200 [−40,264, 26,135], −14,686 | −162,250 [−233,094, −91,997], −131,167 | −178,699 [−258,720, −108,622], −151,216 | −347,148 [−456,791, −237,987], −297,070 |

None passes. The bare form gives the same verdicts (motor S3-L: +9,602 [−14,067, 31,249], +38,746), so the two forms agree and no line goes to the actuary on a disagreement.

Ranking D², S1-L against S3-L (priced): motor 5.68% and 5.73% on A, 6.35% and 6.50% on B; buildings 6.88% and 6.78%, 6.59% and 6.42%; contents 6.51% and 6.20%, 5.81% and 5.75%.

**Reading.**
- The frequency-severity split beats a Tweedie per peril across lines (+63,293 [3,445, 127,129] on A, +29,888 on B), so of the per-peril forms it is the better one.
- Severity relativities matter: dropping them (S4 against S3) costs 143,157 in buildings and 146,691 in contents on A. Sum insured drives severity in property.
- The split pays only where each peril has enough claims. Motor perils carry about 1,000 to 11,000 claims a year; buildings fire, subsidence and flood carry about 200 to 500, against the same set of rating factors in each per-peril GLM.

## The per-peril excess factor (K2 retest)

Per-peril factor against the line factor, within frequency × severity (priced): motor −4,205 [−6,753, −1,404], +132; buildings −6,776 [−10,420, −2,709], −3,682; contents +2,325 [586, 4,156], +329, which passes. Within the Tweedie per peril the signs are the same.

Excess above the cap sits in a few perils: final-fit factors are bodily injury 1.091 (line 1.031), buildings fire 1.087 and flood 1.011 (line 1.022), contents flood 1.073 (line 1.018). The excess is a handful of claims a year, and which rows produce it is not predicted by their bodily-injury or fire share: spreading the load over the whole line ranks better. The line factor applies to S1 in any case, since S1 has no peril split.

## The 2025 weather by row (information; added after the A and B frames were seen)

The priced S1 differs from the bare S1 only by the row-level weather multiplier, since a per-line off-balance does not move ranking. Row-level weather against none: motor +468 [−2,380, 3,243], −60; buildings −4,229 [−18,715, 11,375], −7,438; contents +295 [−789, 1,395], −1,740; all lines −3,466 [−18,910, 11,931], −9,238. It is inside the noise, slightly worse in both years in buildings and across lines. This test was not in the design; it was added once the frames showed the gap and is reported as information for card M3.

## Diagnostic dropped from the exhibits

The design's per-peril diagnostic scores each structure's peril component against that peril's own cost. In motor it is dominated by rows whose cover excludes the peril: the per-peril Tweedie drives third-party-only rows to near-zero own-damage, glass and theft predictions (standard deviation of log prediction 3.2 to 6.2, against 1.3 to 2.2 for frequency × severity). That decides the single-peril score but barely touches the line total, so it says little about structure. It stays in the evidence file and is not used for a decision.

## What frequency × severity shows (final fit 2021 to 2024, motor)

- **Age drives frequency, not severity.** Under-21 own-damage frequency 2.70 against age 30 to 39; own-damage severity flat by age (0.97 to 1.01).
- **Vehicle value drives own-damage severity.** Group 20 against group 11: severity 1.46, frequency 1.04.
- **Claims history predicts how often, not how much.** The log(mod) slope is 0.90 in own-damage frequency and −0.06 in its severity; in bodily injury 0.90 and −0.02.
- **Theft frequency jumps from vehicle group 16** (1.43 to 1.52 against group 11).

These relativities explain the line GLM's factors; they are not adopted as a structure.

## Cards for the actuary (round 1)

- **M1.** Structure: one Tweedie GLM per line in all three lines; in step 7, build frequency × severity in motor only.
- **M2.** Excess factor: the line factor stays (K2 (a)); C2's per-peril form is retired.
- **M3.** 2025 weather: apply it as a line-level factor rather than row by row.

## Verdicts (actuary, 2026-10-19, check-in 3 round 1)

M1 (a), M2 (a), M3 (b): the recommendations, with no notes on the page. Exchange D20.

What changes:
- **Structure.** One capped Tweedie GLM per line stays in all three lines. In step 7, each family builds frequency × severity per peril for motor only; property stays per line.
- **Excess.** The line excess factor applies everywhere, including any per-peril motor model in step 7. C2's per-peril form is retired on evidence.
- **Weather.** The 2025 weather enters as one factor per line: the exposure-weighted average of the 2025 row multipliers (motor −0.04%, buildings +0.06%, contents −0.28%). The level is unchanged; the row-by-row form of K4 is dropped. This is applied when the final pricing chain is assembled.
