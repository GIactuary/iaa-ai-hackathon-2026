# Phase 5 check-in 1: cap and load, trend and level, weather (2026-10-19)

Runs: `scripts/phase5_steps123.py` (first run, `outputs/phase5_steps123.json`) and `scripts/phase5_steps123_run2.py` (second run, `outputs/phase5_steps123_run2.json`). Every test is out of time on non-hold-out customers. Rankings are compared at a common level per line. The keep rule requires lower deviance in both scorings and a 95% customer-bootstrap interval above zero on scoring A. The decisions are register items K1 to K4.

## Conclusion first

- **Capping with a line excess factor (V1) is the new base.** It passes overall and clearly in motor.
- **My first form of the per-peril load failed.** It used separate large-loss and other GLMs (V2).
- **Climate normals fail as rating factors**, even once each row's own year is left out.
- **The weather-year multiplier does nothing for ranking.** But it brings the dry 2024 level closer to actual, and for 2025 it nets to almost nothing.
- **Trend improves the level in five of six line-scorings.** The exception is motor on 2023, hit by the bodily injury drop that could not be seen from 2021 to 2022.
- **One C4 stop.** The motor indication sits 5.3% below the model's own 2025 projection.

## Step 1: cap and load

| Test | Combined, scoring A: D² and deviance saved [95% interval] | Combined, scoring B | Keep |
|---|---|---|---|
| V1 (capped, line factor) against V0 (uncapped) | 9.97% → 10.02%, +32,825 [6,688, 58,429] | 9.41% → 9.50%, +53,044 | Yes |
| V2 (two capped GLMs, group factors) against V1 | 10.02% → 9.93%, −54,597 [−99,284, −11,519] | 9.50% → 9.45%, −28,880 | No |

By line, V1 against V0:

| Line | Saved on A [95% interval] | Saved on B | Keep |
|---|---|---|---|
| Motor | +32,148 [8,842, 55,740] | +36,365 | Yes |
| Buildings | +607 [−11,079, 11,454] | +15,427 | No (interval includes zero) |
| Contents | +70 [−2,765, 2,864] | +1,252 | No (interval includes zero) |

Excess factors on 2021 to 2024: motor 1.031, buildings 1.022, contents 1.018.

**How to read this.** Capping gives the models a steadier target. The factor puts the capped-off cost back in proportion to each row's predicted cost. In motor this ranks clearly better. In property it ranks slightly better in both years, but within noise.

## Step 2: trend and level

Selected trends (2021 to 2024, per year; frequency on completed exposure, severity on capped claims):

| Line | Peril | Frequency | Severity |
|---|---|---|---|
| Motor | Bodily injury | −1.7% | step: level 16,402 from 2023 |
| Motor | Third-party property | −2.5% | +8.0% |
| Motor | Own damage | −2.9% | +5.4% |
| Motor | Glass and hail | +2.9% | +8.2% |
| Motor | Theft | −1.3% | +4.1% |
| Buildings | Escape of water | +0.5% | +8.7% |
| Buildings | Flood | −4.4% | +7.4% |
| Buildings | Storm | +2.1% | +8.4% |
| Buildings | Fire | +4.7% | −3.9% |
| Buildings | Subsidence | +12.0% (fitted 13.4%, held at the bound) | +5.4% |
| Buildings | Accidental damage | +3.1% | +1.0% |
| Contents | Theft | +3.2% | +4.6% |
| Contents | Escape of water | +2.1% | +2.6% |
| Contents | Flood | −1.3% | +3.9% |
| Contents | Accidental damage | +1.2% | +2.2% |

Line loss-cost trend: motor +2.5%, buildings +7.6%, contents +5.6% a year.

Level ratios (prediction × completion over actual):

| Scoring | Line | No trend | Model's own trend | Indication |
|---|---|---|---|---|
| A (2023) | Motor | 1.054 | 1.107 | 1.108 |
| A (2023) | Buildings | 0.958 | 1.003 | 0.992 |
| A (2023) | Contents | 0.927 | 0.947 | 0.937 |
| B (2024) | Motor | 0.929 | 0.954 | 0.944 |
| B (2024) | Buildings | 0.919 | 0.974 | 0.974 |
| B (2024) | Contents | 0.980 | 1.033 | 0.995 |

The 2025 indication, with the C4 check (gap of the indication to the model's own projection):

| Line | Indicated loss cost per policy-year (capped × excess factor) | C4 gap | Against the 2025 premium rate |
|---|---|---|---|
| Motor | 571.7 | −5.3%: outside ±3%, stops for the actuary (K3) | 596, so about 0.96 |
| Buildings | 506.0 | −0.0% | 368, so about 1.37 |
| Contents | 129.1 | −2.1% | 154, so about 0.84 |

## Step 3: weather

| Test | Combined, scoring A [95% interval] | Combined, scoring B | Keep |
|---|---|---|---|
| Normals including own year (first run) against V2 | −125,119 [−194,096, −57,576] | −61,049 | No |
| V5: normals leaving out each row's own year, against V1 | −67,365 [−127,048, −7,144] | −35,164 | No |
| V7: V1 × year-weather multiplier, against V1 (ranking) | +367 [−7,273, 8,250] | −5,747 | No |

The year-weather multiplier, judged on level instead:
- **Scoring A (2023):** no change, since 2023 sat close to its normal.
- **Scoring B (2024):** buildings 0.974 → 0.991, contents 0.995 → 0.998, motor 0.944 → 0.946.

Elasticities (2021 to 2024): storm 0.12 per unit of wind index; subsidence 1.36 per unit of dry-summer index; escape of water 0.008 per freezing day; glass 0.058 per hail day.

For 2025:
- storm +5.6%, subsidence −3.4%, escape of water −1.2%, glass −1.2%
- net by line: motor −0.05%, buildings −0.08%, contents −0.24%

## What the AI got wrong in these steps (exchange D15)

1. **The first form of the per-peril load (C2) failed its test.** Separate large-loss GLMs have too few claims to rank with.
2. **Climate normals that include the row's own year let that year's actual weather in.** The damage was ten times larger on A, with two-year normals, than on B, with three-year normals. Rebuilt, the normals still failed, so this was not the only reason.
3. **The subsidence elasticity was stuck at exactly zero, because its deviation was not standardised.** The bug was found from an implausible number, fixed, and covered by a test.
4. **The weather multiplier was first judged on ranking, which by design cannot see a level effect.** It was re-judged on level.

## Round 2 (2026-10-19): the actuary's questions on K3 and K4

**K3. The motor gap was a bug, now fixed.**
- Line completion weighted every year by the 2021-2022 peril mix. Bodily injury, the slowest peril, fell from 42% to 33% of motor cost after its 2023 step, so 2024 motor completion was understated by 4.8% (0.785 against 0.823).
- With each year's own mix, estimated without the hold-out:
  - The motor C4 gap falls from −5.3% to −0.7%.
  - Buildings moves to −1.5% and contents to −2.0%.
  - On scoring B, motor's level moves from 0.944 to 0.991, and all lines from 0.961 to 0.980.
- The indication (554.5), a plain projection of 2024 (553.6) and the model's projection (558.2) now agree. The AI's round 1 recommendation (the model projection) is withdrawn.
- Run: `scripts/phase5_k3k4.py`. Before the fix: `outputs/phase5_k3k4_before_fix.json`; after: `outputs/phase5_k3k4.json`.

**K4. The actuary's weather-normalised experience method.**

Trends change where weather drifted:

| Peril | Frequency trend, raw | Weather-normalised |
|---|---|---|
| Subsidence | +13.4% | +6.5% |
| Storm | +2.1% | +3.6% |
| Glass | +2.9% | +0.6% |
| Escape of water | +0.5% | +1.3% |

Level error:

| | Scoring A (2023) | Scoring B (2024) |
|---|---|---|
| All lines, raw → normalised | 5.5% → 6.3% | 2.0% → 0.3% |
| Buildings, raw → normalised | 0.8% → 2.1% | 4.0% → 0.1% |

- Normalising the GLM's relativities adds nothing (−2,381 on A, interval −17,207 to 13,162), so the method applies to level and trends only.
- 2025 under the method:
  - Motor 552.3, against 554.5 raw.
  - Buildings 487.7, against 495.1 raw.
  - Contents 126.8, unchanged.
