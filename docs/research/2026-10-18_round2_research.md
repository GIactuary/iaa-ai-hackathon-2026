# Research and data checks for brief round 2 (2026-10-18)

Answers to the actuary's notes on the round 1 brief. Four literature reviews were run by research subagents and their citations are kept below. The data checks were run on `data/raw/` with scripts in the session scratchpad. They will be rebuilt as `src/` modules in phase 1. Where a source was read only through summaries, that is said.

## 1. Claim-level development (actuary's note on P1)

**Conclusion.** Development in this data means reporting delay, and it can be handled at row level by peril. Case estimates look unbiased in aggregate, and there is no incurred history to estimate claim-level IBNER factors. So no claim-level IBNER factor is applied. Development is checked in aggregate by peril.

Data checks:

**Reporting delay by peril.** The delay curve comes from accident year 2021 losses, all observable for at least three years. Expected unreported claims for accident year 2024, as a share of reported:

| Peril | Unreported share |
|---|---|
| Motor bodily injury | 81.5% (only 55% reported) |
| Buildings subsidence | 46.6% (68% reported) |
| Escape of water, contents | 8.1% |
| Escape of water, buildings | 7.3% |
| Motor third-party property | 7.2% |
| Every other peril | 1.6% to 5.4% |

**Backtest.** Pseudo-valuation at 31 Dec 2023, predicting accident year 2023 ultimate counts, compared with today's estimate:
- 13 of 15 perils fall within ±1.5%.
- Bodily injury is +2.2% and subsidence +3.3%. At the pseudo-valuation only 55% and 68% of those claims had been reported.

**Late claims are not larger.** Accident years 2021 and 2022, mean cost by report-delay bucket relative to the peril mean: every bucket with 50 or more claims falls between 0.80 and 1.17. Bodily injury reported after more than a year runs 1.17. Count completion is therefore a fair proxy for amount completion.

**Case estimates.** For 11 of 15 perils, the 2024 average incurred sits within ±6% of the 2021 to 2023 severity trend projected to 2024. The four outside:
- Bodily injury at 0.74. This is a level drop from accident year 2023 that shows in closed claims too: closed means of 21.5k and 22.8k for 2021 and 2022, against 17.7k and 17.3k for 2023 and 2024. It is not a reserving effect.
- Subsidence at 1.13.
- Buildings accidental damage at 1.11.
- Contents theft at 1.07.

**Rows exposed late in 2024 are less complete.** 2024 motor new-business rows, most of which start after Q1, have 34% of their expected bodily injury claims reported, against 59% for renewals. Own damage is 89% against 97%. Buildings subsidence is 45% against 72%. An accident-year factor corrects the average only, so it would understate new business.

Literature:
- **Reporting-delay models (pure IBNR from one snapshot).** Kaminsky (1987), Lawless (1994), Jewell (1989, 1990), and Verbelen, Antonio, Claeskens and Crevecoeur (2022, *Statistical Science*). The last is a daily occurrence model in which exposure times the reporting probability is the intensity.
- **Row-level exposure reduction ("take away exposure instead of adding claims").** Precedent exists in the NetSimR package's PureIBNR vignette (Parizas) and in Lewis and Rossouw (2007, "IBNR versus EBNER").
- **Micro-level reserving needs event or payment histories this data does not have.** Antonio and Plat (2014), Wüthrich (2018), Taylor, McGuire and Sullivan (2008), and Delong, Lindholm and Wüthrich (2022) all fall in this group. Censored-regression IBNER (Lopez, Milhaud and Thérond 2016, 2019) needs closure dates, and the data has none.
- **CAS Monograph 5, §5.1.** A year variable is "usually sufficient" to control development, trend and rate changes for relativities. If factors are applied, still include the year variable as a check.

## 2. Base rate and validation practice, and matching problems (actuary's note on P2)

**Conclusion.** The plan matches the standard texts. Research adds two changes: set the level as an explicit off-balance step, and compare models only at a common level.

- **Werner and Modlin, *Basic Ratemaking* (5th ed., 2016).**
  - The pure premium method projects developed and trended loss per exposure (ch. 8).
  - The trend period runs from the average accident date of the experience to the average accident date of the future policy period, with two-step trending where needed (ch. 6).
  - The off-balance factor backs the base rate out of the target average given the proposed relativities (ch. 14).
  - In multi-year classification work, apply aggregate trend and development unless rates differ materially by class (p. 162).
- **CAS Monograph 5 (Goldburd, Khare, Tevet, Guller, 2nd ed.).**
  - Out-of-time splits matter most for event-driven perils such as wind (§4.3.1).
  - Model perils separately, then combine them on the expected mix (§5.1.2).
  - Lift, double lift and the Gini index must be measured on hold-out data, and Gini says nothing about the level (§7.2).
- **Balance and calibration.**
  - A log-link Tweedie GLM and an early-stopped boosting model do not reproduce the training total (Wüthrich and Merz 2023, §5.1.5 and §7.4.2).
  - Tweedie deviance is a strictly consistent score for the mean. An error in the level is penalised directly as miscalibration (Fissler, Lorentzen and Mayer, arXiv 2202.12780).
  - Isotonic recalibration (Wüthrich and Ziegel 2024).
- **Closest matching problem: the AIcrowd Insurance Pricing Game (2020 to 2021).**
  - 60,000 motor policies with four years of history, and the task was to price year five, scored out of time.
  - The winner used claim-history features and a large-claim detector. Seven models were stacked under cross-validation with a Tweedie GLM as the meta-learner (repository davidlkl/Insurance-Pricing-Game).
- **Actuarial Loss Prediction (Kaggle).** The winner needed an explicit bias correction to get the level right.

## 3. Tweedie (actuary's note on P3)

A Tweedie distribution with power between 1 and 2 is a compound Poisson-gamma. The number of claims is Poisson, each claim is gamma, and the row's total cost has a point mass at zero plus a continuous positive part. A Tweedie GLM is a one-step pure premium model.

The power p sets how variance grows with the mean, variance = φ × mean^p:
- p = 1 behaves like a frequency model.
- p = 2 behaves like a severity model.
- p = 1.5 sits between the two.

The unit deviance at p = 1.5 reduces to d(y, μ) = 4 (√y − √μ)² / √μ. D² = 1 − total deviance of the model / total deviance of a constant-rate prediction. It works like R² for loss cost.

Worked example:

| Expected cost μ | Actual y = 0 | Actual y = 5,000 |
|---|---|---|
| 100 CU | d = 40 | d = 1,474 |
| 50 CU | d = 28 | d = 2,291 |

Halving the price saves 12 points on a row with no claim and costs 817 points if the row has a claim. Under-pricing costs more under this metric too. A prediction of zero on a row with a claim gives infinite deviance.

Quijano Xacur and Garrido (2015, *European Actuarial Journal*) show a single Tweedie GLM ties frequency and severity to move together. That fails where a factor raises frequency but lowers severity. Frequency × severity by peril avoids this. CAS practitioner material reports the two are often very similar in practice. Our P3 (c) tests both.

## 4. Weather (actuary's note on J-weather)

**Conclusion.** Weather in the year moves claim frequency for four perils. It does not move severity for any. For flood, the place matters and the year does not.

The same address is compared across years, so location is held constant. Each figure compares the top fifth with the bottom fifth of the year's deviation from the address's own 2021 to 2024 average:

| Peril | Weather measure | Frequency | Large-claim frequency (above the peril's 90th percentile) | Severity |
|---|---|---|---|---|
| Storm | Wind | 1.27 | 1.04 | 0.96 |
| Subsidence | Dry summer | 1.63 | 2.10 | 1.04 |
| Escape of water, buildings | Freezing days | 1.19 | 1.04 | 0.97 |
| Escape of water, contents | Freezing days | 1.19 | 1.04 | 0.97 |
| Motor glass and hail | Hail days | 1.66 | 1.97 | 1.04 |

Weather shows no effect on flood (year 0.99, against a place effect of 3.3 to 3.8), fire, motor own damage or third-party property.

Weather in 2025 against each address's own average:

| Measure | 2025 deviation | Expected effect |
|---|---|---|
| Wind | +0.42 (about 0.9 standard deviations) | More storm |
| Freezing days | −2.2 | Slightly less escape of water |
| Hail days | −0.46 | Slightly fewer glass claims |
| Dry summer | −0.04 | Slightly less subsidence |

Literature:
- Klawa and Ulbrich (2003, *NHESS*): storm losses scale with gust speed above a high local threshold, so the effect is non-linear.
- Scheel et al. (2013, *JRSS C*) and Haug et al. (2011, *Scandinavian Actuarial Journal*): water-damage claim frequency follows precipitation. Effect sizes were not verified.
- ABI 2022 and 2025: dry summers drive subsidence surges, mostly through claim counts.
- Freezing and burst pipes: the ABI recorded 103,000 claims in late 2010.

## 5. External data and location (actuary's note on J-ext)

**Conclusion.** The vocabulary is UK-like, but the data does not follow the real UK. Nothing joins to synthetic addresses, and real-world series would point the wrong way.

For the UK:
- The UK has 12 ITL1 regions.
- "Listed building" is UK usage.
- The A to G energy ratings match UK EPCs.
- Comparison sites sold a third of UK motor in 2013 (FCA).
- Clay subsidence and escape of water are leading UK perils.

Against using real data:
- Regional mean temperatures run from 6.6 °C to 17.3 °C, and freezing days from 3.6 to 69. That is far wider than the UK.
- Regions R04 and R05 have identical climates.
- Flood zones run 1 to 5, while England uses 1, 2, 3a and 3b.
- Real 2025 was a record UK subsidence year: GBP 307m, with the warmest summer on record. Our file's 2025 dry-summer index is slightly below normal.
- Real UK motor claims rose 13% in average cost in 2024 (ABI). Our bodily injury severity fell about 20% from 2023.

Free benchmarks that exist: ONS CPI motor parts and repair, ONS construction repair price indices, ABI headline releases and Met Office HadUK-Grid. They could serve only as a reasonableness check, and the checks above already show divergence.

## 6. Model families (actuary's note on P7)

**Conclusion.** Boosting beats a good GLM on frequency by a small absolute margin that is a large share of what the GLM explains. On severity the families tie.

Accuracy gaps in published studies:

| Study | Data | Finding |
|---|---|---|
| Wüthrich, "Yes, we CANN!" slides (2019) | French motor | GLM D² 5.1%, GLM-anchored network 6.8% |
| Henckaerts, Côté, Antonio and Verbelen (*NAAJ* 2021) | Belgian motor | Boosting lowest frequency deviance on every fold; severity "no clear winner" |
| Holvoet, Antonio and Henckaerts (arXiv 2310.12671) | Four datasets | Boosting-anchored CANN best on three of four for frequency |
| Krúpová, Rachdi and Guibert (arXiv 2503.21321) | French motor severity | Binned GLM, EBM, XGBoost and EBM with pairs all within 0.6% |

What makes a non-GLM model filing-ready: the NAIC tree-based model appendix (CASTF) asks for a PDP, ALE or Shapley plot for every variable, importance with its method, interaction rationale, lift and double lift, and support from model output to rates.

Ranking for our pool:
1. Boosting started from the GLM prediction.
2. Plain boosting.
3. EBM with pairwise terms.
4. Spline GLM as the anchor and filing baseline.

The three choices most likely to move D²:
1. Exposure entering the loss exactly as the metric uses it: an offset of log(exposure) on row cost.
2. Tweedie or frequency × severity, by peril.
3. Make-model encoding (credibility-weighted inside folds) and large-loss treatment.

## References

- Antonio and Plat (2014), *Scandinavian Actuarial Journal* 2014(7). https://dare.uva.nl/id/08cafb3d-f730-4920-a0ba-98b3a2f3b333
- CAS Monograph 5, Goldburd, Khare, Tevet, Guller. https://www.casact.org/monograph/cas-monograph-no-5
- Werner and Modlin, *Basic Ratemaking* (2016). https://www.casact.org/sites/default/files/old/studynotes_werner_modlin_ratemaking.pdf
- Verbelen et al. (2022). https://arxiv.org/abs/1909.08336
- NetSimR PureIBNR vignette. https://cran.uni-muenster.de/web/packages/NetSimR/vignettes/PureIBNR.html
- Wüthrich and Merz (2023), *Statistical Foundations of Actuarial Learning*. https://link.springer.com/book/10.1007/978-3-031-12409-9
- Fissler, Lorentzen and Mayer. https://arxiv.org/abs/2202.12780
- Wüthrich and Ziegel (2024). https://arxiv.org/abs/2301.02692
- AIcrowd Insurance Pricing Game, winner. https://github.com/davidlkl/Insurance-Pricing-Game
- Henckaerts et al. (2021). https://arxiv.org/abs/1904.10890
- Holvoet, Antonio and Henckaerts. https://arxiv.org/abs/2310.12671
- Krúpová, Rachdi and Guibert. https://arxiv.org/html/2503.21321v2
- Wüthrich, "Yes, we CANN!" (2019). https://www.zhaw.ch/storage/engineering/institute-zentren/iamp/upload/AI_conference_2019/Wuthrich_2019.pdf
- NAIC CASTF tree-based model appendix. https://content.naic.org/sites/default/files/inline-files/CASTF%20Tree-based%20Model%20Appendix%20(B-Trees).pdf
- Quijano Xacur and Garrido (2015). https://totweedieornot.sourceforge.io/
- Klawa and Ulbrich (2003). https://nhess.copernicus.org/articles/3/725/2003/
- Scheel et al. (2013). https://pmc.ncbi.nlm.nih.gov/articles/PMC3563012/
- ABI 2022 subsidence release. https://www.abi.org.uk/news/news-articles/2023/3/sinking-uk--last-summers-record-breaking-heatwave-leads-to-surge-in-insurance-payouts-for-subsidence/
- ABI 2024 motor release. https://www.abi.org.uk/news/news-articles/2025/2/motor-claims-hit-record-11.7-billion-in-2024/
- Met Office, summer 2025. https://www.metoffice.gov.uk/about-us/news-and-media/media-centre/weather-and-climate-news/2025/summer-2025-is-the-warmest-on-record-for-the-uk
- ONS geography (ITL1). https://www.ons.gov.uk/methodology/geography/ukgeographies/eurostat
