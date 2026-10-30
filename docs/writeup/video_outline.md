# Video outline: Part 1 half (2 minutes 30 seconds)

*Draft for the actuary, 2026-10-20. The graded video is the first 5 minutes; Part 1 takes 0:00 to 2:30 and Part 2 takes 2:30 to 5:00. Narration is written at about 150 words a minute (about 375 words for this half) for the actuary to say in his own words. One exhibit per scene; each carries an on-screen caption that reads it with one real number. Numbers are those in `docs/writeup/part1.md`, with the same sources.*

| Scene | Time | Exhibit on screen | Caption (how to read it) |
|---|---|---|---|
| 1. The question | 0:00 to 0:20 | Loss ratio by line and year, 2021 to 2024 | Buildings paid out 1.29 to 1.36 of its earned premium every year |
| 2. Building the target | 0:20 to 0:45 | Motor loss cost per exposure by year, before and after the completion adjustment | 2024 motor rises from 454 to 578 per exposure once unreported claims are allowed for, in line with earlier years |
| 3. The keep rule | 0:45 to 1:10 | Exhibit 1 as a bar ladder: ranking D² on 2023 for each kept step | Claims history moved 10.02% to 10.34%; a step stays only if it wins on 2023 and 2024 and its interval is above zero |
| 4. The price | 1:10 to 1:40 | Exhibit 2 as a chain: GLM rate × excess factor × off-balance × 2025 weather, and expected cost over current premium | A 500 CU motor rate becomes 500 × 1.058 × 1.009 × 0.9996 = 533.5 CU; buildings losses are 1.43 times the current premium |
| 5. The sealed envelope | 1:40 to 2:05 | Exhibit 3: blend against the working GLM on the hold-out, by line and year | Across lines +41,250 on 2023 [2,450, 103,420]; one 11.9M claim is 43.8% of the hold-out's 2023 motor cost |
| 6. Where the AI was challenged | 2:05 to 2:30 | The D16 exchange card, then Exhibit 4's list of AI errors | Asked to explain its motor recommendation, the AI found its own bug: 2024 motor completion understated by 4.8% |

## Scene by scene

**1. The question (0:00 to 0:20).** Open on the loss ratio table: motor about 1.0, buildings 1.29 to 1.36, contents about 0.84.
Narration: "We are the pricing team of an insurer whose buildings book has lost money every year for four years, while its 2025 rates rose under 6%. Our job: the expected cost of every 2025 policy, built so it can be filed, explained and moved to a new market."
*Source: `competition/brief.md` section 6.*

**2. Building the target (0:20 to 0:45).** The completion chart for motor: 2024 before and after.
Narration: "The answer we are scored on is not in the data. Claims are valued at the end of 2024, so recent years are short. Each row's exposure is scaled by the share of its claims we expect to have been reported: IBNR at the level of one policy. Nine data rules repair the planted errors, and no record was thrown away."
*Source: `ai/ai_log.json` D12; `decisions/data_rules.md`.*

**3. The keep rule (0:45 to 1:10).** The ladder of Exhibit 1, bars rising from the current premium (8.01%) to the blend (10.67%).
Narration: "Every idea was tested the way a 2025 price is set: fit on earlier years, score the next one. It stayed only if it won in both test years and the win survived 2,000 resamples of customers. Sixty-seven features were tried; three stayed. Claims history, weighted by credibility, gave the largest step."
*Source: Exhibit 1.*

**4. The price (1:10 to 1:40).** The pricing chain for one motor row, then the last row of Exhibit 2.
Narration: "Each line's price is a weighted blend of model families: GLMs, an explainable additive model, and a small boosted correction to the GLM. The level is set separately, from weather-normalised experience, the method the actuary proposed. A 500 CU motor rate becomes 533.5. And the answer to the opening question: expected buildings losses are 1.43 times what the book charges."
*Source: Exhibit 2.*

**5. The sealed envelope (1:40 to 2:05).** Exhibit 3, then a highlight on motor 2023.
Narration: "Fifteen percent of customers were locked away before any model was fitted and opened once, after the freeze. On them the blend beat our working GLM in every line and year. Motor 2023 looks odd for one reason: a single 11.9 million injury claim is 44% of that year's cost. We kept it as a real loss and showed its effect."
*Source: Exhibit 3.*

**6. Where the AI was challenged (2:05 to 2:30).** The D16 card full screen, then the five AI-error rows of Exhibit 4.
Narration: "AI proposed and tested; the actuary made every decision, and 29 exchanges are logged. When he asked the AI to explain a recommendation, it found a bug in its own code and withdrew it. Eleven exchanges record AI errors, each caught by a check before the final file. The same pipeline now runs on Part 2."
*Source: `ai/ai_log.json` D16; Exhibit 4.*

## Production notes

- Scenes 1 and 2 need two small charts that the write-up does not carry (loss ratio by year; motor completion before and after). Both numbers are already in the brief and the log; the charts should be built from `outputs/` or a script in the repo so they are reproducible.
- Keep the leaderboard score to one mention at most (0.12908, 5th at the time of writing); the private leaderboard decides the accuracy grade.
- The narration says "the actuary" in the third person; if the actuary narrates, change it to "I".
