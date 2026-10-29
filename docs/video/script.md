# Video script: five minutes, narrated by the appointed actuary

*Draft for the appointed actuary (VID (a): he narrates in his own words over the exhibits). The graded video is the first 5 minutes: Part 1 takes 0:00 to 2:30, Part 2 2:30 to 5:00. Narration runs at about 150 words a minute, about 375 words per half. One slide per scene, in `docs/video/slides.html` (open it in a browser, press F11, arrow keys to move); each slide carries a caption that reads its exhibit with one real number. Every number is in the write-up (`docs/writeup/writeup.md`) with its source. Suggested recording: one take per half, screen and voice, then join.*

## Part 1 (0:00 to 2:30)

| Scene | Time | Slide | Caption |
|---|---|---|---|
| 1. The question | 0:00 to 0:20 | Loss ratio by line | Buildings paid out 1.29 to 1.36 of its premium every year from 2021 to 2024 |
| 2. Building the target | 0:20 to 0:45 | How the target was built | Nine data rules; claims capped; exposure scaled by expected reporting, IBNR at row level |
| 3. The keep rule | 0:45 to 1:10 | Exhibit 1, the ladder | Claims history moved ranking D² from 10.02% to 10.34% on 2023; a step stays only if its interval is above zero |
| 4. The price | 1:10 to 1:40 | The pricing chain | 500 × 1.058 × 1.009 × 0.9996 = 533.5 CU; buildings losses 1.43 times the current premium |
| 5. The sealed envelope | 1:40 to 2:05 | Exhibit 2, hold-out | Across lines +41,250 on 2023 [2,450, 103,420]; one 11.9M claim is 43.8% of the hold-out's 2023 motor cost |
| 6. The result | 2:05 to 2:30 | Part 1 result and the D16 card | Private 0.13641, 15th; first 0.13923. Asked to explain its recommendation, the AI found its own bug |

**1. The question.** "I'm the appointed actuary for this entry. Part 1 put us in the pricing team of an insurer whose buildings book had lost money every year for four years. The job: the expected cost of every 2025 policy, built so it can be filed, explained, and moved to a new market."

**2. Building the target.** "The answer we are scored on isn't in the data. Claims are valued at the end of 2024, so recent years are short. Each row's exposure is scaled by the share of its claims we expect to have been reported: IBNR at the level of one policy. Nine stated rules repair the planted errors, and no record was thrown away."

**3. The keep rule.** "Every idea was tested the way a 2025 price is set: fit on earlier years, score the next one. It stayed only if it won in both test years and the win survived 2,000 resamples of customers. Sixty-seven features were tried; three stayed. Claims history, weighted by credibility, gave the largest step."

**4. The price.** "Each line is a blend of model families. The level is set separately, from weather-normalised experience, a method I proposed when the AI's own recommendation turned out to rest on a bug. A 500 CU motor rate becomes 533.5. And buildings losses are expected to run 1.43 times what the book charges."

**5. The sealed envelope.** "Fifteen percent of customers were locked away before any model was fitted and opened once, after the freeze. The blend beat our working GLM in every line and year. Motor 2023 is noisy for one reason: a single 11.9 million injury claim, which we kept as a real loss."

**6. The result.** "Part 1 finished fifteenth of the private leaderboard, at 0.136 against 0.139 for the winner. The same code, the same rule and the same way of deciding then went to Part 2."

## Part 2 (2:30 to 5:00)

| Scene | Time | Slide | Caption |
|---|---|---|---|
| 7. A new market | 2:30 to 2:55 | What changed | Two years of data; the 2025 book is 76% new business against 14% in training |
| 8. Borrow the shape, learn the level | 2:55 to 3:25 | Exhibit 3, what passed | Fitted on 2023, scored on 2024: Part 1's buildings relativities D² 11.5%, an emerging GLM −144% |
| 9. The hold-out decision | 3:25 to 3:55 | Hold-out by line (chart) | Motor and contents hold; buildings loses in three of four readings, so it goes back to Part 1's GLM |
| 10. Reviews that found our inconsistency | 3:55 to 4:20 | Motor by peril | Rejected in round 1 on a pooled rule; on the line rule it passes: +13,443 [4,917, 23,084] |
| 11. The price of one policy | 4:20 to 4:45 | Build-up for B00229821 (chart) | 370.1 risk + 44.4 cat + 85 fixed = 499.6, ÷ 0.49 = 1,019.5 CU, against 285.6 charged today |
| 12. What the AI got wrong | 4:45 to 5:00 | Exhibit 5 | 48 logged exchanges; the AI's errors were caught by tests, reviews and questions |

**7. A new market (2:30 to 2:55).** "Part 2 is an emerging market: two years of data, a book a twentieth the size, ages in bands, no credit score or telematics, and gender, which this market allows. The hard part is the 2025 book: 76 percent of it is new business from a broker portfolio the insurer has just bought. Our training data is 86 percent renewals."
*About 60 words.*

**8. Borrow the shape, learn the level (2:55 to 3:25).** "Two years can't carry a full rating plan. In buildings, a GLM fitted on one emerging year scored minus 144 percent on the next; Part 1's relativities scored plus 11.5. So we used credibility at the level of the whole plan: Part 1's shape is the complement, and the emerging data sets the level and adds only what it proves. Of more than a dozen changes tested, the ones that passed were gender in motor, own-year weather, fire and flood with address hazards, and the insured's premium as our fields predict it."
*About 90 words.*

**9. The hold-out decision (3:25 to 3:55).** "We locked away 15 percent of customers and opened them once, on a rule written down first. Pooled, the model fell just short. By line, motor and contents held; buildings lost in three of four readings. Buildings here is a line of twelve fires. I kept motor and contents and took buildings back to the plain Part 1 GLM, and we report that as a decision made after the hold-out was opened."
*About 75 words.*

**10. Reviews that found our inconsistency (3:55 to 4:20).** "Then I had four independent AI reviews look for what we missed. Three found the same thing on their own: in round one we rejected pricing motor by peril on a pooled test, where buildings swamped it. On the line-by-line rule I had used for buildings, it passes, and it holds on the hold-out customers. Two reviews also found we'd used a claim-count credibility standard on a cost ratio. I adopted both."
*About 75 words.*

**11. The price of one policy (4:20 to 4:45).** "Here's one reference buildings policy. Expected claims cost 370 a year, plus 44 for catastrophe reinsurance and 85 of fixed expense. Divide by the 49 percent left after commission, expenses, tax and profit, and the office premium is about 1,020. Today it's charged 286. Every buildings reference policy costs more in claims alone than it's charged."
*About 60 words.*

**12. What the AI got wrong (4:45 to 5:00).** "Forty-eight exchanges are logged, with the AI's mistakes in plain view: a bug it found in its own code, a wrong diagnosis my question exposed, a threshold its own evidence contradicted. AI built and tested. The judgments were mine."
*About 45 words.*

## Production notes

- Total Part 2 narration is about 405 words: trim scene 8 if a take runs long.
- The Part 2 public and private scores are not in the script, since the private result is unknown at recording. If the actuary wants one number, use the final file's public score once it is known.
- If K17 or K18 changes the final file, scenes 10 and 11 change; the exhibits and slides regenerate from `scripts/exhibits_part2.py` and `scripts/build_slides.py`.
