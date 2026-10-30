# Video script: five minutes, narrated by the actuary

*For the actuary to narrate (VID (a)), written in his voice: what we did and what we found, step by step, as he would explain it to a board. The graded video is the first 5 minutes: Part 1 takes 0:00 to 2:30, Part 2 2:30 to 5:00. Narration runs at about 150 words a minute, about 375 words per half. One slide per scene, in `docs/video/slides.html` (open it in a browser, press F11, arrow keys to move); each slide carries a caption that reads its exhibit with one real number. Every number is in the write-up (`docs/writeup/writeup.md`) with its source. Suggested recording: one take per half, screen and voice, then join.*

## Part 1 (0:00 to 2:30)

| Scene | Time | Slide | Caption |
|---|---|---|---|
| 1. The question | 0:00 to 0:20 | Loss ratio by line | Buildings paid out 1.29 to 1.36 of its premium every year from 2021 to 2024 |
| 2. Building the target | 0:20 to 0:45 | How the target was built | Exposure scaled by expected reporting, IBNR for one policy; written rules repair the data's errors; no record excluded |
| 3. The keep rule | 0:45 to 1:10 | Exhibit 1, the ladder | Claims history moved ranking D² from 10.02% to 10.34% on 2023; a step stays only if its interval is above zero |
| 4. The price | 1:10 to 1:40 | The pricing chain | 500 × 1.058 × 1.009 × 0.9996 = 533.5 CU; buildings losses 1.43 times the current premium |
| 5. The sealed envelope | 1:40 to 2:05 | Exhibit 2, hold-out | Across lines +41,250 on 2023 [2,450, 103,420]; one 11.9M claim is 43.8% of the hold-out's 2023 motor cost |
| 6. The result | 2:05 to 2:30 | Part 1 result and the D16 card | Private 0.13641, 15th; first 0.13923. Asked to explain its recommendation, the AI found its own bug |

**1. The question.** "Hi, I'm Bavan Amirthalingam, an actuary. In Part 1 we were the pricing team of an insurer that writes motor, home buildings and home contents. Buildings had lost money every single year: claims ran at 1.29 to 1.36 times the premium. Our job was to predict the expected claims cost of every 2025 policy."

**2. Building the target.** "First I got the data and checked it, and two things needed fixing before any model. Claims were valued at the end of 2024, so the latest year wasn't fully reported. I scaled each policy's exposure by how much of its claims we expect to see by then: that's IBNR, done policy by policy. And the data had errors, like sums insured a thousand times too large. Each fix is a written rule, and no record was thrown away."

**3. The keep rule.** "Then I set one test for every idea. Fit on earlier years and predict the next year, the same way we'd set a 2025 price. An idea stayed only if it won in both test years, and still won when we resampled customers 2,000 times. We tried 67 features and kept three. The biggest gain came from each customer's claims history, weighted by credibility."

**4. The price.** "For each line we blended different types of model, and set the overall rate level separately, from experience adjusted for weather. That level method was mine. The AI had recommended a different one, and when I asked it to explain, it found a bug in its own code. So a motor rate of 500 currency units becomes 533.50. And buildings claims are expected to run 1.43 times what the book charges today."

**5. The sealed envelope.** "Before any modelling, we locked away 15 percent of customers, like a sealed envelope, and opened it once, after the model was frozen. Our final blend beat our working GLM in every line and year. Motor 2023 looks noisy for one reason: a single 11.9 million injury claim. We checked it and kept it as a real loss."

**6. The result.** "Part 1 finished fifteenth on the private leaderboard, at 0.136 against 0.139 for the winner. For Part 2, I reused all of it: the same code, the same test, and the same way of making decisions."

## Part 2 (2:30 to 5:00)

| Scene | Time | Slide | Caption |
|---|---|---|---|
| 7. A new market | 2:30 to 2:55 | What changed | Two years of data; the 2025 book is 76% new business against 14% in training |
| 8. Borrow the shape, learn the level | 2:55 to 3:20 | Exhibit 3, what passed | Part 1's relativities beat an emerging-only GLM in every line; the emerging data adds only what passes |
| 9. The hold-out decision | 3:20 to 3:50 | Hold-out by line (chart) | Motor holds; buildings loses in three of four readings, so it goes back to Part 1's shape |
| 10. Reviews that found our inconsistency | 3:50 to 4:20 | Motor by peril | Rejected in round 1 on a pooled rule; on the line rule it passes: +13,443 [4,917, 23,084] |
| 11. The price of one policy | 4:20 to 4:45 | Build-up for B00229821 (chart) | 359.0 risk + 43.1 cat + 85 fixed = 487.1, ÷ 0.49 = 994.0 CU, against 285.6 charged today |
| 12. What the AI got wrong | 4:45 to 5:00 | Exhibit 5 | 52 logged exchanges; the AI's errors were caught by tests, reviews and questions |

**7. A new market (2:30 to 2:55).** "Part 2 moved us into an emerging market, with thinner data: two years instead of four, ages only in bands, no credit score or telematics. Gender is there, and this market allows it. The big change is the 2025 book: 76 percent is new business from a broker portfolio the insurer just bought. Our training data was 86 percent renewals."
*About 60 words.*

**8. Borrow the shape, learn the level (2:55 to 3:20).** "Two years of data can't carry a full rating plan. Part 1's relativities predicted better than a model fitted only on emerging data, in every line. So I used credibility on the whole plan: Part 1 gives the shape, the emerging data sets the level, and it adds a factor only if it passes our test. What passed: gender in motor, buildings by peril, Part 1's tree in contents, and each year's own weather."
*About 70 words.*

**9. The hold-out decision (3:20 to 3:50).** "We locked away 15 percent of customers, with the rule for opening them written down first. Overall, the model fell just short. Line by line, motor held up, but buildings lost in three of four readings, and buildings here is driven by just twelve large fires. So I took buildings back to Part 1's shape, and we report that as decided after the envelope was opened."
*About 70 words.*

**10. Reviews that found our inconsistency (3:50 to 4:20).** "Then four separate AI reviewers, each starting fresh, looked for what we had missed. Three found the same thing. In round one we had rejected pricing motor by peril, because we tested all lines together and buildings drowned it out. Tested line by line, as I had done for buildings, it passes, and it holds on the locked-away customers. So I adopted it. Our final file scores 0.157 on the public board."
*About 70 words.*

**11. The price of one policy (4:20 to 4:45).** "The hosts asked us to build the full premium for twenty policies, using their expense figures. Here's one buildings policy. Expected claims are 359. Add 43 for catastrophe reinsurance and 85 of fixed expense. The host's commission, expenses, tax and profit take 51 percent, so we divide by 0.49 and get 994. Today it's charged 286. All seven buildings policies cost more in claims alone than they're charged."
*About 70 words.*

**12. What the AI got wrong (4:45 to 5:00).** "Every decision is logged: 52 exchanges, including the AI's mistakes, like a bug in its own code and a wrong diagnosis my question exposed. The AI built and tested. The judgments were mine."
*About 30 words.*

## Production notes

- Narration is 363 words in Part 1 and 373 in Part 2, about 2 minutes 25 to 30 seconds each at 150 words a minute. Keep Part 2 under 2:30 so scene 12 falls inside the graded 5 minutes.
- Scene 10 gives the public score only; the private result is unknown at recording.
- The slides regenerate from the evidence files with `scripts/exhibits_part2.py` then `scripts/build_slides.py`.
