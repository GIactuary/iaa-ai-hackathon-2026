# Competition overview (notes)

Read from the Kaggle competition pages on 2026-10-16, about 3 hours after Part 1 opened. Source: https://www.kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-1 (Overview tab). These are notes, not a copy; check the page for changes.

## The task

- Setting: the pricing team of an established insurer in developed markets, reviewing its technical pricing across three lines: private motor, home buildings, home contents.
- Data: synthetic, four policy years (2021 to 2024), one developed market.
- Predict, for each 2025 policy row, the expected ultimate claim cost: all claims with a loss date in the row's period, net of the policy excess and net of recoveries, excluding declined (repudiated) claims. The page calls this "the risk premium, not the premium charged".
- The claim model need not use AI; the workflow must show AI working alongside actuaries.

## Two parts

| Part | Window | Content |
|---|---|---|
| Part 1 (this page) | Opened 2026-10-16; the page showed "7 days to go" | Build the pricing approach on the developed-market data |
| Part 2 (separate Kaggle page) | Opens when Part 1 closes; link given at the end of week 1 | Expansion into a new (emerging) market with different characteristics and data problems. Judged on how we identify what changed, respond, and communicate remaining uncertainty |

## Evaluation (overall, decided after Part 2)

| Category | Weight | What it covers |
|---|---|---|
| Pricing performance and adaptation | 40% | Accuracy on both weeks' data (Part 1 and Part 2 private leaderboards, equally weighted, per rules 2.4.c), modelling choices, adaptation to the new market, treatment of data quality, limitations and uncertainty |
| Effective use of AI | 25% | AI through the workflow (EDA, features, modelling, diagnostics, testing); shown improvement in quality, efficiency, challenge or reliability; rigorous verification of AI output |
| Trust, validation and guardrails | 20% | Validation checks, automated tests, model diagnostics; governance controls, reproducibility, human review and escalation protocols |
| Communication and documentation | 15% | A clear, concise video demo and written report on method and key decisions; a well-structured code repository with version control |

AI-assisted review tools screen the videos, documents and code first; the judging panel makes the final assessment. The Part 1 leaderboard alone does not decide prizes.

## Metric

Tweedie D² score with power 1.5 (higher is better): the share of Tweedie deviance explained relative to a constant-rate baseline. Scored on actual ultimate claim cost as defined above.

## Submission file

One CSV covering every row of `test_motor`, `test_buildings` and `test_contents`, header `row_id,prediction`, prediction in CU, zero or above. See `sample_submission.parquet`. Ten submissions a day (the rules first said five; the host confirmed ten on the forum); one final submission may be selected (otherwise the best public score is used).

## Final deliverables (end of Part 2, through a form to be published)

- Team name and members
- Week 1 and week 2 prediction files (as well as the Kaggle submissions)
- A short video demo: approach, results, AI-assisted workflow, and how the solution changed for the new market. Only the first 5 minutes are graded
- A concise write-up: modelling approach, major decisions, use of AI, assumptions, limitations, adaptation to the new market, guardrails. Only the first 5 pages are graded
- The code (scripts and/or notebooks) under version control, so the development can be understood and reproduced

## Prizes

CAD 500 / 100 / 50 for the top three after Part 2, for members or students of a Full or Associate Member Association of the IAA.

## Not yet known

- Exact Part 1 and Part 2 deadlines and the team merger deadline: the rules (section 2.3) still show "[DATE, TIME UTC]" placeholders.
- The share of test rows on the public leaderboard.
