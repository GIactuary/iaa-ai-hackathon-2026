# Part 2 (emerging market): notes from the Kaggle pages

Read on 24 October 2026 from kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-2 (Overview, Data, Rules, Discussion, Leaderboard, Code, Team, Submissions) and the Part 1 Discussion tab. These are notes in our own words, not a copy of the pages.

## Dates

- Part 2 opened Friday 23 October 2026, 12:00 EDT, when Part 1 closed.
- Part 2 closes **Friday 30 October 2026, 23:59:59 EDT** (from the page's timestamp).
- The final submission form (video, write-up, code, both prediction files) is still "link to be provided".

## Task

- Predict the 2025 expected ultimate claim cost (risk premium, net of excess and recoveries, including IBNR and development, excluding declined claims) for every row of `emerging/test_motor`, `emerging/test_buildings` and `emerging/test_contents`. The prediction is a total for the row, not an annual rate.
- One CSV, header `row_id,prediction`, predictions zero or above. The Part 2 leaderboard takes emerging-market rows only.
- Metric unchanged: Tweedie D2, power 1.5.

## The new market

- Two training years (2023 and 2024), regions E01 to E06, claims valued at 31 December 2024 as in Part 1. The page lists weather for 2023 and 2024; the downloaded weather file also holds 2025 (as the developed one does).
- At the start of 2025 the insurer takes on a large broker portfolio, so most 2025 test rows are new business. The training data is the insurer's own book; the test book is mostly a different source.
- The hosts say the customers, climate, regulation and costs differ, and there is less data.
- The `developed/` folder holds the Part 1 data for reference. Ids are unique only within a market: never join across markets. Test row_ids are unique across both.

## Column differences (emerging against developed)

| Area | Developed | Emerging |
|---|---|---|
| Customer age | date of birth; driver_age, policyholder_age | age bands only: 17-24, 25-34, 35-49, 50-64, 65+ |
| Gender | not present | present (M/F); the page says it may be used for pricing in this market |
| Credit band | present | not present |
| Occupation | coded | free text as typed by the customer |
| Motor mileage | annual_mileage | not collected |
| Telematics | opt-in and score | not offered |
| Construction type | brick, stone, timber, non-standard | local codes BRK, STN, WOD, MIX |

## New files in `emerging/`

- `expenses`: fixed expense per policy, variable expense, commission by channel, premium tax, catastrophe reinsurance (as a share of property risk premium) and target profit margin, for both markets.
- `reference_policies`: 20 test policies. The written report must show the office premium build-up from our risk premium for each. Judged by the panel, not scored on Kaggle.
- `sample_submission`: the page says CSV, but the downloaded file is `emerging/sample_submission.parquet`. The submission we upload is still a CSV.

Download: 24 files, 224,722,689 bytes in all (both folders), all parquet. Licence CC BY-NC 4.0. Downloaded 24 October 2026 to `data/raw_part2/` (archive kept beside the unpacked `developed/` and `emerging/` folders); sizes and SHA-256 hashes in `competition/data_manifest_part2.json`. All 11 `developed/` files are byte-identical to the Part 1 files in `competition/data_manifest.json`.

## Scoring and deliverables (both parts)

- Pricing performance and adaptation 40%: Part 1 and Part 2 private leaderboards weighted equally, plus the panel's view of modelling choices, adaptation and treatment of data quality and uncertainty. Effective use of AI 25%. Trust, validation and guardrails 20%. Communication and documentation 15%.
- Final submission form (link to come): team and member names, both prediction files, a video (first 5 minutes graded), a write-up (first 5 pages graded), and version-controlled code covering both parts.
- Limits: 10 submissions a day, one final submission selected per part.

## Rules points that matter to us

- The rules page shown on the Part 2 site is still the Part 1 text (title, website and timeline placeholders). Nothing in it changes external data, AI use or team rules from Part 1.
- AI tools, including hosted LLMs, may process the data (2.5.a). External data must be public and equally accessible, or of minimal cost (2.7).
- Teams must be the same in both parts.

## Leaderboard and our entry (24 October 2026)

- Part 1 closed with our final file at 0.13641 private, 15th; first place 0.13923. Places 5 to 20 sit within 0.0019 (`docs/HANDOFF.md`).

- Public leaderboard uses about 33% of the test rows; the private 67% decides.
- 8 teams on the board after one day. Top public score 0.16446, eighth 0.13631. The host's average-cost baseline scores 0.07319.
- Our team on Part 2: "Bavan Amir", one member (team leader), no submissions yet. Code tab: no public notebooks.

## Host answers on the forums

Part 2 Discussion:

- **Submission row count (fixed).** The scorer first expected 626,639 rows (the Part 1 test size). The host says it is fixed. A Part 2 file has 209,063 rows, one per row of `emerging/sample_submission.parquet`, which matches the three emerging test files.
- **Download All (fixed).** The host removed a stray duplicate of the developed data from the dataset; Download All works now. Our archive (24 files) holds only `developed/` and `emerging/`.

Part 1 Discussion (answers that carry into Part 2):

- **Recoveries.** `paid_to_date` and `incurred_to_date` are net of excess and before recoveries. Recoveries are booked only when a claim settles, so open claims show none. For a settled claim the target is incurred less recoveries; for open and unreported claims, estimating the ultimate is part of the modelling. The host adds that teams may make this adjustment in Part 2. Our treatment so far: `decisions/postmortem_design.md` (PM5).
- **construction_type T and W.** The host will not say what individual codes mean (resolving them is part of the challenge, and the panel assesses how data issues are identified, treated and documented), but confirms a system change in 2023 affected some codes. This supports our Part 1 rule R3 (T and W both read as timber, `decisions/data_rules.md`). The emerging market uses different codes (BRK, STN, WOD, MIX), and `src/data_rules.py` stops the run on any code outside its map, so the Part 2 map has to be written first.
- **Excess.** The policy excess applies to every peril, third-party perils included.
- **Daily limit.** 10 submissions a day, not the 5 first written in the rules.
- **Open question, unanswered.** Whether the Part 1 and Part 2 private scores enter the 40% criterion as raw scores, normalised, or as ranks.
