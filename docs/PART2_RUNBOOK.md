# Part 2 runbook

What to do when the Part 2 files arrive. Design: `decisions/part2_kit_design.md`. Every threshold and judgment for the new market still goes to the actuary before the test that uses it (CLAUDE.md rules 2 and 4).

## Day one, first hour

1. **Files.** Put the Part 2 files, exactly as downloaded, in `data/raw_part2/` (never edit them). The archive unpacks to `developed/` (the Part 1 files, byte-identical) and `emerging/` (the new market); hashes in `competition/data_manifest_part2.json`. Check the rules page for anything new on external data, AI use and deliverables (`/intake`).
2. **What changed.** Run the report against Part 1, before anything else:

   ```
   .venv/Scripts/python.exe -I scripts/what_changed.py --ref data/raw_part2/developed --new data/raw_part2/emerging --name part2 --codes emerging
   ```

   It writes `outputs/what_changed_part2.md`: files and columns added or removed, the largest distribution shifts, the Part 1 traps re-tested (exposure in days, sum insured unit errors, lost id padding, undocumented codes, post-event columns), claims patterns, and training against test rows. Bring it to the actuary as the first Part 2 check-in.
3. **Configuration.** Copy `config/part1.json` to `config/part2.json`, set `"name": "part2"`, `"raw_dir": "data/raw_part2/emerging"`, `"codes": "emerging"` (the documented code set `scripts/what_changed.py` checks against), `"open_recoveries": true` (rule R10, Y3), and change every value the report shows is different: lines and perils, valuation date, years, large-loss perils, step perils, weather perils, size segments. `decisions/thresholds_part2.md` already exists (2026-10-25): every Part 1 row is copied with its date cleared, and the R10 rows are dated. Point `"thresholds_file"` at it, and date each row only once the actuary re-confirms it for the new market: the code refuses an undated one.

## Then

4. **Data rules.** If the report shows a trap Part 1 did not have, write the rule in `decisions/` with its evidence and add it to `src/data_rules.py` first.
5. **One command:**

   ```
   MARKET=part2 .venv/Scripts/python.exe -I scripts/run_market.py
   ```

   Stages: lock, build, what_changed, history, weights (fast mode), level, blend, referral. Each stage's log is in `data/work/markets/part2/run_logs/`; a failed stage stops the run, and `--from STAGE` resumes. The file lands in `outputs/markets/part2/submissions/`. The dry run on a synthetic market took about 32 minutes.
6. **Fast or full.** Fast mode keeps Part 1's tuned settings and refits the family weights; its blend scores are apparent, not out of sample. If the actuary chooses to spend the time, the full step 7 tuning (`scripts/phase5_step7.py`, about 6 hours) and the step 8 cross-fitted weights (`scripts/phase5_step8.py`) run on the market's own data.
7. **Log** every exchange that changes a decision in `ai/ai_log.json`, as in Part 1.

## Proven so far

- `config/part1.json` reproduces all six Part 1 files byte for byte through the refactored code.
- On a synthetic market with six planted changes (`scripts/make_test_market.py`), the report found all six, and the full run produced a valid file.
