# A technical price, built with AI and decided by an actuary

Entry by Bavan Amirthalingam, FCAS, FCIA (appointed actuary for the entry), to the IAA AI-Assisted Hackathon 2026 on Kaggle. Part 1 prices 2025 motor, home buildings and home contents for an established insurer in a developed market. Part 2 prices the same three lines for an emerging market with two years of data and a 2025 book that is 76% new business from an acquired broker portfolio.

The AI proposed, built and tested; the actuary made every judgment. Each decision is in the judgment register with its options, evidence and verdict, and each exchange that changed a decision is a card in the AI log, including the ones where the AI was wrong.

## The story in five lines

1. **Part 1.** One rule decided every modelling choice: a change stays only if it lowers deviance on the next year out of time, with a 95% customer-bootstrap interval above zero. Of 67 engineered features, three stayed; claims history weighted by credibility gave the largest step. A blend of model families per line, set to a weather-normalised 2025 level, beat our working GLM on customers locked away until the end. Private leaderboard 0.13641, 15th (first 0.13923).
2. **Part 2.** Two emerging years could not carry their own risk shape, so the price borrows Part 1's relativities (credibility at the level of a whole rating plan) and lets the emerging data set the level and add only what it proves: gender in motor, own-year weather, fire and flood with address hazards, a credibility-weighted new-business factor.
3. **The hold-out decided buildings.** Opened once, on a rule written before opening, it held for motor and contents and failed for buildings, which went back to Part 1's shape.
4. **Office premium.** The 20 reference policies are built up from risk premium to office premium with the host's expense table.
5. **Guardrails.** Every test design is committed before its run (see the history of `decisions/`), every threshold is dated before use, and every reported number is out of fold.

The write-up is `docs/writeup/writeup.md` (the five graded pages, then appendices). The video script is `docs/video/script.md`.

## Rebuild a final file (one command per part)

The data is not included: the competition data is licensed CC BY-NC 4.0 and may not be redistributed. Download it from the two Kaggle competition pages and unpack it unchanged:

- Part 1 files into `data/raw/` (11 parquet files);
- Part 2 files into `data/raw_part2/` (the `developed/` and `emerging/` folders).

File sizes and SHA-256 hashes are in `competition/data_manifest.json` and `competition/data_manifest_part2.json`.

Set up once (Python 3.14.5; versions pinned in `requirements.txt`):

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
```

Then, for Part 1 (about [time from the clean rerun]):

```bash
.venv/Scripts/python.exe -I scripts/reproduce.py --part 1
```

and for Part 2 (about [time from the clean rerun]):

```bash
.venv/Scripts/python.exe -I scripts/reproduce.py --part 2
```

Each command runs every stage from the raw files, writes the submission file to `outputs/` (Part 1) or `outputs/markets/part2/` (Part 2), and checks its SHA-256 against the file that was uploaded (`outputs/submission_hashes.json`). Part 2 takes `--motor by_peril` (motor priced by peril) or `--motor final` (one motor GLM); register item H-MP records which file was submitted. Logs go to `data/work/reproduce/`.

The exhibits in the write-up regenerate from the committed evidence files with:

```bash
MARKET=part2 .venv/Scripts/python.exe -I scripts/exhibits_part2.py
```

Tests: `.venv/Scripts/python.exe -m pytest`.

## Where things are

| Path | What it holds |
|---|---|
| `docs/writeup/` | The write-up and its exhibits (`exhibits/` is generated) |
| `docs/video/` | Video script and slides |
| `docs/reviews/` | The decision briefs the actuary decided on, one per round |
| `decisions/judgment_register.md` | Every judgment: options, evidence, recommendation, the actuary's verdict |
| `decisions/*_design.md` | Test designs, each committed before the test ran; results appended after |
| `decisions/thresholds.md`, `thresholds_part2.md` | Every threshold, dated before the test that uses it |
| `ai/ai_log.json` | The AI log: 46 exchange cards (question, AI answer, challenge, test, result, decision) |
| `ai/prompt_library.md` | The standing prompts for each AI role |
| `src/` | Data rules, targets, GLM, level chain, weather, validation, evidence files |
| `scripts/` | Each analysis step, named by phase; `reproduce.py` for the final files |
| `config/` | One file per market: Part 2 runs the Part 1 code with its own settings |
| `outputs/` | Evidence files (JSON) behind every number; submission files are not committed |
| `competition/` | Our notes on the competition pages and the data manifests |
| `reference/2025/` | Context from a preparation run on the 2025 competition |
| `tests/` | Tests for `src/` and the notebook builder |

## How the work was done

The entry was built in Claude Code sessions. Standing instructions are in `CLAUDE.md`; the AI worked in roles (data steward, modeller, filing writer, independent reviewer) with prompts in `ai/prompt_library.md`. Decisions went to the actuary as briefs (`docs/reviews/`), one round at a time, and his verdicts were recorded in the register and the AI log in the same turn. Independent reviewers that had not seen the build read the work before each submission.

## Licence

The code and documents in this repository are released under the MIT licence (`LICENSE`). The competition data is not part of the repository and keeps its own licence.
