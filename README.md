# A technical price, built with AI and decided by an actuary

Entry by Bavan Amirthalingam, FCAS, FCIA (appointed actuary for the entry), to the IAA AI-Assisted Hackathon 2026 on Kaggle. Part 1 prices 2025 motor, home buildings and home contents for an established insurer in a developed market. Part 2 prices the same three lines for an emerging market with two years of data and a 2025 book that is 76% new business from an acquired broker portfolio.

The AI proposed, built and tested; the actuary made every judgment. Each decision is in the judgment register with its options, evidence and verdict, and each exchange that changed a decision is a card in the AI log, including the ones where the AI was wrong.

## The story in five lines

1. **Part 1.** One rule decided every modelling choice: a change stays only if it lowers deviance on the next year out of time, with a 95% customer-bootstrap interval above zero. Of 67 engineered features, three stayed; claims history weighted by credibility gave the largest step. A blend of model families per line, set to a weather-normalised 2025 level, beat our working GLM on customers locked away until the end. Private leaderboard 0.13641, 15th (first 0.13923).
2. **Part 2.** Two emerging years could not carry their own risk shape, so the price borrows Part 1's relativities (credibility at the level of a whole rating plan) and lets the emerging data set the level and add only what it proves: in the final file, motor priced by peril with a gender factor and own-year weather, Part 1's tree in contents with own-year weather, and a credibility-weighted new-business factor. Public leaderboard 0.15712.
3. **The hold-out decided buildings.** Opened on a rule written before opening, it showed the emerging additions holding in motor and losing in buildings, which went back to Part 1's shape alone. Three of four independent AI reviews then found that motor priced by peril passes the same line-by-line rule.
4. **Office premium.** The 20 reference policies are built up from risk premium to office premium with the host's expense table.
5. **Guardrails.** Every test design is committed before its run (see the history of `decisions/`), every threshold is dated before use, and every reported number is out of fold unless labelled apparent. Choices made after a hold-out was opened, or after a public-board reading, are reported as such.

The write-up is `docs/writeup/writeup.md` (the five graded pages, then appendices). The video script is `docs/video/script.md`.

## Rebuild a final file (one command per part)

The data is not included: the competition data is licensed CC BY-NC 4.0 and may not be redistributed. Download it from the two Kaggle competition pages and unpack it unchanged:

- Part 1 files into `data/raw/` (11 parquet files);
- Part 2 files into `data/raw_part2/` (the `developed/` and `emerging/` folders).

File sizes and SHA-256 hashes are in `competition/data_manifest.json` and `competition/data_manifest_part2.json`.

On Windows, clone to a short folder path (for example `C:\iaa`): some output paths are long, and Windows limits a full path to 260 characters.

Set up once (Python 3.14.5; versions pinned in `requirements.txt`):

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
```

Then, for Part 1 (about 30 minutes):

```bash
.venv/Scripts/python.exe -I scripts/reproduce.py --part 1
```

and for Part 2 (about 7 minutes, Part 1's data build included):

```bash
.venv/Scripts/python.exe -I scripts/reproduce.py --part 2
```

Each command runs every stage from the raw files, writes the submission file to `outputs/submissions/` (Part 1) or `outputs/markets/part2/submissions/` (Part 2), and checks its SHA-256 against the file that was uploaded (`outputs/submission_hashes.json`). Logs go to `data/work/reproduce/`. Both commands were run on 29 October 2026 in an empty copy of the repository with only the raw files added, and both files matched the uploaded files byte for byte.

The exhibits, the write-up PDF and the video slides regenerate from the committed evidence files with:

```bash
MARKET=part2 .venv/Scripts/python.exe -I scripts/exhibits_part2.py
```

```bash
.venv/Scripts/python.exe -I scripts/build_writeup.py
```

```bash
.venv/Scripts/python.exe -I scripts/build_slides.py
```

Tests: `.venv/Scripts/python.exe -m pytest`.

## Where things are

| Path | What it holds |
|---|---|
| `docs/writeup/` | The write-up and its exhibits (`exhibits/` is generated) |
| `docs/video/` | Video script and slides |
| `docs/reviews/` | The decision briefs the actuary decided on; the register and AI log also cite their hosted copies, which may not open for every reader |
| `decisions/judgment_register.md` | Every judgment: options, evidence, recommendation, the actuary's verdict |
| `decisions/*_design.md` | Test designs, each committed before the test ran; results appended after |
| `decisions/thresholds.md`, `thresholds_part2.md` | Every threshold, dated before the test that uses it |
| `ai/ai_log.json` | The AI log: 51 exchange cards (question, AI answer, the actuary's challenge where there was one, test, result, decision) |
| `ai/prompt_library.md` | The standing prompts for each AI role |
| `src/` | Data rules, targets, GLM, level chain, weather, validation, evidence files |
| `scripts/` | Each analysis step, named by phase; `reproduce.py` for the final files |
| `config/` | One file per market: Part 2 runs the Part 1 code with its own settings |
| `outputs/` | Evidence files (JSON) behind every number; submission files are not committed |
| `competition/` | Our notes on the competition pages and the data manifests; the copy of the host's pages was removed from the whole history before publication |
| `reference/2025/` | Context from a preparation run on the 2025 competition: our own code and results. Reruns of other teams' notebooks were removed from the whole history before publication (`reference/2025/README.md`) |
| `tests/` | Tests for `src/` and the notebook builder |

## How the work was done

The entry was built in Claude Code sessions. Standing instructions are in `CLAUDE.md`; the AI worked in roles (data steward, modeller, filing writer, independent reviewer) with prompts in `ai/prompt_library.md`. Decisions went to the actuary as briefs (`docs/reviews/`), one round at a time, and his verdicts were recorded in the register and the AI log in the same turn. Independent reviewers in fresh sessions, which had not seen the build, read the work at set points: four gap reviews of the Part 2 file and a review of the write-up and repository (`outputs/review_2026-10-29.md`).

## Licence

The code and documents in this repository are released under the MIT licence (`LICENSE`). The competition data is not part of the repository and keeps its own licence.
