# IAA AI Hackathon 2026 entry

A Claude Code workspace for the 2026 IAA AI for Actuaries hackathon. It carries everything learned from the 2025 competition: what each prize-winner did, how each scored when rebuilt on identical folds, a principle-based playbook, a rate-filing documentation template, and the 2025 code.

## Set up (once)

1. Create a private GitHub repo and copy this folder's contents to its root.
2. `pip install -r requirements.txt` (Python 3.13), or use Kaggle's preinstalled versions.
3. Open Claude Code at the repo root. It reads `CLAUDE.md` automatically.

## When the competition opens

1. Copy the overview, rules and evaluation pages into `competition/` as markdown or text.
2. Put the data files in `data/raw/` unchanged.
3. In Claude Code, run `/intake`. It writes the brief and the first judgments, then waits for you.
4. Work through the phases in `CLAUDE.md`. Use `/screen` for phase 2, `/log-decision` after any exchange that changes a decision, and `/review` before submission.
5. Submit the leaderboard file and the notebook yourself.

## What is where

- `CLAUDE.md`: role, rules, phases and gates for the coding session.
- `docs/PLAYBOOK.md`: the 11 principles, 2025 evidence, traps, time plan.
- `docs/LESSONS_2025.md`: each winner in detail, the houses that decided the leaderboard, what the AI got wrong and how it was caught.
- `docs/FILING_TEMPLATE.md`: the notebook's section structure and the judgment register.
- `docs/JUDGING_AND_AI_EVIDENCE.md`: criteria, and how to evidence creative AI use.
- `.claude/`: the `independent-reviewer` and `data-steward` subagents and four slash commands.
- `ai/`, `decisions/`: the log, prompts, transcript, judgment register, thresholds, exclusions.
- `reference/2025/`: code and results from 2025.
