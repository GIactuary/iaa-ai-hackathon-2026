# Pre-publication scan of the whole git history

*29 October 2026, before the repository is made public (register REPO (a)). Scope: every commit on every branch (125 commits on `main` and `intake-and-plan` at the time of the scan), every path ever committed (259), every blob's content. Nothing has been rewritten or pushed; the clean-up below waits for the appointed actuary's decision.*

## Conclusion

Two things must not be published as they stand, and both sit in history as well as in the current files:

1. **A copy of the host's pages.** `competition/kaggle pages.docx` (44,741 bytes), the actuary's copy of the Part 1 Overview, Data and Rules pages, committed on 18 October (`19364c8`) and present in every later commit. Deleting it in a new commit leaves it in history, so history must be rewritten to drop it.
2. **Third-party code copied from public 2025 notebooks.** `reference/2025/winner_reproduction/repro_winner.py` states that its "code logic and hyperparameters [are] copied from the notebook's published cells"; `reference/2025/other_winners/second_place.py` and `third_place.py` are faithful reruns of other teams' notebooks, including their row-specific fixes to the 2025 data. Kaggle publishes notebooks under Apache 2.0, which cannot be relicensed as MIT, and the row-specific fixes carry 2025 competition data values. No 2026 code imports anything from `reference/`.

Nothing else was found: no competition data, no credentials, no employer names, data or context.

## What was checked, and how

| Check | Method | Result |
|---|---|---|
| Data files | Every path ever committed, filtered for data and document types (parquet, csv, xlsx, zip, pickle, docx, pdf) and for `data/`, `competition/` and submission paths | No competition data file was ever committed. `data/` holds only `README.md` and `.gitkeep`. One CSV: `reference/2025/results/cv_comparison_identical_folds.csv`, scores only. Submission evidence files (`outputs/submission_*.json`) hold summaries, not row predictions |
| Row-level data inside text files | Count of identifier-shaped tokens (policy, claim, address ids) per file across every commit | At most 4 in any file (individual records named in decision notes, for example the 11.9M claim). No file lists rows |
| Large blobs | Every blob over 300 KB | One: `outputs/phase5_step6.json` (470 KB), an evidence file of scores |
| Host-page copies | `competition/` contents | `kaggle pages.docx` is a copy (above). `overview.md`, `rules_notes.md`, `data_description.md` and `part2_notes.md` are notes in our own words with short quotes, and say so |
| Credentials | All revisions searched for API keys, tokens, passwords, private keys, cloud and GitHub key formats, `kaggle.json` | None. The only hits are the word "token" in the notebook builder's template code |
| Employer content | All revisions searched for Canadian and global insurer names, work email domains, corporate file-share paths, "confidential", "proprietary", "internal use" | None. "Proprietary" appears only in the rules notes (external data rule 2.7); "economical" only as an ordinary word |
| Authors | Every commit's author and committer | One identity: Bavan Amirthalingam, personal address |
| Local paths | All revisions searched for Windows user paths | `outputs/what_changed_test_market.md` holds the local project path (the actuary's Windows user name, no employer). Harmless; can stay |

## Proposed clean-up (for the actuary to approve)

1. Rewrite history with `git filter-repo` to drop `competition/kaggle pages.docx` and, under option (a) of PUB-R, `reference/2025/winner_reproduction/`, `reference/2025/other_winners/` and `reference/2025/results/leaderboard.json`, from every commit on both branches. Commit order, dates and messages stay, so the record that each design was committed before its test still holds; commit hashes change, and the hashes quoted in evidence files' `meta.git_commit` and in the AI log will then point to rewritten commits.
2. Re-run this scan on the rewritten history and confirm both paths are gone from every commit (`git log --all -- <path>` empty).
3. Add the removed paths to `.gitignore` so they cannot come back.
4. The actuary creates the public GitHub repository and pushes; nothing is pushed from here.

## Outcome: history rewritten, 29 October 2026

The appointed actuary approved option (a) on both cards (documentation brief round 3, finalized; register PUB-H, PUB-R; exchange D51). The rewrite was run with `git filter-repo` 2.47.0 on both branches, after a full backup of the repository as it stood (a git bundle and a mirror clone, kept outside the repository). It dropped four paths from every commit:

- `competition/kaggle pages.docx`
- `reference/2025/winner_reproduction/`
- `reference/2025/other_winners/`
- `reference/2025/results/leaderboard.json`

| Check after the rewrite | Result |
|---|---|
| Commits on all branches | 137 before (136 at the scan plus the commit recording the verdicts), 137 after; no commit consisted only of the removed paths, so none was dropped |
| Each commit against its original | All 137 have the same tree apart from the four paths, and the same author, committer, dates and message (one message quoted a commit hash; filter-repo rewrote it to the new hash) |
| `git log --all` on the four paths | Empty |
| Any `.docx` ever committed | None |
| Objects for the removed paths, reachable or not | None; reflog expired and the repository repacked |
| The copied-code statement | Appears only in this report and the decision brief, which quote it |

Every commit hash changed. Hashes quoted before the rewrite (evidence files' `meta.git_commit`, the AI log, the register, decision notes) are old hashes: `docs/history_commit_map.txt` maps each to its new commit. The removed paths are in `.gitignore`. Nothing has been pushed; the actuary creates the public repository and pushes.
