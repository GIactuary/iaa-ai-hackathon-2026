# Judging, and how to show creative use of AI

## 2025 criteria and process

From the 2025 Kaggle overview page: private-leaderboard accuracy 30%, creative use of AI tools 30%, interpretability and explanation 20%, code and prompt documentation and reproducibility 20%. Prompts, AI-usage summaries and prompt transcripts were required. The leaderboard file and the notebook link (by Google Form) were both due at the deadline. Judging: stage 1 AI-assisted screening, stage 2 AI and human hybrid ranking, stage 3 human actuaries. Confirm all of this for 2026 in `/intake`.

Citations for the 2025 prizes: 1st, "strongest holistic GenAI-enhanced workflow; stood out in interpretability and AI-supported reasoning". 2nd, "most creative and pervasive use of GenAI". 3rd, "one of the best predictive models". Special mention, "forward-thinking agentic workflow".

## What counts as evidence of creative AI use

Judges cannot see a chat. They see the notebook. The evidence is a record of AI changing decisions under challenge.

1. **Roles.** Each AI role has a set-up prompt (data steward, modeller, filing writer, plain-language explainer, independent reviewer) in `ai/prompt_library.md`. Roles are prompts and subagents, not a separate agent product.
2. **Exchange cards.** For every exchange that changed a decision: the question, the AI's answer, the actuary's challenge, the test run, the result, the decision, the effect, and the notebook section. Stored in `ai/ai_log.json`, rendered as cards next to the analysis they changed.
3. **Decision ledger.** The cards summarised as one table in notebook section 2.
4. **Cards where the AI was wrong.** These are the most persuasive. In 2025: a false leakage claim withdrawn after matching rows; a capping rule that failed its own test; a disagreement flag with a blind spot. Keep them.
5. **Independent review.** A fresh-context reviewer scores the notebook against the criteria and tries to break it (`/review`). Record what it found and what changed.
6. **Transcript.** A condensed transcript of the sessions in `ai/transcript.md`, attached to the notebook.

## Challenge prompts the actuary uses

- "Prove it, don't assert it. What check would show this is wrong?"
- "Is this out of fold? Which rows was the model fitted on?"
- "Could this be luck of one split? Rerun on two more seeds."
- "Would this input exist at the moment the model is used?"
- "Where do the two models disagree most, and who is right there?"
- "What is the weakest claim in this section? Make it more cautious or test it."
- "Did we write this threshold down before we saw what it catches?"
