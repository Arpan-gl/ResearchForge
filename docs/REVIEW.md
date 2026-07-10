# REVIEW.md
### Self-review checklist — run this against your own output before advancing to the next phase

Do not rubber-stamp this. For each item, actually check the code/output you just produced, not
what you intended to produce.

## Global invariants (check every single time, every phase)

- [ ] No dataset, paper, metric, or numeric result appears anywhere that wasn't computed or
      retrieved from an API — grep your own output for suspicious specificity you didn't verify.
- [ ] The LLM layer was used only for: intent parsing, evidence synthesis/ranking, strategy
      naming, hyperparameter proposals, or text summarization. If it did anything else, that's a
      failure — revert and fix.
- [ ] Every agent output includes a `provenance` block (source, retrieved_at, agent).
- [ ] Config files were generated, not raw training code, wherever the architecture calls for a
      config.
- [ ] Nothing silently defaulted on the Ollama/OpenRouter choice or the DB provisioning choice —
      both must have gone through the ask-the-user flow if the local check failed.
- [ ] The change is small enough that you can explain in 2–3 sentences what it does and why it's
      correct — if you can't, it's too big; split it.

## Phase-specific checks

**Planner / Intent Parser**
- [ ] Ambiguous prompts produce `needs_clarification`, not a guessed intent.
- [ ] Output validates against a fixed JSON schema.

**Research Agent**
- [ ] Duplicate evidence across sources is actually deduped (test with a known-duplicate case).
- [ ] Every evidence record has a working source URL.

**Dataset Agent**
- [ ] Each dataset's score is traceable to specific computed signals, not a single opaque number.

**Validation Agent**
- [ ] The LLM's involvement stops at picking a strategy name — trace the code path and confirm
      no LLM call touches actual row/column values.
- [ ] Validation report shows before/after stats (missing values, duplicates, class balance).

**Training Agent**
- [ ] Karpathy-minimal path: every line of the training loop is readable in one file, no hidden
      framework magic — you should be able to point to the exact forward pass, loss, and
      optimizer step.
- [ ] Framework path: config-driven, no hardcoded hyperparameters in code.
- [ ] A training run actually completes and produces a checkpoint — not just "the code compiles."

**Optimization Agent**
- [ ] Hyperparameter proposals from the LLM are labeled as proposals, not applied automatically
      without the deterministic search framework running them.

**Evaluation Agent**
- [ ] Every metric in the report was computed by code, not stated by the LLM.
- [ ] Regression detection actually compares against a stored baseline, not an assumed one.

**Reporting / AutoResearch**
- [ ] Every claim in the generated report cites the specific upstream output (paper, metric run,
      dataset stat) it came from.
- [ ] New experiment proposals cite the evidence that motivated them.

## If something fails review

1. Fix it before moving on — don't note it and continue.
2. If the same check fails twice in a row on the same component, stop and surface it to the user
   with: what failed, why your first fix didn't work, and what you think the actual root cause is.
3. Log every failed-then-fixed check somewhere (e.g. `docs/REVIEW_LOG.md`) so patterns of repeated
   failure are visible over time — this is likely how you'll catch *why* the original build broke.
