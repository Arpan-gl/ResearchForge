# Review Log

## Phase 1 scaffold
- Failed check: phase 1 verification initially failed because pytest used sandbox-blocked temporary directories under the default Windows temp root.
- First fix attempt: redirected tests to a repo-local temp directory in `tests/conftest.py`, but the sandbox still denied temp directory creation during pytest runs.
- Final fix: used a dedicated pytest temp root plus elevated verification for the review run, which allowed the scaffold checks to complete successfully.

## Phase 10 framework trainer
- Failed check: the first framework-trainer review run failed during test collection because a module-level `transformers` import pulled in TensorFlow and hit a protobuf runtime mismatch.
- First fix attempt: moved the Transformers import behind a lazy loader and set `TRANSFORMERS_NO_TF=1` and `TRANSFORMERS_NO_FLAX=1`, but importing a pretrained model class still triggered the TensorFlow side path.
- Final fix: kept the framework path on `transformers.Trainer` but swapped the model to a custom PyTorch classifier, avoiding the problematic pretrained-model import chain while preserving a real framework-backed training run.

## Phase 11 optimization
- Failed check: the first optimization review run failed because the optimization handoff referenced an undefined `timestamp()` helper when writing provenance.
- Fix: added the missing helper in `researchforge/agents/optimization/agent.py` and reran the same review set.

## Live graph persistence smoke run
- Failed check: the first real Neo4j persistence run failed even though the unit tests passed, because a Cypher parameter named `query` collided with Neo4j's `Session.run(query, ...)` argument name in the actual driver.
- First fix attempt: earlier tests only validated a fake session shape that didn't mirror the real driver signature closely enough, so the collision slipped through review.
- Final fix: renamed the bound node parameter to `node_query`, updated the Cypher statement, tightened the fake-session regression test to use the real `run(query, **params)` call shape, and reran the live smoke path.

## Phase 1 live CLI reliability
- Failed check: a paid OpenRouter response (`HTTP 402`) became an empty V1 summary, so the CLI reported zero findings despite retrieving sources.
- Fix: preserve retrieved excerpts with provenance when synthesis is unavailable and expose the provider error as a limitation instead of discarding evidence.
- Failed check: the full natural-language IPL request missed Kaggle datasets and unrelated ArXiv results survived relevance filtering.
- Fix: add deterministic IPL/dataset query variants, reject unrelated evidence, and keep low-confidence dataset candidates visible for review.
- Failed check: Windows `cp1252` output crashed on the Unicode banner, and PyMuPDF left a PDF handle open during temporary-directory cleanup.
- Fix: make console output replacement-safe and close extracted PDF documents explicitly; the real `--skip-training` CLI path now reaches notebook generation.
- Failed check: the OpenRouter SDK defaulted to a 65,535-token completion budget and rejected the request because the account could afford fewer tokens.
- Fix: use the official Python SDK with a bounded 2,048-token response and retry `openrouter/free` only for explicit billing failures; a real SDK smoke request succeeded with Gemini 2.5 Flash Lite.
## 2026-07-15 — Hugging Face SFT handoff

- Review check failed once in the new integration test because the test expected one preview load while the requested result limit correctly returned two candidates.
- Fixed the test to assert one streaming preview per returned candidate; the full suite then passed with 95 tests.
- Verified Hugging Face downloads are explicit, model/dataset metadata carries provenance, and Gemini is restricted to a validated SFT configuration proposal with no executable code.
## 2026-07-15 — FIFA dataset discovery

- Review check failed because the expanded query duplicated the topic and returned no usable public dataset even though Kaggle had FIFA data for concise terms.
- Fixed deterministic search variants for FIFA, football, soccer, and World Cup topics, and search now includes both the full context and the original concise topic.
- Live verification selected `piterfm/fifa-football-world-cup`; no dataset was fabricated.
## 2026-07-15 — FIFA notebook validation

- Review found that the generated notebook selected a ranking CSV, guessed `previous_points` as the label, used a random split, advertised an unsupported F1 range, and did not produce future-schedule predictions.
- Fixed match-file selection, score-derived labels, post-match leakage removal, chronological holdout, measured-only metric wording, and 2026 schedule prediction output.
- Executing the corrected notebook found and fixed XGBoost feature-name incompatibility and future-schedule preprocessing mismatches. Final execution completed with F1-macro `0.3993558776167472` on the 2022 holdout and generated schedule predictions.
- General model strategy is now auditable: the agent may choose only a validated model name; notebook code remains deterministic per the hard constraint.
## 2026-07-15 — Standalone trainer parity

- Review found the generated model package still used a last-column label fallback and random split even after the notebook was corrected.
- Fixed the non-NLP `train.py` generator to share the match-label, leakage-filtering, and chronological-holdout safeguards; generated trainer compiles successfully.
