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
