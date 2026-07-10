# ResearchForge — Architecture (Rebuild Spec)

This reconciles `ResearchForge_Master_Product_Specification.pdf` and
`ResearchForge_Training_OS_System_Design.pdf` (both kept verbatim in `spec/` — see below) into one
buildable architecture. Three changes from the original docs, per explicit decision:

1. **Reasoning layer**: Gemini → **OpenRouter** (model-agnostic, swappable via env var).
2. **Execution engine**: adds a **Karpathy-style minimal trainer** (nanoGPT/micrograd philosophy —
   a small, fully-readable, from-scratch training loop) as a first-class path, not just
   Lightning/Transformers/TRL. This is the path used when the task is "train/finetune a small
   model from scratch to understand it," vs. the heavy-framework path used for production-scale
   finetuning.
3. **Default OpenRouter model**: `qwen/qwen3-coder-next` — an open-weight MoE model (80B total /
   3B active params, 256k context, non-thinking mode) purpose-built for agentic coding and tool
   use. It's the default for every LLM call in this system (intent parsing, config generation,
   evidence synthesis, strategy naming) unless a call specifically needs a larger context or a
   reasoning/"thinking" model, in which case swap the model string only — never the call pattern.

## Source-of-Truth Specs

The two original PDFs are kept, unmodified, under `spec/` with their original filenames:

```
spec/
  ResearchForge_Master_Product_Specification.pdf
  ResearchForge_Training_OS_System_Design.pdf
```

Any time this architecture doc and a source PDF disagree, treat it as a **flagged deviation**, not
a silent override — call it out to the user (this doc already lists the three intentional
deviations above; any other mismatch found while building should be reported, not just patched).

## Why the original build likely broke

Both source docs already name the failure mode without knowing it: **the LLM was probably being
asked to do things that must be deterministic** — inventing dataset stats, writing raw training
code instead of configs, or deciding DB schema on the fly. Section 21/"Hallucination Prevention" in
the spec is the fix; it just was never enforced as *code*, only as intent. This rebuild enforces it
structurally: the LLM layer (OpenRouter) is only ever allowed to (a) parse intent into JSON,
(b) rank/justify from retrieved evidence, (c) summarize results. It never generates: dataset
values, training code, or infra decisions. Those come from libraries, templates, and the user.

## System Flow

```
CLI (researchforge)
  │
  ▼
Intent Parser  ──────────────► OpenRouter (structured JSON output only)
  │
  ▼
Parallel Evidence Search  ───► ArXiv / Semantic Scholar / OpenAlex / GitHub /
  │                             PapersWithCode / HF / Kaggle / OpenML
  ▼
Evidence Store  ─────────────► Postgres (metadata) + object storage (raw artifacts)
  │
  ▼
Knowledge Graph  ────────────► Neo4j / KuzuDB (nodes: Paper, Dataset, Model, Task,
  │                             Metric, Benchmark, Repository, Author)
  ▼
Reasoning Layer  ────────────► OpenRouter — synthesizes evidence into ranked,
  │                             cited recommendations. Never invents facts.
  ▼
Dataset Validation  ─────────► Great Expectations / Pandera / Cleanlab / Polars+DuckDB
  │                             (LLM proposes strategy name only; libraries execute it)
  ▼
Training Planner (rule engine) ─► maps (task, hardware, dataset) → pipeline template
  │
  ▼
Config Generator  ───────────► YAML/JSON only — never raw training code from the LLM
  │
  ▼
Execution Engine   ──┬───────► Path A: Karpathy-minimal trainer (readable, from-scratch,
  │                   │          single-file loop — default for small/educational/custom models)
  │                   └───────► Path B: Lightning / Transformers / TRL / Accelerate
  │                             (default for standard finetuning at scale)
  ▼
Experiment Manager  ─────────► Optuna / Ray Tune
  │
  ▼
Evaluation Engine  ───────────► metrics, significance tests, regression detection
  │
  ▼
AutoResearch Agent  ──────────► literature watch loop → new experiment proposals
  │
  ▼
Reports & Deployment
```

## Storage Decision Flow (new — resolves your DB-provisioning ambiguity)

This is the concrete logic the CLI must implement on `researchforge init`:

```
1. Check for a running local Ollama instance (http://localhost:11434/api/tags)
   │
   ├─ REACHABLE ─────────────────────────────────────────────────────────┐
   │                                                                      │
   └─ NOT REACHABLE                                                      │
        │                                                                │
        ▼                                                                │
   Prompt the user directly:                                            │
   "Ollama isn't running locally. Where should I get your DB            │
    connection details (Postgres + Neo4j/KuzuDB)?"                      │
        [a] I'll provide connection strings now                         │
        [b] Spin them up for me via Docker Compose                      │
        │                                                                │
        ├─ (a) → read from --db-url / --graph-url flags or .env,        │
        │        validate connection before proceeding                  │
        │                                                                │
        └─ (b) → docker compose up -d (postgres, neo4j/kuzu, redis)     │
                 write generated creds to .researchforge/.env            │
                                                                          │
   ◄──────────────────────────────────────────────────────────────────────┘
```

Rule: **the agent never silently assumes a DB location.** If Ollama (the intended local LLM
runtime) is down, that's the trigger to stop and ask — because it means the whole local-first
assumption may be wrong, and DB provisioning should follow the same "ask, don't assume" principle
as the rest of the hallucination-prevention design.

## LLM Layer

- All reasoning/intent calls go through **OpenRouter** (`OPENROUTER_API_KEY` env var), default
  model `qwen/qwen3-coder-next` — chosen because it's an agentic/tool-use model, which matches
  what every call in this system actually is (structured JSON output, config generation, tool
  calls), not free-form chat. Model is still configurable per call via env var override
  (e.g. a larger-context or thinking model for deep evidence synthesis, if 256k isn't enough).
- Local Ollama is the **preferred default** for cost/latency on repeated calls (intent parsing,
  dataset-strategy naming) — pull an equivalent local Qwen3-Coder weight if running fully offline;
  OpenRouter (`qwen/qwen3-coder-next`) is the **fallback/production** path or used directly if the
  user opts out of local.
- Neither is ever used to generate: dataset content, training code, or numeric results.

## Folder Structure

```
researchforge/
  cli/
  agents/
    planner/
    research/
    dataset/
    validation/
    training/
    optimization/
    evaluation/
    reporting/
  knowledge_graph/
  search/
  datasets/
  training/
    karpathy_minimal/      # new: single-file, from-scratch trainer
    frameworks/            # Lightning / Transformers / TRL wrappers
  evaluation/
  experiments/
  sdk/
  configs/
  templates/
  infra/
    docker-compose.yml      # postgres, neo4j/kuzu, redis
  spec/
    ResearchForge_Master_Product_Specification.pdf
    ResearchForge_Training_OS_System_Design.pdf
  docs/
    ARCHITECTURE.md
    AGENTS.md
    AGENT_PROMPT.md
    REVIEW.md
```

## MVP Phases (unchanged from spec, now with exit criteria)

| Phase | Scope | Exit Criteria |
|---|---|---|
| 1 | Research + Dataset Discovery | Can search all evidence sources, dedupe, store with provenance |
| 2 | Validation + Planning | Deterministic validation runs end-to-end on a real dataset; planner produces a config, not code |
| 3 | Training + Evaluation | Both Karpathy-minimal and framework paths train successfully on one real task each |
| 4 | AutoResearch | Literature loop produces at least one reproducible, cited proposal |
| 5 | Collaboration/hosted | Out of scope for this rebuild pass |
