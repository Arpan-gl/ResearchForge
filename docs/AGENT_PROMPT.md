# AGENT_PROMPT.md
### Paste this into your coding agent (Claude Code, etc.) to build ResearchForge

You are building **ResearchForge**, a CLI-first AI training operating system. Before writing any
code, read, in this order:
1. `spec/ResearchForge_Master_Product_Specification.pdf` and
   `spec/ResearchForge_Training_OS_System_Design.pdf` — the original product specs.
2. `docs/ARCHITECTURE.md` and `docs/AGENTS.md` — the rebuilt, buildable version of those specs,
   with the provider split called out at the top of ARCHITECTURE.md (OpenRouter using Gemini 2.5 Flash Lite for research reasoning,
   Ollama-first/OpenRouter-fallback for general agent work, the added Karpathy-minimal training
   path, and `qwen/qwen3-coder-next` as the default remote model).

The `docs/` files are the source of truth for anything they cover; the `spec/` PDFs are the source
of truth for anything they don't. If you find a real conflict between them beyond the three known
deviations, flag it to the user — do not silently pick one.

## Working method: build in a loop, not in one pass

For each phase below:
1. **Build** the smallest working slice of that phase.
2. **Self-review** it against `docs/REVIEW.md` before moving on — literally re-read the checklist
   and check your own output against each line. Fix anything that fails.
3. **Report** to the user what was built, what passed review, and what you flagged as uncertain.
4. Only then move to the next phase.

Never skip the self-review step to save time. If a phase fails review twice in a row, stop and ask
the user instead of guessing.

## Hard constraints (non-negotiable, from ARCHITECTURE.md / AGENTS.md)

- The LLM layer (OpenRouter using Gemini 2.5 Flash Lite for research reasoning; Ollama/OpenRouter for general agent work) is used **only** for: intent parsing into
  structured JSON, ranking/synthesizing already-retrieved evidence, choosing a validation strategy
  *name* from a fixed menu, proposing hyperparameters to search, and writing human-readable
  summaries/reports.
- **Default model for every one of those calls: `qwen/qwen3-coder-next`** (via OpenRouter, or the
  local Ollama equivalent if available). It's an agentic/tool-use MoE model with 256k context and
  non-thinking mode — matches the structured-output, tool-calling nature of every call this system
  makes. Only override to a different model where a specific call needs more context or explicit
  reasoning traces, and do so via env var, not a hardcoded string.
- The LLM **never**: invents a dataset, paper, or metric value; writes raw training code; decides
  DB/infra location on its own; touches actual data values during cleaning.
- Use **OpenRouter** with `google/gemini-2.5-flash-lite` for multi-source research queries, evidence
  summaries, and contradiction explanations. Use local **Ollama** for general agent calls when
  reachable, with **OpenRouter** as the fallback.
- Config generation (YAML/JSON) is the deliverable of the Training Planner — not code generation.
- Training Execution Engine must support **two paths**:
  - `training/karpathy_minimal/` — a single, fully-readable, from-scratch training loop
    (own forward/backward/optimizer step, no framework abstraction hiding what's happening —
    in the spirit of Karpathy's nanoGPT/micrograd). Use this as the default scaffold so the system
    is inspectable and debuggable, which is likely where the previous build failed silently.
  - `training/frameworks/` — Lightning / Transformers / TRL / Accelerate wrappers for
    standard-scale finetuning.
- Every agent output must carry a `provenance` block (source, retrieved_at, agent) per
  `docs/AGENTS.md`'s Handoff Rule. Reject/flag any downstream input missing provenance.

## Storage provisioning — ask, don't assume

On `researchforge init`:
1. Ping local Ollama (`http://localhost:11434/api/tags`).
2. If unreachable, **stop and ask the user directly**:
   > "Ollama isn't running locally. Where should I get your DB connection details
   > (Postgres for evidence store, Neo4j/KuzuDB for the knowledge graph)? You can (a) give me
   > connection strings now, or (b) I can spin up Postgres + Neo4j/KuzuDB + Redis via Docker
   > Compose for you."
3. If (a): read from `.env` / CLI flags, validate the connection before continuing.
4. If (b): run `docker compose up -d` against `infra/docker-compose.yml`, persist generated
   credentials to `.researchforge/.env`.
5. Never default silently to either path — this exact ambiguity is a suspected cause of the
   original build's failures.

## Build order

1. Scaffold folder structure exactly as in `ARCHITECTURE.md`.
2. Planner Agent + Intent Parser (OpenRouter/Ollama switch + storage decision flow).
3. Research Agent (start with 2–3 evidence sources, not all 10, to get a working slice fast —
   ArXiv + Semantic Scholar + GitHub is a good first cut).
4. Evidence Store (Postgres schema + provenance columns).
5. Knowledge Graph (Neo4j/KuzuDB schema from `ARCHITECTURE.md`).
6. Dataset Agent + scoring.
7. Validation Agent (deterministic libraries; LLM only names the strategy).
8. Training Planner + Config Generator.
9. Karpathy-minimal training path (get this working before the heavy-framework path — it's
   simpler to debug and validates the rest of the pipeline).
10. Framework training path.
11. Optimization Agent, Evaluation Agent.
12. Reporting Agent + AutoResearch loop.

After each numbered step, run the self-review loop against `docs/REVIEW.md` before moving to the
next number.
