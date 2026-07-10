# ResearchForge — Agent Workflow

Eight agents, each with a narrow job and a hard boundary on what it's allowed to decide vs. what it
must retrieve/compute deterministically. Every agent writes its output plus a provenance record
before handing off — the next agent refuses to run on unprovenanced input.

**Default LLM for every agent below**: OpenRouter, model `qwen/qwen3-coder-next` (via local Ollama
first if available). Any agent-level LLM call not otherwise noted uses this model. Source specs
for this whole system live at `spec/ResearchForge_Master_Product_Specification.pdf` and
`spec/ResearchForge_Training_OS_System_Design.pdf`.

---

### 1. Planner Agent
- **Input**: raw user prompt (natural language research/training goal)
- **Does**: calls OpenRouter to extract structured intent JSON — objective, task type, modality,
  labels, eval metric, constraints, GPU availability, framework preference, expected output.
- **Must NOT**: guess datasets, models, or metrics not stated or implied by the user — if unclear,
  emit `needs_clarification` and stop.
- **Output**: `intent.json`

### 2. Research Agent
- **Input**: `intent.json`
- **Does**: parallel calls to ArXiv, Semantic Scholar, OpenAlex, CrossRef, GitHub,
  PapersWithCode, Hugging Face, Kaggle, OpenML, Google Dataset Search. Merges + dedupes.
- **Must NOT**: fabricate a paper, dataset, or stat that didn't come back from an API.
- **Output**: `evidence/*.json` (raw), written into Postgres + object storage with source URL and
  retrieval timestamp on every record.

### 3. Dataset Agent
- **Input**: evidence store
- **Does**: scores candidate datasets on measurable signals only — task match, modality, citation
  count, downloads, last-maintained date, license, missing-value rate, duplicate rate, class
  balance, annotation quality (computed, not guessed).
- **Must NOT**: assign a quality score without a computed signal behind it.
- **Output**: `dataset_ranking.json` with each score traceable to its source metric.

### 4. Validation Agent
- **Input**: chosen dataset
- **Does**: names a cleaning/validation *strategy* (may use OpenRouter/Ollama to pick a strategy
  name from a fixed menu), then executes it with Great Expectations, Pandera, Cleanlab,
  Polars/DuckDB.
- **Must NOT**: let the LLM touch the actual data values, ever. LLM picks the recipe; the library
  runs it.
- **Output**: cleaned dataset + `validation_report.json`

### 5. Training Agent
- **Input**: validated dataset, `intent.json`, hardware constraints
- **Does**: Training Planner (rule engine, no LLM) selects a pipeline template. Config Generator
  emits YAML/JSON. Execution Engine runs one of:
  - **Karpathy-minimal path**: single-file, from-scratch loop (own optimizer step, own training
    loop, no framework magic) — used for small/custom/educational models where you want to see
    every line execute.
  - **Framework path**: Lightning / Transformers / TRL / Accelerate — used for standard-scale
    finetuning jobs.
- **Must NOT**: ever have the LLM write the training loop itself. It only fills in a config
  template's fields.
- **Output**: trained checkpoint + `run_config.yaml` + logs

### 6. Optimization Agent
- **Input**: trained baseline + config
- **Does**: Optuna/Ray Tune search hyperparameters. LLM (OpenRouter) may propose *which*
  hyperparameters are worth searching and *why*, based on the evidence — the search itself is
  deterministic.
- **Output**: best-config + trial history

### 7. Evaluation Agent
- **Input**: trained model(s)
- **Does**: computes benchmark metrics, statistical significance, regression detection vs.
  baseline, calibration, robustness checks — all deterministic computation.
- **Must NOT**: let the LLM state a metric value; it only summarizes/explains computed values.
- **Output**: `eval_report.json`

### 8. Reporting Agent (incl. AutoResearch loop)
- **Input**: all prior outputs
- **Does**: OpenRouter drafts a human-readable report/paper from the computed results and cited
  evidence. AutoResearch mode re-runs Research Agent on a schedule, flags new relevant papers, and
  proposes new experiments — each proposal must cite the paper/evidence that motivated it.
- **Must NOT**: state a result, comparison, or claim not backed by a prior agent's output.
- **Output**: `report.md` / `paper.tex`, plus new experiment proposals fed back to Planner.

---

## Handoff Rule (applies to every agent)

Each agent's output includes:
```json
{
  "data": {...},
  "provenance": { "source": "...", "retrieved_at": "...", "agent": "..." },
  "confidence": "computed | llm_summarized"
}
```
Anything marked `llm_summarized` can never become an input to a *computation* in a downstream
agent — only to text the human eventually reads.
