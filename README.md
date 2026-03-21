# ResearchForge CLI

> Topic → Research → Datasets → ML Notebook → Trained Model. Fully automated.

Your LLM (Ollama, local) handles all intelligence.  
Your GPU handles all training.  
Zero cloud costs.

---

## Install

```bash
pip install researchforge
```

Or from source:
```bash
git clone https://github.com/yourname/researchforge
cd researchforge
pip install -e .

# Optional extras
pip install -e ".[gnn]"     # for GNN (PyTorch Geometric)
pip install -e ".[nlp]"     # for transformers / fine-tuning
pip install -e ".[kaggle]"  # for Kaggle dataset auto-download
```

---

## Requirements

| Requirement | Purpose |
|---|---|
| Python 3.9+ | Core runtime |
| [Ollama](https://ollama.ai) running locally | All LLM calls (V1 extraction, V3 generation, autoresearch suggestions) |
| GPU with CUDA (optional but recommended) | Autoresearch training experiments |
| Jupyter | Open generated notebooks |

Install and start Ollama:
```bash
# Install Ollama: https://ollama.ai
ollama pull llama3
ollama serve
```

---

## First-time setup

```bash
researchforge init
```
This saves your Ollama URL, model name, and optional Kaggle credentials to `~/.researchforge/config.json`.

---

## Usage

### Run the full pipeline
```bash
researchforge run "NCAA basketball shot prediction using GNN"
```

### Run with your own dataset
```bash
researchforge run "NCAA basketball GNN" --dataset ./ncaa_tracking.csv
```

### Skip GPU training (generate notebook only)
```bash
researchforge run "topic" --skip-training
```

### Override model selection
```bash
researchforge run "topic" --model gnn
researchforge run "topic" --model xgboost
```

### Interactive chat
```bash
researchforge chat
```
Chat with your local LLM about your research topic, pipeline results, or notebook improvements.

### Check last run status
```bash
researchforge status
```

### Open generated notebook
```bash
researchforge notebook
```

---

## What happens when you run

```
[1] V1 Research
    ├── Rewrites your topic into 4 search queries
    ├── Searches web (DuckDuckGo) + arXiv in parallel
    ├── Reranks by recency + credibility + depth
    ├── Asks Ollama to extract: findings, metrics, datasets, contradictions
    └── Outputs: structured JSON research report

[2] V2 Datasets
    ├── If you gave --dataset: audits it (size, balance, missing values, risks)
    ├── Otherwise: searches Kaggle + HuggingFace, scores each 0–1
    └── Outputs: best dataset + risk report

[3] V3 Notebook
    ├── Auto-detects problem type (classification / regression / graph / NLP)
    ├── Selects model based on dataset size
    ├── Generates 5-section .ipynb: Load+EDA → Split → Preprocess → Train+CV → Summary
    └── Outputs: ready-to-run .ipynb on your GPU

[4] Autoresearch (overnight on your GPU)
    ├── Asks Ollama to suggest one experiment at a time
    ├── Runs each with 3 random seeds
    ├── Accepts if mean > baseline + 1 std
    ├── Commits improvements to git
    └── Outputs: best model commit + % improvement
```

---

## Configuration

`~/.researchforge/config.json`:
```json
{
  "ollama_url": "http://localhost:11434",
  "model": "llama3",
  "kaggle_username": "your_username",
  "kaggle_key": "your_api_key"
}
```

Override with environment variables:
```bash
export OLLAMA_URL=http://localhost:11434
export RF_MODEL=mistral
```

---

## Example output

```
  ResearchForge pipeline
  Topic : NCAA basketball GNN shot prediction
  Dataset: ncaa_tracking.csv

  [1] V1 Research — hybrid retrieval + LLM extraction
  ✓ 11 sources · 4 findings · 1 contradiction flagged

  [2] V2 Datasets — scoring + risk audit
  ✓ Dataset: ncaa_tracking.csv · Score: 0.89 · Risks: 2

  [3] V3 Notebook — generating runnable .ipynb
  ✓ Notebook: researchforge_NCAA_basketball_GNN.ipynb
    Expected F1-macro: 0.71–0.82

  [4] Autoresearch — running experiments on your GPU
    Experiment 1/100 → Switch GCN to GAT... ✓ +0.021 → committed a3f7b2
    Experiment 2/100 → Add shot_clock feature... ✓ +0.012 → committed c19de4
    Experiment 3/100 → Increase dropout 0.3→0.5... ✗ 0.701 (threshold 0.724)
    ...

  ┌─────────────────────────────────────────────┐
  │           RESEARCHFORGE SUMMARY              │
  ├─────────────────────────────────────────────┤
  │  Sources found     : 11                      │
  │  Dataset           : ncaa_tracking.csv       │
  │  Dataset score     : 0.89                    │
  │  Notebook          : researchforge_NCAA.ipynb│
  │  Baseline F1-macro : 0.710                   │
  │  Best F1-macro     : 0.814                   │
  │  Experiments run   : 100                     │
  └─────────────────────────────────────────────┘
```

---

## License
MIT
