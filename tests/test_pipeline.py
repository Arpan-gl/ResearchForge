"""
Integration tests for the ResearchForge pipeline.
All external calls (Ollama, arXiv, DuckDuckGo, Kaggle, HuggingFace) are mocked.
Tests verify that data flows correctly: V1 → V2 → V3 and notebook is created.
"""

import json
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock


# ── Shared fixtures ───────────────────────────────────────────────

MOCK_V1_RESULT = {
    "topic": "basketball GNN shot prediction",
    "key_findings": ["GNNs achieve F1=0.81 on player tracking data [1]"],
    "metrics": [{"name": "F1", "value": "0.81", "unit": "", "source": 1}],
    "datasets": [{"name": "ncaa_tracking", "size": "10k rows", "task": "classification"}],
    "limitations": ["Small dataset coverage"],
    "recommended_models": ["GATConv", "GraphSAGE"],
    "problem_type": "classification",
    "contradictions": [],
    "sources": [
        {"title": "GNN for Basketball", "url": "http://arxiv.org/1234",
         "snippet": "GNNs outperform classical ML on player tracking data.", "source": "arxiv"}
    ],
}

MOCK_V2_RESULT = {
    "name": "mock_dataset.csv",
    "score": 0.82,
    "shape": "5000 rows × 10 cols",
    "label_column": "shot_made",
    "risks": [],
    "ml_tasks": ["XGBoost baseline", "LightGBM"],
    "source": "user-provided",
    "problem_type": "classification",
}

MOCK_LLM_QUERIES = json.dumps([
    "basketball GNN query 1",
    "basketball GNN keywords",
    "GNN basketball title",
    "graph neural basketball technical",
])
MOCK_LLM_FINDINGS = json.dumps(MOCK_V1_RESULT)


# ── V1 integration ────────────────────────────────────────────────

def test_v1_run_returns_expected_keys():
    from researchforge.stages.v1_research import V1Research
    v1 = V1Research.__new__(V1Research)
    v1.ollama_url = "http://localhost:11434"
    v1.model = "llama3"

    with patch.object(v1, "_ask_llm", side_effect=[
        MOCK_LLM_QUERIES,   # query rewriting
        MOCK_LLM_FINDINGS,  # findings extraction
        "[]",               # semantic contradictions
    ]):
        with patch.object(v1, "_web_search", return_value=[]):
            with patch.object(v1, "_arxiv_search", return_value=[]):
                result = v1.run("basketball GNN shot prediction")

    assert "key_findings" in result
    assert "contradictions" in result
    assert "sources" in result
    assert "topic" in result
    assert isinstance(result["contradictions"], list)


# ── V3 integration ────────────────────────────────────────────────

def test_v3_creates_notebook_file():
    from researchforge.stages.v3_notebook import V3Notebook

    with tempfile.TemporaryDirectory() as tmp:
        original_dir = os.getcwd()
        os.chdir(tmp)
        try:
            v3 = V3Notebook.__new__(V3Notebook)
            v3.settings = MagicMock()

            result = v3.generate(
                topic="basketball gnn",
                v1_findings=MOCK_V1_RESULT,
                v2_dataset=MOCK_V2_RESULT,
                model_override=None,
            )
            assert os.path.exists(result["notebook_path"])
            assert result["notebook_path"].endswith(".ipynb")
            assert result["metric_name"] in ("F1-macro", "Accuracy", "RMSE")
        finally:
            os.chdir(original_dir)


def test_v3_notebook_has_correct_structure():
    """Generated notebook should have a title + 6 code sections."""
    import nbformat
    from researchforge.stages.v3_notebook import V3Notebook

    with tempfile.TemporaryDirectory() as tmp:
        original_dir = os.getcwd()
        os.chdir(tmp)
        try:
            v3 = V3Notebook.__new__(V3Notebook)
            v3.settings = MagicMock()
            result = v3.generate(
                topic="test topic",
                v1_findings=MOCK_V1_RESULT,
                v2_dataset=MOCK_V2_RESULT,
            )
            with open(result["notebook_path"]) as f:
                nb = nbformat.read(f, as_version=4)
            # 1 markdown title + 6 code sections = 7 cells
            assert len(nb.cells) == 7
            assert nb.cells[0].cell_type == "markdown"
            code_cells = [c for c in nb.cells if c.cell_type == "code"]
            assert len(code_cells) == 6
        finally:
            os.chdir(original_dir)


def test_v3_gnn_notebook_uses_graph_builder():
    """GNN problem type should reference GraphBuilder in SECTION 4 (code cell index 3)."""
    import nbformat
    from researchforge.stages.v3_notebook import V3Notebook

    with tempfile.TemporaryDirectory() as tmp:
        original_dir = os.getcwd()
        os.chdir(tmp)
        try:
            v3 = V3Notebook.__new__(V3Notebook)
            v3.settings = MagicMock()
            gnn_v2 = dict(MOCK_V2_RESULT, problem_type="graph")
            gnn_v1 = dict(MOCK_V1_RESULT, problem_type="graph")
            result = v3.generate(
                topic="gnn topic",
                v1_findings=gnn_v1,
                v2_dataset=gnn_v2,
            )
            with open(result["notebook_path"], encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)
            code_cells = [c for c in nb.cells if c.cell_type == "code"]
            # SECTION 4 is the 4th code cell (0-indexed = 3)
            model_cell = code_cells[3].source
            assert "GraphBuilder" in model_cell
        finally:
            os.chdir(original_dir)


# ── State persistence ─────────────────────────────────────────────

def test_save_and_load_state():
    from researchforge.utils.state import save_state, load_state

    test_data = {"v1": MOCK_V1_RESULT, "v2": MOCK_V2_RESULT}
    with tempfile.TemporaryDirectory() as tmp:
        from pathlib import Path
        state_path = Path(tmp) / ".researchforge" / "last_run.json"
        with patch(
            "researchforge.utils.state._STATE_PATH",
            state_path
        ):
            save_state(test_data)
            assert state_path.exists()
            loaded = load_state()
            assert loaded["v1"]["topic"] == MOCK_V1_RESULT["topic"]
            assert "timestamp" in loaded


def test_load_state_returns_empty_dict_when_missing():
    from researchforge.utils.state import load_state
    from pathlib import Path
    with patch("researchforge.utils.state._STATE_PATH", Path("/nonexistent/path.json")):
        result = load_state()
    assert result == {}
