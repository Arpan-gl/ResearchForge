"""
Integration tests for the ResearchForge pipeline.
All external calls (Ollama, arXiv, DuckDuckGo, Kaggle, HuggingFace) are mocked.
Tests verify that data flows correctly: V1 → V2 → V3 and notebook is created.
"""

import json
import os
import tempfile
import pytest
import nbformat
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
    """Generated notebook should have a title + 7 code sections."""
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
            # 1 markdown title + 7 code sections = 8 cells
            assert len(nb.cells) == 8
            assert nb.cells[0].cell_type == "markdown"
            code_cells = [c for c in nb.cells if c.cell_type == "code"]
            assert len(code_cells) == 7
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


def test_pipeline_stops_early_when_no_dataset_found():
    from researchforge.core.pipeline import Pipeline

    pipeline = Pipeline.__new__(Pipeline)
    pipeline.v1 = MagicMock()
    pipeline.v2 = MagicMock()
    pipeline.v3 = MagicMock()
    pipeline.auto = MagicMock()
    pipeline._print_summary = MagicMock()

    pipeline.v1.run.return_value = MOCK_V1_RESULT
    pipeline.v2.discover_and_score.return_value = {
        "name": "No dataset found",
        "score": 0.0,
        "source": "none",
        "risks": ["No matching public datasets — provide your own with --dataset"],
    }

    with patch("researchforge.core.pipeline.Display"):
        with patch("researchforge.core.pipeline.save_state") as save_state:
            result = pipeline.run("test topic", skip_training=False, budget=5)

    assert result["status"] == "dataset_unavailable"
    pipeline.v3.generate.assert_not_called()
    pipeline.auto.run.assert_not_called()
    save_state.assert_called_once()


def test_cli_run_passes_budget_to_pipeline():
    from researchforge import cli

    fake_pipeline = MagicMock()
    with patch("researchforge.core.pipeline.Pipeline", return_value=fake_pipeline):
        with patch("sys.argv", ["researchforge", "run", "topic", "--budget", "7"]):
            cli.main()

    kwargs = fake_pipeline.run.call_args.kwargs
    assert kwargs["budget"] == 7


def test_v3_data_loading_cell_uses_candidates_and_target_fallback():
    import nbformat
    from researchforge.stages.v3_notebook import V3Notebook

    with tempfile.TemporaryDirectory() as tmp:
        original_dir = os.getcwd()
        os.chdir(tmp)
        try:
            v3 = V3Notebook.__new__(V3Notebook)
            v3.settings = MagicMock()
            v2 = dict(MOCK_V2_RESULT)
            v2["path"] = "custom/path/data.csv"
            result = v3.generate(
                topic="test topic",
                v1_findings=MOCK_V1_RESULT,
                v2_dataset=v2,
            )
            with open(result["notebook_path"], encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)
            load_cell = [c for c in nb.cells if c.cell_type == "code"][0].source
            preprocess_cell = [c for c in nb.cells if c.cell_type == "code"][2].source

            assert "DATASET_CANDIDATES" in load_cell
            assert "RF_DATASET_PATH" in load_cell
            assert "_maybe_download_kaggle_dataset" in load_cell
            assert "TARGET_COL" in load_cell
            assert "X = df.drop(columns=[TARGET_COL])" in preprocess_cell
            assert result["dataset_candidates"][0] == "custom/path/data.csv"
        finally:
            os.chdir(original_dir)


def test_autoresearch_extract_error_snippet_keeps_deeper_traceback_context():
    from researchforge.stages.autoresearch import Autoresearch

    auto = Autoresearch.__new__(Autoresearch)
    long_trace = "\n".join([
        "Traceback (most recent call last):",
        "  File 'launcher.py', line 1, in <module>",
        "    main()",
        "  File 'app.py', line 10, in main",
        "    run()",
        "  File 'runner.py', line 50, in run",
        "    step()",
        "  File 'step.py', line 99, in step",
        "    load_data()",
        "  File 'data.py', line 5, in load_data",
        "    raise FileNotFoundError('missing data file')",
        "FileNotFoundError: missing data file",
    ])
    snippet = auto._extract_error_snippet(long_trace, max_lines=25)
    assert "FileNotFoundError: missing data file" in snippet


def test_autoresearch_parse_suggestion_response_strict_json_and_errors():
    from researchforge.stages.autoresearch import Autoresearch

    auto = Autoresearch.__new__(Autoresearch)
    valid, reason = auto._parse_suggestion_response(
        '{"description":"tune model","target_section":"training","code_change":"a → b","expected_gain":"2%"}'
    )
    assert reason == "ok"
    assert valid["target_section"] == "training"

    invalid, reason = auto._parse_suggestion_response("not json")
    assert invalid is None
    assert reason == "malformed_json"

    missing, reason = auto._parse_suggestion_response('{"description":"x"}')
    assert missing is None
    assert reason.startswith("missing_keys:")


def test_autoresearch_uses_fallback_for_reserved_budget_and_parse_failures():
    from researchforge.stages.autoresearch import Autoresearch

    auto = Autoresearch.__new__(Autoresearch)

    suggestion, source, reason = auto._next_suggestion(
        index=0,
        fallback_budget=2,
        history=[],
        metric="F1-macro",
        best_score=0.7,
    )
    assert source == "fallback"
    assert reason == "ok"
    assert "description" in suggestion

    with patch.object(auto, "_suggest_modification", return_value=(None, "empty_response")):
        suggestion, source, reason = auto._next_suggestion(
            index=3,
            fallback_budget=1,
            history=[],
            metric="F1-macro",
            best_score=0.7,
        )
    assert source == "fallback"
    assert reason == "empty_response"
    assert suggestion["target_section"] in {"model", "training", "preprocessing", "features"}


def test_autoresearch_resolve_kernel_name_prefers_python3():
    from researchforge.stages.autoresearch import Autoresearch

    auto = Autoresearch.__new__(Autoresearch)
    kernelspecs = {
        "python3": {"resource_dir": "x", "spec": {"display_name": "Python 3"}},
        "ir": {"resource_dir": "y", "spec": {"display_name": "R"}},
    }
    assert auto._resolve_kernel_name(kernelspecs) == "python3"


def test_autoresearch_preflight_fails_when_jupyter_missing():
    from researchforge.stages.autoresearch import Autoresearch

    with tempfile.TemporaryDirectory() as tmp:
        nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print('ok')")])
        nb_path = os.path.join(tmp, "test.ipynb")
        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        auto = Autoresearch.__new__(Autoresearch)
        auto.kernel_name = "python3"
        with patch("researchforge.stages.autoresearch.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="Jupyter executable not found"):
                auto._preflight_checks(nb_path)
