import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from researchforge.agents.dataset import DatasetAgent


INTENT_HANDOFF = {
    "data": {
        "objective": "Train a support ticket classifier",
        "task_type": "classification",
        "modality": "text",
        "labels": ["priority"],
        "evaluation_metric": "f1",
        "constraints": [],
        "gpu_availability": "yes",
        "framework_preference": "transformers",
        "expected_output": "config",
        "needs_clarification": False,
        "clarification_reason": "",
    },
    "provenance": {
        "source": "planner:fake",
        "retrieved_at": "2026-07-07T00:00:00+00:00",
        "agent": "planner",
    },
    "confidence": "computed",
}


def make_backend():
    backend = MagicMock()
    backend._search_kaggle.return_value = [
        {
            "name": "good-dataset",
            "title": "Support ticket priority dataset",
            "size_mb": 120,
            "downloads": 25000,
            "source": "kaggle",
            "url": "https://kaggle.com/datasets/good-dataset",
        }
    ]
    backend._search_huggingface.return_value = [
        {
            "name": "weak-dataset",
            "title": "generic text",
            "size_mb": 1,
            "downloads": 0,
            "source": "huggingface",
            "url": "https://huggingface.co/datasets/weak-dataset",
        }
    ]
    backend._topic_similarity.side_effect = lambda topic, text: 0.8 if "Support ticket" in text else 0.1
    backend._accessibility_score.side_effect = lambda ds: 0.9 if ds["downloads"] > 0 else 0.4
    backend._metadata_completeness_score.side_effect = lambda ds: 1.0 if ds["downloads"] > 0 else 0.6
    backend._score_dataset.side_effect = lambda ds, topic, findings: {
        **ds,
        "score": 0.87 if ds["downloads"] > 0 else 0.22,
        "selection_confidence": "high" if ds["downloads"] > 0 else "low",
        "selection_rationale": ["deterministic score"],
        "why_recommended": ["deterministic score"],
        "risks": [],
        "problem_type": findings["problem_type"],
        "ml_tasks": ["XGBoost baseline"],
    }
    return backend


def test_dataset_agent_requires_provenance():
    agent = DatasetAgent(dataset_backend=make_backend())
    with pytest.raises(ValueError, match="missing provenance"):
        agent.discover_and_rank({"data": INTENT_HANDOFF["data"]})


def test_dataset_agent_returns_ranked_datasets_with_score_trace():
    agent = DatasetAgent(dataset_backend=make_backend())
    handoff = agent.discover_and_rank(INTENT_HANDOFF)
    datasets = handoff["data"]["datasets"]

    assert datasets[0]["score"] >= datasets[1]["score"]
    assert "score_trace" in datasets[0]
    assert datasets[0]["score_trace"]["final_score"] == datasets[0]["score"]
    assert datasets[0]["provenance"]["agent"] == "dataset"


def test_cli_datasets_command_writes_output_file():
    intent_path = Path("tests") / ".dataset_intent_test.json"
    output_path = Path("tests") / ".dataset_output_test.json"
    try:
        intent_path.write_text(json.dumps(INTENT_HANDOFF), encoding="utf-8")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("researchforge.agents.dataset.DatasetAgent.discover_and_rank", lambda self, payload: {
                "data": {"datasets": []},
                "provenance": {"source": "planner:fake", "retrieved_at": "now", "agent": "dataset"},
                "confidence": "computed",
            })
            mp.setattr("sys.argv", ["researchforge", "datasets", str(intent_path), "--output", str(output_path)])
            from researchforge import cli

            cli.main()

        assert output_path.exists()
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        assert saved["provenance"]["agent"] == "dataset"
    finally:
        intent_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
