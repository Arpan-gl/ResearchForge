import json
from pathlib import Path

from researchforge.agents.training import TrainingPlannerAgent


INTENT_HANDOFF = {
    "data": {
        "objective": "Train a support ticket classifier",
        "task_type": "classification",
        "modality": "tabular",
        "labels": ["priority"],
        "evaluation_metric": "f1",
        "constraints": [],
        "gpu_availability": "no",
        "framework_preference": "sklearn",
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

VALIDATION_REPORT = {
    "data": {
        "strategy_name": "tabular_classification_basic",
        "dataset_path": "data/raw.csv",
        "cleaned_dataset_path": "validation/cleaned_dataset.csv",
        "before": {"rows": 8000, "cols": 12, "missing_cells": 10, "duplicate_rows": 20, "class_balance": {"0": 4000, "1": 4000}},
        "after": {"rows": 7800, "cols": 12, "missing_cells": 0, "duplicate_rows": 0, "class_balance": {"0": 3900, "1": 3900}},
        "label_column": "label",
    },
    "provenance": {
        "source": "data/raw.csv",
        "retrieved_at": "2026-07-07T00:00:00+00:00",
        "agent": "validation",
    },
    "confidence": "computed",
}


def test_training_planner_chooses_karpathy_path_for_small_or_cpu_runs():
    agent = TrainingPlannerAgent()
    handoff = agent.create_plan(INTENT_HANDOFF, VALIDATION_REPORT)
    config = handoff["data"]
    assert config["execution_path"] == "training/karpathy_minimal"
    assert config["framework"] == "minimal_loop"
    assert handoff["provenance"]["agent"] == "training_planner"


def test_training_planner_chooses_framework_path_for_large_gpu_text_runs():
    agent = TrainingPlannerAgent()
    intent = json.loads(json.dumps(INTENT_HANDOFF))
    validation = json.loads(json.dumps(VALIDATION_REPORT))
    intent["data"]["gpu_availability"] = "yes"
    intent["data"]["modality"] = "text"
    intent["data"]["task_type"] = "nlp"
    validation["data"]["after"]["rows"] = 200000
    handoff = agent.create_plan(intent, validation)
    config = handoff["data"]
    assert config["execution_path"] == "training/frameworks"
    assert config["framework"] == "transformers"
    assert "model.fit(" not in json.dumps(config)


def test_cli_plan_train_command_writes_config_file():
    intent_path = Path("tests") / ".train_intent_test.json"
    validation_path = Path("tests") / ".train_validation_test.json"
    output_path = Path("tests") / ".train_config_test.json"
    try:
        intent_path.write_text(json.dumps(INTENT_HANDOFF), encoding="utf-8")
        validation_path.write_text(json.dumps(VALIDATION_REPORT), encoding="utf-8")
        from researchforge import cli
        import sys
        old_argv = sys.argv
        sys.argv = ["researchforge", "plan-train", str(intent_path), str(validation_path), "--output", str(output_path)]
        try:
            cli.main()
        finally:
            sys.argv = old_argv

        assert output_path.exists()
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        assert saved["provenance"]["agent"] == "training_planner"
    finally:
        intent_path.unlink(missing_ok=True)
        validation_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
