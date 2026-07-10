import json
from pathlib import Path

import pandas as pd

from researchforge.agents.evaluation import EvaluationAgent
from researchforge.agents.optimization import OptimizationAgent
from researchforge.training.karpathy_minimal import MinimalTrainer


def make_files(base: Path):
    dataset_path = base / "opt_eval.csv"
    df = pd.DataFrame(
        {
            "feat_a": [0.0, 0.2, 1.0, 1.1, 2.0, 2.2],
            "feat_b": [0.1, 0.0, 1.2, 1.0, 1.9, 2.1],
            "label": ["low", "low", "mid", "mid", "high", "high"],
        }
    )
    df.to_csv(dataset_path, index=False)
    config_path = base / "config.json"
    config = {
        "data": {
            "dataset_path": str(dataset_path),
            "label_column": "label",
            "epochs": 4,
            "learning_rate": 0.2,
        },
        "provenance": {"source": str(dataset_path), "retrieved_at": "2026-07-07T00:00:00+00:00", "agent": "training_planner"},
        "confidence": "computed",
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return dataset_path, config_path


def cleanup(base: Path):
    if base.exists():
        for path in sorted(base.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        base.rmdir()


def test_optimization_agent_produces_trial_history_and_best_config():
    base = Path("tests") / ".optimization_case"
    base.mkdir(exist_ok=True)
    try:
        _, config_path = make_files(base)
        handoff = OptimizationAgent().optimize(str(config_path), output_dir=str(base / "artifacts"))
        assert len(handoff["data"]["trial_history"]) == 3
        assert handoff["data"]["best_config"]["final_loss"] == min(t["final_loss"] for t in handoff["data"]["trial_history"])
        assert handoff["provenance"]["agent"] == "optimization"
    finally:
        cleanup(base)


def test_evaluation_agent_computes_metrics_and_regression_flag():
    base = Path("tests") / ".evaluation_case"
    base.mkdir(exist_ok=True)
    try:
        dataset_path, config_path = make_files(base)
        train = MinimalTrainer().run(str(config_path), output_dir=str(base / "minimal"))
        baseline_path = base / "baseline.json"
        baseline_path.write_text(json.dumps({"f1_macro": 0.99}), encoding="utf-8")
        report = EvaluationAgent().evaluate(
            train["data"]["checkpoint_path"],
            str(dataset_path),
            "label",
            baseline_path=str(baseline_path),
            output_path=str(base / "eval_report.json"),
        )
        assert 0.0 <= report["data"]["accuracy"] <= 1.0
        assert 0.0 <= report["data"]["f1_macro"] <= 1.0
        assert report["data"]["regression_detected"] is True
        assert report["provenance"]["agent"] == "evaluation"
    finally:
        cleanup(base)


def test_cli_tune_and_evaluate_commands_run_end_to_end():
    base = Path("tests") / ".opt_eval_cli_case"
    base.mkdir(exist_ok=True)
    try:
        dataset_path, config_path = make_files(base)
        from researchforge import cli
        import sys

        old_argv = sys.argv
        sys.argv = ["researchforge", "tune", str(config_path), "--output-dir", str(base / "opt")]
        try:
            cli.main()
        finally:
            sys.argv = old_argv

        train = MinimalTrainer().run(str(config_path), output_dir=str(base / "minimal"))
        old_argv = sys.argv
        sys.argv = [
            "researchforge",
            "evaluate",
            train["data"]["checkpoint_path"],
            str(dataset_path),
            "--label-column",
            "label",
            "--output",
            str(base / "eval_report.json"),
        ]
        try:
            cli.main()
        finally:
            sys.argv = old_argv

        assert (base / "eval_report.json").exists()
    finally:
        cleanup(base)
