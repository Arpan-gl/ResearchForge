import json
from pathlib import Path

from researchforge.agents.reporting import ReportingAgent


INTENT = {
    "data": {"objective": "Train a support ticket classifier", "task_type": "classification", "modality": "text", "expected_output": "config"},
    "provenance": {"source": "user", "retrieved_at": "2026-07-07T00:00:00+00:00", "agent": "planner"},
    "confidence": "computed",
}
RESEARCH = {
    "data": {
        "evidence": [
            {"title": "Ticket routing paper", "url": "http://arxiv.org/abs/1234", "provenance": {"source": "http://arxiv.org/abs/1234", "retrieved_at": "2026-07-07T00:00:00+00:00", "agent": "research"}},
            {"title": "Reference repo", "url": "https://github.com/example/repo", "provenance": {"source": "https://github.com/example/repo", "retrieved_at": "2026-07-07T00:00:00+00:00", "agent": "research"}},
        ]
    },
    "provenance": {"source": "planner:fake", "retrieved_at": "2026-07-07T00:00:00+00:00", "agent": "research"},
    "confidence": "computed",
}
DATASETS = {
    "data": {"datasets": [{"title": "Support ticket dataset", "score_trace": {"final_score": 0.87}}]},
    "provenance": {"source": "planner:fake", "retrieved_at": "2026-07-07T00:00:00+00:00", "agent": "dataset"},
    "confidence": "computed",
}
VALIDATION = {
    "data": {"strategy_name": "text_classification_basic", "before": {"missing_cells": 4}, "after": {"missing_cells": 0}},
    "provenance": {"source": "data.csv", "retrieved_at": "2026-07-07T00:00:00+00:00", "agent": "validation"},
    "confidence": "computed",
}
TRAINING = {
    "data": {"execution_path": "training/frameworks", "framework": "transformers", "epochs": 1},
    "provenance": {"source": "data.csv", "retrieved_at": "2026-07-07T00:00:00+00:00", "agent": "training_planner"},
    "confidence": "computed",
}
EVALUATION = {
    "data": {"accuracy": 0.9, "f1_macro": 0.88, "baseline": {"f1_macro": 0.85}, "regression_detected": False},
    "provenance": {"source": "data.csv", "retrieved_at": "2026-07-07T00:00:00+00:00", "agent": "evaluation"},
    "confidence": "computed",
}


def cleanup(paths):
    for path in paths:
        if path.exists():
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                for child in sorted(path.rglob("*"), reverse=True):
                    if child.is_file():
                        child.unlink()
                    else:
                        child.rmdir()
                path.rmdir()


def test_reporting_agent_builds_cited_report_and_proposals():
    report_path = Path("tests") / ".report_case.md"
    proposals_path = Path("tests") / ".proposals_case.json"
    try:
        agent = ReportingAgent()
        report = agent.build_report(
            {
                "intent": INTENT,
                "research": RESEARCH,
                "datasets": DATASETS,
                "validation": VALIDATION,
                "training": TRAINING,
                "evaluation": EVALUATION,
            },
            output_path=str(report_path),
        )
        proposals = agent.propose_experiments(RESEARCH, EVALUATION, output_path=str(proposals_path))

        assert report_path.exists()
        report_text = report_path.read_text(encoding="utf-8")
        assert "http://arxiv.org/abs/1234" in report_text
        assert "0.88" in report_text
        assert proposals_path.exists()
        assert proposals["data"]["proposals"][0]["citations"]
    finally:
        cleanup([report_path, proposals_path])


def test_cli_report_and_autoresearch_commands_run_end_to_end():
    base = Path("tests") / ".reporting_cli_case"
    base.mkdir(exist_ok=True)
    paths = {
        "intent": base / "intent.json",
        "research": base / "research.json",
        "datasets": base / "datasets.json",
        "validation": base / "validation.json",
        "training": base / "training.json",
        "evaluation": base / "evaluation.json",
        "report": base / "report.md",
        "proposals": base / "proposals.json",
    }
    try:
        paths["intent"].write_text(json.dumps(INTENT), encoding="utf-8")
        paths["research"].write_text(json.dumps(RESEARCH), encoding="utf-8")
        paths["datasets"].write_text(json.dumps(DATASETS), encoding="utf-8")
        paths["validation"].write_text(json.dumps(VALIDATION), encoding="utf-8")
        paths["training"].write_text(json.dumps(TRAINING), encoding="utf-8")
        paths["evaluation"].write_text(json.dumps(EVALUATION), encoding="utf-8")

        from researchforge import cli
        import sys

        old_argv = sys.argv
        sys.argv = [
            "researchforge",
            "report",
            str(paths["intent"]),
            str(paths["research"]),
            str(paths["datasets"]),
            str(paths["validation"]),
            str(paths["training"]),
            str(paths["evaluation"]),
            "--output",
            str(paths["report"]),
        ]
        try:
            cli.main()
        finally:
            sys.argv = old_argv

        old_argv = sys.argv
        sys.argv = ["researchforge", "autoresearch", str(paths["research"]), str(paths["evaluation"]), "--output", str(paths["proposals"])]
        try:
            cli.main()
        finally:
            sys.argv = old_argv

        assert paths["report"].exists()
        assert paths["proposals"].exists()
    finally:
        cleanup([base])
