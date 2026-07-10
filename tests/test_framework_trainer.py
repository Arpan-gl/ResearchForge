import json
from pathlib import Path

import pandas as pd

from researchforge.training.frameworks import FrameworkTrainer


def make_framework_files(base: Path):
    dataset_path = base / "framework_train.csv"
    df = pd.DataFrame(
        {
            "text": [
                "login issue urgent",
                "password reset needed",
                "invoice request",
                "billing question",
                "server outage critical",
                "service down urgent",
            ],
            "label": ["tech", "tech", "billing", "billing", "incident", "incident"],
        }
    )
    df.to_csv(dataset_path, index=False)

    config_path = base / "framework_config.json"
    config = {
        "data": {
            "dataset_path": str(dataset_path),
            "label_column": "label",
            "framework": "transformers",
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 5e-4,
        },
        "provenance": {"source": str(dataset_path), "retrieved_at": "2026-07-07T00:00:00+00:00", "agent": "training_planner"},
        "confidence": "computed",
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return dataset_path, config_path


def cleanup_tree(base: Path):
    if base.exists():
        for path in sorted(base.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        base.rmdir()


def test_framework_trainer_transformers_run_writes_checkpoint():
    base = Path("tests") / ".framework_train_case"
    output_dir = base / "artifacts"
    base.mkdir(exist_ok=True)
    try:
        _, config_path = make_framework_files(base)
        trainer = FrameworkTrainer()
        handoff = trainer.run(str(config_path), output_dir=str(output_dir))

        assert Path(handoff["data"]["checkpoint_path"]).exists()
        assert Path(handoff["data"]["metrics_path"]).exists()
        assert handoff["data"]["framework"] == "transformers"
        assert handoff["provenance"]["agent"] == "training_frameworks"
    finally:
        cleanup_tree(base)


def test_cli_train_framework_command_runs_end_to_end():
    base = Path("tests") / ".framework_cli_case"
    output_dir = base / "artifacts"
    base.mkdir(exist_ok=True)
    try:
        _, config_path = make_framework_files(base)
        from researchforge import cli
        import sys

        old_argv = sys.argv
        sys.argv = ["researchforge", "train-framework", str(config_path), "--output-dir", str(output_dir)]
        try:
            cli.main()
        finally:
            sys.argv = old_argv

        assert (output_dir / "checkpoint").exists()
        assert (output_dir / "metrics.json").exists()
    finally:
        cleanup_tree(base)
