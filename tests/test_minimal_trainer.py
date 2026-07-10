import json
from pathlib import Path

import pandas as pd

from researchforge.training.karpathy_minimal import MinimalTrainer


def make_training_files(base: Path):
    dataset_path = base / "minimal_train.csv"
    df = pd.DataFrame(
        {
            "feat_a": [0.0, 0.2, 1.0, 1.1, 2.0, 2.2],
            "feat_b": [0.1, 0.0, 1.2, 1.0, 1.9, 2.1],
            "label": ["low", "low", "mid", "mid", "high", "high"],
        }
    )
    df.to_csv(dataset_path, index=False)

    config_path = base / "minimal_config.json"
    config = {
        "data": {
            "dataset_path": str(dataset_path),
            "label_column": "label",
            "epochs": 8,
            "learning_rate": 0.2,
        },
        "provenance": {"source": str(dataset_path), "retrieved_at": "2026-07-07T00:00:00+00:00", "agent": "training_planner"},
        "confidence": "computed",
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return dataset_path, config_path


def test_minimal_trainer_completes_and_writes_checkpoint():
    base = Path("tests") / ".minimal_train_case"
    output_dir = base / "artifacts"
    base.mkdir(exist_ok=True)
    try:
        _, config_path = make_training_files(base)
        trainer = MinimalTrainer()
        handoff = trainer.run(str(config_path), output_dir=str(output_dir))

        assert Path(handoff["data"]["checkpoint_path"]).exists()
        assert Path(handoff["data"]["metrics_path"]).exists()
        assert handoff["data"]["final_loss"] > 0
        assert handoff["provenance"]["agent"] == "training_minimal"
    finally:
        if base.exists():
            for path in sorted(base.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            base.rmdir()


def test_cli_train_minimal_command_runs_end_to_end():
    base = Path("tests") / ".minimal_cli_case"
    output_dir = base / "artifacts"
    base.mkdir(exist_ok=True)
    try:
        _, config_path = make_training_files(base)
        from researchforge import cli
        import sys

        old_argv = sys.argv
        sys.argv = ["researchforge", "train-minimal", str(config_path), "--output-dir", str(output_dir)]
        try:
            cli.main()
        finally:
            sys.argv = old_argv

        assert (output_dir / "minimal_checkpoint.npz").exists()
        assert (output_dir / "metrics.json").exists()
    finally:
        if base.exists():
            for path in sorted(base.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            base.rmdir()
