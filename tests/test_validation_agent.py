import json
from pathlib import Path

import pandas as pd
import pytest

from researchforge.agents.validation import VALIDATION_STRATEGIES, ValidationAgent


def make_dataset(path: Path) -> None:
    df = pd.DataFrame(
        {
            "text": ["hello", None, "hello", "world"],
            "label": ["bug", "bug", "bug", None],
            "score": [1.0, None, 1.0, 2.5],
        }
    )
    df.to_csv(path, index=False)


def test_strategy_menu_contains_expected_entries():
    assert "tabular_classification_basic" in VALIDATION_STRATEGIES
    assert "text_classification_basic" in VALIDATION_STRATEGIES


def test_validation_agent_rejects_unknown_strategy():
    agent = ValidationAgent()
    with pytest.raises(ValueError, match="Unknown validation strategy"):
        agent.validate_dataset("missing.csv", "not_a_strategy")


def test_validation_agent_reports_before_and_after_stats():
    dataset_path = Path("tests") / ".validation_input_test.csv"
    output_dir = Path("tests") / ".validation_output"
    try:
        make_dataset(dataset_path)
        agent = ValidationAgent()
        report = agent.validate_dataset(
            str(dataset_path),
            "text_classification_basic",
            label_column="label",
            output_dir=str(output_dir),
        )

        before = report["data"]["before"]
        after = report["data"]["after"]
        assert before["missing_cells"] > after["missing_cells"]
        assert before["duplicate_rows"] > after["duplicate_rows"]
        assert report["provenance"]["agent"] == "validation"
        assert Path(report["data"]["cleaned_dataset_path"]).exists()
    finally:
        if dataset_path.exists():
            dataset_path.unlink()
        if output_dir.exists():
            for child in output_dir.iterdir():
                child.unlink()
            output_dir.rmdir()


def test_cli_validate_command_writes_report():
    dataset_path = Path("tests") / ".validation_cli_input.csv"
    output_dir = Path("tests") / ".validation_cli_output"
    try:
        make_dataset(dataset_path)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "sys.argv",
                [
                    "researchforge",
                    "validate",
                    str(dataset_path),
                    "--strategy",
                    "text_classification_basic",
                    "--label-column",
                    "label",
                    "--output-dir",
                    str(output_dir),
                ],
            )
            from researchforge import cli

            cli.main()

        report_path = output_dir / "validation_report.json"
        assert report_path.exists()
        saved = json.loads(report_path.read_text(encoding="utf-8"))
        assert saved["provenance"]["agent"] == "validation"
    finally:
        if dataset_path.exists():
            dataset_path.unlink()
        if output_dir.exists():
            for child in output_dir.iterdir():
                child.unlink()
            output_dir.rmdir()
