"""Deterministic dataset validation agent."""

import json
from pathlib import Path

import pandas as pd

VALIDATION_STRATEGIES = {
    "tabular_classification_basic": "Drop duplicate rows and impute missing values for classification datasets.",
    "tabular_regression_basic": "Drop duplicate rows and impute missing values for regression datasets.",
    "text_classification_basic": "Drop duplicate rows, fill missing text fields, and impute labels.",
}


class ValidationAgent:
    def validate_dataset(
        self,
        dataset_path: str,
        strategy_name: str,
        label_column: str | None = None,
        output_dir: str = "validation",
    ) -> dict:
        if strategy_name not in VALIDATION_STRATEGIES:
            raise ValueError("Unknown validation strategy.")

        df = pd.read_csv(dataset_path)
        before = self._stats(df, label_column)
        cleaned = self._apply_strategy(df.copy(), strategy_name, label_column)
        after = self._stats(cleaned, label_column)

        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        cleaned_path = output_root / "cleaned_dataset.csv"
        report_path = output_root / "validation_report.json"
        cleaned.to_csv(cleaned_path, index=False)

        report = {
            "data": {
                "strategy_name": strategy_name,
                "dataset_path": dataset_path,
                "cleaned_dataset_path": str(cleaned_path),
                "before": before,
                "after": after,
                "label_column": label_column or "",
            },
            "provenance": {
                "source": dataset_path,
                "retrieved_at": timestamp(),
                "agent": "validation",
            },
            "confidence": "computed",
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def _apply_strategy(self, df: pd.DataFrame, strategy_name: str, label_column: str | None) -> pd.DataFrame:
        df = df.drop_duplicates().reset_index(drop=True)

        for column in df.columns:
            if df[column].isnull().any():
                if pd.api.types.is_numeric_dtype(df[column]):
                    df[column] = df[column].fillna(df[column].median())
                else:
                    if column == label_column:
                        mode = df[column].mode(dropna=True)
                        fill_value = mode.iloc[0] if not mode.empty else "unknown"
                        df[column] = df[column].fillna(fill_value)
                    elif strategy_name == "text_classification_basic":
                        df[column] = df[column].fillna("")
                    else:
                        mode = df[column].mode(dropna=True)
                        fill_value = mode.iloc[0] if not mode.empty else "unknown"
                        df[column] = df[column].fillna(fill_value)
        return df

    def _stats(self, df: pd.DataFrame, label_column: str | None) -> dict:
        rows, cols = df.shape
        missing_cells = int(df.isnull().sum().sum())
        duplicate_rows = int(df.duplicated().sum())
        class_balance = {}
        if label_column and label_column in df.columns:
            counts = df[label_column].value_counts(dropna=False)
            class_balance = {str(key): int(value) for key, value in counts.items()}
        return {
            "rows": int(rows),
            "cols": int(cols),
            "missing_cells": missing_cells,
            "duplicate_rows": duplicate_rows,
            "class_balance": class_balance,
        }


def timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
