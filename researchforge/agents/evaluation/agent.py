"""Deterministic evaluation agent."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


class EvaluationAgent:
    def evaluate(self, checkpoint_path: str, dataset_path: str, label_column: str, baseline_path: str | None = None, output_path: str = "evaluation/eval_report.json") -> dict:
        weights, bias, feature_names = self._load_checkpoint(checkpoint_path)
        X, y = self._load_dataset(dataset_path, label_column, feature_names)
        logits = X @ weights + bias
        predictions = logits.argmax(axis=1)

        accuracy = float((predictions == y).mean())
        f1_macro = self._f1_macro(y, predictions)
        baseline = self._load_baseline(baseline_path)
        regression = None
        if baseline is not None:
            regression = f1_macro < baseline.get("f1_macro", 0.0)

        report = {
            "data": {
                "accuracy": accuracy,
                "f1_macro": f1_macro,
                "baseline": baseline,
                "regression_detected": regression,
            },
            "provenance": {
                "source": dataset_path,
                "retrieved_at": timestamp(),
                "agent": "evaluation",
            },
            "confidence": "computed",
        }
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    @staticmethod
    def _load_checkpoint(checkpoint_path: str):
        payload = np.load(checkpoint_path, allow_pickle=True)
        return payload["weights"], payload["bias"], [str(item) for item in payload["feature_names"]]

    @staticmethod
    def _load_dataset(dataset_path: str, label_column: str, feature_names: list[str]):
        df = pd.read_csv(dataset_path)
        y_raw = df[label_column].astype(str)
        labels = {label: idx for idx, label in enumerate(sorted(y_raw.unique()))}
        y = np.array([labels[item] for item in y_raw], dtype=int)
        X_df = pd.get_dummies(df.drop(columns=[label_column]), drop_first=False)
        X_df = X_df.reindex(columns=feature_names, fill_value=0.0)
        X = X_df.to_numpy(dtype=float)
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        X = (X - mean) / std
        return X, y

    @staticmethod
    def _f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        scores = []
        for label in labels:
            tp = int(((y_true == label) & (y_pred == label)).sum())
            fp = int(((y_true != label) & (y_pred == label)).sum())
            fn = int(((y_true == label) & (y_pred != label)).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            score = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
            scores.append(score)
        return float(sum(scores) / len(scores)) if scores else 0.0

    @staticmethod
    def _load_baseline(baseline_path: str | None):
        if not baseline_path:
            return None
        return json.loads(Path(baseline_path).read_text(encoding="utf-8"))


def timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
