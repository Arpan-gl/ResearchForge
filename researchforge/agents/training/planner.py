"""Deterministic training planner and config generator."""

import json
from pathlib import Path


class TrainingPlannerAgent:
    def create_plan(self, intent_handoff: dict, validation_report: dict) -> dict:
        self._validate_handoff(intent_handoff, "planner")
        self._validate_handoff(validation_report, "validation")

        intent = intent_handoff.get("data") or {}
        validation = validation_report.get("data") or {}
        dataset_rows = int((validation.get("after") or {}).get("rows", 0))
        task_type = intent.get("task_type", "unknown") or "unknown"
        modality = intent.get("modality", "") or ""
        gpu = intent.get("gpu_availability", "unknown") or "unknown"

        execution_path = self._select_execution_path(task_type, modality, gpu, dataset_rows)
        framework = self._select_framework(task_type, modality, execution_path)
        config = {
            "objective": intent.get("objective", ""),
            "task_type": task_type,
            "modality": modality,
            "dataset_path": validation.get("cleaned_dataset_path", validation.get("dataset_path", "")),
            "label_column": validation.get("label_column", ""),
            "dataset_rows": dataset_rows,
            "validation_strategy": validation.get("strategy_name", ""),
            "execution_path": execution_path,
            "framework": framework,
            "batch_size": self._batch_size(gpu, modality),
            "epochs": self._epochs(dataset_rows),
            "learning_rate": self._learning_rate(task_type, modality),
        }
        return {
            "data": config,
            "provenance": {
                "source": validation_report["provenance"]["source"],
                "retrieved_at": validation_report["provenance"]["retrieved_at"],
                "agent": "training_planner",
            },
            "confidence": "computed",
        }

    def save_config(self, handoff: dict, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(handoff, indent=2), encoding="utf-8")

    @staticmethod
    def _validate_handoff(handoff: dict, expected_agent: str) -> None:
        provenance = handoff.get("provenance") or {}
        if not {"source", "retrieved_at", "agent"}.issubset(provenance):
            raise ValueError(f"{expected_agent} input is missing provenance.")

    @staticmethod
    def _select_execution_path(task_type: str, modality: str, gpu: str, dataset_rows: int) -> str:
        if gpu == "no" or dataset_rows <= 10000:
            return "training/karpathy_minimal"
        if modality == "text" or task_type == "nlp":
            return "training/frameworks"
        return "training/frameworks"

    @staticmethod
    def _select_framework(task_type: str, modality: str, execution_path: str) -> str:
        if execution_path == "training/karpathy_minimal":
            return "minimal_loop"
        if modality == "text" or task_type == "nlp":
            return "transformers"
        if task_type == "graph":
            return "pytorch_geometric"
        return "lightning"

    @staticmethod
    def _batch_size(gpu: str, modality: str) -> int:
        if gpu == "no":
            return 8
        if modality == "text":
            return 16
        return 32

    @staticmethod
    def _epochs(dataset_rows: int) -> int:
        if dataset_rows <= 5000:
            return 5
        if dataset_rows <= 50000:
            return 3
        return 2

    @staticmethod
    def _learning_rate(task_type: str, modality: str) -> float:
        if modality == "text" or task_type == "nlp":
            return 2e-5
        if task_type == "graph":
            return 1e-3
        return 3e-4
