"""Reporting agent and evidence-backed autoresearch proposals."""

import json
from pathlib import Path


class ReportingAgent:
    def build_report(self, inputs: dict, output_path: str = "reports/report.md") -> dict:
        required = ["intent", "research", "datasets", "validation", "training", "evaluation"]
        for key in required:
            if key not in inputs:
                raise ValueError(f"Missing report input: {key}")

        report_text = self._render_report(inputs)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report_text, encoding="utf-8")
        return {
            "data": {
                "report_path": str(path),
                "report_markdown": report_text,
            },
            "provenance": {
                "source": inputs["evaluation"]["provenance"]["source"],
                "retrieved_at": inputs["evaluation"]["provenance"]["retrieved_at"],
                "agent": "reporting",
            },
            "confidence": "computed",
        }

    def propose_experiments(self, research_handoff: dict, evaluation_handoff: dict, output_path: str = "reports/proposals.json") -> dict:
        evidence = ((research_handoff.get("data") or {}).get("evidence")) or []
        eval_data = evaluation_handoff.get("data") or {}
        proposals = []
        for item in evidence[:2]:
            proposals.append(
                {
                    "description": f"Reproduce or adapt ideas from {item.get('title', 'evidence item')}.",
                    "motivation": f"Evidence source: {item.get('url', '')}",
                    "expected_metric_target": max(eval_data.get("f1_macro", 0.0), eval_data.get("accuracy", 0.0)),
                    "citations": [item.get("url", "")],
                }
            )
        payload = {
            "data": {
                "proposals": proposals,
            },
            "provenance": {
                "source": evaluation_handoff["provenance"]["source"],
                "retrieved_at": evaluation_handoff["provenance"]["retrieved_at"],
                "agent": "reporting",
            },
            "confidence": "computed",
        }
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    @staticmethod
    def _render_report(inputs: dict) -> str:
        intent = inputs["intent"]["data"]
        research = inputs["research"]["data"]
        datasets = inputs["datasets"]["data"]
        validation = inputs["validation"]["data"]
        training = inputs["training"]["data"]
        evaluation = inputs["evaluation"]["data"]

        top_dataset = (datasets.get("datasets") or [{}])[0]
        top_evidence = (research.get("evidence") or [{}])[:2]
        evidence_lines = "\n".join(
            f"- {item.get('title', 'Unknown')} ({item.get('url', '')})"
            for item in top_evidence
        ) or "- None"

        return "\n".join(
            [
                f"# ResearchForge Report: {intent.get('objective', '')}",
                "",
                "## Intent",
                f"- Task type: {intent.get('task_type', '')}",
                f"- Modality: {intent.get('modality', '')}",
                f"- Expected output: {intent.get('expected_output', '')}",
                "",
                "## Evidence",
                evidence_lines,
                "",
                "## Dataset Ranking",
                f"- Top dataset: {top_dataset.get('title', top_dataset.get('name', ''))}",
                f"- Score trace: {json.dumps(top_dataset.get('score_trace', {}), sort_keys=True)}",
                "",
                "## Validation",
                f"- Strategy: {validation.get('strategy_name', '')}",
                f"- Before stats: {json.dumps(validation.get('before', {}), sort_keys=True)}",
                f"- After stats: {json.dumps(validation.get('after', {}), sort_keys=True)}",
                "",
                "## Training",
                f"- Execution path: {training.get('execution_path', training.get('framework', ''))}",
                f"- Config: {json.dumps(training, sort_keys=True)}",
                "",
                "## Evaluation",
                f"- Accuracy: {evaluation.get('accuracy', '')}",
                f"- F1 macro: {evaluation.get('f1_macro', '')}",
                f"- Baseline comparison: {json.dumps(evaluation.get('baseline', {}), sort_keys=True)}",
            ]
        )
