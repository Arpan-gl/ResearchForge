"""Dataset ranking agent with traceable scoring."""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from researchforge.stages.v2_datasets import V2Datasets


class DatasetAgent:
    def __init__(self, dataset_backend: V2Datasets | None = None):
        self.dataset_backend = dataset_backend or V2Datasets()

    def discover_and_rank(self, intent_handoff: dict) -> dict:
        provenance = intent_handoff.get("provenance") or {}
        if not {"source", "retrieved_at", "agent"}.issubset(provenance):
            raise ValueError("Planner input is missing provenance.")

        intent = intent_handoff.get("data") or {}
        topic = intent.get("objective", "")
        problem_type = intent.get("task_type", "unknown") or "unknown"
        findings = {"problem_type": problem_type}

        candidates = []
        candidates.extend(self.dataset_backend._search_kaggle(topic))
        candidates.extend(self.dataset_backend._search_huggingface(topic))

        ranked = [self._score_with_trace(candidate, topic, findings) for candidate in candidates]
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return {
            "data": {
                "datasets": ranked,
            },
            "provenance": {
                "source": provenance["source"],
                "retrieved_at": provenance["retrieved_at"],
                "agent": "dataset",
            },
            "confidence": "computed",
        }

    def save_ranking(self, handoff: dict, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(handoff, indent=2), encoding="utf-8")

    def _score_with_trace(self, candidate: dict, topic: str, findings: dict) -> dict:
        size_score = min(1.0, math.log10(max(candidate.get("size_mb", 1) * 1000, 1)) / 6)
        popularity_score = min(1.0, math.log10(max(candidate.get("downloads", 1), 1)) / 5)
        topic_score = self.dataset_backend._topic_similarity(topic, candidate.get("title", ""))
        accessibility_score = self.dataset_backend._accessibility_score(candidate)
        metadata_score = self.dataset_backend._metadata_completeness_score(candidate)

        raw_score = (
            0.28 * topic_score
            + 0.20 * popularity_score
            + 0.12 * size_score
            + 0.22 * accessibility_score
            + 0.18 * metadata_score
        )
        penalty = 0.0
        if candidate.get("downloads", 0) <= 0:
            penalty += 0.12
        if topic_score < 0.2:
            penalty += 0.10
        if len((candidate.get("title") or candidate.get("name") or "").strip()) < 6:
            penalty += 0.06

        scored = self.dataset_backend._score_dataset(candidate, topic, findings)
        scored["score_trace"] = {
            "topic_score": round(topic_score, 4),
            "popularity_score": round(popularity_score, 4),
            "size_score": round(size_score, 4),
            "accessibility_score": round(accessibility_score, 4),
            "metadata_score": round(metadata_score, 4),
            "raw_score": round(raw_score, 4),
            "penalty": round(penalty, 4),
            "final_score": scored["score"],
        }
        scored["provenance"] = {
            "source": candidate.get("url", ""),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "agent": "dataset",
        }
        return scored
