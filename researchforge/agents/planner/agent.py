"""Planner agent implementation."""

import json
from datetime import datetime, timezone

from researchforge.agents.planner.llm import LLMRouter
from researchforge.agents.planner.schema import normalize_intent_payload, validate_intent_payload
from researchforge.config.settings import Settings


class PlannerAgent:
    def __init__(self, settings: Settings | None = None, llm: LLMRouter | None = None):
        self.settings = settings or Settings()
        self.llm = llm or LLMRouter(self.settings)

    def parse_intent(self, user_prompt: str) -> dict:
        prompt = user_prompt.strip()
        if not prompt:
            payload = {
                "objective": "",
                "task_type": "",
                "modality": "",
                "labels": [],
                "evaluation_metric": "",
                "constraints": [],
                "gpu_availability": "unknown",
                "framework_preference": "",
                "expected_output": "",
                "needs_clarification": True,
                "clarification_reason": "No research or training goal was provided.",
            }
            return self._handoff(payload)

        router_prompt = self._build_prompt(prompt)
        payload, provider = self.llm.parse_json(router_prompt)
        intent = normalize_intent_payload(payload, prompt)
        valid, errors = validate_intent_payload(intent)
        if not valid:
            raise ValueError("Intent payload failed schema validation: " + "; ".join(errors))

        handoff = self._handoff(intent)
        handoff["provenance"]["source"] = f"planner:{provider}"
        return handoff

    def save_intent(self, handoff: dict, output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(handoff, handle, indent=2)

    @staticmethod
    def _build_prompt(user_prompt: str) -> str:
        return (
            "Extract structured intent for an AI training operating system. "
            "Return JSON only with these keys: objective, task_type, modality, labels, "
            "evaluation_metric, constraints, gpu_availability, framework_preference, "
            "expected_output, needs_clarification, clarification_reason. "
            "Use gpu_availability as one of yes, no, unknown, limited. "
            "If the prompt is ambiguous or missing key information, set needs_clarification to true "
            "and explain why in clarification_reason instead of guessing.\n\n"
            f"User prompt: {user_prompt}"
        )

    @staticmethod
    def _handoff(intent: dict) -> dict:
        return {
            "data": intent,
            "provenance": {
                "source": "user_prompt",
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "agent": "planner",
            },
            "confidence": "computed",
        }
