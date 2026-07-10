"""Planner schema helpers."""

from copy import deepcopy

INTENT_TEMPLATE = {
    "objective": "",
    "task_type": "",
    "modality": "",
    "labels": [],
    "evaluation_metric": "",
    "constraints": [],
    "gpu_availability": "unknown",
    "framework_preference": "",
    "expected_output": "",
    "needs_clarification": False,
    "clarification_reason": "",
}


def default_intent() -> dict:
    return deepcopy(INTENT_TEMPLATE)


def normalize_intent_payload(payload: dict, user_prompt: str) -> dict:
    intent = default_intent()
    if isinstance(payload, dict):
        intent.update(payload)

    intent["objective"] = str(intent.get("objective") or user_prompt).strip()
    intent["task_type"] = str(intent.get("task_type") or "").strip()
    intent["modality"] = str(intent.get("modality") or "").strip()
    intent["evaluation_metric"] = str(intent.get("evaluation_metric") or "").strip()
    intent["framework_preference"] = str(intent.get("framework_preference") or "").strip()
    intent["expected_output"] = str(intent.get("expected_output") or "").strip()
    intent["clarification_reason"] = str(intent.get("clarification_reason") or "").strip()
    intent["needs_clarification"] = bool(intent.get("needs_clarification", False))

    labels = intent.get("labels")
    if isinstance(labels, list):
        intent["labels"] = [str(item).strip() for item in labels if str(item).strip()]
    elif labels:
        intent["labels"] = [str(labels).strip()]
    else:
        intent["labels"] = []

    constraints = intent.get("constraints")
    if isinstance(constraints, list):
        intent["constraints"] = [str(item).strip() for item in constraints if str(item).strip()]
    elif constraints:
        intent["constraints"] = [str(constraints).strip()]
    else:
        intent["constraints"] = []

    gpu_value = intent.get("gpu_availability", "unknown")
    intent["gpu_availability"] = str(gpu_value).strip() or "unknown"
    return intent


def validate_intent_payload(payload: dict) -> tuple[bool, list[str]]:
    errors = []
    if not isinstance(payload, dict):
        return False, ["payload must be an object"]

    required = set(INTENT_TEMPLATE)
    missing = sorted(required - set(payload))
    if missing:
        errors.append(f"missing keys: {', '.join(missing)}")

    if not isinstance(payload.get("labels"), list):
        errors.append("labels must be a list")
    if not isinstance(payload.get("constraints"), list):
        errors.append("constraints must be a list")
    if not isinstance(payload.get("needs_clarification"), bool):
        errors.append("needs_clarification must be a bool")

    allowed_gpu = {"yes", "no", "unknown", "limited"}
    if payload.get("gpu_availability") not in allowed_gpu:
        errors.append("gpu_availability must be one of: yes, no, unknown, limited")

    if payload.get("needs_clarification") and not payload.get("clarification_reason"):
        errors.append("clarification_reason is required when needs_clarification is true")

    return not errors, errors
