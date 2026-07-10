import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from researchforge.agents.planner.agent import PlannerAgent
from researchforge.agents.planner.llm import LLMRouter
from researchforge.agents.planner.schema import validate_intent_payload


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def parse_json(self, prompt: str):
        return self.payload, "fake"


def test_planner_agent_returns_provenanced_handoff():
    payload = {
        "objective": "Train a classifier for support tickets",
        "task_type": "classification",
        "modality": "text",
        "labels": ["priority"],
        "evaluation_metric": "f1",
        "constraints": ["single GPU"],
        "gpu_availability": "yes",
        "framework_preference": "transformers",
        "expected_output": "config files",
        "needs_clarification": False,
        "clarification_reason": "",
    }
    handoff = PlannerAgent(llm=FakeLLM(payload)).parse_intent("Train a classifier for support tickets")

    assert handoff["provenance"]["agent"] == "planner"
    assert handoff["provenance"]["source"] == "planner:fake"
    assert handoff["confidence"] == "computed"
    valid, errors = validate_intent_payload(handoff["data"])
    assert valid, errors


def test_planner_agent_emits_needs_clarification_without_guessing():
    payload = {
        "objective": "Help me with research",
        "task_type": "",
        "modality": "",
        "labels": [],
        "evaluation_metric": "",
        "constraints": [],
        "gpu_availability": "unknown",
        "framework_preference": "",
        "expected_output": "",
        "needs_clarification": True,
        "clarification_reason": "The task and modality are not specified.",
    }
    handoff = PlannerAgent(llm=FakeLLM(payload)).parse_intent("Help me with research")
    assert handoff["data"]["needs_clarification"] is True
    assert handoff["data"]["clarification_reason"]


def test_llm_router_prefers_ollama_when_available():
    settings = SimpleNamespace(
        llm_provider="auto",
        ollama_url="http://localhost:11434",
        llm_model="qwen/qwen3-coder-next",
        openrouter_api_key="key",
        openrouter_base_url="https://openrouter.ai/api/v1",
    )
    router = LLMRouter(settings)

    with patch("researchforge.agents.planner.llm.requests.get") as get_mock:
        with patch("researchforge.agents.planner.llm.requests.post") as post_mock:
            get_mock.return_value.ok = True
            post_mock.return_value.json.return_value = {"response": json.dumps({"ok": True})}
            post_mock.return_value.raise_for_status.return_value = None
            payload, provider = router.parse_json("prompt")

    assert payload == {"ok": True}
    assert provider == "ollama"
    called_url = post_mock.call_args.args[0]
    assert called_url.endswith("/api/generate")


def test_llm_router_falls_back_to_openrouter_when_ollama_is_down():
    settings = SimpleNamespace(
        llm_provider="auto",
        ollama_url="http://localhost:11434",
        llm_model="qwen/qwen3-coder-next",
        openrouter_api_key="key",
        openrouter_base_url="https://openrouter.ai/api/v1",
    )
    router = LLMRouter(settings)

    def fake_post(url, **kwargs):
        response = SimpleNamespace()
        response.raise_for_status = lambda: None
        if url.endswith("/chat/completions"):
            response.json = lambda: {
                "choices": [{"message": {"content": json.dumps({"ok": True})}}]
            }
            return response
        raise AssertionError("Unexpected Ollama POST call")

    with patch("researchforge.agents.planner.llm.requests.get", side_effect=RuntimeError("down")):
        with patch("researchforge.agents.planner.llm.requests.post", side_effect=fake_post):
            payload, provider = router.parse_json("prompt")

    assert payload == {"ok": True}
    assert provider == "openrouter"


def test_cli_plan_command_writes_intent_file():
    output_path = Path("tests") / ".planner_intent_test.json"
    payload = {
        "objective": "Train a classifier",
        "task_type": "classification",
        "modality": "tabular",
        "labels": ["label"],
        "evaluation_metric": "accuracy",
        "constraints": [],
        "gpu_availability": "no",
        "framework_preference": "sklearn",
        "expected_output": "yaml",
        "needs_clarification": False,
        "clarification_reason": "",
    }

    try:
        with patch("researchforge.agents.planner.PlannerAgent.parse_intent", return_value={
            "data": payload,
            "provenance": {"source": "planner:fake", "retrieved_at": "now", "agent": "planner"},
            "confidence": "computed",
        }):
            with patch("sys.argv", ["researchforge", "plan", "Train a classifier", "--output", str(output_path)]):
                from researchforge import cli

                cli.main()

        assert output_path.exists()
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        assert saved["data"]["task_type"] == "classification"
    finally:
        output_path.unlink(missing_ok=True)
