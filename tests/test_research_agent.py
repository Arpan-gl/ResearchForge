import json
from pathlib import Path

import pytest

from researchforge.agents.research import ResearchAgent


class FakeResponse:
    def __init__(self, text="", payload=None):
        self.text = text
        self._payload = payload or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeSession:
    def get(self, url, **kwargs):
        if "arxiv.org" in url:
            return FakeResponse(
                text="""
                <feed xmlns=\"http://www.w3.org/2005/Atom\">
                  <entry>
                    <title>Graph Models for Ticket Routing</title>
                    <summary>Uses graph neural networks for support tickets.</summary>
                    <id>http://arxiv.org/abs/1234.5678</id>
                    <published>2026-01-02T00:00:00Z</published>
                    <author><name>Alice</name></author>
                  </entry>
                </feed>
                """
            )
        if "semanticscholar" in url:
            return FakeResponse(
                payload={
                    "data": [
                        {
                            "title": "Graph Models for Ticket Routing",
                            "abstract": "Same paper indexed in Semantic Scholar.",
                            "url": "http://arxiv.org/abs/1234.5678",
                            "year": 2026,
                            "authors": [{"name": "Alice"}],
                        },
                        {
                            "title": "Benchmarking Ticket Classification",
                            "abstract": "Compares classical baselines.",
                            "url": "https://www.semanticscholar.org/paper/abc",
                            "year": 2025,
                            "authors": [{"name": "Bob"}],
                        },
                    ]
                }
            )
        if "api.github.com" in url:
            return FakeResponse(
                payload={
                    "items": [
                        {
                            "full_name": "openai/researchforge-demo",
                            "html_url": "https://github.com/openai/researchforge-demo",
                            "description": "Training pipeline reference implementation.",
                            "updated_at": "2026-02-01T00:00:00Z",
                            "owner": {"login": "openai"},
                        }
                    ]
                }
            )
        raise AssertionError(f"Unexpected URL: {url}")


INTENT_HANDOFF = {
    "data": {
        "objective": "Train a support ticket classifier",
        "task_type": "classification",
        "modality": "text",
        "labels": ["priority"],
        "evaluation_metric": "f1",
        "constraints": [],
        "gpu_availability": "yes",
        "framework_preference": "transformers",
        "expected_output": "config",
        "needs_clarification": False,
        "clarification_reason": "",
    },
    "provenance": {
        "source": "planner:fake",
        "retrieved_at": "2026-07-07T00:00:00+00:00",
        "agent": "planner",
    },
    "confidence": "computed",
}


def test_research_agent_requires_provenanced_input():
    agent = ResearchAgent(session=FakeSession())
    with pytest.raises(ValueError, match="missing provenance"):
        agent.run({"data": INTENT_HANDOFF["data"]})


def test_research_agent_deduplicates_cross_source_results_and_keeps_urls():
    agent = ResearchAgent(session=FakeSession())
    handoff = agent.run(INTENT_HANDOFF)
    evidence = handoff["data"]["evidence"]

    assert len(evidence) == 3
    assert all(item["url"] for item in evidence)
    assert all(item["provenance"]["agent"] == "research" for item in evidence)
    urls = [item["url"] for item in evidence]
    assert len(urls) == len(set(urls))


def test_research_agent_builds_at_most_three_queries():
    agent = ResearchAgent(session=FakeSession())
    queries = agent._build_queries(INTENT_HANDOFF["data"])
    assert 1 <= len(queries) <= 3
    assert queries[0] == "Train a support ticket classifier"


def test_cli_research_command_writes_output_file():
    intent_path = Path("tests") / ".intent_input_test.json"
    output_path = Path("tests") / ".research_output_test.json"
    try:
        intent_path.write_text(json.dumps(INTENT_HANDOFF), encoding="utf-8")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("researchforge.agents.research.ResearchAgent.run", lambda self, payload: {
                "data": {"queries": ["q1"], "evidence": []},
                "provenance": {"source": "planner:fake", "retrieved_at": "now", "agent": "research"},
                "confidence": "computed",
            })
            mp.setattr("sys.argv", ["researchforge", "research", str(intent_path), "--output", str(output_path)])
            from researchforge import cli

            cli.main()

        assert output_path.exists()
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        assert saved["provenance"]["agent"] == "research"
    finally:
        intent_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
