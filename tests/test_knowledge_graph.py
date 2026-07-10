import json
from pathlib import Path

import pytest

from researchforge.knowledge_graph import (
    EDGE_TYPES,
    KUZU_SCHEMA_DDL,
    NEO4J_SCHEMA_CYPHER,
    NODE_LABELS,
    KnowledgeGraphProjector,
    Neo4jKnowledgeGraphStore,
)


RESEARCH_HANDOFF = {
    "data": {
        "queries": ["support ticket classification"],
        "evidence": [
            {
                "title": "Graph Models for Ticket Routing",
                "url": "http://arxiv.org/abs/1234.5678",
                "snippet": "Uses graph neural networks.",
                "published": "2026-01-02T00:00:00Z",
                "authors": ["Alice"],
                "source": "arxiv",
                "query": "support ticket classification",
                "provenance": {
                    "source": "http://arxiv.org/abs/1234.5678",
                    "retrieved_at": "2026-07-07T00:00:00+00:00",
                    "agent": "research",
                },
            },
            {
                "title": "openai/researchforge-demo",
                "url": "https://github.com/openai/researchforge-demo",
                "snippet": "Training pipeline reference implementation.",
                "published": "2026-02-01T00:00:00Z",
                "authors": ["openai"],
                "source": "github",
                "query": "support ticket classification",
                "provenance": {
                    "source": "https://github.com/openai/researchforge-demo",
                    "retrieved_at": "2026-07-07T00:00:00+00:00",
                    "agent": "research",
                },
            },
        ],
    },
    "provenance": {
        "source": "planner:fake",
        "retrieved_at": "2026-07-07T00:00:00+00:00",
        "agent": "research",
    },
    "confidence": "computed",
}


class FakeNeo4jSession:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, query, **params):
        self.calls.append((query, params))


class FakeNeo4jDriver:
    def __init__(self):
        self.session_obj = FakeNeo4jSession()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def session(self):
        return self.session_obj


def test_schema_lists_required_node_labels_and_edge_types():
    assert {"Paper", "Dataset", "Repository", "Author"}.issubset(set(NODE_LABELS))
    assert {"implements", "cites", "authored"}.issubset(set(EDGE_TYPES))
    assert "CREATE CONSTRAINT paper_id" in NEO4J_SCHEMA_CYPHER
    assert "CREATE NODE TABLE Paper" in KUZU_SCHEMA_DDL


def test_projector_requires_research_provenance():
    projector = KnowledgeGraphProjector()
    with pytest.raises(ValueError, match="missing provenance"):
        projector.project_research_handoff({"data": {"evidence": []}})


def test_projector_creates_paper_repository_and_author_nodes():
    projector = KnowledgeGraphProjector()
    projection = projector.project_research_handoff(RESEARCH_HANDOFF)
    nodes = projection["data"]["nodes"]
    labels = {node["label"] for node in nodes}
    assert {"Paper", "Repository", "Author"}.issubset(labels)
    assert all("provenance" in node for node in nodes)
    assert any(edge["type"] == "authored" for edge in projection["data"]["edges"])


def test_neo4j_store_persists_projected_nodes_and_edges():
    driver = FakeNeo4jDriver()
    store = Neo4jKnowledgeGraphStore(driver=driver)
    projection = store.projector.project_research_handoff(RESEARCH_HANDOFF)
    result = store.persist_projection(projection)

    assert result["nodes_written"] == len(projection["data"]["nodes"])
    assert result["edges_written"] == len(projection["data"]["edges"])
    assert any("MERGE (n:Paper" in query for query, _ in driver.session_obj.calls)
    assert any("MERGE (src)-[r:authored]->(dst)" in query for query, _ in driver.session_obj.calls)
    node_param_sets = [params for query, params in driver.session_obj.calls if "MERGE (n:" in query]
    assert any("node_query" in params for params in node_param_sets)


def test_cli_graph_command_writes_output_file():
    evidence_path = Path("tests") / ".graph_input_test.json"
    output_path = Path("tests") / ".graph_output_test.json"
    try:
        evidence_path.write_text(json.dumps(RESEARCH_HANDOFF), encoding="utf-8")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("sys.argv", ["researchforge", "graph", str(evidence_path), "--output", str(output_path)])
            from researchforge import cli

            cli.main()

        assert output_path.exists()
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        assert saved["provenance"]["agent"] == "knowledge_graph"
    finally:
        evidence_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


def test_cli_persist_graph_command_uses_neo4j_store():
    research_path = Path("tests") / ".persist_graph_input.json"
    projection_output = Path("tests") / ".persist_graph_projection.json"
    try:
        research_path.write_text(json.dumps(RESEARCH_HANDOFF), encoding="utf-8")
        with pytest.MonkeyPatch.context() as mp:
            fake_driver = FakeNeo4jDriver()
            mp.setattr("researchforge.knowledge_graph.Neo4jKnowledgeGraphStore", lambda: Neo4jKnowledgeGraphStore(driver=fake_driver))
            mp.setattr(
                "sys.argv",
                [
                    "researchforge",
                    "persist-graph",
                    str(research_path),
                    "--projection-output",
                    str(projection_output),
                ],
            )
            from researchforge import cli

            cli.main()

        assert projection_output.exists()
        assert fake_driver.session_obj.calls
    finally:
        research_path.unlink(missing_ok=True)
        projection_output.unlink(missing_ok=True)
