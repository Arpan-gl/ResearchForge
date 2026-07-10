"""Persistence helpers for Neo4j-backed knowledge graph storage."""

from contextlib import nullcontext

from researchforge.config.settings import Settings
from researchforge.knowledge_graph.projector import KnowledgeGraphProjector
from researchforge.knowledge_graph.schema import NEO4J_SCHEMA_CYPHER


class Neo4jKnowledgeGraphStore:
    def __init__(self, settings: Settings | None = None, driver=None, projector: KnowledgeGraphProjector | None = None):
        self.settings = settings or Settings()
        self.driver = driver
        self.projector = projector or KnowledgeGraphProjector()

    def initialize(self) -> None:
        driver_cm = nullcontext(self.driver) if self.driver is not None else self._connect()
        with driver_cm as driver:
            with driver.session() as session:
                for statement in _split_statements(NEO4J_SCHEMA_CYPHER):
                    session.run(statement)

    def persist_research_handoff(self, handoff: dict) -> dict:
        projection = self.projector.project_research_handoff(handoff)
        self.persist_projection(projection)
        return projection

    def persist_projection(self, projection: dict) -> dict:
        nodes = ((projection.get("data") or {}).get("nodes")) or []
        edges = ((projection.get("data") or {}).get("edges")) or []

        driver_cm = nullcontext(self.driver) if self.driver is not None else self._connect()
        with driver_cm as driver:
            with driver.session() as session:
                for statement in _split_statements(NEO4J_SCHEMA_CYPHER):
                    session.run(statement)
                for node in nodes:
                    session.run(
                        (
                            "MERGE (n:{label} {{id: $id}}) "
                            "SET n.title = $title, n.source = $source, n.published = $published, "
                            "n.query = $node_query, n.provenance_source = $provenance_source, "
                            "n.provenance_retrieved_at = $provenance_retrieved_at, "
                            "n.provenance_agent = $provenance_agent"
                        ).format(label=node["label"]),
                        **_node_params(node),
                    )
                for edge in edges:
                    session.run(
                        (
                            "MATCH (src {{id: $from_id}}) "
                            "MATCH (dst {{id: $to_id}}) "
                            "MERGE (src)-[r:{edge_type}]->(dst) "
                            "SET r.provenance_source = $provenance_source, "
                            "r.provenance_retrieved_at = $provenance_retrieved_at, "
                            "r.provenance_agent = $provenance_agent"
                        ).format(edge_type=edge["type"]),
                        **_edge_params(edge),
                    )

        return {
            "nodes_written": len(nodes),
            "edges_written": len(edges),
        }

    def _connect(self):
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("Install neo4j to persist the knowledge graph.") from exc

        graph_url = self.settings.graph_url
        if not graph_url.startswith("bolt://"):
            raise RuntimeError("GRAPH_URL must be a Neo4j bolt URL for graph persistence.")

        payload = graph_url.removeprefix("bolt://")
        creds, host = payload.split("@", 1)
        username, password = creds.split(":", 1)
        return GraphDatabase.driver(f"bolt://{host}", auth=(username, password))


def _split_statements(sql_blob: str) -> list[str]:
    return [line.strip() for line in sql_blob.splitlines() if line.strip()]


def _node_params(node: dict) -> dict:
    provenance = node.get("provenance") or {}
    return {
        "id": node.get("id", ""),
        "title": node.get("title", ""),
        "source": node.get("source", ""),
        "published": node.get("published", ""),
        "node_query": node.get("query", ""),
        "provenance_source": provenance.get("source", ""),
        "provenance_retrieved_at": provenance.get("retrieved_at", ""),
        "provenance_agent": provenance.get("agent", ""),
    }


def _edge_params(edge: dict) -> dict:
    provenance = edge.get("provenance") or {}
    return {
        "from_id": edge.get("from", ""),
        "to_id": edge.get("to", ""),
        "provenance_source": provenance.get("source", ""),
        "provenance_retrieved_at": provenance.get("retrieved_at", ""),
        "provenance_agent": provenance.get("agent", ""),
    }
