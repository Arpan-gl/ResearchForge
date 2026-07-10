"""Projection from evidence records into graph nodes."""

import json
from pathlib import Path


class KnowledgeGraphProjector:
    def project_research_handoff(self, handoff: dict) -> dict:
        provenance = handoff.get("provenance") or {}
        if not {"source", "retrieved_at", "agent"}.issubset(provenance):
            raise ValueError("Research handoff is missing provenance.")

        evidence = ((handoff.get("data") or {}).get("evidence")) or []
        nodes = {}
        edges = []

        for record in evidence:
            record_provenance = record.get("provenance") or {}
            if not {"source", "retrieved_at", "agent"}.issubset(record_provenance):
                raise ValueError("Evidence record is missing provenance.")

            label = "Repository" if record.get("source") == "github" else "Paper"
            node_id = record.get("url") or record.get("title")
            if node_id:
                nodes[node_id] = {
                    "id": node_id,
                    "label": label,
                    "title": record.get("title", ""),
                    "source": record.get("source", ""),
                    "published": record.get("published", ""),
                    "query": record.get("query", ""),
                    "provenance": record_provenance,
                }

            if label == "Paper":
                for author_name in record.get("authors", []):
                    author_id = f"author:{author_name.strip().lower()}"
                    if not author_name.strip():
                        continue
                    nodes[author_id] = {
                        "id": author_id,
                        "label": "Author",
                        "title": author_name,
                        "source": record.get("source", ""),
                        "published": "",
                        "query": record.get("query", ""),
                        "provenance": record_provenance,
                    }
                    edges.append(
                        {
                            "type": "authored",
                            "from": author_id,
                            "to": node_id,
                            "provenance": record_provenance,
                        }
                    )

        return {
            "data": {
                "nodes": list(nodes.values()),
                "edges": edges,
            },
            "provenance": {
                "source": provenance["source"],
                "retrieved_at": provenance["retrieved_at"],
                "agent": "knowledge_graph",
            },
            "confidence": "computed",
        }

    def save_projection(self, projection: dict, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(projection, indent=2), encoding="utf-8")
