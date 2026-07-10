"""Research agent for evidence retrieval."""

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import requests


class ResearchAgent:
    def __init__(self, session=None):
        self.session = session or requests.Session()

    def run(self, intent_handoff: dict) -> dict:
        self._validate_handoff(intent_handoff)
        intent = intent_handoff["data"]
        queries = self._build_queries(intent)
        evidence = []
        for query in queries:
            evidence.extend(self._search_arxiv(query))
            evidence.extend(self._search_semantic_scholar(query))
            evidence.extend(self._search_github(query))

        deduped = self._deduplicate(evidence)
        ranked = self._rank(deduped)
        return {
            "data": {
                "queries": queries,
                "evidence": ranked,
            },
            "provenance": {
                "source": intent_handoff["provenance"]["source"],
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "agent": "research",
            },
            "confidence": "computed",
        }

    def save_evidence(self, handoff: dict, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(handoff, handle, indent=2)

    @staticmethod
    def _validate_handoff(intent_handoff: dict) -> None:
        provenance = intent_handoff.get("provenance") or {}
        if not {"source", "retrieved_at", "agent"}.issubset(provenance):
            raise ValueError("Planner input is missing provenance.")
        if "data" not in intent_handoff:
            raise ValueError("Planner input is missing data.")

    @staticmethod
    def _build_queries(intent: dict) -> list[str]:
        objective = (intent.get("objective") or "").strip()
        task_type = (intent.get("task_type") or "").strip()
        modality = (intent.get("modality") or "").strip()
        labels = intent.get("labels") or []

        queries = [objective]
        if task_type:
            queries.append(f"{objective} {task_type}".strip())
        if modality:
            queries.append(f"{objective} {modality}".strip())
        if labels:
            queries.append(f"{objective} {' '.join(labels[:2])}".strip())

        cleaned = []
        seen = set()
        for query in queries:
            normalized = " ".join(query.split())
            if normalized and normalized.lower() not in seen:
                seen.add(normalized.lower())
                cleaned.append(normalized)
        return cleaned[:3]

    def _search_arxiv(self, query: str) -> list[dict]:
        url = (
            "http://export.arxiv.org/api/query?"
            f"search_query=all:{quote_plus(query)}&start=0&max_results=3"
        )
        response = self.session.get(url, timeout=20)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        records = []
        for entry in root.findall("atom:entry", ns):
            title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
            summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
            link = entry.findtext("atom:id", default="", namespaces=ns)
            published = entry.findtext("atom:published", default="", namespaces=ns)
            authors = [author.findtext("atom:name", default="", namespaces=ns) for author in entry.findall("atom:author", ns)]
            records.append(self._record("arxiv", title, link, summary, published, authors, query))
        return records

    def _search_semantic_scholar(self, query: str) -> list[dict]:
        url = (
            "https://api.semanticscholar.org/graph/v1/paper/search?"
            f"query={quote_plus(query)}&limit=3&fields=title,abstract,url,year,authors"
        )
        response = self.session.get(url, timeout=20)
        response.raise_for_status()
        payload = response.json()
        records = []
        for item in payload.get("data", []):
            authors = [author.get("name", "") for author in item.get("authors", [])]
            records.append(
                self._record(
                    "semantic_scholar",
                    item.get("title", ""),
                    item.get("url", ""),
                    item.get("abstract", ""),
                    str(item.get("year", "")),
                    authors,
                    query,
                )
            )
        return records

    def _search_github(self, query: str) -> list[dict]:
        url = (
            "https://api.github.com/search/repositories?"
            f"q={quote_plus(query)}&sort=stars&order=desc&per_page=3"
        )
        response = self.session.get(url, timeout=20, headers={"Accept": "application/vnd.github+json"})
        response.raise_for_status()
        payload = response.json()
        records = []
        for item in payload.get("items", []):
            owner = item.get("owner", {}).get("login", "")
            records.append(
                self._record(
                    "github",
                    item.get("full_name", ""),
                    item.get("html_url", ""),
                    item.get("description", ""),
                    item.get("updated_at", ""),
                    [owner] if owner else [],
                    query,
                )
            )
        return records

    def _record(self, source: str, title: str, url: str, snippet: str, published: str, authors: list[str], query: str) -> dict:
        return {
            "title": title,
            "url": url,
            "snippet": snippet,
            "published": published,
            "authors": authors,
            "source": source,
            "query": query,
            "provenance": {
                "source": url,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "agent": "research",
            },
        }

    @staticmethod
    def _deduplicate(records: list[dict]) -> list[dict]:
        seen = set()
        deduped = []
        for record in records:
            key = (record.get("url") or "").strip().lower()
            if not key:
                key = (record.get("title") or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(record)
        return deduped

    @staticmethod
    def _rank(records: list[dict]) -> list[dict]:
        source_weight = {
            "arxiv": 3,
            "semantic_scholar": 2,
            "github": 1,
        }
        for record in records:
            snippet_len = len(record.get("snippet") or "")
            record["score"] = source_weight.get(record.get("source"), 0) + min(snippet_len / 500, 1)
        return sorted(records, key=lambda item: item.get("score", 0), reverse=True)
