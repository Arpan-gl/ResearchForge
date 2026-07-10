"""Postgres evidence store for research outputs."""

import json
from contextlib import nullcontext

from researchforge.config.settings import Settings

EVIDENCE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS evidence_records (
    id BIGSERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    url TEXT NOT NULL UNIQUE,
    snippet TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL,
    published TEXT NOT NULL DEFAULT '',
    authors JSONB NOT NULL DEFAULT '[]'::jsonb,
    query TEXT NOT NULL DEFAULT '',
    raw_record JSONB NOT NULL,
    provenance_source TEXT NOT NULL,
    provenance_retrieved_at TIMESTAMPTZ NOT NULL,
    provenance_agent TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_evidence_records_source ON evidence_records(source);
CREATE INDEX IF NOT EXISTS idx_evidence_records_provenance_agent ON evidence_records(provenance_agent);
""".strip()

INSERT_EVIDENCE_SQL = """
INSERT INTO evidence_records (
    title,
    url,
    snippet,
    source,
    published,
    authors,
    query,
    raw_record,
    provenance_source,
    provenance_retrieved_at,
    provenance_agent
)
VALUES (
    %(title)s,
    %(url)s,
    %(snippet)s,
    %(source)s,
    %(published)s,
    %(authors)s::jsonb,
    %(query)s,
    %(raw_record)s::jsonb,
    %(provenance_source)s,
    %(provenance_retrieved_at)s,
    %(provenance_agent)s
)
ON CONFLICT (url) DO UPDATE SET
    title = EXCLUDED.title,
    snippet = EXCLUDED.snippet,
    source = EXCLUDED.source,
    published = EXCLUDED.published,
    authors = EXCLUDED.authors,
    query = EXCLUDED.query,
    raw_record = EXCLUDED.raw_record,
    provenance_source = EXCLUDED.provenance_source,
    provenance_retrieved_at = EXCLUDED.provenance_retrieved_at,
    provenance_agent = EXCLUDED.provenance_agent;
""".strip()


class EvidenceStore:
    def __init__(self, settings: Settings | None = None, connection=None):
        self.settings = settings or Settings()
        self.connection = connection

    def initialize(self) -> None:
        connection_cm = nullcontext(self.connection) if self.connection is not None else self._connect()
        with connection_cm as conn:
            with conn.cursor() as cursor:
                cursor.execute(EVIDENCE_SCHEMA_SQL)
            conn.commit()

    def store_research_handoff(self, handoff: dict) -> int:
        rows = self._flatten_handoff(handoff)
        connection_cm = nullcontext(self.connection) if self.connection is not None else self._connect()
        with connection_cm as conn:
            with conn.cursor() as cursor:
                cursor.execute(EVIDENCE_SCHEMA_SQL)
                cursor.executemany(INSERT_EVIDENCE_SQL, rows)
            conn.commit()
        return len(rows)

    def count_records(self) -> int:
        connection_cm = nullcontext(self.connection) if self.connection is not None else self._connect()
        with connection_cm as conn:
            with conn.cursor() as cursor:
                cursor.execute(EVIDENCE_SCHEMA_SQL)
                cursor.execute("SELECT COUNT(*) FROM evidence_records")
                row = cursor.fetchone()
        return int(row[0]) if row else 0

    def _connect(self):
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("Install psycopg to use the Postgres evidence store.") from exc
        if not self.settings.postgres_url:
            raise RuntimeError("POSTGRES_URL is not configured.")
        return psycopg.connect(self.settings.postgres_url)

    @staticmethod
    def _flatten_handoff(handoff: dict) -> list[dict]:
        top_provenance = handoff.get("provenance") or {}
        if not {"source", "retrieved_at", "agent"}.issubset(top_provenance):
            raise ValueError("Research handoff is missing provenance.")

        evidence = ((handoff.get("data") or {}).get("evidence")) or []
        rows = []
        for record in evidence:
            provenance = record.get("provenance") or {}
            if not {"source", "retrieved_at", "agent"}.issubset(provenance):
                raise ValueError("Evidence record is missing provenance.")
            rows.append(
                {
                    "title": record.get("title", ""),
                    "url": record.get("url", ""),
                    "snippet": record.get("snippet", ""),
                    "source": record.get("source", ""),
                    "published": record.get("published", ""),
                    "authors": json.dumps(record.get("authors", [])),
                    "query": record.get("query", ""),
                    "raw_record": json.dumps(record),
                    "provenance_source": provenance["source"],
                    "provenance_retrieved_at": provenance["retrieved_at"],
                    "provenance_agent": provenance["agent"],
                }
            )
        return rows
