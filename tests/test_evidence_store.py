import json
from pathlib import Path

import pytest

from researchforge.sdk.evidence_store import EVIDENCE_SCHEMA_SQL, EvidenceStore


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
            }
        ],
    },
    "provenance": {
        "source": "planner:fake",
        "retrieved_at": "2026-07-07T00:00:00+00:00",
        "agent": "research",
    },
    "confidence": "computed",
}


class FakeCursor:
    def __init__(self):
        self.executed = []
        self.executemany_calls = []
        self.fetchone_result = (1,)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql):
        self.executed.append(sql)

    def executemany(self, sql, rows):
        self.executemany_calls.append((sql, rows))

    def fetchone(self):
        return self.fetchone_result


class FakeConnection:
    def __init__(self):
        self.cursor_obj = FakeCursor()
        self.commit_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.commit_calls += 1


def test_schema_contains_provenance_columns():
    assert "provenance_source TEXT NOT NULL" in EVIDENCE_SCHEMA_SQL
    assert "provenance_retrieved_at TIMESTAMPTZ NOT NULL" in EVIDENCE_SCHEMA_SQL
    assert "provenance_agent TEXT NOT NULL" in EVIDENCE_SCHEMA_SQL


def test_flatten_handoff_requires_record_provenance():
    broken = json.loads(json.dumps(RESEARCH_HANDOFF))
    broken["data"]["evidence"][0].pop("provenance")
    with pytest.raises(ValueError, match="missing provenance"):
        EvidenceStore._flatten_handoff(broken)


def test_flatten_handoff_serializes_raw_record_and_authors():
    rows = EvidenceStore._flatten_handoff(RESEARCH_HANDOFF)
    assert len(rows) == 1
    row = rows[0]
    assert row["provenance_agent"] == "research"
    assert json.loads(row["authors"]) == ["Alice"]
    assert json.loads(row["raw_record"])["title"] == "Graph Models for Ticket Routing"


def test_store_research_handoff_initializes_schema_and_inserts_rows():
    connection = FakeConnection()
    store = EvidenceStore(connection=connection)
    inserted = store.store_research_handoff(RESEARCH_HANDOFF)

    assert inserted == 1
    assert connection.commit_calls == 1
    assert connection.cursor_obj.executed[0].startswith("CREATE TABLE IF NOT EXISTS evidence_records")
    sql, rows = connection.cursor_obj.executemany_calls[0]
    assert "INSERT INTO evidence_records" in sql
    assert rows[0]["url"] == "http://arxiv.org/abs/1234.5678"


def test_count_records_uses_real_table_count_query():
    connection = FakeConnection()
    connection.cursor_obj.fetchone_result = (7,)
    store = EvidenceStore(connection=connection)
    assert store.count_records() == 7
    assert connection.cursor_obj.executed[-1] == "SELECT COUNT(*) FROM evidence_records"


def test_cli_store_evidence_command_reports_inserted_and_total_counts():
    research_path = Path("tests") / ".store_evidence_input.json"
    try:
        research_path.write_text(json.dumps(RESEARCH_HANDOFF), encoding="utf-8")
        with pytest.MonkeyPatch.context() as mp:
            fake_connection = FakeConnection()
            mp.setattr("researchforge.sdk.EvidenceStore", lambda: EvidenceStore(connection=fake_connection))
            mp.setattr("sys.argv", ["researchforge", "store-evidence", str(research_path)])
            from researchforge import cli

            cli.main()

        assert fake_connection.cursor_obj.executemany_calls
    finally:
        research_path.unlink(missing_ok=True)
