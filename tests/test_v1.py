"""
Tests for V1 Research pipeline components.
All Ollama/network calls are mocked.
"""

import json
import pytest
from unittest.mock import patch, MagicMock


# ── Query rewriting ───────────────────────────────────────────────

def make_v1():
    from researchforge.stages.v1_research import V1Research
    v = V1Research.__new__(V1Research)
    v.ollama_url = "http://localhost:11434"
    v.model = "llama3"
    return v


def test_rewrite_queries_valid_response():
    v = make_v1()
    mock_resp = json.dumps(["q1", "q2", "q3", "q4"])
    with patch.object(v, "_ask_llm", return_value=mock_resp):
        result = v._rewrite_queries("basketball GNN")
    assert len(result) == 4
    assert result[0] == "q1"


def test_rewrite_queries_fallback_on_bad_json():
    v = make_v1()
    with patch.object(v, "_ask_llm", return_value="NOT JSON"):
        result = v._rewrite_queries("basketball GNN")
    assert "basketball GNN" in result
    assert len(result) == 4


def test_rewrite_queries_strips_markdown_fences():
    v = make_v1()
    fenced = '```json\n["a","b","c","d"]\n```'
    with patch.object(v, "_ask_llm", return_value=fenced):
        result = v._rewrite_queries("topic")
    assert result == ["a", "b", "c", "d"]


# ── Reranking ─────────────────────────────────────────────────────

def test_rerank_deduplicates():
    v = make_v1()
    results = [
        {"url": "http://a.com", "snippet": "x" * 100, "source": "web"},
        {"url": "http://a.com", "snippet": "x" * 100, "source": "web"},  # duplicate
        {"url": "http://b.com", "snippet": "y" * 200, "source": "arxiv"},
    ]
    ranked = v._rerank(results)
    urls = [r["url"] for r in ranked]
    assert len(urls) == 2
    assert len(set(urls)) == 2


def test_rerank_arxiv_scores_higher_than_web():
    v = make_v1()
    results = [
        {"url": "http://web.com", "snippet": "a" * 100, "source": "web"},
        {"url": "http://arxiv.org/abs/1234", "snippet": "a" * 100, "source": "arxiv",
         "published": "2025-01-01"},
    ]
    ranked = v._rerank(results)
    assert ranked[0]["source"] == "arxiv"


# ── Contradiction detection ───────────────────────────────────────

def test_detect_numeric_contradictions_fires():
    v = make_v1()
    findings = {
        "metrics": [
            {"name": "F1", "value": "0.90", "unit": "", "source": 1},
            {"name": "F1", "value": "0.60", "unit": "", "source": 2},
        ],
        "key_findings": [],
    }
    contradictions = v._detect_contradictions(findings)
    assert len(contradictions) == 1
    assert "F1" in contradictions[0]


def test_detect_numeric_contradictions_no_fire_on_small_variance():
    v = make_v1()
    findings = {
        "metrics": [
            {"name": "F1", "value": "0.80"},
            {"name": "F1", "value": "0.82"},
        ],
        "key_findings": [],
    }
    contradictions = v._detect_contradictions(findings)
    assert len(contradictions) == 0


def test_semantic_contradictions_returns_list():
    v = make_v1()
    mock_response = json.dumps([
        {"finding_a": 0, "finding_b": 1, "contradiction": "directed vs undirected edges"}
    ])
    findings = {
        "key_findings": [
            "Directed edges improve GNN performance [1]",
            "Undirected edges work better for this dataset [2]",
        ]
    }
    with patch.object(v, "_ask_llm", return_value=mock_response):
        result = v._detect_semantic_contradictions(findings)
    assert len(result) == 1
    assert "directed" in result[0].lower() or "contradiction" in result[0].lower()


def test_semantic_contradictions_empty_on_bad_llm():
    v = make_v1()
    findings = {"key_findings": ["finding A", "finding B"]}
    with patch.object(v, "_ask_llm", return_value="NOT JSON"):
        result = v._detect_semantic_contradictions(findings)
    assert result == []


def test_semantic_contradictions_skips_single_finding():
    v = make_v1()
    findings = {"key_findings": ["only one finding"]}
    result = v._detect_semantic_contradictions(findings)
    assert result == []


# ── JSON extraction fallback ──────────────────────────────────────

def test_extract_findings_fallback_on_unparseable():
    v = make_v1()
    with patch.object(v, "_ask_llm", return_value="sorry, I cannot help with that"):
        result = v._extract_findings("test topic", [])
    assert "key_findings" in result
    assert isinstance(result["key_findings"], list)
    assert "problem_type" in result
