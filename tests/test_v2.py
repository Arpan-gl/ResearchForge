"""
Tests for V2 Dataset Discovery and scoring.
"""

import io
import math
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


def make_v2():
    from researchforge.stages.v2_datasets import V2Datasets
    v = V2Datasets.__new__(V2Datasets)
    # Mock settings
    settings = MagicMock()
    v.settings = settings
    return v


def make_df(n_rows=500, n_cols=5, with_imbalance=False, with_missing=False,
            with_duplicates=False, missing_frac=0.12):
    data = {f"feat_{i}": np.random.randn(n_rows) for i in range(n_cols - 1)}
    if with_imbalance:
        data["target"] = [0] * (n_rows - 10) + [1] * 10
    else:
        data["target"] = [0] * (n_rows // 2) + [1] * (n_rows - n_rows // 2)
    df = pd.DataFrame(data)
    if with_missing:
        # spread across all columns to exceed the 10% overall threshold
        n_cells = n_rows * n_cols
        n_missing = int(n_cells * missing_frac)
        flat = df.values.flatten().astype(object)
        indices = np.random.choice(len(flat), n_missing, replace=False)
        flat[indices] = np.nan
        df = pd.DataFrame(flat.reshape(n_rows, n_cols), columns=df.columns)
    if with_duplicates:
        df = pd.concat([df, df.iloc[:int(n_rows * 0.1)]], ignore_index=True)
    return df


# ── Label column detection ────────────────────────────────────────

def test_detect_label_column_by_name():
    v = make_v2()
    df = pd.DataFrame({"feat_a": [1], "label": [0], "feat_b": [2]})
    assert v._detect_label_column(df) == "label"


def test_detect_label_column_target():
    v = make_v2()
    df = pd.DataFrame({"feat_a": [1], "target": [0]})
    assert v._detect_label_column(df) == "target"


def test_detect_label_column_falls_back_to_last():
    v = make_v2()
    df = pd.DataFrame({"feat_a": [1], "feat_b": [2], "col_z": [0]})
    assert v._detect_label_column(df) == "col_z"


# ── Risk detection ────────────────────────────────────────────────

def test_risk_detects_class_imbalance():
    v = make_v2()
    df = make_df(n_rows=100, with_imbalance=True)
    v1_findings = {"problem_type": "classification"}
    result = v._score_dataframe(df, "test.csv", v1_findings)
    risk_text = " ".join(result["risks"])
    assert "imbalance" in risk_text.lower()


def test_risk_detects_missing_values():
    v = make_v2()
    # 2000 rows so "small dataset" risk doesn't fire; 12% missing via make_df default
    df = make_df(n_rows=2000, with_missing=True)
    v1_findings = {"problem_type": "classification"}
    result = v._score_dataframe(df, "test.csv", v1_findings)
    risk_text = " ".join(result["risks"])
    assert "missing" in risk_text.lower()


def test_risk_detects_small_dataset():
    v = make_v2()
    df = make_df(n_rows=50)
    v1_findings = {"problem_type": "classification"}
    result = v._score_dataframe(df, "test.csv", v1_findings)
    risk_text = " ".join(result["risks"])
    assert "1k" in risk_text.lower() or "small" in risk_text.lower()


def test_risk_detects_duplicates():
    v = make_v2()
    df = make_df(n_rows=200, with_duplicates=True)
    v1_findings = {"problem_type": "classification"}
    result = v._score_dataframe(df, "test.csv", v1_findings)
    risk_text = " ".join(result["risks"])
    assert "duplicate" in risk_text.lower()


# ── Scoring formula ───────────────────────────────────────────────

def test_score_is_between_0_and_1():
    v = make_v2()
    df = make_df(n_rows=1000)
    result = v._score_dataframe(df, "data.csv", {"problem_type": "classification"})
    assert 0.0 <= result["score"] <= 1.0


def test_larger_dataset_scores_higher():
    v = make_v2()
    small = make_df(n_rows=100)
    large = make_df(n_rows=10000)
    vf = {"problem_type": "classification"}
    r_small = v._score_dataframe(small, "s.csv", vf)
    r_large = v._score_dataframe(large, "l.csv", vf)
    assert r_large["score"] >= r_small["score"]


# ── Suggested tasks ───────────────────────────────────────────────

def test_suggest_tasks_classification():
    v = make_v2()
    tasks = v._suggest_tasks("classification", 5000)
    assert any("XGBoost" in t for t in tasks)


def test_suggest_tasks_graph():
    v = make_v2()
    tasks = v._suggest_tasks("graph", 5000)
    assert any("GCN" in t or "GAT" in t for t in tasks)


def test_suggest_tasks_unknown_returns_defaults():
    v = make_v2()
    tasks = v._suggest_tasks("unknown_type", 0)
    assert len(tasks) > 0


# ── Discover and score ────────────────────────────────────────────

def test_discover_returns_no_dataset_sentinel_when_empty():
    v = make_v2()
    with patch.object(v, "_search_kaggle", return_value=[]):
        with patch.object(v, "_search_huggingface", return_value=[]):
            result = v.discover_and_score("obscure topic", {"problem_type": "classification"})
    assert "No dataset" in result["name"]
    assert result["score"] == 0.0


def test_score_dataset_includes_confidence_and_rationale():
    v = make_v2()
    ds = {
        "name": "owner/good-dataset",
        "title": "Basketball shot prediction tracking dataset",
        "size_mb": 120,
        "downloads": 25000,
        "source": "kaggle",
        "url": "https://kaggle.com/datasets/owner/good-dataset",
    }
    scored = v._score_dataset(ds, "basketball shot prediction", {"problem_type": "classification"})
    assert scored["selection_confidence"] in {"low", "medium", "high"}
    assert isinstance(scored["selection_rationale"], list)
    assert len(scored["selection_rationale"]) > 0


def test_discover_and_score_warns_for_low_confidence_pick():
    v = make_v2()
    low_signal = {
        "name": "x",
        "title": "x",
        "size_mb": 0,
        "downloads": 0,
        "source": "huggingface",
        "url": "",
    }
    with patch.object(v, "_search_kaggle", return_value=[]):
        with patch.object(v, "_search_huggingface", return_value=[low_signal]):
            result = v.discover_and_score("very specific akkadian cuneiform task", {"problem_type": "classification"})

    risk_text = " ".join(result.get("risks", []))
    assert result.get("selection_confidence") == "low"
    assert "Low-confidence dataset match" in risk_text
