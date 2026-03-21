"""
V2 Dataset Discovery
--------------------
- If user provides dataset: audit it (quality + risk check)
- If no dataset: search Kaggle + HuggingFace, score, return top pick
- Score: 0.3*log(size) + 0.3*topic_similarity + 0.2*task_match + 0.2*usability
"""

import os
import math
import json
import requests
from researchforge.config.settings import Settings


class V2Datasets:
    def __init__(self):
        self.settings = Settings()

    # ── Public interface ──────────────────────────────────────────

    def audit_user_dataset(self, path: str, v1_findings: dict) -> dict:
        """Audit a user-provided CSV/parquet dataset."""
        try:
            import pandas as pd
            if path.startswith("http") or "kaggle.com" in path:
                return self._handle_kaggle_url(path, v1_findings)

            if path.endswith(".csv"):
                df = pd.read_csv(path)
            elif path.endswith(".parquet"):
                df = pd.read_parquet(path)
            else:
                # Try CSV by default
                df = pd.read_csv(path)

            return self._score_dataframe(df, path, v1_findings)

        except Exception as e:
            return {
                "name": os.path.basename(path),
                "score": 0.5,
                "risks": [f"Could not fully audit: {str(e)}"],
                "shape": "unknown",
                "ml_tasks": v1_findings.get("recommended_models", []),
                "source": "user-provided",
                "problem_type": v1_findings.get("problem_type", "unknown"),
            }

    def discover_and_score(self, topic: str, v1_findings: dict) -> dict:
        """Search Kaggle + HuggingFace, score each dataset, return best."""
        candidates = []
        candidates.extend(self._search_kaggle(topic))
        candidates.extend(self._search_huggingface(topic))

        if not candidates:
            return {
                "name": "No dataset found",
                "score": 0.0,
                "risks": ["No matching public datasets — provide your own with --dataset"],
                "ml_tasks": [],
                "source": "none",
                "problem_type": v1_findings.get("problem_type", "unknown"),
            }

        scored = [self._compute_score(c, topic, v1_findings) for c in candidates]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[0]

    # ── Private: search backends ──────────────────────────────────

    def _search_kaggle(self, topic: str) -> list:
        """Requires: pip install kaggle + ~/.kaggle/kaggle.json"""
        try:
            import kaggle
            datasets = kaggle.api.dataset_list(
                search=topic, sort_by="relevance", max_size=None
            )
            results = []
            for ds in datasets[:5]:
                results.append({
                    "name":     ds.ref,
                    "title":    ds.title,
                    "size_mb":  getattr(ds, "totalBytes", 0) / 1e6,
                    "downloads": getattr(ds, "downloadCount", 0),
                    "source":   "kaggle",
                    "url":      f"https://kaggle.com/datasets/{ds.ref}",
                })
            return results
        except Exception:
            return []

    def _search_huggingface(self, topic: str) -> list:
        """HuggingFace Datasets API — free, no auth."""
        try:
            resp = requests.get(
                "https://huggingface.co/api/datasets",
                params={"search": topic, "limit": 5, "sort": "downloads"},
                timeout=10,
            )
            results = []
            for ds in resp.json():
                results.append({
                    "name":      ds.get("id", ""),
                    "title":     ds.get("id", ""),
                    "downloads": ds.get("downloads", 0),
                    "size_mb":   0,
                    "source":    "huggingface",
                    "url":       f"https://huggingface.co/datasets/{ds.get('id')}",
                })
            return results
        except Exception:
            return []

    def _handle_kaggle_url(self, url: str, v1_findings: dict) -> dict:
        """
        Download a dataset from a Kaggle URL if credentials are configured,
        otherwise return a stub with the URL for manual download.
        """
        try:
            import kaggle
            import re
            # Extract owner/dataset from URL
            match = re.search(r"kaggle\.com/datasets?/([^/?]+/[^/?]+)", url)
            if not match:
                raise ValueError("Could not parse Kaggle dataset ref from URL")
            ref = match.group(1)

            import tempfile, zipfile, glob
            download_dir = tempfile.mkdtemp(prefix="rf_kaggle_")
            kaggle.api.dataset_download_files(ref, path=download_dir, unzip=True)

            # Find the first CSV
            csv_files = glob.glob(os.path.join(download_dir, "**", "*.csv"), recursive=True)
            if not csv_files:
                raise FileNotFoundError("No CSV found in downloaded Kaggle dataset")

            import pandas as pd
            df = pd.read_csv(csv_files[0])
            result = self._score_dataframe(df, csv_files[0], v1_findings)
            result["source"] = "kaggle-url"
            result["kaggle_ref"] = ref
            return result

        except Exception as e:
            return {
                "name": url.split("/")[-1] or "kaggle-dataset",
                "score": 0.5,
                "risks": [
                    f"Kaggle URL download failed ({e}). "
                    "Configure Kaggle API creds via `researchforge init`."
                ],
                "shape": "unknown",
                "ml_tasks": v1_findings.get("recommended_models", []),
                "source": "kaggle-url",
                "problem_type": v1_findings.get("problem_type", "unknown"),
            }

    # ── Private: scoring ──────────────────────────────────────────

    def _score_dataframe(self, df, path: str, v1_findings: dict) -> dict:
        """Score a loaded dataframe for quality + task relevance."""
        rows, cols = df.shape
        size_score = min(1.0, math.log10(max(rows * cols, 1)) / 6)

        risks = []
        label_col = self._detect_label_column(df)

        # Class imbalance check
        if label_col and df[label_col].dtype in ["object", "int64"]:
            counts = df[label_col].value_counts()
            if len(counts) > 1:
                ratio = counts.iloc[0] / counts.iloc[-1]
                if ratio > 2:
                    risks.append(f"Class imbalance {ratio:.1f}:1 — recommend SMOTE")

        # Missing value check
        missing_pct = df.isnull().sum().sum() / max(rows * cols, 1)
        if missing_pct > 0.1:
            risks.append(f"Missing values: {missing_pct:.0%} of cells")

        # Small dataset warning
        if rows < 1000:
            risks.append("Small dataset (<1k rows) — model may overfit")

        # Duplicate rows
        dup_pct = df.duplicated().sum() / max(rows, 1)
        if dup_pct > 0.05:
            risks.append(f"Duplicate rows: {dup_pct:.0%} — consider deduplication")

        problem_type = v1_findings.get("problem_type", "unknown")
        task_score = 0.8 if problem_type != "unknown" else 0.4

        final_score = (
            0.3 * size_score
            + 0.3 * 0.85       # user-provided = high topic match assumed
            + 0.2 * task_score
            + 0.2 * (1.0 if not risks else 0.5)
        )

        return {
            "name":          os.path.basename(path),
            "score":         round(final_score, 2),
            "shape":         f"{rows:,} rows × {cols} cols",
            "label_column":  label_col,
            "risks":         risks,
            "ml_tasks":      self._suggest_tasks(problem_type, rows),
            "source":        "user-provided",
            "problem_type":  problem_type,
        }

    def _compute_score(self, candidate: dict, topic: str, v1_findings: dict) -> dict:
        size_score = min(
            1.0, math.log10(max(candidate.get("size_mb", 1) * 1000, 1)) / 6
        )
        pop_score = min(
            1.0, math.log10(max(candidate.get("downloads", 1), 1)) / 5
        )
        topic_score = 0.7  # placeholder — production: cosine similarity via embedding

        final_score = (
            0.3 * size_score
            + 0.3 * topic_score
            + 0.2 * pop_score
            + 0.2 * 0.8  # assume public datasets have acceptable license
        )

        candidate["score"]    = round(final_score, 2)
        candidate["risks"]    = [
            "Verify label column before training",
            "Check license for commercial use",
        ]
        candidate["ml_tasks"] = self._suggest_tasks(
            v1_findings.get("problem_type", "unknown"), 0
        )
        candidate["problem_type"] = v1_findings.get("problem_type", "unknown")
        return candidate

    def _detect_label_column(self, df) -> str:
        """Heuristic: find the most likely target column by name hints."""
        target_hints = [
            "label", "target", "class", "y", "output",
            "result", "made", "shot", "outcome", "y_true",
        ]
        for col in df.columns:
            if any(hint in col.lower() for hint in target_hints):
                return col
        # fallback: last column
        return df.columns[-1] if len(df.columns) > 0 else None

    def _suggest_tasks(self, problem_type: str, n_rows: int) -> list:
        tasks = {
            "classification": [
                "XGBoost baseline", "LightGBM", "PyTorch MLP",
                "fine-tune BERT (text)",
            ],
            "regression": [
                "XGBoost regressor", "LightGBM regressor", "TabNet",
            ],
            "graph": [
                "GCN (PyTorch Geometric)", "GAT (attention)", "GraphSAGE",
            ],
            "nlp": [
                "fine-tune DistilBERT", "fine-tune Llama (Ollama)", "seq2seq T5",
            ],
        }
        return tasks.get(problem_type, ["XGBoost baseline", "LightGBM", "MLP"])
