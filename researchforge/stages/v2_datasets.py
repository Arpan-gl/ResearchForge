"""
V2 Dataset Discovery
--------------------
- Production-ready dataset discovery and auditing.
- Discover returns top ranked candidates with explanations.
- Backward-compatible wrappers are kept for existing pipeline/tests.
"""

import os
import math
import re
import requests
from researchforge.config.settings import Settings


class V2Datasets:
    def __init__(self, ollama_url: str = None, model: str = None):
        self.settings = Settings()
        self.ollama_url = ollama_url or self.settings.ollama_url
        self.model = model or self.settings.llm_model

    # ── Public interface ──────────────────────────────────────────

    def discover(self, topic: str, v1_findings: dict) -> dict:
        """Discover and rank datasets from Kaggle + HuggingFace."""
        candidates = []
        candidates += self._search_kaggle(topic)
        candidates += self._search_huggingface(topic)

        if not candidates:
            return {"datasets": []}

        scored = [self._score_dataset(c, topic, v1_findings) for c in candidates]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return {"datasets": scored[:5]}

    def audit(self, df, topic: str, v1_findings: dict) -> dict:
        """Audit a loaded dataframe for quality risks."""
        return self._audit_dataframe(df, topic, v1_findings)

    # ── Backward-compatible public wrappers ─────────────────────

    def audit_user_dataset(self, path: str, v1_findings: dict) -> dict:
        """Audit a user-provided CSV/parquet dataset from path or URL."""
        try:
            import pandas as pd
            if path.startswith("http") or "kaggle.com" in path:
                return self._handle_kaggle_url(path, v1_findings)

            if path.endswith(".csv"):
                df = pd.read_csv(path)
            elif path.endswith(".parquet"):
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
            result = self.audit(df, topic=os.path.basename(path), v1_findings=v1_findings)
            result.update({
                "name": os.path.basename(path),
                "shape": f"{result['rows']:,} rows × {result['cols']} cols",
                "ml_tasks": self._suggest_tasks(result.get("problem_type", "unknown"), result["rows"]),
                "source": "user-provided",
                "score": self._score_audited_dataframe(df, v1_findings, result.get("risks", [])),
            })
            return result

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
        """Backward-compatible: return best single candidate or sentinel."""
        discovered = self.discover(topic, v1_findings)
        datasets = discovered.get("datasets", [])
        if not datasets:
            return {
                "name": "No dataset found",
                "score": 0.0,
                "risks": ["No matching public datasets — provide your own with --dataset"],
                "ml_tasks": [],
                "source": "none",
                "problem_type": v1_findings.get("problem_type", "unknown"),
            }
        best = datasets[0]
        best.setdefault("ml_tasks", self._suggest_tasks(best.get("problem_type", "unknown"), 0))
        confidence = best.get("selection_confidence", "low")
        if confidence == "low":
            best.setdefault("risks", [])
            best["risks"].append(
                "Low-confidence dataset match — consider passing an explicit --dataset path/URL"
            )
        return best

    # ── Private: search backends ──────────────────────────────────

    def _search_kaggle(self, topic: str) -> list:
        """Requires: pip install kaggle + ~/.kaggle/kaggle.json"""
        try:
            import kaggle
            datasets = kaggle.api.dataset_list(search=topic)
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
                params={"search": topic, "limit": 5},
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
            result = self.audit(df, topic=ref, v1_findings=v1_findings)
            result.update({
                "name": os.path.basename(csv_files[0]),
                "shape": f"{result['rows']:,} rows × {result['cols']} cols",
                "ml_tasks": self._suggest_tasks(result.get("problem_type", "unknown"), result["rows"]),
                "score": self._score_audited_dataframe(df, v1_findings, result.get("risks", [])),
            })
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

    def _score_dataset(self, ds: dict, topic: str, v1_findings: dict) -> dict:
        size_score = min(1.0, math.log10(max(ds.get("size_mb", 1) * 1000, 1)) / 6)
        pop_score = min(1.0, math.log10(max(ds.get("downloads", 1), 1)) / 5)
        topic_score = self._topic_similarity(topic, ds.get("title", ""))

        access_score = self._accessibility_score(ds)
        metadata_score = self._metadata_completeness_score(ds)

        raw_score = (
            0.28 * topic_score
            + 0.20 * pop_score
            + 0.12 * size_score
            + 0.22 * access_score
            + 0.18 * metadata_score
        )

        penalty = 0.0
        if ds.get("downloads", 0) <= 0:
            penalty += 0.12
        if topic_score < 0.2:
            penalty += 0.10
        if len((ds.get("title") or ds.get("name") or "").strip()) < 6:
            penalty += 0.06

        final_score = max(0.0, min(1.0, raw_score - penalty))
        confidence = self._confidence_label(final_score, topic_score, access_score, metadata_score)

        rationale = self._why_recommended(ds, topic, v1_findings)
        rationale.append(
            f"Selection confidence: {confidence} (topic={topic_score:.2f}, access={access_score:.2f}, metadata={metadata_score:.2f})"
        )

        return {
            **ds,
            "score": round(final_score, 2),
            "selection_confidence": confidence,
            "selection_rationale": rationale,
            "why_recommended": rationale,
            "risks": [
                "Verify label column",
                "Check for imbalance",
                "Validate real-world relevance",
            ],
            "problem_type": v1_findings.get("problem_type", "unknown"),
            "ml_tasks": self._suggest_tasks(v1_findings.get("problem_type", "unknown"), 0),
        }

    def _accessibility_score(self, ds: dict) -> float:
        source = (ds.get("source") or "").lower()
        has_url = bool(ds.get("url"))
        downloads = ds.get("downloads", 0) or 0

        base = 0.2
        if source in {"kaggle", "kaggle-url", "huggingface"}:
            base += 0.35
        if has_url:
            base += 0.25
        if downloads > 0:
            base += 0.20
        return min(1.0, base)

    def _metadata_completeness_score(self, ds: dict) -> float:
        checks = [
            bool((ds.get("name") or "").strip()),
            bool((ds.get("title") or "").strip()),
            bool(ds.get("url")),
            ds.get("downloads") is not None,
            ds.get("source") is not None,
        ]
        return sum(1 for ok in checks if ok) / len(checks)

    def _confidence_label(
        self,
        final_score: float,
        topic_score: float,
        access_score: float,
        metadata_score: float,
    ) -> str:
        if (
            final_score >= 0.72
            and topic_score >= 0.35
            and access_score >= 0.60
            and metadata_score >= 0.70
        ):
            return "high"
        if final_score >= 0.50 and access_score >= 0.40 and metadata_score >= 0.50:
            return "medium"
        return "low"

    def _topic_similarity(self, topic: str, text: str) -> float:
        topic_words = set(re.findall(r"\w+", topic.lower()))
        text_words = set(re.findall(r"\w+", (text or "").lower()))
        overlap = len(topic_words & text_words)
        return min(1.0, overlap / max(len(topic_words), 1))

    def _why_recommended(self, ds: dict, topic: str, v1_findings: dict) -> list:
        reasons = []

        if ds.get("downloads", 0) > 10000:
            reasons.append("Popular dataset with high usage")

        if ds.get("size_mb", 0) > 50:
            reasons.append("Large dataset suitable for training")

        title = (ds.get("title") or "").lower()
        if any(word in title for word in topic.lower().split()):
            reasons.append("Strong keyword match with topic")

        if not reasons:
            reasons.append("Relevant to problem domain")

        return reasons

    def _score_audited_dataframe(self, df, v1_findings: dict, risks: list) -> float:
        rows, cols = df.shape
        size_score = min(1.0, math.log10(max(rows * cols, 1)) / 6)
        task_score = 0.8 if v1_findings.get("problem_type", "unknown") != "unknown" else 0.4
        final_score = (
            0.3 * size_score
            + 0.3 * 0.85
            + 0.2 * task_score
            + 0.2 * (1.0 if not risks else 0.5)
        )
        return round(final_score, 2)

    def _audit_dataframe(self, df, topic: str, v1_findings: dict) -> dict:
        rows, cols = df.shape
        risks = []
        label_col = self._detect_label(df)

        missing_pct = df.isnull().sum().sum() / max(rows * cols, 1)
        if missing_pct > 0.1:
            risks.append(f"{missing_pct:.0%} missing values")

        dup_pct = df.duplicated().sum() / max(rows, 1)
        if dup_pct > 0.05:
            risks.append(f"{dup_pct:.0%} duplicate rows")

        if label_col and df[label_col].nunique() < 20:
            counts = df[label_col].value_counts()
            if len(counts) > 1:
                ratio = counts.iloc[0] / counts.iloc[-1]
                if ratio > 2:
                    risks.append(f"Class imbalance {ratio:.1f}:1")

        return {
            "rows": rows,
            "cols": cols,
            "label_column": label_col,
            "risks": risks,
            "problem_type": v1_findings.get("problem_type", "unknown"),
        }

    def _score_dataframe(self, df, path: str, v1_findings: dict) -> dict:
        """Backward-compatible dataframe scoring API."""
        audited = self._audit_dataframe(df, path, v1_findings)
        rows = audited["rows"]
        cols = audited["cols"]
        problem_type = audited["problem_type"]
        label_col = audited["label_column"]
        risks = list(audited["risks"])
        if rows < 1000:
            risks.append("Small dataset (<1k rows) — model may overfit")

        return {
            "name":          os.path.basename(path),
            "score":         self._score_audited_dataframe(df, v1_findings, risks),
            "shape":         f"{rows:,} rows × {cols} cols",
            "label_column":  label_col,
            "risks":         risks,
            "ml_tasks":      self._suggest_tasks(problem_type, rows),
            "source":        "user-provided",
            "problem_type":  problem_type,
        }

    def _compute_score(self, candidate: dict, topic: str, v1_findings: dict) -> dict:
        """Backward-compatible single-candidate scoring API."""
        return self._score_dataset(candidate, topic, v1_findings)

    def _detect_label(self, df) -> str:
        hints = ["target", "label", "class", "y"]
        for col in df.columns:
            if any(h in col.lower() for h in hints):
                return col
        if len(df.columns) == 0:
            return None
        return min(df.columns, key=lambda c: df[c].nunique())

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
