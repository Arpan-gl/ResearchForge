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
import json
import requests
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus
from researchforge.config.settings import Settings


class V2Datasets:
    def __init__(self, ollama_url: str = None, model: str = None):
        self.settings = Settings()
        self.ollama_url = ollama_url or self.settings.ollama_url
        self.model = model or self.settings.llm_model

    # ── Query helpers ───────────────────────────────────────────

    def _normalize_text(self, value: str | None) -> str:
        return (value or "").strip()

    def _tokenize(self, text: str) -> set:
        cleaned = self._normalize_text(text).lower()
        for char in ("/", "-", "_", ",", ".", "(", ")", ":"):
            cleaned = cleaned.replace(char, " ")
        return {token for token in cleaned.split() if token}

    def _flatten_keywords(self, value) -> list[str]:
        if not value:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            flattened = []
            for item in value:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("title") or ""
                    if name:
                        flattened.append(str(name))
                else:
                    flattened.append(str(item))
            return flattened
        if isinstance(value, dict):
            return [str(value.get("name") or value.get("title") or value)]
        return [str(value)]

    def build_search_query(self, query: str, extra_inputs: dict | None = None) -> str:
        extra_inputs = extra_inputs or {}
        parts = [self._normalize_text(query)]
        useful_keys = ["task", "domain", "keywords"]

        for key in useful_keys:
            value = extra_inputs.get(key)
            if not value:
                continue
            if isinstance(value, list):
                parts.append(" ".join(str(v) for v in value if v))
            else:
                parts.append(str(value))
        return " ".join(part for part in parts if part)

    def _build_search_context(self, topic: str, v1_findings: dict) -> dict:
        keywords = self._flatten_keywords(v1_findings.get("datasets"))
        return {
            "task": v1_findings.get("problem_type"),
            "domain": v1_findings.get("topic", topic),
            "keywords": keywords[:6],
        }

    # ── Public interface ──────────────────────────────────────────

    def discover(self, topic: str, v1_findings: dict) -> dict:
        """Discover and rank datasets from Kaggle + HuggingFace."""
        search_context = self._build_search_context(topic, v1_findings)
        search_query = self.build_search_query(topic, search_context)

        candidates = []
        candidates += self._search_kaggle(search_query)
        candidates += self._search_huggingface(search_query)

        if not candidates:
            return {
                "datasets": [],
                "search_query": search_query,
                "search_urls": self._build_search_urls(search_query),
            }

        scored = [self._score_dataset(c, topic, v1_findings) for c in candidates]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return {
            "datasets": scored[:5],
            "search_query": search_query,
            "search_urls": self._build_search_urls(search_query),
        }

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
            package = self._build_dataset_package(
                df=df,
                dataset_name=os.path.basename(path),
                label_column=result.get("label_column"),
                risks=result.get("risks", []),
            )
            if package:
                result["package"] = package
                result["package_dir"] = package.get("package_dir")
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
                "search_query": discovered.get("search_query"),
                "search_urls": discovered.get("search_urls", {}),
            }
        best = datasets[0]
        best.setdefault("ml_tasks", self._suggest_tasks(best.get("problem_type", "unknown"), 0))
        confidence = best.get("selection_confidence", "low")
        if confidence == "low":
            best.setdefault("risks", [])
            best["risks"].append(
                "Low-confidence dataset match — consider passing an explicit --dataset path/URL"
            )
        best["search_query"] = discovered.get("search_query")
        best["search_urls"] = discovered.get("search_urls", {})

        if best.get("source") == "huggingface":
            sample = self._preview_hf_sample(best.get("name") or best.get("id"))
            if sample is not None:
                best["sample"] = sample

        best["related_papers"] = self._get_arxiv_papers(
            discovered.get("search_query") or topic
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
            from huggingface_hub import HfApi
            api = HfApi()
            datasets = list(api.list_datasets(search=topic, limit=10, full=True))
            datasets.sort(key=lambda ds: self._hf_relevance_score(topic, ds), reverse=True)
            results = []
            for ds in datasets[:5]:
                results.append({
                    "name":      ds.id,
                    "title":     ds.id,
                    "downloads": getattr(ds, "downloads", 0) or 0,
                    "likes":     getattr(ds, "likes", 0) or 0,
                    "size_mb":   0,
                    "source":    "huggingface",
                    "url":       f"https://huggingface.co/datasets/{ds.id}",
                    "description": getattr(ds, "description", "") or "",
                    "tags": getattr(ds, "tags", []) or [],
                })
            return results
        except Exception:
            return self._search_huggingface_http(topic)

    def _search_huggingface_http(self, topic: str) -> list:
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
                    "description": ds.get("description", ""),
                    "tags": ds.get("tags", []) or [],
                })
            return results
        except Exception:
            return []

    def _hf_relevance_score(self, query: str, ds) -> tuple:
        query_tokens = self._tokenize(query)
        searchable_text = f"{getattr(ds, 'id', '')} {getattr(ds, 'description', '')} {' '.join(getattr(ds, 'tags', []) or [])}"
        overlap = len(query_tokens.intersection(self._tokenize(searchable_text)))
        return overlap, getattr(ds, "likes", 0) or 0, getattr(ds, "downloads", 0) or 0

    def _preview_hf_sample(self, repo_id: str | None):
        if not repo_id:
            return None
        try:
            from datasets import get_dataset_split_names, load_dataset
            splits = get_dataset_split_names(repo_id)
            if not splits:
                return "Could not load preview (no splits found)."
            split = "train" if "train" in splits else splits[0]
            ds_stream = load_dataset(repo_id, split=split, streaming=True, trust_remote_code=True)
            return next(iter(ds_stream))
        except Exception as e:
            return f"Could not load preview ({e})."

    def _get_arxiv_papers(self, query: str, max_results: int = 3) -> list:
        try:
            import arxiv
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            papers = []
            for res in client.results(search):
                papers.append({
                    "title": res.title,
                    "url": res.entry_id,
                    "pdf": res.pdf_url,
                })
            return papers
        except Exception:
            return []

    def download_hf_dataset(self, repo_id: str, output_dir: str = "outputs/datasets/huggingface") -> str | None:
        """Physically download the dataset files to a local folder."""
        if not repo_id:
            return None
        try:
            from huggingface_hub import snapshot_download
            safe_name = repo_id.replace("/", "_")
            local_dir = os.path.join(output_dir, safe_name)
            os.makedirs(local_dir, exist_ok=True)
            return snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir)
        except Exception:
            return None

    def _build_search_urls(self, query: str) -> dict:
        return {
            "huggingface": f"https://huggingface.co/datasets?search={quote_plus(query)}",
            "kaggle": f"https://www.kaggle.com/search?q={quote_plus(query)}+in%3Adatasets+sortBy%3Adate",
        }

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
            package = self._build_dataset_package(
                df=df,
                dataset_name=ref,
                label_column=result.get("label_column"),
                risks=result.get("risks", []),
            )
            if package:
                result["package"] = package
                result["package_dir"] = package.get("package_dir")
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

    # ── Dataset optimization outputs ─────────────────────────────

    def _build_dataset_package(
        self,
        df,
        dataset_name: str,
        label_column: str | None,
        risks: list,
        output_dir: str = "outputs/datasets",
    ) -> dict | None:
        if df is None:
            return None

        safe_name = self._safe_name(dataset_name or "dataset")
        package_dir = Path(output_dir) / safe_name
        package_dir.mkdir(parents=True, exist_ok=True)

        cleaned, cleaning_report = self._clean_dataframe(df)
        splits = self._split_dataframe(cleaned, label_column)

        cleaned_path = package_dir / "cleaned.csv"
        cleaned.to_csv(cleaned_path, index=False)

        split_paths = {}
        for split_name, split_df in splits.items():
            split_path = package_dir / f"{split_name}.csv"
            split_df.to_csv(split_path, index=False)
            split_paths[split_name] = str(split_path)

        feature_report = self._feature_report(cleaned, label_column)
        feature_path = package_dir / "feature_report.json"
        feature_path.write_text(json.dumps(feature_report, indent=2), encoding="utf-8")

        bias_report = self._bias_report(cleaned, label_column)
        bias_path = package_dir / "bias_report.json"
        bias_path.write_text(json.dumps(bias_report, indent=2), encoding="utf-8")

        leakage_report = self._leakage_report(cleaned, label_column)
        leakage_path = package_dir / "leakage_report.json"
        leakage_path.write_text(json.dumps(leakage_report, indent=2), encoding="utf-8")

        feature_plan = self._feature_plan(cleaned, label_column)
        feature_plan_path = package_dir / "feature_plan.json"
        feature_plan_path.write_text(json.dumps(feature_plan, indent=2), encoding="utf-8")

        health_report = self._health_report(cleaned, label_column, bias_report, leakage_report)
        health_path = package_dir / "health_report.json"
        health_path.write_text(json.dumps(health_report, indent=2), encoding="utf-8")

        risks_path = package_dir / "risks.json"
        risks_path.write_text(json.dumps(risks or [], indent=2), encoding="utf-8")

        preprocess_path = package_dir / "preprocessing.py"
        preprocess_path.write_text(self._preprocessing_script(label_column), encoding="utf-8")

        card_path = package_dir / "dataset_card.md"
        card_path.write_text(
            self._dataset_card_text(
                dataset_name,
                cleaned.shape,
                label_column,
                risks,
                cleaning_report,
                health_report,
            ),
            encoding="utf-8",
        )

        return {
            "package_dir": str(package_dir),
            "cleaned_path": str(cleaned_path),
            "split_paths": split_paths,
            "feature_report": str(feature_path),
            "feature_plan": str(feature_plan_path),
            "bias_report": str(bias_path),
            "leakage_report": str(leakage_path),
            "health_report": str(health_path),
            "risks_file": str(risks_path),
            "preprocessing_script": str(preprocess_path),
            "dataset_card": str(card_path),
            "cleaning_report": cleaning_report,
            "dataset_health_score": health_report.get("score", 0),
            "rows_before": int(df.shape[0]),
            "rows_after": int(cleaned.shape[0]),
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        }

    def _clean_dataframe(self, df):
        cleaned = df.copy()
        original_rows = cleaned.shape[0]
        cleaned = cleaned.drop_duplicates()

        missing_before = int(cleaned.isnull().sum().sum())
        for col in cleaned.columns:
            if cleaned[col].dtype.kind in "biufc":
                cleaned[col] = cleaned[col].fillna(cleaned[col].median())
            else:
                mode = cleaned[col].mode(dropna=True)
                fill_value = mode.iloc[0] if not mode.empty else ""
                cleaned[col] = cleaned[col].fillna(fill_value)
        missing_after = int(cleaned.isnull().sum().sum())

        return cleaned, {
            "rows_before": int(original_rows),
            "rows_after": int(cleaned.shape[0]),
            "duplicates_removed": int(original_rows - cleaned.shape[0]),
            "missing_before": missing_before,
            "missing_after": missing_after,
        }

    def _split_dataframe(self, df, label_column: str | None) -> dict:
        try:
            from sklearn.model_selection import train_test_split
        except Exception:
            return {"train": df, "val": df.iloc[:0], "test": df.iloc[:0]}

        stratify_col = None
        if label_column and label_column in df.columns:
            if df[label_column].nunique() < 25:
                stratify_col = df[label_column]

        train_df, temp_df = train_test_split(
            df, test_size=0.3, random_state=42, stratify=stratify_col
        )
        stratify_temp = None
        if stratify_col is not None:
            stratify_temp = temp_df[label_column]
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=42, stratify=stratify_temp
        )
        return {"train": train_df, "val": val_df, "test": test_df}

    def _feature_report(self, df, label_column: str | None) -> dict:
        report = {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "label_column": label_column,
            "columns": [],
        }
        for col in df.columns:
            series = df[col]
            report["columns"].append({
                "name": col,
                "dtype": str(series.dtype),
                "missing": int(series.isnull().sum()),
                "unique": int(series.nunique()),
            })
        return report

    def _bias_report(self, df, label_column: str | None) -> dict:
        report = {
            "label_column": label_column,
            "class_distribution": {},
            "gini": None,
            "minority_ratio": None,
        }
        if not label_column or label_column not in df.columns:
            return report

        counts = df[label_column].value_counts(dropna=False)
        report["class_distribution"] = {str(k): int(v) for k, v in counts.items()}

        values = counts.values.tolist()
        if values:
            report["gini"] = round(self._gini(values), 4)
            if len(values) > 1:
                report["minority_ratio"] = round(min(values) / max(values), 4)
        return report

    def _leakage_report(self, df, label_column: str | None) -> dict:
        report = {"label_column": label_column, "suspected_leakage": []}
        if not label_column or label_column not in df.columns:
            return report

        target = df[label_column]
        for col in df.columns:
            if col == label_column:
                continue
            series = df[col]

            if series.dtype.kind in "biufc" and target.dtype.kind in "biufc":
                if series.nunique() > 1 and target.nunique() > 1:
                    corr = series.corr(target)
                    if corr is not None and abs(corr) > 0.98:
                        report["suspected_leakage"].append(
                            {"column": col, "reason": f"high correlation ({corr:.3f})"}
                        )
            else:
                try:
                    agreement = (series.astype(str) == target.astype(str)).mean()
                    if agreement > 0.98:
                        report["suspected_leakage"].append(
                            {"column": col, "reason": f"label leakage (agreement {agreement:.2%})"}
                        )
                except Exception:
                    continue
        return report

    def _feature_plan(self, df, label_column: str | None) -> dict:
        plan = {"label_column": label_column, "features": []}
        for col in df.columns:
            if col == label_column:
                continue
            series = df[col]
            if series.dtype.kind in "biufc":
                transform = "scale"
            elif series.nunique() <= 50:
                transform = "label_encode"
            else:
                transform = "drop_high_cardinality"
            plan["features"].append({"name": col, "dtype": str(series.dtype), "transform": transform})
        return plan

    def _health_report(self, df, label_column: str | None, bias_report: dict, leakage_report: dict) -> dict:
        rows = max(int(df.shape[0]), 1)
        missing_ratio = 1.0 - (df.isnull().sum().sum() / (rows * max(df.shape[1], 1)))

        class_balance = 0.5
        if bias_report.get("gini") is not None:
            class_balance = max(0.0, 1.0 - float(bias_report.get("gini", 0.0)))

        diversity_scores = []
        for col in df.columns:
            unique_ratio = df[col].nunique() / rows
            diversity_scores.append(min(1.0, unique_ratio))
        diversity = sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0.0

        leakage_penalty = 0.0 if leakage_report.get("suspected_leakage") else 1.0

        score = (
            0.25 * max(0.0, missing_ratio)
            + 0.25 * class_balance
            + 0.2 * diversity
            + 0.2 * (1.0 if rows >= 1000 else rows / 1000)
            + 0.1 * leakage_penalty
        )
        return {
            "score": round(score * 100, 2),
            "missing_ratio": round(missing_ratio, 4),
            "class_balance": round(class_balance, 4),
            "diversity": round(diversity, 4),
            "leakage_penalty": leakage_penalty,
            "rows": rows,
        }

    def _preprocessing_script(self, label_column: str | None) -> str:
        label_literal = label_column or "target"
        return (
            "import pandas as pd\n"
            "\n"
            "def preprocess(df):\n"
            "    df = df.drop_duplicates().copy()\n"
            "    for col in df.columns:\n"
            "        if pd.api.types.is_numeric_dtype(df[col]):\n"
            "            df[col] = df[col].fillna(df[col].median())\n"
            "        else:\n"
            "            mode = df[col].mode(dropna=True)\n"
            "            fill_value = mode.iloc[0] if not mode.empty else ''\n"
            "            df[col] = df[col].fillna(fill_value)\n"
            "    return df\n"
            "\n"
            f"LABEL_COLUMN = '{label_literal}'\n"
        )

    def _dataset_card_text(
        self,
        dataset_name: str,
        shape: tuple,
        label_column: str | None,
        risks: list,
        cleaning_report: dict,
        health_report: dict | None,
    ) -> str:
        risks_list = "\n".join(f"- {r}" for r in (risks or [])) or "- None"
        health_score = health_report.get("score") if health_report else None
        health_line = f"- Health score: {health_score}\n" if health_score is not None else ""
        return (
            f"# Dataset Card: {dataset_name}\n\n"
            f"- Rows: {shape[0]}\n"
            f"- Columns: {shape[1]}\n"
            f"- Label column: {label_column or 'unknown'}\n\n"
            f"{health_line}\n"
            "## Cleaning Summary\n"
            f"- Rows before: {cleaning_report.get('rows_before', shape[0])}\n"
            f"- Rows after: {cleaning_report.get('rows_after', shape[0])}\n"
            f"- Duplicates removed: {cleaning_report.get('duplicates_removed', 0)}\n"
            f"- Missing values before: {cleaning_report.get('missing_before', 0)}\n"
            f"- Missing values after: {cleaning_report.get('missing_after', 0)}\n\n"
            "## Risks\n"
            f"{risks_list}\n"
        )

    def _safe_name(self, value: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
        return cleaned.strip("_") or "dataset"

    def _gini(self, values: list[int]) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        total = sum(sorted_vals)
        if total == 0:
            return 0.0
        cumulative = sum((i + 1) * v for i, v in enumerate(sorted_vals))
        return (2 * cumulative) / (n * total) - (n + 1) / n

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
