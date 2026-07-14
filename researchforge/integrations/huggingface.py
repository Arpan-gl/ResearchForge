"""Hugging Face dataset and model discovery.

The integration deliberately keeps discovery metadata separate from downloads.
Search and previews are read-only; callers must explicitly request a snapshot.
"""

from datetime import datetime, timezone
from pathlib import Path


class HuggingFaceIntegration:
    def __init__(self, token: str = ""):
        self.token = token or None

    @staticmethod
    def _tokenize(text) -> set[str]:
        cleaned = str(text or "").strip().lower()
        for char in ("/", "-", "_", ",", ".", "(", ")", ":"):
            cleaned = cleaned.replace(char, " ")
        return {token for token in cleaned.split() if token}

    def _score(self, query: str, item) -> tuple:
        tags = getattr(item, "tags", []) or []
        searchable = f"{getattr(item, 'id', '')} {getattr(item, 'description', '')} {' '.join(map(str, tags))}"
        overlap = len(self._tokenize(query).intersection(self._tokenize(searchable)))
        return overlap, getattr(item, "likes", 0) or 0, getattr(item, "downloads", 0) or 0

    @staticmethod
    def _provenance(url: str, agent: str = "dataset") -> dict:
        return {
            "source": url,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "agent": agent,
        }

    def search_datasets(self, query: str, limit: int = 5) -> list[dict]:
        try:
            from datasets import get_dataset_split_names, load_dataset
            from huggingface_hub import HfApi

            api = HfApi(token=self.token)
            datasets = list(api.list_datasets(search=query, limit=limit, full=True))
            datasets.sort(key=lambda item: self._score(query, item), reverse=True)
            results = []
            for item in datasets[:limit]:
                repo_id = item.id
                sample = None
                splits = []
                try:
                    splits = get_dataset_split_names(repo_id, token=self.token)
                    split = "train" if "train" in splits else (splits[0] if splits else None)
                    if split:
                        stream = load_dataset(
                            repo_id,
                            split=split,
                            streaming=True,
                            trust_remote_code=True,
                            token=self.token,
                        )
                        sample = next(iter(stream), None)
                except Exception as exc:
                    sample = f"Preview unavailable: {exc}"

                url = f"https://huggingface.co/datasets/{repo_id}"
                results.append({
                    "id": repo_id,
                    "name": repo_id,
                    "title": repo_id,
                    "downloads": getattr(item, "downloads", 0) or 0,
                    "likes": getattr(item, "likes", 0) or 0,
                    "size_mb": 0,
                    "description": getattr(item, "description", "") or "",
                    "tags": getattr(item, "tags", []) or [],
                    "splits": splits,
                    "sample": sample,
                    "source": "huggingface",
                    "url": url,
                    "provenance": self._provenance(url),
                })
            return results
        except Exception:
            return []

    def search_models(self, query: str, limit: int = 5) -> list[dict]:
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=self.token)
            models = list(api.list_models(search=query, limit=limit, full=True))
            models.sort(key=lambda item: self._score(query, item), reverse=True)
            results = []
            for item in models[:limit]:
                repo_id = item.id
                url = f"https://huggingface.co/{repo_id}"
                results.append({
                    "id": repo_id,
                    "name": repo_id,
                    "pipeline_tag": getattr(item, "pipeline_tag", None),
                    "library_name": getattr(item, "library_name", None),
                    "tags": getattr(item, "tags", []) or [],
                    "downloads": getattr(item, "downloads", 0) or 0,
                    "likes": getattr(item, "likes", 0) or 0,
                    "source": "huggingface",
                    "url": url,
                    "provenance": self._provenance(url, agent="training_planner"),
                })
            return results
        except Exception:
            return []

    def download_dataset(self, repo_id: str, output_dir: str = "outputs/datasets/huggingface") -> str:
        from huggingface_hub import snapshot_download

        local_dir = Path(output_dir) / repo_id.replace("/", "_")
        local_dir.mkdir(parents=True, exist_ok=True)
        return snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            token=self.token,
        )

    def download_model(self, repo_id: str, output_dir: str = "outputs/models/huggingface") -> str:
        from huggingface_hub import snapshot_download

        local_dir = Path(output_dir) / repo_id.replace("/", "_")
        local_dir.mkdir(parents=True, exist_ok=True)
        return snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(local_dir),
            token=self.token,
        )
