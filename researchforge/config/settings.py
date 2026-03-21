"""
Settings — reads from ~/.researchforge/config.json or environment variables
"""

import os
import json
from pathlib import Path


class Settings:
    def __init__(self):
        self.config_path = Path.home() / ".researchforge" / "config.json"
        self._config = self._load()

    def _load(self) -> dict:
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    @property
    def ollama_url(self) -> str:
        return (
            os.environ.get("OLLAMA_URL") or
            self._config.get("ollama_url") or
            "http://localhost:11434"
        )

    @property
    def llm_model(self) -> str:
        return (
            os.environ.get("RF_MODEL") or
            self._config.get("model") or
            "llama3"
        )

    @property
    def kaggle_username(self) -> str:
        return os.environ.get("KAGGLE_USERNAME") or self._config.get("kaggle_username", "")

    @property
    def kaggle_key(self) -> str:
        return os.environ.get("KAGGLE_KEY") or self._config.get("kaggle_key", "")

    @property
    def mlflow_tracking_uri(self) -> str:
        return (
            os.environ.get("MLFLOW_TRACKING_URI") or
            self._config.get("mlflow_tracking_uri") or
            "mlruns"
        )

    @classmethod
    def init_wizard(cls):
        """Run on first install: researchforge init"""
        config_dir = Path.home() / ".researchforge"
        config_dir.mkdir(parents=True, exist_ok=True)

        print("\n  ResearchForge — First-time Setup")
        print("  ─────────────────────────────────────────")

        existing = {}
        config_path = config_dir / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    existing = json.load(f)
                print(f"  (Existing config found — press Enter to keep current values)\n")
            except Exception:
                pass

        ollama_url = input(
            f"  Ollama URL [{existing.get('ollama_url', 'http://localhost:11434')}]: "
        ).strip() or existing.get("ollama_url", "http://localhost:11434")

        model = input(
            f"  LLM model [{existing.get('model', 'llama3')}]: "
        ).strip() or existing.get("model", "llama3")

        kaggle_user = input(
            f"  Kaggle username (optional) [{existing.get('kaggle_username', '')}]: "
        ).strip() or existing.get("kaggle_username", "")

        if kaggle_user:
            kaggle_key = input(
                f"  Kaggle API key [{existing.get('kaggle_key', '')}]: "
            ).strip() or existing.get("kaggle_key", "")
        else:
            kaggle_key = ""

        mlflow_uri = input(
            f"  MLflow tracking URI [{existing.get('mlflow_tracking_uri', 'mlruns')}]: "
        ).strip() or existing.get("mlflow_tracking_uri", "mlruns")

        config = {
            "ollama_url": ollama_url,
            "model": model,
            "kaggle_username": kaggle_user,
            "kaggle_key": kaggle_key,
            "mlflow_tracking_uri": mlflow_uri,
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"\n  ✓ Config saved to {config_path}")
        print(f"  ✓ Ollama: {ollama_url}  Model: {model}")
        if kaggle_user:
            print(f"  ✓ Kaggle: {kaggle_user}")
        print(f"  ✓ MLflow tracking URI: {mlflow_uri}")
        print("\n  Ready. Try: researchforge run \"your research topic\"\n")
