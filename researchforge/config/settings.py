"""
Settings and init flow for ResearchForge.
"""

import json
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import requests

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_LLM_MODEL = "qwen/qwen3-coder-next"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_POSTGRES_URL = "postgresql://researchforge:researchforge@localhost:5432/researchforge"
DEFAULT_GRAPH_URL = "bolt://neo4j:researchforge@localhost:7687"
DEFAULT_REDIS_URL = "redis://localhost:6379/0"


class Settings:
    def __init__(self):
        self.config_dir = Path.home() / ".researchforge"
        self.config_path = self.config_dir / "config.json"
        self.runtime_env_path = self.config_dir / ".env"
        self._config = self._load()

    def _load(self) -> dict:
        if self.config_path.exists():
            try:
                with open(self.config_path, encoding="utf-8") as handle:
                    return json.load(handle)
            except Exception:
                pass
        return {}

    @property
    def ollama_url(self) -> str:
        return os.environ.get("OLLAMA_URL") or self._config.get("ollama_url") or DEFAULT_OLLAMA_URL

    @property
    def llm_model(self) -> str:
        return os.environ.get("RF_MODEL") or self._config.get("model") or DEFAULT_LLM_MODEL

    @property
    def llm_provider(self) -> str:
        return os.environ.get("RF_LLM_PROVIDER") or self._config.get("llm_provider") or "auto"

    @property
    def openrouter_api_key(self) -> str:
        return os.environ.get("OPENROUTER_API_KEY") or self._config.get("openrouter_api_key") or ""

    @property
    def openrouter_base_url(self) -> str:
        return (
            os.environ.get("OPENROUTER_BASE_URL")
            or self._config.get("openrouter_base_url")
            or DEFAULT_OPENROUTER_BASE_URL
        )

    @property
    def postgres_url(self) -> str:
        return os.environ.get("POSTGRES_URL") or self._config.get("postgres_url") or ""

    @property
    def graph_url(self) -> str:
        return os.environ.get("GRAPH_URL") or self._config.get("graph_url") or ""

    @property
    def redis_url(self) -> str:
        return os.environ.get("REDIS_URL") or self._config.get("redis_url") or ""

    @property
    def kaggle_username(self) -> str:
        return os.environ.get("KAGGLE_USERNAME") or self._config.get("kaggle_username", "")

    @property
    def kaggle_key(self) -> str:
        return os.environ.get("KAGGLE_KEY") or self._config.get("kaggle_key", "")

    @property
    def mlflow_tracking_uri(self) -> str:
        return os.environ.get("MLFLOW_TRACKING_URI") or self._config.get("mlflow_tracking_uri") or "mlruns"

    @property
    def tavily_api_key(self) -> str:
        return os.environ.get("TAVILY_API_KEY") or self._config.get("tavily_api_key", "")

    @property
    def huggingface_token(self) -> str:
        return os.environ.get("HUGGINGFACE_TOKEN") or self._config.get("huggingface_token", "")

    @property
    def openai_api_key(self) -> str:
        return os.environ.get("OPENAI_API_KEY") or self._config.get("openai_api_key", "")

    @classmethod
    def _load_existing_config(cls, config_path: Path) -> dict:
        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as handle:
                    return json.load(handle)
            except Exception:
                pass
        return {}

    @staticmethod
    def _prompt_value(label: str, default: str = "") -> str:
        suffix = f" [{default}]" if default else ""
        return input(f"  {label}{suffix}: ").strip() or default

    @staticmethod
    def _looks_like_connection_detail(value: str) -> bool:
        if not value:
            return False
        if "://" in value:
            parsed = urlparse(value)
            return bool(parsed.scheme and (parsed.netloc or parsed.path))
        return any(token in value for token in ("/", "\\", ":"))

    @classmethod
    def _require_connection_detail(cls, label: str, default: str = "") -> str:
        while True:
            value = cls._prompt_value(label, default)
            if cls._looks_like_connection_detail(value):
                return value
            print(f"  {label} must look like a connection string or local path.")

    @staticmethod
    def _write_runtime_env(runtime_env_path: Path, values: dict) -> None:
        lines = [f"{key}={value}" for key, value in values.items() if value]
        runtime_env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _ping_ollama(ollama_url: str) -> bool:
        try:
            response = requests.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=3)
            return response.ok
        except Exception:
            return False

    @staticmethod
    def _run_compose_stack(compose_path: Path) -> None:
        subprocess.run(
            ["docker", "compose", "-f", str(compose_path), "up", "-d"],
            check=True,
        )

    @classmethod
    def init_wizard(cls):
        settings = cls()
        settings.config_dir.mkdir(parents=True, exist_ok=True)
        existing = cls._load_existing_config(settings.config_path)

        print("\n  ResearchForge first-time setup")
        print("  --------------------------------")
        if existing:
            print("  Existing config found. Press Enter to keep current values.\n")

        ollama_url = cls._prompt_value("Ollama URL", existing.get("ollama_url", DEFAULT_OLLAMA_URL))
        ollama_reachable = cls._ping_ollama(ollama_url)

        storage_mode = existing.get("storage_mode", "")
        postgres_url = existing.get("postgres_url", "")
        graph_url = existing.get("graph_url", "")
        redis_url = existing.get("redis_url", "")

        if kaggle_user:
            kaggle_key = input(
                f"  Kaggle API key [{existing.get('kaggle_key', '')}]: "
            ).strip() or existing.get("kaggle_key", "")
        else:
            kaggle_key = ""

        mlflow_uri = input(
            f"  MLflow tracking URI [{existing.get('mlflow_tracking_uri', 'mlruns')}]: "
        ).strip() or existing.get("mlflow_tracking_uri", "mlruns")

        print("\n  Optional API integrations (press Enter to skip):")

        tavily_key = input(
            f"  Tavily API key [{existing.get('tavily_api_key', '')}]: "
        ).strip() or existing.get("tavily_api_key", "")

        hf_token = input(
            f"  Hugging Face token [{existing.get('huggingface_token', '')}]: "
        ).strip() or existing.get("huggingface_token", "")

        openai_key = input(
            f"  OpenAI API key [{existing.get('openai_api_key', '')}]: "
        ).strip() or existing.get("openai_api_key", "")

        config = {
            "ollama_url": ollama_url,
            "llm_provider": llm_provider,
            "model": model,
            "kaggle_username": kaggle_user,
            "kaggle_key": kaggle_key,
            "mlflow_tracking_uri": mlflow_uri,
            "tavily_api_key": tavily_key,
            "huggingface_token": hf_token,
            "openai_api_key": openai_key,
        }

        with open(settings.config_path, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)

        print(f"\n  Saved config to {settings.config_path}")
        print(f"  Ollama reachable: {'yes' if ollama_reachable else 'no'}")
        print(f"  LLM provider: {llm_provider}")
        print(f"  Default model: {model}")
        if storage_mode:
            print(f"  Storage mode: {storage_mode}")
        if postgres_url:
            print("  DB connection details saved for later phases.")
        return config
