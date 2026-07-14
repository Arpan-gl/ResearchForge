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
DEFAULT_RESEARCH_MODEL = "google/gemini-2.5-flash-lite"
DEFAULT_OPENROUTER_FREE_MODEL = "openrouter/free"
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
    def research_model(self) -> str:
        return os.environ.get("RF_RESEARCH_MODEL") or self._config.get("research_model") or DEFAULT_RESEARCH_MODEL

    @property
    def openrouter_free_model(self) -> str:
        return os.environ.get("RF_OPENROUTER_FREE_MODEL") or self._config.get("openrouter_free_model") or DEFAULT_OPENROUTER_FREE_MODEL

    @property
    def semantic_scholar_key(self) -> str:
        return os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or self._config.get("semantic_scholar_key", "")

    @property
    def github_token(self) -> str:
        return os.environ.get("GITHUB_TOKEN") or self._config.get("github_token", "")

    @property
    def enable_multi_source(self) -> bool:
        value = os.environ.get("RF_ENABLE_MULTI_SOURCE")
        if value is None:
            value = self._config.get("enable_multi_source", True)
        return str(value).strip().lower() not in {"0", "false", "no", "off"}

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
        return os.environ.get("MLFLOW_TRACKING_URI") or self._config.get("mlflow_tracking_uri") or "http://localhost:5000"

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
    def _prompt_secret(label: str, default: str = "") -> str:
        suffix = " [configured]" if default else ""
        return input(f"  {label}{suffix} (paste allowed): ").strip() or default

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
        llm_provider = existing.get("llm_provider", "auto")
        model = existing.get("model", DEFAULT_LLM_MODEL)
        openrouter_api_key = existing.get("openrouter_api_key", "")
        openrouter_base_url = existing.get("openrouter_base_url", DEFAULT_OPENROUTER_BASE_URL)
        research_model = existing.get("research_model", DEFAULT_RESEARCH_MODEL)
        kaggle_user = existing.get("kaggle_username", "")
        kaggle_key = existing.get("kaggle_key", "")
        tavily_key = existing.get("tavily_api_key", "")
        semantic_scholar_key = existing.get("semantic_scholar_key", "")
        github_token = existing.get("github_token", "")
        huggingface_token = existing.get("huggingface_token", "")
        openai_key = existing.get("openai_api_key", "")
        mlflow_uri = existing.get("mlflow_tracking_uri", "http://localhost:5000")
        enable_multi_source = existing.get("enable_multi_source", True)

        if not ollama_reachable:
            print("\n  Ollama isn't running locally. Choose storage setup:")
            print("  [a] I'll provide connection strings now")
            print("  [b] Spin them up for me via Docker Compose")
            storage_choice = cls._prompt_value("Storage choice", storage_mode[:1] or "a").lower()
            while storage_choice not in {"a", "b"}:
                print("  Enter 'a' for manual connection strings or 'b' for Docker Compose.")
                storage_choice = cls._prompt_value("Storage choice", "a").lower()

            if storage_choice == "a":
                storage_mode = "manual"
                postgres_url = cls._require_connection_detail("Postgres URL", postgres_url)
                graph_url = cls._require_connection_detail("Graph URL or KuzuDB path", graph_url)
                redis_url = cls._require_connection_detail("Redis URL", redis_url)
            else:
                storage_mode = "docker"
                compose_path = Path.cwd() / "infra" / "docker-compose.yml"
                cls._run_compose_stack(compose_path)
                postgres_url = DEFAULT_POSTGRES_URL
                graph_url = DEFAULT_GRAPH_URL
                redis_url = DEFAULT_REDIS_URL

            cls._write_runtime_env(
                settings.runtime_env_path,
                {
                    "POSTGRES_URL": postgres_url,
                    "GRAPH_URL": graph_url,
                    "REDIS_URL": redis_url,
                },
            )

        llm_provider = cls._prompt_value("LLM provider", llm_provider)
        model = cls._prompt_value("Default model", model)
        if llm_provider in {"auto", "openrouter"}:
            openrouter_api_key = cls._prompt_secret("OpenRouter API key", openrouter_api_key)
            openrouter_base_url = cls._prompt_value("OpenRouter base URL", openrouter_base_url)
        research_model = cls._prompt_value("Research model via OpenRouter", research_model)

        print("\n  Optional integrations (press Enter to keep current value or skip):")
        kaggle_user = cls._prompt_value("Kaggle username", kaggle_user)
        if kaggle_user:
            kaggle_key = cls._prompt_secret("Kaggle API key", kaggle_key)
        else:
            kaggle_key = ""
        tavily_key = cls._prompt_secret("Tavily API key", tavily_key)
        semantic_scholar_key = cls._prompt_secret("Semantic Scholar API key", semantic_scholar_key)
        github_token = cls._prompt_secret("GitHub token", github_token)
        huggingface_token = cls._prompt_secret("Hugging Face token", huggingface_token)
        openai_key = cls._prompt_secret("OpenAI API key", openai_key)
        mlflow_uri = cls._prompt_value("MLflow tracking URI", mlflow_uri)
        multi_source_default = "yes" if enable_multi_source else "no"
        multi_source_value = cls._prompt_value("Enable multi-source research (yes/no)", multi_source_default).lower()
        while multi_source_value not in {"yes", "no", "y", "n"}:
            print("  Enter yes or no.")
            multi_source_value = cls._prompt_value("Enable multi-source research (yes/no)", multi_source_default).lower()
        enable_multi_source = multi_source_value in {"yes", "y"}

        config = {
            "ollama_url": ollama_url,
            "storage_mode": storage_mode,
            "postgres_url": postgres_url,
            "graph_url": graph_url,
            "redis_url": redis_url,
            "llm_provider": llm_provider,
            "model": model,
            "openrouter_api_key": openrouter_api_key,
            "openrouter_base_url": openrouter_base_url,
            "research_model": research_model,
            "kaggle_username": kaggle_user,
            "kaggle_key": kaggle_key,
            "tavily_api_key": tavily_key,
            "semantic_scholar_key": semantic_scholar_key,
            "github_token": github_token,
            "huggingface_token": huggingface_token,
            "openai_api_key": openai_key,
            "mlflow_tracking_uri": mlflow_uri,
            "enable_multi_source": enable_multi_source,
        }

        cls._write_runtime_env(
            settings.runtime_env_path,
            {
                "POSTGRES_URL": postgres_url,
                "GRAPH_URL": graph_url,
                "REDIS_URL": redis_url,
                "OPENROUTER_API_KEY": openrouter_api_key,
                "OPENROUTER_BASE_URL": openrouter_base_url,
                "RF_RESEARCH_MODEL": research_model,
                "KAGGLE_USERNAME": kaggle_user,
                "KAGGLE_KEY": kaggle_key,
                "TAVILY_API_KEY": tavily_key,
                "SEMANTIC_SCHOLAR_API_KEY": semantic_scholar_key,
                "GITHUB_TOKEN": github_token,
                "HUGGINGFACE_TOKEN": huggingface_token,
                "OPENAI_API_KEY": openai_key,
                "MLFLOW_TRACKING_URI": mlflow_uri,
                "RF_ENABLE_MULTI_SOURCE": "true" if enable_multi_source else "false",
            },
        )

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
