"""
Tests for Settings config loading, env var overrides, and defaults.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch


def test_defaults():
    """Settings should return default values when no config exists."""
    with tempfile.TemporaryDirectory() as tmp:
        fake_home = Path(tmp)
        with patch.object(Path, "home", return_value=fake_home):
            env = {k: v for k, v in os.environ.items() if k not in ("OLLAMA_URL", "RF_MODEL")}
            with patch.dict(os.environ, env, clear=True):
                from researchforge.config.settings import Settings

                s = Settings()
                assert s.ollama_url == "http://localhost:11434"
                assert s.llm_model == "qwen/qwen3-coder-next"
                assert s.kaggle_username == ""


def test_config_file_is_loaded():
    """Settings should load values from config.json."""
    with tempfile.TemporaryDirectory() as tmp:
        fake_home = Path(tmp)
        config_dir = fake_home / ".researchforge"
        config_dir.mkdir()
        config = {
            "ollama_url": "http://othermachine:11434",
            "model": "mistral",
            "kaggle_username": "testuser",
            "kaggle_key": "abc123",
        }
        with open(config_dir / "config.json", "w", encoding="utf-8") as handle:
            json.dump(config, handle)

        with patch.object(Path, "home", return_value=fake_home):
            env = {k: v for k, v in os.environ.items() if k not in ("OLLAMA_URL", "RF_MODEL", "KAGGLE_USERNAME", "KAGGLE_KEY")}
            with patch.dict(os.environ, env, clear=True):
                from importlib import reload
                import researchforge.config.settings as sm

                reload(sm)
                s = sm.Settings()
                assert s.ollama_url == "http://othermachine:11434"
                assert s.llm_model == "mistral"
                assert s.kaggle_username == "testuser"


def test_env_vars_override_config():
    """Environment variables should take priority over config.json."""
    with tempfile.TemporaryDirectory() as tmp:
        fake_home = Path(tmp)
        config_dir = fake_home / ".researchforge"
        config_dir.mkdir()
        config = {"ollama_url": "http://config:11434", "model": "phi3"}
        with open(config_dir / "config.json", "w", encoding="utf-8") as handle:
            json.dump(config, handle)

        with patch.object(Path, "home", return_value=fake_home):
            with patch.dict(os.environ, {"OLLAMA_URL": "http://env:11434", "RF_MODEL": "gemma"}):
                from importlib import reload
                import researchforge.config.settings as sm

                reload(sm)
                s = sm.Settings()
                assert s.ollama_url == "http://env:11434"
                assert s.llm_model == "gemma"


def test_broken_config_falls_back_to_defaults():
    """Broken config.json should not crash and should fall back to defaults."""
    with tempfile.TemporaryDirectory() as tmp:
        fake_home = Path(tmp)
        config_dir = fake_home / ".researchforge"
        config_dir.mkdir()
        with open(config_dir / "config.json", "w", encoding="utf-8") as handle:
            handle.write("NOT VALID JSON {{{{")

        with patch.object(Path, "home", return_value=fake_home):
            env = {k: v for k, v in os.environ.items() if k not in ("OLLAMA_URL", "RF_MODEL")}
            with patch.dict(os.environ, env, clear=True):
                from importlib import reload
                import researchforge.config.settings as sm

                reload(sm)
                s = sm.Settings()
                assert s.ollama_url == "http://localhost:11434"
                assert s.llm_model == "qwen/qwen3-coder-next"


def test_init_wizard_prompts_for_manual_storage_when_ollama_unreachable():
    with tempfile.TemporaryDirectory() as tmp:
        fake_home = Path(tmp)
        answers = iter(
            [
                "http://localhost:11434",
                "a",
                "postgresql://db.example/researchforge",
                "bolt://graph.example:7687",
                "redis://redis.example:6379/0",
                "auto",
                "qwen/qwen3-coder-next",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "mlruns",
                "yes",
            ]
        )

        with patch.object(Path, "home", return_value=fake_home):
            with patch("researchforge.config.settings.requests.get", side_effect=RuntimeError("down")):
                with patch("builtins.input", side_effect=lambda _: next(answers)):
                    from importlib import reload
                    import researchforge.config.settings as sm

                    reload(sm)
                    config = sm.Settings.init_wizard()

        env_path = fake_home / ".researchforge" / ".env"
        assert config["storage_mode"] == "manual"
        assert env_path.exists()
        env_text = env_path.read_text(encoding="utf-8")
        assert "POSTGRES_URL=postgresql://db.example/researchforge" in env_text
        assert "GRAPH_URL=bolt://graph.example:7687" in env_text


def test_cli_warning_filter_is_targeted():
    from researchforge import cli

    with patch("warnings.filterwarnings") as fw:
        cli._configure_warning_filters()

    fw.assert_called_once()
    kwargs = fw.call_args.kwargs
    assert kwargs["module"] == r"pydantic\._internal\._fields"
    assert "protected namespace" in kwargs["message"]
