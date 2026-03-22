"""
Tests for Settings config loading, env var overrides, and defaults.
"""

import os
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch


def test_defaults():
    """Settings should return default values when no config exists."""
    with tempfile.TemporaryDirectory() as tmp:
        fake_home = Path(tmp)
        with patch.object(Path, "home", return_value=fake_home):
            # Also clear env vars
            env = {k: v for k, v in os.environ.items()
                   if k not in ("OLLAMA_URL", "RF_MODEL")}
            with patch.dict(os.environ, env, clear=True):
                from researchforge.config.settings import Settings
                s = Settings()
                assert s.ollama_url == "http://localhost:11434"
                assert s.llm_model == "llama3"
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
        with open(config_dir / "config.json", "w") as f:
            json.dump(config, f)

        with patch.object(Path, "home", return_value=fake_home):
            env = {k: v for k, v in os.environ.items()
                   if k not in ("OLLAMA_URL", "RF_MODEL", "KAGGLE_USERNAME", "KAGGLE_KEY")}
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
        with open(config_dir / "config.json", "w") as f:
            json.dump(config, f)

        with patch.object(Path, "home", return_value=fake_home):
            with patch.dict(os.environ, {"OLLAMA_URL": "http://env:11434", "RF_MODEL": "gemma"}):
                from importlib import reload
                import researchforge.config.settings as sm
                reload(sm)
                s = sm.Settings()
                assert s.ollama_url == "http://env:11434"
                assert s.llm_model == "gemma"


def test_broken_config_falls_back_to_defaults():
    """Broken/corrupt config.json should not crash — fall back to defaults."""
    with tempfile.TemporaryDirectory() as tmp:
        fake_home = Path(tmp)
        config_dir = fake_home / ".researchforge"
        config_dir.mkdir()
        with open(config_dir / "config.json", "w") as f:
            f.write("NOT VALID JSON {{{{")

        with patch.object(Path, "home", return_value=fake_home):
            env = {k: v for k, v in os.environ.items()
                   if k not in ("OLLAMA_URL", "RF_MODEL")}
            with patch.dict(os.environ, env, clear=True):
                from importlib import reload
                import researchforge.config.settings as sm
                reload(sm)
                s = sm.Settings()
                assert s.ollama_url == "http://localhost:11434"


def test_cli_warning_filter_is_targeted():
    from researchforge import cli

    with patch("warnings.filterwarnings") as fw:
        cli._configure_warning_filters()

    fw.assert_called_once()
    kwargs = fw.call_args.kwargs
    assert kwargs["module"] == r"pydantic\._internal\._fields"
    assert "protected namespace" in kwargs["message"]
