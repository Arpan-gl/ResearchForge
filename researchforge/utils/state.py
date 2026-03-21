"""
State — save/load last pipeline run to ~/.researchforge/last_run.json
"""

import json
import datetime
from pathlib import Path

_STATE_PATH = Path.home() / ".researchforge" / "last_run.json"


def save_state(results: dict) -> None:
    """Persist the pipeline result dict to disk."""
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(results)
    payload["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(_STATE_PATH, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def load_state() -> dict:
    """Load the last run result from disk, or return empty dict."""
    if _STATE_PATH.exists():
        try:
            with open(_STATE_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}
