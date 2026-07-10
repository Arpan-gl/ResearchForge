"""Deterministic optimization agent for minimal training configs."""

import json
from copy import deepcopy
from pathlib import Path

from researchforge.training.karpathy_minimal import MinimalTrainer


class OptimizationAgent:
    def __init__(self, trainer: MinimalTrainer | None = None):
        self.trainer = trainer or MinimalTrainer()

    def optimize(self, config_path: str, output_dir: str = "artifacts/optimization") -> dict:
        base_config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        config_data = base_config["data"] if "data" in base_config else base_config
        search_space = self._search_space(config_data)

        trials = []
        best = None
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        for idx, params in enumerate(search_space):
            trial_config = deepcopy(base_config)
            trial_data = trial_config["data"] if "data" in trial_config else trial_config
            trial_data.update(params)
            trial_config_path = output_root / f"trial_{idx}.json"
            trial_output_dir = output_root / f"trial_{idx}"
            trial_config_path.write_text(json.dumps(trial_config), encoding="utf-8")
            result = self.trainer.run(str(trial_config_path), output_dir=str(trial_output_dir))
            final_loss = float(result["data"]["final_loss"])
            trial = {
                "trial_index": idx,
                "params": params,
                "final_loss": final_loss,
                "checkpoint_path": result["data"]["checkpoint_path"],
            }
            trials.append(trial)
            if best is None or final_loss < best["final_loss"]:
                best = trial

        return {
            "data": {
                "best_config": best,
                "trial_history": trials,
            },
            "provenance": {
                "source": str(config_path),
                "retrieved_at": timestamp(),
                "agent": "optimization",
            },
            "confidence": "computed",
        }

    @staticmethod
    def _search_space(config_data: dict) -> list[dict]:
        base_lr = float(config_data.get("learning_rate", 3e-4))
        base_epochs = int(config_data.get("epochs", 5))
        return [
            {"learning_rate": base_lr, "epochs": base_epochs},
            {"learning_rate": base_lr * 0.5, "epochs": max(1, base_epochs + 1)},
            {"learning_rate": base_lr * 1.5, "epochs": max(1, base_epochs - 1)},
        ]


def timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
