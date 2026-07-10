"""Readable minimal trainer with explicit forward, loss, gradients, and optimizer step."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


class MinimalTrainer:
    def run(self, config_path: str, output_dir: str = "artifacts/minimal") -> dict:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        run_config = config["data"] if "data" in config else config
        dataset_path = run_config["dataset_path"]
        label_column = run_config.get("label_column") or None
        epochs = int(run_config.get("epochs", 5))
        learning_rate = float(run_config.get("learning_rate", 0.1))

        X, y, feature_names, label_column = self._load_dataset(dataset_path, label_column)
        n_samples, n_features = X.shape
        n_classes = int(np.max(y)) + 1

        rng = np.random.default_rng(42)
        weights = rng.normal(0, 0.01, size=(n_features, n_classes))
        bias = np.zeros((1, n_classes))

        loss_history = []
        for _ in range(epochs):
            logits = self._forward(X, weights, bias)
            probabilities = self._softmax(logits)
            loss = self._cross_entropy(probabilities, y)
            grad_w, grad_b = self._backward(X, probabilities, y)
            weights, bias = self._step(weights, bias, grad_w, grad_b, learning_rate)
            loss_history.append(float(loss))

        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_root / "minimal_checkpoint.npz"
        metrics_path = output_root / "metrics.json"
        np.savez(checkpoint_path, weights=weights, bias=bias, feature_names=np.array(feature_names), label_column=label_column)
        metrics_path.write_text(json.dumps({"loss_history": loss_history}, indent=2), encoding="utf-8")

        return {
            "data": {
                "checkpoint_path": str(checkpoint_path),
                "metrics_path": str(metrics_path),
                "epochs": epochs,
                "final_loss": loss_history[-1],
                "label_column": label_column,
            },
            "provenance": {
                "source": dataset_path,
                "retrieved_at": timestamp(),
                "agent": "training_minimal",
            },
            "confidence": "computed",
        }

    def _load_dataset(self, dataset_path: str, label_column: str | None):
        df = pd.read_csv(dataset_path)
        if label_column is None:
            label_column = df.columns[-1]
        y_raw = df[label_column]
        X_df = df.drop(columns=[label_column])
        X_df = pd.get_dummies(X_df, drop_first=False)
        X = X_df.to_numpy(dtype=float)
        X = self._standardize(X)
        labels, y = np.unique(y_raw.astype(str), return_inverse=True)
        return X, y.astype(int), list(X_df.columns), label_column

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        return (X - mean) / std

    def _forward(self, X: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
        return X @ weights + bias

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def _cross_entropy(self, probabilities: np.ndarray, y: np.ndarray) -> float:
        sample_indices = np.arange(len(y))
        clipped = np.clip(probabilities[sample_indices, y], 1e-9, 1.0)
        return float(-np.mean(np.log(clipped)))

    def _backward(self, X: np.ndarray, probabilities: np.ndarray, y: np.ndarray):
        n_samples = X.shape[0]
        grad_logits = probabilities.copy()
        grad_logits[np.arange(n_samples), y] -= 1.0
        grad_logits /= n_samples
        grad_w = X.T @ grad_logits
        grad_b = grad_logits.sum(axis=0, keepdims=True)
        return grad_w, grad_b

    def _step(self, weights: np.ndarray, bias: np.ndarray, grad_w: np.ndarray, grad_b: np.ndarray, learning_rate: float):
        weights = weights - learning_rate * grad_w
        bias = bias - learning_rate * grad_b
        return weights, bias


def timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
