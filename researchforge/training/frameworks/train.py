"""Framework-backed trainer using Hugging Face Trainer for text classification."""

import json
import os
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=32):
        self.texts = list(texts)
        self.labels = list(labels)
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        token_ids = [self.vocab.get(token, self.vocab["[UNK]"]) for token in text.lower().split()]
        token_ids = token_ids[: self.max_length]
        attention = [1] * len(token_ids)
        while len(token_ids) < self.max_length:
            token_ids.append(self.vocab["[PAD]"])
            attention.append(0)
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.float32),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class BagOfWordsClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_labels: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        embedded = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        logits = self.classifier(pooled)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


class FrameworkTrainer:
    def run(self, config_path: str, output_dir: str = "artifacts/frameworks") -> dict:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        run_config = config["data"] if "data" in config else config
        framework = run_config.get("framework", "")
        if framework == "transformers":
            return self._run_transformers(run_config, output_dir)
        if framework == "lightning":
            raise RuntimeError("Lightning framework path requires the `lightning` package, which is not installed in this environment.")
        raise ValueError(f"Unsupported framework path: {framework}")

    def _run_transformers(self, run_config: dict, output_dir: str) -> dict:
        Trainer, TrainingArguments = self._load_transformers()

        dataset_path = run_config["dataset_path"]
        label_column = run_config.get("label_column") or "label"
        batch_size = int(run_config.get("batch_size", 8))
        epochs = int(run_config.get("epochs", 1))
        learning_rate = float(run_config.get("learning_rate", 2e-5))

        df = pd.read_csv(dataset_path)
        text_column = self._pick_text_column(df, label_column)
        labels, encoded_labels = self._encode_labels(df[label_column])
        vocab = self._build_vocab(df[text_column])
        dataset = TextClassificationDataset(df[text_column].fillna(""), encoded_labels, vocab)
        model = BagOfWordsClassifier(vocab_size=len(vocab), embed_dim=32, num_labels=len(labels))

        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        training_args = TrainingArguments(
            output_dir=str(output_root / "trainer_output"),
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            logging_strategy="no",
            save_strategy="no",
            eval_strategy="no",
            report_to=[],
            disable_tqdm=True,
            seed=42,
        )
        trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
        train_result = trainer.train()

        checkpoint_dir = output_root / "checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / "pytorch_model.bin")
        metrics_path = output_root / "metrics.json"
        metrics = {
            "train_loss": float(train_result.training_loss),
            "text_column": text_column,
            "label_column": label_column,
            "num_labels": len(labels),
            "framework": "transformers",
        }
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        return {
            "data": {
                "checkpoint_path": str(checkpoint_dir),
                "metrics_path": str(metrics_path),
                "framework": "transformers",
                "train_loss": metrics["train_loss"],
            },
            "provenance": {
                "source": dataset_path,
                "retrieved_at": timestamp(),
                "agent": "training_frameworks",
            },
            "confidence": "computed",
        }

    @staticmethod
    def _load_transformers():
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("USE_FLAX", "0")
        from transformers import Trainer, TrainingArguments

        return Trainer, TrainingArguments

    @staticmethod
    def _pick_text_column(df: pd.DataFrame, label_column: str) -> str:
        text_candidates = [column for column in df.columns if column != label_column and df[column].dtype == object]
        if text_candidates:
            return text_candidates[0]
        non_label = [column for column in df.columns if column != label_column]
        if not non_label:
            raise ValueError("No feature column available for framework training.")
        return non_label[0]

    @staticmethod
    def _encode_labels(series):
        values = series.fillna("unknown").astype(str)
        labels = sorted(values.unique())
        index = {label: idx for idx, label in enumerate(labels)}
        return labels, [index[value] for value in values]

    @staticmethod
    def _build_vocab(texts) -> dict:
        vocab = {"[PAD]": 0, "[UNK]": 1}
        for text in texts.fillna("").astype(str):
            for token in text.lower().split():
                if token not in vocab:
                    vocab[token] = len(vocab)
        return vocab


def timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
