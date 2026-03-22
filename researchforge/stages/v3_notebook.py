"""
V3 Notebook Generation
-----------------------
- Infers problem type from V1 + V2.
- Selects model based on task type and dataset size.
- Generates a runnable notebook with separated data/EDA/train/eval sections.
"""

import json
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from researchforge.config.settings import Settings


class V3Notebook:
    def __init__(self):
        self.settings = Settings()

    def generate(
        self,
        topic: str,
        v1_findings: dict,
        v2_dataset: dict,
        model_override: str = None,
    ) -> dict:
        problem_type = (
            v2_dataset.get("problem_type")
            or v1_findings.get("problem_type", "classification")
        )
        dataset_name = (
            v2_dataset.get("path")
            or v2_dataset.get("local_path")
            or v2_dataset.get("name", "dataset.csv")
        )
        label_col = v2_dataset.get("label_column", "target")
        dataset_candidates = self._build_dataset_candidates(v2_dataset, dataset_name)

        model_name, model_reason = self._select_model(problem_type, v2_dataset, model_override)
        metric_name, expected_range = self._expected_metrics(problem_type, model_name)

        nb = new_notebook()
        nb.cells = [
            self._title_cell(topic, model_name, model_reason),
            self._data_loading_cell(dataset_candidates, label_col),
            self._eda_cell(),
            self._preprocessing_cell(),
            self._model_cell(problem_type, model_name),
            self._training_cell(problem_type),
            self._evaluation_cell(problem_type, metric_name),
            self._summary_cell(topic, model_name, model_reason, v1_findings),
        ]

        safe_topic = "".join(c if c.isalnum() else "_" for c in topic)[:30]
        nb_path = f"rf_{safe_topic}.ipynb"
        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        return {
            "notebook_path": nb_path,
            "model": model_name,
            "model_reason": model_reason,
            "metric_name": metric_name,
            "expected_range": expected_range,
            "problem_type": problem_type,
            "dataset_candidates": dataset_candidates,
            "label_column_requested": label_col,
        }

    def _build_dataset_candidates(self, v2_dataset: dict, dataset_name: str) -> list[str]:
        candidates = []
        for key in ["path", "local_path", "download_path", "name", "title"]:
            value = (v2_dataset.get(key) or "").strip()
            if value:
                candidates.append(value)
        if dataset_name:
            candidates.append(dataset_name)
        # Preserve order while deduplicating.
        return list(dict.fromkeys(candidates))

    def _select_model(self, problem_type: str, v2_dataset: dict, model_override: str = None) -> tuple[str, str]:
        override = (model_override or "").strip().lower()
        if override in {"gnn", "gatconv"}:
            return "GATConv", "User override selected graph model"
        if override in {"bert", "distilbert"}:
            return "DistilBERT", "User override selected NLP transformer"
        if override in {"xgboost", "xgb"}:
            return "XGBoost", "User override selected gradient boosting baseline"
        if override in {"lightgbm", "lgbm"}:
            return "LightGBM", "User override selected efficient tabular booster"

        rows = self._parse_rows(v2_dataset.get("shape", ""))

        if problem_type == "graph":
            return "GATConv", "Graph task detected from research/dataset signals"
        if problem_type == "nlp":
            return "DistilBERT", "Pretrained transformer for text-heavy tasks"
        if problem_type == "regression":
            return "LightGBMRegressor", "Good default for tabular regression"

        if rows < 10_000:
            return "XGBoost", "Small dataset: robust baseline with strong bias/variance tradeoff"
        return "LightGBM", "Efficient for medium-to-large structured datasets"

    def _parse_rows(self, shape_str: str) -> int:
        try:
            return int(shape_str.split("rows")[0].replace(",", "").strip())
        except Exception:
            return 10_000

    def _expected_metrics(self, problem_type: str, model: str):
        defaults = {
            "classification": ("F1-macro", "0.78-0.86"),
            "regression": ("RMSE", "model-dependent"),
            "graph": ("F1-macro", "0.71-0.82"),
            "nlp": ("Accuracy", "0.82-0.90"),
        }
        return defaults.get(problem_type, ("F1-macro", "0.75-0.85"))

    def _title_cell(self, topic, model, reason):
        return new_markdown_cell(
            f"# AI Generated ML Notebook\n\n"
            f"**Topic:** {topic}  \n"
            f"**Model:** {model}  \n"
            f"**Why this model:** {reason}  \n\n"
            "This notebook is auto-generated. Review before running.\n"
        )

    def _data_loading_cell(self, dataset_candidates, label_col):
        candidates_literal = json.dumps(dataset_candidates)
        requested_label = json.dumps(label_col)
        return new_code_cell(f"""\
# Load Data

import pandas as pd
import os
import tempfile
import glob
from pathlib import Path

DATASET_CANDIDATES = {candidates_literal}
REQUESTED_LABEL = {requested_label}


def _maybe_download_kaggle_dataset(dataset_ref: str):
    # Try to download a Kaggle dataset ref and return first CSV/Parquet path.
    try:
        import kaggle
    except Exception:
        return None

    try:
        tmp_dir = tempfile.mkdtemp(prefix="rf_kaggle_nb_")
        kaggle.api.dataset_download_files(dataset_ref, path=tmp_dir, unzip=True)
        csv_files = glob.glob(os.path.join(tmp_dir, "**", "*.csv"), recursive=True)
        parquet_files = glob.glob(os.path.join(tmp_dir, "**", "*.parquet"), recursive=True)
        all_files = csv_files + parquet_files
        if all_files:
            print(f"Downloaded Kaggle dataset '{{dataset_ref}}' to '{{tmp_dir}}'")
            return all_files[0]
    except Exception as e:
        print(f"Kaggle download failed for '{{dataset_ref}}': {{e}}")
    return None

candidate_paths = []
for raw in DATASET_CANDIDATES:
    raw = (raw or "").strip()
    if not raw:
        continue
    candidate_paths.append(raw)
    candidate_paths.append(str(Path.cwd() / raw))

env_dataset = os.getenv("RF_DATASET_PATH", "").strip()
if env_dataset:
    candidate_paths.insert(0, env_dataset)

seen = set()
ordered_candidates = []
for path in candidate_paths:
    if path and path not in seen:
        ordered_candidates.append(path)
        seen.add(path)

DATASET_PATH = None
for candidate in ordered_candidates:
    if os.path.exists(candidate):
        DATASET_PATH = candidate
        break

if DATASET_PATH is None:
    csv_candidates = sorted(str(p) for p in Path.cwd().glob("*.csv"))
    parquet_candidates = sorted(str(p) for p in Path.cwd().glob("*.parquet"))
    inferred = csv_candidates + parquet_candidates
    if len(inferred) == 1:
        DATASET_PATH = inferred[0]
        print(f"Using inferred dataset path: {{DATASET_PATH}}")

# As a final fallback, try treating candidates like Kaggle dataset refs: owner/dataset
if DATASET_PATH is None:
    for raw in DATASET_CANDIDATES:
        raw = (raw or "").strip()
        if "/" in raw and not raw.lower().endswith((".csv", ".parquet")):
            downloaded = _maybe_download_kaggle_dataset(raw)
            if downloaded and os.path.exists(downloaded):
                DATASET_PATH = downloaded
                break

if DATASET_PATH is None:
    attempted = "\\n".join(f"- {{p}}" for p in ordered_candidates[:10])
    raise FileNotFoundError(
        "Could not locate a dataset file. Tried candidate paths:\\n"
        + (attempted or "- (no candidates provided)")
        + "\\nSet RF_DATASET_PATH or edit DATASET_CANDIDATES."
    )

if DATASET_PATH.lower().endswith(".parquet"):
    df = pd.read_parquet(DATASET_PATH)
else:
    df = pd.read_csv(DATASET_PATH)

target_hints = [
    "target", "label", "class", "y", "output", "result", "outcome", "made", "shot"
]

if REQUESTED_LABEL in df.columns:
    TARGET_COL = REQUESTED_LABEL
else:
    hinted_cols = [c for c in df.columns if any(h in c.lower() for h in target_hints)]
    if hinted_cols:
        TARGET_COL = hinted_cols[0]
    else:
        TARGET_COL = df.columns[-1]
    print(f"Requested label '{{REQUESTED_LABEL}}' not found. Using '{{TARGET_COL}}' instead.")

print(f"Dataset path: {{DATASET_PATH}}")
print(f"Target column: {{TARGET_COL}}")
print(df.shape)
df.head()
""")

    def _eda_cell(self):
        return new_code_cell(f"""\
# Basic EDA

import matplotlib.pyplot as plt

print(\"Missing values:\")
print(df.isnull().sum())

print(\"\\nDuplicates:\", df.duplicated().sum())

if TARGET_COL in df.columns:
    df[TARGET_COL].value_counts().plot(kind=\"bar\")
    plt.title(\"Target Distribution\")
    plt.show()
""")

    def _preprocessing_cell(self):
        return new_code_cell(f"""\
# Preprocessing

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np

# Drop likely ID columns
df = df[[col for col in df.columns if \"id\" not in col.lower()]]

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Encode categorical features conservatively
for col in X.select_dtypes(include=[\"object\"]).columns:
    if X[col].nunique() < 50:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    else:
        X = X.drop(columns=[col])

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
if num_cols:
    imputer = SimpleImputer(strategy=\"median\")
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(imputer.fit_transform(X[num_cols]))

X = X.fillna(0)

stratify_arg = y if y.nunique() < 20 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify_arg
)
""")

    def _model_cell(self, problem_type, model_name):
        if problem_type == "graph" or model_name == "GATConv":
            return new_code_cell(f"""\
# Graph Model (GATConv)

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import DataLoader
from researchforge.utils.graph_builder import GraphBuilder

device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")
print(f\"Using device: {{device}}\")

class ResearchForgeGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, dropout=0.3)
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=1)
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return self.classifier(x[0])

POSITION_COLS = [\"x\", \"y\", \"velocity_x\", \"velocity_y\"]
LABEL_COL = TARGET_COL

builder = GraphBuilder()
graph_list = builder.build_from_dataframe(
    df,
    position_cols=POSITION_COLS,
    label_col=LABEL_COL,
    proximity_threshold=3.0,
)

loader = DataLoader(graph_list, batch_size=32, shuffle=True)
n_features = graph_list[0].x.shape[1]
n_classes = df[LABEL_COL].nunique()
model = ResearchForgeGNN(n_features, hidden_channels=64, out_channels=n_classes).to(device)
""")

        if problem_type == "nlp" or model_name == "DistilBERT":
            return new_code_cell(f"""\
# NLP Model (DistilBERT)

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")
print(f\"Using device: {{device}}\")

TEXT_COL = \"text\"
LABEL_COL = TARGET_COL

if TEXT_COL not in df.columns:
    raise ValueError(\"Expected a 'text' column for NLP notebooks\")

tokenizer = DistilBertTokenizerFast.from_pretrained(\"distilbert-base-uncased\")
n_labels = df[LABEL_COL].nunique()
model = DistilBertForSequenceClassification.from_pretrained(
    \"distilbert-base-uncased\", num_labels=n_labels
).to(device)
""")

        if problem_type == "regression" or model_name == "LightGBMRegressor":
            return new_code_cell("""\
from lightgbm import LGBMRegressor
model = LGBMRegressor()
""")

        if model_name == "XGBoost":
            return new_code_cell("""\
import xgboost as xgb
model = xgb.XGBClassifier()
""")

        if model_name == "LightGBM":
            return new_code_cell("""\
import lightgbm as lgb
model = lgb.LGBMClassifier()
""")

        return new_code_cell("""\
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
""")

    def _training_cell(self, problem_type):
        return new_code_cell("""\
# Training

model.fit(X_train, y_train)
""")

    def _evaluation_cell(self, problem_type, metric_name):
        if problem_type == "regression":
            return new_code_cell("""\
from sklearn.metrics import mean_squared_error

preds = model.predict(X_test)
print("RMSE:", mean_squared_error(y_test, preds, squared=False))
""")

        return new_code_cell(f"""\
from sklearn.metrics import classification_report, f1_score

preds = model.predict(X_test)
print(classification_report(y_test, preds))
print("{metric_name}:", f1_score(y_test, preds, average=\"macro\"))
""")

    def _summary_cell(self, topic, model, reason, v1_findings):
        return new_code_cell(f"""\
# Summary

import json

summary = {{
    \"topic\": \"{topic}\",
    \"model\": \"{model}\",
    \"reason\": \"{reason}\",
    \"research_insights\": {json.dumps(v1_findings.get("key_findings", [])[:3])}
}}

print(json.dumps(summary, indent=2))
""")
