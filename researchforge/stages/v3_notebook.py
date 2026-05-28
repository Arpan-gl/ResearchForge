"""
V3 Notebook Generation
-----------------------
- Infers problem type from V1 + V2.
- Selects model based on task type and dataset size.
- Generates a runnable notebook with separated data/EDA/train/eval sections.
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
import nbformat
import requests
from researchforge.utils.pdf_parser import parse_pdf_bytes
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

        model_package = self._generate_model_package(
            topic=topic,
            problem_type=problem_type,
            model_name=model_name,
            v1_findings=v1_findings,
            v2_dataset=v2_dataset,
            dataset_candidates=dataset_candidates,
            label_col=label_col,
        )

        return {
            "notebook_path": nb_path,
            "model": model_name,
            "model_reason": model_reason,
            "metric_name": metric_name,
            "expected_range": expected_range,
            "problem_type": problem_type,
            "dataset_candidates": dataset_candidates,
            "label_column_requested": label_col,
            "model_package_dir": model_package.get("package_dir") if model_package else None,
            "model_package_files": model_package.get("files") if model_package else None,
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

    # ── Model package generation ─────────────────────────────────

    def _generate_model_package(
        self,
        topic: str,
        problem_type: str,
        model_name: str,
        v1_findings: dict,
        v2_dataset: dict,
        dataset_candidates: list[str],
        label_col: str,
        output_dir: str = "outputs/model",
    ) -> dict:
        safe_topic = "".join(c if c.isalnum() else "_" for c in topic)[:30] or "topic"
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        package_dir = Path(output_dir) / f"{safe_topic}_{stamp}"
        package_dir.mkdir(parents=True, exist_ok=True)

        config_path = package_dir / "config.yaml"
        model_path = package_dir / "model.py"
        train_path = package_dir / "train.py"
        infer_path = package_dir / "inference.py"
        metrics_path = package_dir / "metrics.json"
        card_path = package_dir / "model_card.md"

        config_path.write_text(
            self._model_config_text(
                topic=topic,
                problem_type=problem_type,
                model_name=model_name,
                v1_findings=v1_findings,
                v2_dataset=v2_dataset,
                dataset_candidates=dataset_candidates,
                label_col=label_col,
            ),
            encoding="utf-8",
        )
        model_path.write_text(self._model_py_text(problem_type, model_name), encoding="utf-8")
        train_path.write_text(self._train_py_text(problem_type, model_name, dataset_candidates, label_col), encoding="utf-8")
        infer_path.write_text(self._inference_py_text(problem_type, model_name, label_col), encoding="utf-8")
        metrics_path.write_text(json.dumps({"metric": "", "value": None}, indent=2), encoding="utf-8")
        card_path.write_text(
            self._model_card_text(topic, model_name, problem_type, v1_findings, v2_dataset),
            encoding="utf-8",
        )

        research_path = package_dir / "research_extract.json"
        research_path.write_text(
            json.dumps(self._research_extract(v1_findings), indent=2),
            encoding="utf-8",
        )

        paper_blueprints = self._extract_paper_blueprints(v1_findings)
        paper_path = package_dir / "paper_blueprints.json"
        paper_path.write_text(json.dumps(paper_blueprints, indent=2), encoding="utf-8")

        return {
            "package_dir": str(package_dir),
            "files": {
                "config": str(config_path),
                "model": str(model_path),
                "train": str(train_path),
                "inference": str(infer_path),
                "metrics": str(metrics_path),
                "model_card": str(card_path),
                "research_extract": str(research_path),
                "paper_blueprints": str(paper_path),
            },
        }

    def _model_config_text(
        self,
        topic: str,
        problem_type: str,
        model_name: str,
        v1_findings: dict,
        v2_dataset: dict,
        dataset_candidates: list[str],
        label_col: str,
    ) -> str:
        candidates = ", ".join(dataset_candidates[:3]) if dataset_candidates else ""
        blueprint = json.dumps(v1_findings.get("research_blueprint", {}), indent=2)
        return (
            f"topic: {topic}\n"
            f"problem_type: {problem_type}\n"
            f"model: {model_name}\n"
            f"dataset_candidates: [{candidates}]\n"
            f"label_column: {label_col}\n"
            f"source_dataset: {v2_dataset.get('name', '')}\n"
            "training:\n"
            "  epochs: 3\n"
            "  batch_size: 32\n"
            "  learning_rate: 0.1\n"
            "  n_estimators: 200\n"
            "  max_depth: 6\n"
            "research_blueprint: |\n"
            + "\n".join(f"  {line}" for line in blueprint.splitlines())
            + "\n"
        )

    def _model_py_text(self, problem_type: str, model_name: str) -> str:
        if problem_type == "nlp" or model_name == "DistilBERT":
            return (
                "from transformers import DistilBertForSequenceClassification\n\n"
                "def build_model(num_labels, **kwargs):\n"
                "    return DistilBertForSequenceClassification.from_pretrained(\n"
                "        'distilbert-base-uncased', num_labels=num_labels\n"
                "    )\n"
            )
        if problem_type == "graph" or model_name == "GATConv":
            return (
                "import torch\n"
                "from torch_geometric.nn import GATConv\n\n"
                "class ResearchForgeGNN(torch.nn.Module):\n"
                "    def __init__(self, in_channels, hidden_channels, out_channels):\n"
                "        super().__init__()\n"
                "        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, dropout=0.3)\n"
                "        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=1)\n"
                "        self.classifier = torch.nn.Linear(hidden_channels, out_channels)\n\n"
                "    def forward(self, data):\n"
                "        x, edge_index = data.x, data.edge_index\n"
                "        x = self.conv1(x, edge_index)\n"
                "        x = self.conv2(x, edge_index)\n"
                "        return self.classifier(x[0])\n\n"
                "def build_model(in_channels, out_channels, **kwargs):\n"
                "    return ResearchForgeGNN(in_channels, 64, out_channels)\n"
            )
        if "Regressor" in model_name:
            return (
                "from lightgbm import LGBMRegressor\n\n\n"
                "def build_model(**kwargs):\n"
                "    return LGBMRegressor(**kwargs)\n"
            )
        if model_name == "XGBoost":
            return (
                "import xgboost as xgb\n\n\n"
                "def build_model(**kwargs):\n"
                "    return xgb.XGBClassifier(**kwargs)\n"
            )
        if model_name == "LightGBM":
            return (
                "import lightgbm as lgb\n\n\n"
                "def build_model(**kwargs):\n"
                "    return lgb.LGBMClassifier(**kwargs)\n"
            )
        return (
            "from sklearn.ensemble import RandomForestClassifier\n\n\n"
            "def build_model(**kwargs):\n"
            "    return RandomForestClassifier(**kwargs)\n"
        )

    def _train_py_text(
        self,
        problem_type: str,
        model_name: str,
        dataset_candidates: list[str],
        label_col: str,
    ) -> str:
        first_candidate = dataset_candidates[0] if dataset_candidates else "dataset.csv"
        if problem_type == "nlp" or model_name == "DistilBERT":
            return (
                "import os\n"
                "import yaml\n"
                "import pandas as pd\n"
                "from datasets import Dataset\n"
                "from transformers import DistilBertTokenizerFast, Trainer, TrainingArguments\n"
                "from model import build_model\n\n"
                "def _load_config(path='config.yaml'):\n"
                "    if not os.path.exists(path):\n"
                "        return {}\n"
                "    try:\n"
                "        with open(path, 'r', encoding='utf-8') as f:\n"
                "            return yaml.safe_load(f) or {}\n"
                "    except Exception:\n"
                "        return {}\n\n"
                "cfg = _load_config()\n"
                "train_cfg = cfg.get('training', {})\n"
                "epochs = int(train_cfg.get('epochs', 1))\n"
                "batch_size = int(train_cfg.get('batch_size', 8))\n"
                f"DATASET_PATH = os.getenv('RF_DATASET_PATH', '{first_candidate}')\n"
                f"LABEL_COL = os.getenv('RF_TARGET_COL', '{label_col}')\n"
                "TEXT_COL = os.getenv('RF_TEXT_COL', 'text')\n\n"
                "df = pd.read_csv(DATASET_PATH)\n"
                "if TEXT_COL not in df.columns:\n"
                "    raise ValueError('Expected a text column for NLP training')\n\n"
                "tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')\n"
                "dataset = Dataset.from_pandas(df[[TEXT_COL, LABEL_COL]])\n\n"
                "def tokenize(batch):\n"
                "    return tokenizer(batch[TEXT_COL], truncation=True)\n\n"
                "dataset = dataset.map(tokenize, batched=True)\n"
                "dataset = dataset.rename_column(LABEL_COL, 'labels')\n"
                "dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])\n\n"
                "model = build_model(num_labels=df[LABEL_COL].nunique())\n"
                "args = TrainingArguments(output_dir='checkpoints', num_train_epochs=epochs, per_device_train_batch_size=batch_size)\n"
                "trainer = Trainer(model=model, args=args, train_dataset=dataset)\n"
                "trainer.train()\n"
            )
        return (
            "import os\n"
            "import json\n"
            "import pandas as pd\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.metrics import f1_score, mean_squared_error\n"
            "import joblib\n"
            "import yaml\n"
            "from model import build_model\n\n"
            "def _load_config(path='config.yaml'):\n"
            "    if not os.path.exists(path):\n"
            "        return {}\n"
            "    try:\n"
            "        with open(path, 'r', encoding='utf-8') as f:\n"
            "            return yaml.safe_load(f) or {}\n"
            "    except Exception:\n"
            "        return {}\n\n"
            "cfg = _load_config()\n"
            "train_cfg = cfg.get('training', {})\n"
            "params = {\n"
            "    'learning_rate': train_cfg.get('learning_rate', 0.1),\n"
            "    'n_estimators': int(train_cfg.get('n_estimators', 200)),\n"
            "    'max_depth': int(train_cfg.get('max_depth', 6)),\n"
            "}\n"
            f"DATASET_PATH = os.getenv('RF_DATASET_PATH', '{first_candidate}')\n"
            f"TARGET_COL = os.getenv('RF_TARGET_COL', '{label_col}')\n\n"
            "df = pd.read_csv(DATASET_PATH)\n"
            "if TARGET_COL not in df.columns:\n"
            "    TARGET_COL = df.columns[-1]\n\n"
            "X = df.drop(columns=[TARGET_COL])\n"
            "y = df[TARGET_COL]\n"
            "seed = int(os.getenv('RF_SEED', '42'))\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)\n\n"
            "params['random_state'] = seed\n"
            "model = build_model(**params)\n"
            "model.fit(X_train, y_train)\n"
            "preds = model.predict(X_test)\n\n"
            "metrics = {}\n"
            "if y.dtype.kind in 'biu' and y.nunique() < 50:\n"
            "    metrics['f1_macro'] = float(f1_score(y_test, preds, average='macro'))\n"
            "else:\n"
            "    metrics['rmse'] = float(mean_squared_error(y_test, preds, squared=False))\n\n"
            "joblib.dump(model, 'model.joblib')\n"
            "with open('metrics.json', 'w', encoding='utf-8') as f:\n"
            "    json.dump(metrics, f, indent=2)\n"
        )

    def _inference_py_text(self, problem_type: str, model_name: str, label_col: str) -> str:
        if problem_type == "nlp" or model_name == "DistilBERT":
            return (
                "import torch\n"
                "from transformers import DistilBertTokenizerFast\n"
                "from model import build_model\n\n"
                "tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')\n"
                "model = build_model(num_labels=2)\n"
                "model.eval()\n\n"
                "def predict(texts):\n"
                "    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)\n"
                "    with torch.no_grad():\n"
                "        logits = model(**inputs).logits\n"
                "    return logits.argmax(dim=1).tolist()\n"
            )
        return (
            "import joblib\n"
            "import pandas as pd\n\n"
            "model = joblib.load('model.joblib')\n\n"
            "def predict(df: pd.DataFrame):\n"
            f"    if '{label_col}' in df.columns:\n"
            f"        df = df.drop(columns=['{label_col}'])\n"
            "    return model.predict(df)\n"
        )

    def _model_card_text(
        self,
        topic: str,
        model_name: str,
        problem_type: str,
        v1_findings: dict,
        v2_dataset: dict,
    ) -> str:
        findings = v1_findings.get("key_findings", [])[:5]
        findings_text = "\n".join(f"- {f}" for f in findings) if findings else "- None"
        return (
            f"# Model Card: {model_name}\n\n"
            f"- Topic: {topic}\n"
            f"- Problem type: {problem_type}\n"
            f"- Dataset: {v2_dataset.get('name', 'unknown')}\n\n"
            "## Research Findings\n"
            f"{findings_text}\n"
        )

    def _research_extract(self, v1_findings: dict) -> dict:
        return {
            "problem_definition": v1_findings.get("topic"),
            "key_findings": v1_findings.get("key_findings", []),
            "metrics": v1_findings.get("metrics", []),
            "datasets": v1_findings.get("datasets", []),
            "recommended_models": v1_findings.get("recommended_models", []),
            "limitations": v1_findings.get("limitations", []),
            "contradictions": v1_findings.get("contradictions", []),
            "research_blueprint": v1_findings.get("research_blueprint", {}),
        }

    # ── Paper extraction ─────────────────────────────────────────

    def _extract_paper_blueprints(self, v1_findings: dict, max_papers: int = 3) -> list:
        sources = v1_findings.get("sources", []) if isinstance(v1_findings, dict) else []
        papers = []
        for src in sources:
            if len(papers) >= max_papers:
                break
            url = src.get("url") if isinstance(src, dict) else None
            title = src.get("title") if isinstance(src, dict) else ""
            pdf_url = self._resolve_pdf_url(url)
            if not pdf_url:
                continue
            paper = {
                "title": title,
                "url": url,
                "pdf_url": pdf_url,
            }
            text, tables, error = self._download_pdf_text(pdf_url)
            if error:
                paper["error"] = error
            else:
                paper["text_snippet"] = text[:4000]
                paper["sections"] = self._extract_pdf_sections(text)
                if tables:
                    paper["tables"] = [t[:2000] for t in tables]
            papers.append(paper)
        return papers

    def _resolve_pdf_url(self, url: str | None) -> str | None:
        if not url:
            return None
        if url.endswith(".pdf"):
            return url
        if "arxiv.org/abs/" in url:
            return url.replace("/abs/", "/pdf/") + ".pdf"
        return None

    def _download_pdf_text(self, pdf_url: str) -> tuple[str, list, str | None]:
        try:
            resp = requests.get(pdf_url, timeout=20)
            resp.raise_for_status()
            content = resp.content
            if len(content) > 25_000_000:
                return "", [], "PDF too large to process"
        except Exception as e:
            return "", [], f"download failed: {e}"

        try:
            text, tables, err = parse_pdf_bytes(content)
            return text, tables, err
        except Exception as e:
            return "", [], f"parse error: {e}"

    def _extract_pdf_sections(self, text: str) -> dict:
        if not text:
            return {}
        lower = text.lower()
        sections = {}
        for header in ["abstract", "introduction", "method", "methods", "results", "discussion", "conclusion"]:
            idx = lower.find(header)
            if idx == -1:
                continue
            snippet = text[idx:idx + 2000].strip()
            sections[header] = "\n".join(snippet.splitlines()[:30])
        return sections
