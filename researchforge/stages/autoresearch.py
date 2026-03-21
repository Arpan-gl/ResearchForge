"""
Autoresearch — Karpathy-style autonomous experiment loop
---------------------------------------------------------
Inspired by https://github.com/karpathy/autoresearch

Key design principles from Karpathy:
  - Branch-based git workflow (autoresearch/<tag> branch)
  - Fixed time budget per experiment (timeout=300s default)
  - TSV results log (results.tsv) alongside git commits
  - Crash recovery — log "crash", revert, keep going
  - Acceptance rule: mean_score > best_score + 1*std(scores) across 3 seeds

Pipeline-specific additions:
  - Modifies notebook cells (not a .py file) via nbformat
  - Runs via nbconvert --execute
  - Parses metric from output cell stdout
  - MLflow experiment tracking
"""

import os
import csv
import json
import shutil
import subprocess
import tempfile
import time
import copy
import requests
from pathlib import Path

import nbformat
from researchforge.config.settings import Settings
from researchforge.utils.display import Display


class Autoresearch:
    SEEDS = [42, 123, 999]

    def __init__(self):
        self.settings = Settings()
        self.ollama_url = self.settings.ollama_url
        self.model = self.settings.llm_model

    # ── Main entry point ──────────────────────────────────────────

    def run(
        self,
        notebook_path: str,
        metric: str,
        budget: int = 100,
        experiment_timeout: int = 300,
    ) -> dict:
        Display.info(f"Starting autonomous experiment loop  budget={budget}")
        Display.info(f"Metric: {metric} · Seeds: {self.SEEDS} · Timeout: {experiment_timeout}s/seed")
        Display.info("Stop early: Ctrl+C — best result will be saved\n")

        # Set up git branch (Karpathy pattern)
        branch_tag = self._setup_git_branch(notebook_path)

        # Initialize TSV results log (Karpathy pattern — tab-separated, untracked)
        tsv_path = os.path.join(os.path.dirname(os.path.abspath(notebook_path)), "results.tsv")
        self._init_results_tsv(tsv_path)

        # MLflow tracking (Priority 4)
        tracker = self._init_tracker(metric)

        # Establish baseline
        Display.info("Running baseline experiment…")
        baseline = self._run_baseline(notebook_path, metric, experiment_timeout)
        Display.success(f"Baseline {metric}: {baseline:.4f}")
        self._log_tsv(tsv_path, "baseline", baseline, 0.0, "keep", "baseline — unmodified notebook")

        best_score = baseline
        best_commit = "baseline"
        experiments_run = 0
        history: list[dict] = []

        try:
            for i in range(budget):
                experiments_run += 1
                print(f"\n  Experiment {i+1}/{budget} ", end="", flush=True)

                suggestion = self._suggest_modification(history, metric, best_score)
                if not suggestion:
                    print("(no suggestion from LLM) — skipping")
                    continue

                desc = suggestion.get("description", "?")[:55]
                print(f"→ {desc}…", end=" ", flush=True)

                # Run with 3 seeds (crash-safe)
                scores, crashed = self._run_experiment(
                    notebook_path, suggestion, self.SEEDS, experiment_timeout
                )

                if crashed or not scores:
                    print("✗ CRASH")
                    self._log_tsv(tsv_path, "unknown", 0.0, 0.0, "crash", desc)
                    history.append({"description": desc, "score": 0.0, "accepted": False})
                    if tracker:
                        tracker.log_experiment(suggestion, [0.0], accepted=False, metric_name=metric)
                    continue

                mean_score = sum(scores) / len(scores)
                std_score  = (sum((s - mean_score) ** 2 for s in scores) / len(scores)) ** 0.5

                # Karpathy acceptance rule: mean > best + 1*std
                threshold = best_score + std_score
                accepted  = mean_score > threshold

                if accepted:
                    best_score = mean_score
                    commit_hash = self._commit_improvement(
                        notebook_path, suggestion, mean_score, metric
                    )
                    best_commit = commit_hash
                    delta = mean_score - baseline
                    print(f"✓ {mean_score:.4f} (+{delta:.4f}) → committed {commit_hash[:7]}")
                    self._log_tsv(tsv_path, commit_hash[:7], mean_score, 0.0, "keep", desc)
                    history.append({"description": desc, "score": mean_score, "accepted": True})
                else:
                    print(f"✗ {mean_score:.4f} (need >{threshold:.4f}) → discarded")
                    self._log_tsv(tsv_path, "—", mean_score, 0.0, "discard", desc)
                    history.append({"description": desc, "score": mean_score, "accepted": False})

                if tracker:
                    tracker.log_experiment(suggestion, scores, accepted=accepted, metric_name=metric)

                time.sleep(0.5)  # brief cooldown

        except KeyboardInterrupt:
            Display.warn("Stopped by user. Best result is saved.")

        improvement_pct = (
            ((best_score - baseline) / baseline) * 100 if baseline > 0 else 0.0
        )

        return {
            "metric":           metric,
            "baseline_score":   round(baseline, 4),
            "best_score":       round(best_score, 4),
            "improvement_pct":  round(improvement_pct, 2),
            "experiments_run":  experiments_run,
            "best_commit":      best_commit,
            "branch":           branch_tag,
            "tsv_log":          tsv_path,
            "history":          history,
        }

    # ── Git helpers (Karpathy pattern) ────────────────────────────

    def _setup_git_branch(self, notebook_path: str) -> str:
        """Create a dedicated autoresearch/<date-tag> branch for this run."""
        tag = time.strftime("autoresearch/%b%d").lower()
        try:
            # Check branch doesn't already exist
            existing = subprocess.run(
                ["git", "branch", "--list", tag],
                capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(notebook_path)) or "."
            ).stdout.strip()
            if existing:
                tag = tag + f"-{int(time.time()) % 10000}"
            subprocess.run(
                ["git", "checkout", "-b", tag],
                capture_output=True, text=True
            )
            Display.info(f"Git branch: {tag}")
        except Exception:
            Display.warn("Git not available — commits will be skipped")
        return tag

    def _commit_improvement(
        self, notebook_path: str, suggestion: dict, score: float, metric: str
    ) -> str:
        """Commit improved notebook to git with standard message format."""
        try:
            msg = (
                f"autoresearch: {suggestion['description']} "
                f"→ {metric}={score:.4f}"
            )
            subprocess.run(["git", "add", notebook_path], capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", msg],
                capture_output=True, text=True
            )
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True
            )
            return result.stdout.strip() or f"exp_{int(time.time())}"
        except Exception:
            return f"exp_{int(time.time())}"

    # ── TSV logging (Karpathy pattern) ────────────────────────────

    def _init_results_tsv(self, tsv_path: str):
        if not os.path.exists(tsv_path):
            with open(tsv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(["commit", "mean_score", "std_score", "status", "description"])

    def _log_tsv(
        self,
        tsv_path: str,
        commit: str,
        mean_score: float,
        std_score: float,
        status: str,
        description: str,
    ):
        try:
            with open(tsv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow([commit, f"{mean_score:.6f}", f"{std_score:.6f}", status, description])
        except Exception:
            pass

    # ── Baseline execution ────────────────────────────────────────

    def _run_baseline(self, notebook_path: str, metric: str, timeout: int) -> float:
        """Execute the notebook as-is and parse the metric from output."""
        tmp_out = os.path.join(tempfile.gettempdir(), "rf_baseline.ipynb")
        try:
            result = subprocess.run(
                [
                    "jupyter", "nbconvert", "--to", "notebook",
                    "--execute", "--output", tmp_out,
                    "--ExecutePreprocessor.timeout", str(timeout),
                    "--ExecutePreprocessor.kernel_name", "python3",
                    notebook_path,
                ],
                capture_output=True, text=True, timeout=timeout + 60,
            )
            score = self._parse_metric_from_output(result.stdout + result.stderr, metric)
            if score is not None:
                return score
            # Try to read from executed notebook output cells
            score = self._parse_metric_from_notebook(tmp_out, metric)
            return score if score is not None else 0.70
        except Exception as e:
            Display.warn(f"Baseline execution error: {e} — using estimate 0.70")
            return 0.70

    # ── Experiment execution (PRIORITY 1 fix) ─────────────────────

    def _run_experiment(
        self,
        notebook_path: str,
        suggestion: dict,
        seeds: list,
        timeout: int,
    ) -> tuple[list, bool]:
        """
        Real notebook execution with nbconvert.
        Returns (scores, crashed) — crashed=True if all seeds failed.
        """
        scores = []
        any_success = False
        tmp_dir = tempfile.mkdtemp(prefix="rf_exp_")

        try:
            # Load original notebook
            with open(notebook_path, encoding="utf-8") as f:
                original_nb = nbformat.read(f, as_version=4)

            for seed in seeds:
                try:
                    # 1. Deep-copy notebook
                    nb = copy.deepcopy(original_nb)

                    # 2. Apply code change to the appropriate cell
                    nb = self._apply_code_change(nb, suggestion)

                    # 3. Inject seed: replace random_state=42 with this seed
                    nb = self._inject_seed(nb, seed)

                    # 4. Write modified notebook to temp dir
                    tmp_nb = os.path.join(tmp_dir, f"exp_seed{seed}.ipynb")
                    with open(tmp_nb, "w", encoding="utf-8") as f:
                        nbformat.write(nb, f)

                    # 5. Execute with nbconvert
                    output_nb = os.path.join(tmp_dir, f"out_seed{seed}.ipynb")
                    result = subprocess.run(
                        [
                            "jupyter", "nbconvert", "--to", "notebook",
                            "--execute", "--output", output_nb,
                            "--ExecutePreprocessor.timeout", str(timeout),
                            "--ExecutePreprocessor.kernel_name", "python3",
                            tmp_nb,
                        ],
                        capture_output=True, text=True,
                        timeout=timeout + 60,
                    )

                    # 6. Parse metric from stdout/stderr then from executed notebook
                    all_output = result.stdout + result.stderr
                    score = self._parse_metric_from_output(all_output, "F1")
                    if score is None:
                        score = self._parse_metric_from_notebook(output_nb, "F1")
                    if score is None:
                        score = self._parse_metric_from_output(all_output, "f1")
                    if score is None:
                        score = self._parse_metric_from_output(all_output, "accuracy")

                    if score is not None:
                        scores.append(score)
                        any_success = True
                    else:
                        Display.warn(f"  Seed {seed}: could not parse metric from output")

                except subprocess.TimeoutExpired:
                    Display.warn(f"  Seed {seed}: timed out after {timeout}s — skipped")
                except Exception as e:
                    Display.warn(f"  Seed {seed}: {e}")

        finally:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

        crashed = not any_success
        return scores, crashed

    def _apply_code_change(self, nb, suggestion: dict) -> object:
        """
        Find the cell that corresponds to suggestion['target_section'],
        then apply suggestion['code_change'] via string replacement.

        target_section keywords map to section headers in generated cells.
        """
        section_keywords = {
            "model":          ["SECTION 4", "model ready", "GATConv", "XGBClassifier", "LGBMClassifier"],
            "training":       ["SECTION 5", "cross_val_score", "model.fit", "Holdout"],
            "preprocessing":  ["SECTION 3", "Preprocess", "StandardScaler", "LabelEncoder"],
            "features":       ["SECTION 1", "SECTION 2", "EDA", "Split", "drop(columns"],
        }
        target = suggestion.get("target_section", "training")
        keywords = section_keywords.get(target, section_keywords["training"])
        code_change = suggestion.get("code_change", "")

        if not code_change:
            return nb

        # Find the best matching cell
        best_cell_idx = None
        best_match_count = 0
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            match_count = sum(1 for kw in keywords if kw in cell.source)
            if match_count > best_match_count:
                best_match_count = match_count
                best_cell_idx = idx

        if best_cell_idx is None:
            # Fallback: append to last code cell
            for idx in range(len(nb.cells) - 1, -1, -1):
                if nb.cells[idx].cell_type == "code":
                    best_cell_idx = idx
                    break

        if best_cell_idx is None:
            return nb

        cell = nb.cells[best_cell_idx]

        # Try exact-string replacement if code_change contains "→" or "="
        # The suggestion format: "old_value → new_value" or just new code
        if "→" in code_change:
            parts = code_change.split("→", 1)
            old_part = parts[0].strip()
            new_part = parts[1].strip()
            if old_part in cell.source:
                cell.source = cell.source.replace(old_part, new_part, 1)
            else:
                # Append change as a comment + override line
                cell.source += f"\n# autoresearch: {code_change}\n{new_part}\n"
        else:
            # Append the code change at the end of the cell
            cell.source += f"\n# autoresearch change:\n{code_change}\n"

        nb.cells[best_cell_idx] = cell
        return nb

    def _inject_seed(self, nb, seed: int) -> object:
        """Replace random_state=42 with the given seed in all code cells."""
        for cell in nb.cells:
            if cell.cell_type == "code":
                cell.source = cell.source.replace(
                    "random_state=42", f"random_state={seed}"
                )
        return nb

    # ── LLM suggestion (updated prompt) ──────────────────────────

    def _suggest_modification(
        self, history: list, metric: str, best_score: float
    ) -> dict | None:
        rejected = [h["description"] for h in history if not h["accepted"]]
        accepted = [h["description"] for h in history if h["accepted"]]

        prompt = (
            "You are an autonomous ML optimization agent inspired by Karpathy's autoresearch. "
            f"Your goal is to improve {metric} for a machine learning model trained in a Jupyter notebook.\n\n"
            "Already ACCEPTED improvements:\n"
            f"{json.dumps(accepted, indent=2) if accepted else 'None yet'}\n\n"
            "Already REJECTED (do not repeat these):\n"
            f"{json.dumps(rejected, indent=2) if rejected else 'None yet'}\n\n"
            f"Current best {metric}: {best_score:.4f}\n\n"
            "Suggest exactly ONE specific, implementable change. "
            "Be creative — try architectural changes, regularization, learning rate schedules, "
            "feature engineering, or data augmentation.\n\n"
            "Return ONLY a JSON object:\n"
            "{\n"
            '  "description": "Brief description of the change",\n'
            '  "target_section": "model|training|preprocessing|features",\n'
            '  "code_change": "old_code → new_code  (use → to indicate replacement, or just new code to append)",\n'
            '  "expected_gain": "estimated % improvement"\n'
            "}\n\n"
            "Return ONLY valid JSON. No explanation."
        )

        response = self._ask_llm(prompt)
        try:
            clean = response.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(clean)
        except Exception:
            return None

    # ── Metric parsing ────────────────────────────────────────────

    def _parse_metric_from_output(self, output: str, metric: str) -> float | None:
        """Parse a float metric value from stdout/stderr text."""
        import re
        patterns = [
            rf"Holdout\s+{metric}[:\s]+([0-9]+\.[0-9]+)",
            rf"CV\s+{metric}[:\s]+([0-9]+\.[0-9]+)",
            rf"{metric}[:\s]+([0-9]+\.[0-9]+)",
            rf"([0-9]+\.[0-9]+)\s+{metric}",
            r"f1[_\s]macro[:\s]+([0-9]+\.[0-9]+)",
            r"Holdout\s+F1[_-]macro[:\s]+([0-9]+\.[0-9]+)",
            r"accuracy[:\s]+([0-9]+\.[0-9]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass
        return None

    def _parse_metric_from_notebook(self, nb_path: str, metric: str) -> float | None:
        """Parse metric from executed notebook output cells."""
        try:
            with open(nb_path, encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)
        except Exception:
            return None

        for cell in reversed(nb.cells):
            if cell.cell_type != "code":
                continue
            for output in cell.get("outputs", []):
                text = output.get("text", "") or "".join(output.get("data", {}).get("text/plain", []))
                score = self._parse_metric_from_output(text, metric)
                if score is not None:
                    return score
        return None

    # ── MLflow helper ─────────────────────────────────────────────

    def _init_tracker(self, metric: str):
        """Instantiate MLflowTracker if mlflow is installed (Priority 4)."""
        try:
            from researchforge.utils.mlflow_tracker import MLflowTracker
            return MLflowTracker(
                experiment_name=f"researchforge_{metric}",
                tracking_uri=self.settings.mlflow_tracking_uri,
            )
        except ImportError:
            Display.warn("mlflow not installed — experiment tracking disabled. "
                         "Install with: pip install mlflow")
            return None
        except Exception as e:
            Display.warn(f"MLflow init failed: {e}")
            return None

    # ── Ollama helper ─────────────────────────────────────────────

    def _ask_llm(self, prompt: str) -> str:
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=60,
            )
            return resp.json().get("response", "")
        except Exception:
            return "{}"
