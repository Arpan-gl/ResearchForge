"""
MLflowTracker — experiment tracking for Autoresearch
Priority 4 from ResearchForge build plan.

Every accept/reject decision in the autoresearch loop is logged
as an MLflow run with:
  - params: description, target_section, accepted, n_seeds
  - metrics: mean_score, std_score
  - tags: researchforge, metric_name
"""

import statistics
from typing import List


class MLflowTracker:
    def __init__(self, experiment_name: str, tracking_uri: str = "mlruns"):
        try:
            import mlflow
            self._mlflow = mlflow
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._enabled = True
        except ImportError:
            self._enabled = False
            raise

    def log_experiment(
        self,
        suggestion: dict,
        scores: List[float],
        accepted: bool,
        metric_name: str,
    ) -> None:
        """
        Log one autoresearch experiment as an MLflow run.

        Parameters
        ----------
        suggestion  : the dict returned by _suggest_modification()
        scores      : list of per-seed scores
        accepted    : whether this experiment was merged
        metric_name : e.g. "F1-macro"
        """
        if not self._enabled or not scores:
            return

        mlflow = self._mlflow
        mean_score = sum(scores) / len(scores)
        std_score  = (
            statistics.stdev(scores) if len(scores) > 1
            else 0.0
        )

        try:
            with mlflow.start_run():
                # Parameters
                mlflow.log_param("description",    suggestion.get("description", ""))
                mlflow.log_param("target_section", suggestion.get("target_section", "unknown"))
                mlflow.log_param("code_change",    suggestion.get("code_change", "")[:250])
                mlflow.log_param("expected_gain",  suggestion.get("expected_gain", "unknown"))
                mlflow.log_param("accepted",       str(accepted))
                mlflow.log_param("n_seeds",        len(scores))

                # Metrics
                mlflow.log_metric(metric_name, mean_score)
                mlflow.log_metric(f"{metric_name}_std", std_score)
                for i, s in enumerate(scores):
                    mlflow.log_metric(f"{metric_name}_seed{i}", s)

                # Tags
                mlflow.set_tags({
                    "researchforge": "autoresearch",
                    "metric_name":   metric_name,
                    "status":        "keep" if accepted else "discard",
                })
        except Exception:
            # Never crash the pipeline because MLflow fails
            pass
