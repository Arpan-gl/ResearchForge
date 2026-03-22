"""
Pipeline — orchestrates V1 → V2 → V3 → Autoresearch
All LLM calls go to Ollama (user's local LLM)
All training runs on user's GPU via PyTorch
"""

import json
import time
from researchforge.stages.v1_research import V1Research
from researchforge.stages.v2_datasets import V2Datasets
from researchforge.stages.v3_notebook import V3Notebook
from researchforge.stages.autoresearch import Autoresearch
from researchforge.utils.display import Display
from researchforge.utils.state import save_state


class Pipeline:
    def __init__(self):
        self.v1   = V1Research()
        self.v2   = V2Datasets()
        self.v3   = V3Notebook()
        self.auto = Autoresearch()

    def run(
        self,
        topic: str,
        dataset_path: str = None,
        skip_training: bool = False,
        model_override: str = None,
        export: str = None,
        budget: int = 100,
    ):
        Display.banner()
        Display.section("Starting ResearchForge Pipeline")
        Display.info(f"Topic  : {topic}")
        Display.info(f"Dataset: {dataset_path or 'auto-discover'}")
        Display.info(f"GPU training: {'disabled' if skip_training else 'enabled'}")
        if model_override:
            Display.info(f"Model override: {model_override}")
        print()

        results = {}
        stage_name = "setup"

        try:
            # ── STAGE 1: V1 Research ──────────────────────────────
            stage_name = "V1 Research"
            Display.stage(1, "V1 Research — hybrid retrieval + LLM extraction")
            v1_result = self.v1.run(topic)
            results["v1"] = v1_result
            n_sources = len(v1_result.get("sources", []))
            n_findings = len(v1_result.get("key_findings", []))
            n_contradictions = len(v1_result.get("contradictions", []))
            Display.success(
                f"{n_sources} sources · {n_findings} findings · "
                f"{n_contradictions} contradictions flagged"
            )
            if v1_result.get("contradictions"):
                for c in v1_result["contradictions"]:
                    Display.warn(f"  {c}")

            # ── STAGE 2: V2 Dataset Discovery ────────────────────
            stage_name = "V2 Datasets"
            Display.stage(2, "V2 Datasets — scoring + risk audit")
            if dataset_path:
                Display.info("User-provided dataset — running audit only")
                v2_result = self.v2.audit_user_dataset(dataset_path, v1_result)
            else:
                v2_result = self.v2.discover_and_score(topic, v1_result)
            results["v2"] = v2_result
            Display.success(
                f"Dataset: {v2_result['name']} · Score: {v2_result['score']:.2f} · "
                f"Risks: {len(v2_result.get('risks', []))}"
            )
            for r in v2_result.get("risks", []):
                Display.warn(f"  {r}")

            dataset_missing = (
                v2_result.get("score", 0.0) <= 0.0
                and v2_result.get("source") == "none"
            ) or "no dataset found" in str(v2_result.get("name", "")).lower()
            if dataset_missing:
                results["status"] = "dataset_unavailable"
                Display.error("No usable dataset found in Stage 2.")
                Display.info("Provide your own dataset and rerun with: --dataset <path/to/data.csv>")
                save_state(results)
                return results

            # ── STAGE 3: V3 Notebook Generation ──────────────────
            stage_name = "V3 Notebook"
            Display.stage(3, "V3 Notebook — generating runnable .ipynb")
            v3_result = self.v3.generate(
                topic=topic,
                v1_findings=v1_result,
                v2_dataset=v2_result,
                model_override=model_override,
            )
            results["v3"] = v3_result
            Display.success(f"Notebook: {v3_result['notebook_path']}")
            Display.info(f"Model: {v3_result['model']}  ·  Expected {v3_result['metric_name']}: {v3_result['expected_range']}")

            # ── STAGE 4: Autoresearch (GPU training) ─────────────
            if not skip_training:
                stage_name = "Autoresearch"
                Display.stage(4, "Autoresearch — running experiments on your GPU")
                Display.info("This runs until budget is exhausted.  Ctrl+C to stop early.\n")
                auto_result = self.auto.run(
                    notebook_path=v3_result["notebook_path"],
                    metric=v3_result["metric_name"],
                    budget=budget,
                )
                results["autoresearch"] = auto_result
                Display.success(
                    f"Best {auto_result['metric']}: {auto_result['best_score']:.3f} "
                    f"(+{auto_result['improvement_pct']:.1f}% from baseline)"
                )
                Display.info(f"Best commit: {auto_result['best_commit']}  ·  TSV log: {auto_result['tsv_log']}")
            else:
                Display.info("GPU training skipped. Open the notebook to train manually.")

            # ── FINAL OUTPUT ──────────────────────────────────────
            Display.section("Pipeline Complete")
            self._print_summary(results)
            save_state(results)

            output_path = "researchforge_output.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=str)
            Display.success(f"Full result JSON saved to {output_path}")

            # Optional export (Priority 6)
            if export:
                self._export_report(results, export)
            return results

        except KeyboardInterrupt:
            Display.warn("Pipeline interrupted by user. Partial results were saved.")
            results["status"] = "interrupted"
            save_state(results)
            return results
        except Exception as e:
            Display.error(f"Pipeline failed during {stage_name}: {e}")
            results["status"] = "failed"
            results["failed_stage"] = stage_name
            results["error"] = str(e)
            save_state(results)
            raise

    def _export_report(self, results: dict, fmt: str):
        """Export the pipeline results to HTML or PDF."""
        Display.info(f"Generating {fmt.upper()} report…")
        try:
            from researchforge.utils.exporter import Exporter
            exporter = Exporter()
            if fmt == "html":
                path = exporter.to_html(results, "researchforge_report.html")
                Display.success(f"HTML report saved to {path}")
            elif fmt == "pdf":
                path = exporter.to_pdf(results, "researchforge_report.pdf")
                Display.success(f"PDF report saved to {path}")
            else:
                Display.warn(f"Unknown export format: {fmt}. Use 'html' or 'pdf'.")
        except ImportError as e:
            Display.warn(str(e))
        except Exception as e:
            Display.error(f"Export failed: {e}")

    def _print_summary(self, results: dict):
        v1   = results.get("v1", {})
        v2   = results.get("v2", {})
        v3   = results.get("v3", {})
        auto = results.get("autoresearch", {})
        metric = v3.get("metric_name", "metric")

        print("  ┌─────────────────────────────────────────────┐")
        print("  │           RESEARCHFORGE SUMMARY              │")
        print("  ├─────────────────────────────────────────────┤")
        print(f"  │  Sources found     : {len(v1.get('sources', [])):<26}│")
        print(f"  │  Dataset           : {v2.get('name','—')[:26]:<26}│")
        print(f"  │  Dataset score     : {v2.get('score', 0):.2f}{'':<23}│")
        print(f"  │  Model             : {v3.get('model','—')[:26]:<26}│")
        print(f"  │  Notebook          : {v3.get('notebook_path','—')[:26]:<26}│")
        if auto:
            print(f"  │  Baseline {metric:8s}: {auto.get('baseline_score', 0):.3f}{'':<22}│")
            print(f"  │  Best     {metric:8s}: {auto.get('best_score', 0):.3f}{'':<22}│")
            print(f"  │  Improvement      : +{auto.get('improvement_pct', 0):.1f}%{'':<23}│")
            print(f"  │  Experiments run  : {auto.get('experiments_run', 0):<26}│")
        print("  └─────────────────────────────────────────────┘\n")
