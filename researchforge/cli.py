#!/usr/bin/env python3
"""
ResearchForge CLI
-----------------
Usage:
  researchforge init                              # first-time setup
  researchforge run "<topic>"                     # full pipeline
  researchforge run "<topic>" --dataset data.csv  # with user dataset
  researchforge run "<topic>" --skip-training     # notebook only
  researchforge run "<topic>" --model gnn         # force model
  researchforge run "<topic>" --export html       # export report
  researchforge chat                              # interactive chat
  researchforge status                            # last run summary
  researchforge notebook                          # open last notebook
"""

import argparse
import sys
import warnings


def _configure_warning_filters():
    """Hide one noisy third-party warning without muting unrelated warnings."""
    warnings.filterwarnings(
        "ignore",
        message=r'Field "model_name" has conflict with protected namespace "model_"\..*',
        category=UserWarning,
        module=r"pydantic\._internal\._fields",
    )


def main():
    _configure_warning_filters()

    parser = argparse.ArgumentParser(
        prog="researchforge",
        description="ResearchForge — topic to trained model, fully automated.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── researchforge init ────────────────────────────────────────
    subparsers.add_parser(
        "init",
        help="First-time setup wizard (Ollama URL, model, Kaggle creds)",
    )

    # ── researchforge run ─────────────────────────────────────────
    run_parser = subparsers.add_parser(
        "run", help="Run the full pipeline on a topic"
    )
    run_parser.add_argument("topic", type=str, help="Research topic")
    run_parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path to CSV/parquet or Kaggle dataset URL",
    )
    run_parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip GPU training, generate notebook only",
    )
    run_parser.add_argument(
        "--model", type=str, default=None,
        help="Override model selection: xgboost | lightgbm | gnn | bert",
    )
    run_parser.add_argument(
        "--export", type=str, default=None, choices=["html", "pdf"],
        help="Export report after pipeline: html | pdf",
    )
    run_parser.add_argument(
        "--budget", type=int, default=100,
        help="Autoresearch experiment budget (default: 100)",
    )

    # ── researchforge chat ────────────────────────────────────────
    subparsers.add_parser(
        "chat", help="Interactive chat with your local LLM about any topic"
    )

    # ── researchforge status ──────────────────────────────────────
    subparsers.add_parser("status", help="Show last pipeline run status")

    # ── researchforge notebook ────────────────────────────────────
    subparsers.add_parser("notebook", help="Launch last generated notebook in Jupyter")

    args = parser.parse_args()

    # ── Dispatch ──────────────────────────────────────────────────

    if args.command == "init":
        from researchforge.config.settings import Settings
        Settings.init_wizard()

    elif args.command == "run":
        from researchforge.core.pipeline import Pipeline
        pipeline = Pipeline()
        pipeline.run(
            topic=args.topic,
            dataset_path=args.dataset,
            skip_training=args.skip_training,
            model_override=args.model,
            export=args.export,
            budget=args.budget,
        )

    elif args.command == "chat":
        from researchforge.core.chat import ChatSession
        session = ChatSession()
        session.start()

    elif args.command == "status":
        from researchforge.utils.display import Display
        Display.show_status()

    elif args.command == "notebook":
        import subprocess
        from researchforge.utils.state import load_state
        from researchforge.utils.display import Display
        state = load_state()
        nb = state.get("v3", {}).get("notebook_path")
        if nb:
            Display.info(f"Opening notebook: {nb}")
            subprocess.run(["jupyter", "notebook", nb])
        else:
            Display.error("No notebook found. Run a pipeline first: researchforge run \"<topic>\"")
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
