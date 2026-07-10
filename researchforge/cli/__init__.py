#!/usr/bin/env python3
"""
ResearchForge CLI.
"""

import argparse
import json
import sys
import warnings


def _configure_warning_filters():
    warnings.filterwarnings(
        "ignore",
        message=r'Field "model_name" has conflict with protected namespace "model_"\..*',
        category=UserWarning,
        module=r"pydantic\._internal\._fields",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="researchforge", description="ResearchForge - topic to trained model, fully automated.", formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("init", help="First-time setup with Ollama/OpenRouter and storage decisions")

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("prompt", type=str)
    plan_parser.add_argument("--output", type=str, default="intent.json")

    research_parser = subparsers.add_parser("research")
    research_parser.add_argument("intent_path", type=str)
    research_parser.add_argument("--output", type=str, default="evidence/research.json")

    store_evidence_parser = subparsers.add_parser("store-evidence")
    store_evidence_parser.add_argument("research_path", type=str)

    graph_parser = subparsers.add_parser("graph")
    graph_parser.add_argument("evidence_path", type=str)
    graph_parser.add_argument("--output", type=str, default="knowledge_graph/graph.json")

    persist_graph_parser = subparsers.add_parser("persist-graph")
    persist_graph_parser.add_argument("research_path", type=str)
    persist_graph_parser.add_argument("--projection-output", type=str, default=None)

    datasets_parser = subparsers.add_parser("datasets")
    datasets_parser.add_argument("intent_path", type=str)
    datasets_parser.add_argument("--output", type=str, default="datasets/dataset_ranking.json")

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("dataset_path", type=str)
    validate_parser.add_argument("--strategy", type=str, required=True)
    validate_parser.add_argument("--label-column", type=str, default=None)
    validate_parser.add_argument("--output-dir", type=str, default="validation")

    train_plan_parser = subparsers.add_parser("plan-train")
    train_plan_parser.add_argument("intent_path", type=str)
    train_plan_parser.add_argument("validation_report_path", type=str)
    train_plan_parser.add_argument("--output", type=str, default="configs/run_config.json")

    train_minimal_parser = subparsers.add_parser("train-minimal")
    train_minimal_parser.add_argument("config_path", type=str)
    train_minimal_parser.add_argument("--output-dir", type=str, default="artifacts/minimal")

    train_framework_parser = subparsers.add_parser("train-framework")
    train_framework_parser.add_argument("config_path", type=str)
    train_framework_parser.add_argument("--output-dir", type=str, default="artifacts/frameworks")

    tune_parser = subparsers.add_parser("tune")
    tune_parser.add_argument("config_path", type=str)
    tune_parser.add_argument("--output-dir", type=str, default="artifacts/optimization")

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("checkpoint_path", type=str)
    evaluate_parser.add_argument("dataset_path", type=str)
    evaluate_parser.add_argument("--label-column", type=str, required=True)
    evaluate_parser.add_argument("--baseline-path", type=str, default=None)
    evaluate_parser.add_argument("--output", type=str, default="evaluation/eval_report.json")

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("intent_path", type=str)
    report_parser.add_argument("research_path", type=str)
    report_parser.add_argument("datasets_path", type=str)
    report_parser.add_argument("validation_path", type=str)
    report_parser.add_argument("training_path", type=str)
    report_parser.add_argument("evaluation_path", type=str)
    report_parser.add_argument("--output", type=str, default="reports/report.md")

    auto_parser = subparsers.add_parser("autoresearch")
    auto_parser.add_argument("research_path", type=str)
    auto_parser.add_argument("evaluation_path", type=str)
    auto_parser.add_argument("--output", type=str, default="reports/proposals.json")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("topic", type=str)
    run_parser.add_argument("--dataset", type=str, default=None)
    run_parser.add_argument("--skip-training", action="store_true")
    run_parser.add_argument("--model", type=str, default=None)
    run_parser.add_argument("--export", type=str, default=None, choices=["html", "pdf"])
    run_parser.add_argument("--budget", type=int, default=10)
    run_parser.add_argument("--experiment-timeout", type=int, default=300)

    subparsers.add_parser("chat")
    subparsers.add_parser("status")
    subparsers.add_parser("notebook")
    return parser


def _read_json(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    _configure_warning_filters()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "init":
        from researchforge.config.settings import Settings
        Settings.init_wizard()
    elif args.command == "plan":
        from researchforge.agents.planner import PlannerAgent
        planner = PlannerAgent()
        handoff = planner.parse_intent(args.prompt)
        planner.save_intent(handoff, args.output)
        print(json.dumps(handoff, indent=2))
    elif args.command == "research":
        from researchforge.agents.research import ResearchAgent
        agent = ResearchAgent()
        handoff = agent.run(_read_json(args.intent_path))
        agent.save_evidence(handoff, args.output)
        print(json.dumps(handoff, indent=2))
    elif args.command == "store-evidence":
        from researchforge.sdk import EvidenceStore
        store = EvidenceStore()
        inserted = store.store_research_handoff(_read_json(args.research_path))
        print(json.dumps({"inserted_records": inserted, "total_records": store.count_records()}, indent=2))
    elif args.command == "graph":
        from researchforge.knowledge_graph import KnowledgeGraphProjector
        projector = KnowledgeGraphProjector()
        projection = projector.project_research_handoff(_read_json(args.evidence_path))
        projector.save_projection(projection, args.output)
        print(json.dumps(projection, indent=2))
    elif args.command == "persist-graph":
        from researchforge.knowledge_graph import Neo4jKnowledgeGraphStore
        store = Neo4jKnowledgeGraphStore()
        projection = store.projector.project_research_handoff(_read_json(args.research_path))
        if args.projection_output:
            store.projector.save_projection(projection, args.projection_output)
        result = store.persist_projection(projection)
        print(json.dumps(result, indent=2))
    elif args.command == "datasets":
        from researchforge.agents.dataset import DatasetAgent
        agent = DatasetAgent()
        handoff = agent.discover_and_rank(_read_json(args.intent_path))
        agent.save_ranking(handoff, args.output)
        print(json.dumps(handoff, indent=2))
    elif args.command == "validate":
        from researchforge.agents.validation import ValidationAgent
        report = ValidationAgent().validate_dataset(args.dataset_path, args.strategy, label_column=args.label_column, output_dir=args.output_dir)
        print(json.dumps(report, indent=2))
    elif args.command == "plan-train":
        from researchforge.agents.training import TrainingPlannerAgent
        agent = TrainingPlannerAgent()
        handoff = agent.create_plan(_read_json(args.intent_path), _read_json(args.validation_report_path))
        agent.save_config(handoff, args.output)
        print(json.dumps(handoff, indent=2))
    elif args.command == "train-minimal":
        from researchforge.training.karpathy_minimal import MinimalTrainer
        handoff = MinimalTrainer().run(args.config_path, output_dir=args.output_dir)
        print(json.dumps(handoff, indent=2))
    elif args.command == "train-framework":
        from researchforge.training.frameworks import FrameworkTrainer
        handoff = FrameworkTrainer().run(args.config_path, output_dir=args.output_dir)
        print(json.dumps(handoff, indent=2))
    elif args.command == "tune":
        from researchforge.agents.optimization import OptimizationAgent
        handoff = OptimizationAgent().optimize(args.config_path, output_dir=args.output_dir)
        print(json.dumps(handoff, indent=2))
    elif args.command == "evaluate":
        from researchforge.agents.evaluation import EvaluationAgent
        handoff = EvaluationAgent().evaluate(args.checkpoint_path, args.dataset_path, args.label_column, baseline_path=args.baseline_path, output_path=args.output)
        print(json.dumps(handoff, indent=2))
    elif args.command == "report":
        from researchforge.agents.reporting import ReportingAgent
        inputs = {
            "intent": _read_json(args.intent_path),
            "research": _read_json(args.research_path),
            "datasets": _read_json(args.datasets_path),
            "validation": _read_json(args.validation_path),
            "training": _read_json(args.training_path),
            "evaluation": _read_json(args.evaluation_path),
        }
        handoff = ReportingAgent().build_report(inputs, output_path=args.output)
        print(json.dumps(handoff, indent=2))
    elif args.command == "autoresearch":
        from researchforge.agents.reporting import ReportingAgent
        handoff = ReportingAgent().propose_experiments(_read_json(args.research_path), _read_json(args.evaluation_path), output_path=args.output)
        print(json.dumps(handoff, indent=2))
    elif args.command == "run":
        from researchforge.core.pipeline import Pipeline
        Pipeline().run(topic=args.topic, dataset_path=args.dataset, skip_training=args.skip_training, model_override=args.model, export=args.export, budget=args.budget, experiment_timeout=args.experiment_timeout)
    elif args.command == "chat":
        from researchforge.core.chat import ChatSession
        ChatSession().start()
    elif args.command == "status":
        from researchforge.utils.display import Display
        Display.show_status()
    elif args.command == "notebook":
        import subprocess
        from researchforge.utils.display import Display
        from researchforge.utils.state import load_state
        state = load_state()
        nb = state.get("v3", {}).get("notebook_path")
        if nb:
            Display.info(f"Opening notebook: {nb}")
            subprocess.run(["jupyter", "notebook", nb])
        else:
            Display.error('No notebook found. Run a pipeline first: researchforge run "<topic>"')
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
