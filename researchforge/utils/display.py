"""
Display — colored CLI output for ResearchForge
Colors: cyan (info), amber (warn), green (success), red (error)
"""

import json
from pathlib import Path


class Colors:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    AMBER  = "\033[93m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"
    DIM    = "\033[2m"


class Display:
    """All user-facing console output goes through here."""

    @staticmethod
    def banner():
        print(f"\n{Colors.CYAN}{Colors.BOLD}")
        print("  ██████╗ ███████╗███████╗ █████╗ ██████╗  ██████╗██╗  ██╗")
        print("  ██╔══██╗██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝██║  ██║")
        print("  ██████╔╝█████╗  ███████╗███████║██████╔╝██║     ███████║")
        print("  ██╔══██╗██╔══╝  ╚════██║██╔══██║██╔══██╗██║     ██╔══██║")
        print("  ██║  ██║███████╗███████║██║  ██║██║  ██║╚██████╗██║  ██║")
        print("  ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝")
        print(f"\n  {Colors.DIM}FORGE  Topic → Research → Dataset → Notebook → GPU Training{Colors.RESET}\n")

    @staticmethod
    def stage(n: int, title: str):
        print(f"\n{Colors.CYAN}{Colors.BOLD}  [{n}] {title}{Colors.RESET}")
        print(f"  {'─' * (len(title) + 4)}")

    @staticmethod
    def section(title: str):
        bar = "─" * 48
        print(f"\n{Colors.WHITE}{Colors.BOLD}  ┌{bar}┐")
        padding = (48 - len(title) - 2) // 2
        print(f"  │{' ' * padding} {title} {' ' * (48 - len(title) - 2 - padding)}│")
        print(f"  └{bar}┘{Colors.RESET}")

    @staticmethod
    def success(msg: str):
        print(f"{Colors.GREEN}  ✓ {msg}{Colors.RESET}")

    @staticmethod
    def info(msg: str):
        print(f"{Colors.CYAN}  ℹ {msg}{Colors.RESET}")

    @staticmethod
    def warn(msg: str):
        print(f"{Colors.AMBER}  ⚠ {msg}{Colors.RESET}")

    @staticmethod
    def error(msg: str):
        print(f"{Colors.RED}  ✗ {msg}{Colors.RESET}")

    @staticmethod
    def show_status():
        """Print the last pipeline run status from state file."""
        state_path = Path.home() / ".researchforge" / "last_run.json"
        if not state_path.exists():
            Display.warn("No previous run found. Run: researchforge run \"<topic>\"")
            return

        try:
            with open(state_path) as f:
                state = json.load(f)
        except Exception as e:
            Display.error(f"Could not read state: {e}")
            return

        v1   = state.get("v1", {})
        v2   = state.get("v2", {})
        v3   = state.get("v3", {})
        auto = state.get("autoresearch", {})
        ts   = state.get("timestamp", "unknown")

        print(f"\n{Colors.WHITE}{Colors.BOLD}  Last Run Summary{Colors.RESET}  {Colors.DIM}({ts}){Colors.RESET}")
        print("  " + "─" * 46)
        print(f"  Sources found   : {len(v1.get('sources', []))}")
        print(f"  Dataset         : {v2.get('name', '—')}")
        print(f"  Dataset score   : {v2.get('score', 0):.2f}")
        print(f"  Notebook        : {v3.get('notebook_path', '—')}")
        print(f"  Model           : {v3.get('model', '—')}")
        if auto:
            metric = v3.get("metric_name", "metric")
            print(f"  Baseline {metric:10s}: {auto.get('baseline_score', 0):.3f}")
            print(f"  Best {metric:14s}: {auto.get('best_score', 0):.3f}  "
                  f"(+{auto.get('improvement_pct', 0):.1f}%)")
            print(f"  Experiments run : {auto.get('experiments_run', 0)}")
            print(f"  Best commit     : {auto.get('best_commit', '—')}")
        print()
        if v2.get("risks"):
            Display.warn("Dataset risks:")
            for r in v2["risks"]:
                Display.warn(f"  • {r}")
        if v1.get("contradictions"):
            Display.warn("Contradictions flagged:")
            for c in v1["contradictions"]:
                Display.warn(f"  • {c}")
