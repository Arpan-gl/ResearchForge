"""
Chat Session — interactive chat with Ollama LLM
Knows about ResearchForge pipeline context
"""

import requests
from researchforge.config.settings import Settings
from researchforge.utils.state import load_state


class ChatSession:
    def __init__(self):
        self.settings = Settings()
        self.ollama_url = self.settings.ollama_url
        self.model = self.settings.llm_model
        self.history = []
        self.system_prompt = (
            "You are ResearchForge Assistant, an expert in ML research, "
            "datasets, and model training. You help users understand their research topics, "
            "interpret pipeline results, suggest improvements, and debug ML notebooks.\n"
            "You have access to the ResearchForge pipeline which runs:\n"
            "V1 (research retrieval) → V2 (dataset scoring) → V3 (notebook generation) "
            "→ Autoresearch (GPU optimization).\n"
            "Be concise, technical, and actionable."
        )

    def start(self):
        state = load_state()
        last_topic = state.get("v1", {}).get("topic", None)

        print("\n  ResearchForge Chat")
        print("  ─────────────────────────────────────────────")
        print(f"  Model: {self.model} via Ollama at {self.ollama_url}")
        if last_topic:
            print(f"  Context: last pipeline run on \"{last_topic}\"")
        print("  Type 'exit' to quit · 'run <topic>' to start a pipeline · 'status' for last run")
        print("  ─────────────────────────────────────────────\n")

        while True:
            try:
                user_input = input("  You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("  Bye.")
                    break
                if user_input.lower() == "status":
                    from researchforge.utils.display import Display
                    Display.show_status()
                    continue
                if user_input.lower().startswith("run "):
                    topic = user_input[4:].strip()
                    print(f"\n  Run this in your terminal:")
                    print(f"    researchforge run \"{topic}\"\n")
                    continue

                response = self._chat(user_input)
                print(f"\n  RF: {response}\n")

            except KeyboardInterrupt:
                print("\n  Bye.")
                break

    def _chat(self, message: str) -> str:
        self.history.append({"role": "user", "content": message})

        # Build prompt with history (keep last 10 turns)
        conversation = f"System: {self.system_prompt}\n\n"
        for msg in self.history[-10:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation += f"{role}: {msg['content']}\n"
        conversation += "Assistant:"

        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.model, "prompt": conversation, "stream": False},
                timeout=60
            )
            reply = resp.json().get(
                "response",
                "Sorry, could not reach Ollama. Is it running?"
            ).strip()
            self.history.append({"role": "assistant", "content": reply})
            return reply
        except Exception:
            return (
                f"Could not reach Ollama at {self.ollama_url}. "
                "Make sure it's running: `ollama serve`"
            )
