"""LLM routing for planner tasks."""

import json

import requests


class LLMRouter:
    def __init__(self, settings):
        self.settings = settings

    def parse_json(self, prompt: str) -> tuple[dict, str]:
        last_error = None
        if self.settings.llm_provider in {"auto", "ollama"} and self._ollama_available():
            try:
                return self._load_json(self._call_ollama(prompt)), "ollama"
            except Exception as exc:
                last_error = exc

        if self.settings.llm_provider in {"auto", "openrouter"} and self.settings.openrouter_api_key:
            try:
                return self._load_json(self._call_openrouter(prompt)), "openrouter"
            except Exception as exc:
                last_error = exc

        if last_error:
            raise RuntimeError(f"No LLM backend produced valid JSON: {last_error}") from last_error
        raise RuntimeError("No LLM backend is available. Start Ollama or set OPENROUTER_API_KEY.")

    def generate_agent(self, prompt: str) -> tuple[str, str]:
        """Use Ollama for agent work when reachable, then OpenRouter."""
        last_error = None
        if self.settings.llm_provider in {"auto", "ollama"} and self._ollama_available():
            try:
                return self._call_ollama_text(prompt), "ollama"
            except Exception as exc:
                last_error = exc
        if self.settings.llm_provider in {"auto", "openrouter"} and self.settings.openrouter_api_key:
            try:
                return self._call_openrouter(prompt), "openrouter"
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"No agent LLM backend is available: {last_error or 'configure OpenRouter'}")

    def generate_research(self, prompt: str) -> tuple[str, str]:
        """Use Gemini 2.5 Flash Lite through OpenRouter for research reasoning."""
        if not self.settings.openrouter_api_key:
            raise RuntimeError("Research reasoning requires OPENROUTER_API_KEY.")
        try:
            return self._call_openrouter(prompt, model=self.settings.research_model), "openrouter:gemini"
        except Exception as exc:
            error_text = str(exc).lower()
            if not any(token in error_text for token in ("402", "paymentrequired", "payment required", "more credits", "insufficient credit")):
                raise
            return self._call_openrouter(prompt, model=self.settings.openrouter_free_model), "openrouter:free"

    def _ollama_available(self) -> bool:
        try:
            response = requests.get(f"{self.settings.ollama_url.rstrip('/')}/api/tags", timeout=3)
            return response.ok
        except Exception:
            return False

    def _call_ollama(self, prompt: str) -> str:
        response = requests.post(
            f"{self.settings.ollama_url.rstrip('/')}/api/generate",
            json={
                "model": self.settings.llm_model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json().get("response", "")

    def _call_ollama_text(self, prompt: str) -> str:
        response = requests.post(
            f"{self.settings.ollama_url.rstrip('/')}/api/generate",
            json={"model": self.settings.llm_model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        response.raise_for_status()
        return response.json().get("response", "")

    def _call_openrouter(self, prompt: str, model: str | None = None) -> str:
        try:
            from openrouter import OpenRouter
        except ImportError as exc:
            raise RuntimeError("Install the OpenRouter SDK with: pip install openrouter") from exc

        with OpenRouter(
            api_key=self.settings.openrouter_api_key,
            server_url=self.settings.openrouter_base_url,
        ) as client:
            completion = client.chat.send(
                model=model or self.settings.llm_model,
                messages=[
                    {"role": "system", "content": "Return valid JSON only when requested."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=2048,
                timeout_ms=60000,
            )
        return completion.choices[0].message.content or ""


    @staticmethod
    def _load_json(text: str) -> dict:
        cleaned = (text or "").strip()
        if cleaned.startswith("```"):
            lines = [line for line in cleaned.splitlines() if not line.startswith("```")]
            cleaned = "\n".join(lines).strip()
        payload = json.loads(cleaned)
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")
        return payload
