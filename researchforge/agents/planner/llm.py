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

    def _call_openrouter(self, prompt: str) -> str:
        response = requests.post(
            f"{self.settings.openrouter_base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.settings.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.settings.llm_model,
                "messages": [
                    {"role": "system", "content": "Return valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
            },
            timeout=60,
        )
        response.raise_for_status()
        body = response.json()
        return body["choices"][0]["message"]["content"]

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
