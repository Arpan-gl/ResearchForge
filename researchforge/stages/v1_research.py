"""
V1 Research Pipeline
--------------------
1. Rewrite topic into 4 query variants
2. Search via Tavily in parallel
3. Deduplicate + relevance filter + rerank
4. Extract structured findings via Ollama LLM
5. Flag numeric contradictions
6. Flag semantic contradictions via Ollama pairwise comparison
"""

import requests
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from researchforge.config.settings import Settings


class V1Research:
    def __init__(self, ollama_url: str = None, model: str = None, tavily_api_key: str = None):
        self.settings = Settings()
        self.ollama_url = ollama_url or self.settings.ollama_url
        self.model = model or self.settings.llm_model
        self.tavily_api_key = (
            tavily_api_key
            or os.environ.get("TAVILY_API_KEY")
            or self.settings.tavily_api_key
        )

    def run(self, topic: str) -> dict:
        queries = self._rewrite_queries(topic)

        # Step 2: parallel Tavily search
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self._tavily_search, q) for q in queries]
            results = []
            for f in futures:
                results.extend(f.result())

        # Step 3: deduplicate + relevance filter + rerank
        results = self._deduplicate(results)
        results = self._filter_relevance(results, topic)
        ranked = self._rerank(results)
        top_sources = ranked[:10]

        # Step 4: extract and compose
        extracted = self._extract(top_sources, topic)
        findings = self._compose(topic, extracted, top_sources)

        # Step 5/6: contradiction detection
        numeric_contradictions = self._detect_contradictions(findings)
        semantic_contradictions = self._detect_semantic_contradictions(findings)
        findings["contradictions"] = numeric_contradictions + semantic_contradictions
        return findings

    # ──────────────────────────────────────────────────────────────
    def _rewrite_queries(self, topic: str) -> list:
        prompt = (
            f"Generate 4 high-quality search queries for this topic:\n{topic}\n\n"
            "Rules:\n"
            "- Make them diverse\n"
            "- Include one academic-style query\n"
            "- Include one benchmark-focused query\n\n"
            "Return ONLY a JSON array of strings."
        )
        queries = self._safe_json(self._ask_llm(prompt), fallback=[])
        if isinstance(queries, list) and len(queries) >= 4:
            return queries[:4]
        return [
            topic,
            f"{topic} benchmark",
            f"{topic} survey 2024",
            f"{topic} academic paper",
        ]

    def _tavily_search(self, query: str) -> list:
        """Tavily search API (advanced mode)."""
        tavily_api_key = getattr(self, "tavily_api_key", "")
        if not tavily_api_key:
            return []
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": tavily_api_key,
                    "query": query,
                    "search_depth": "advanced",
                    "max_results": 5,
                },
                timeout=10,
            )
            data = resp.json()
            results = []
            for r in data.get("results", []):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", ""),
                    "source": "web",
                    "score": 0.5,
                })
            return results
        except Exception:
            return []

    def _web_search(self, query: str) -> list:
        """Backward-compatible alias used in tests/legacy code."""
        return self._tavily_search(query)

    def _arxiv_search(self, query: str) -> list:
        """Backward-compatible alias used in tests/legacy code."""
        return self._tavily_search(query)

    def _filter_relevance(self, results: list, topic: str) -> list:
        topic_words = {w for w in re.findall(r"\w+", topic.lower()) if len(w) > 2}
        if not topic_words:
            return results

        filtered = []
        for r in results:
            title = r.get("title", "") or ""
            snippet = r.get("snippet", "") or ""
            text = f"{title} {snippet}".lower()
            overlap = sum(1 for w in topic_words if w in text)
            if overlap >= 2:
                filtered.append(r)
        return filtered or results

    def _rerank(self, results: list) -> list:
        """Score = 0.4*depth + 0.3*credibility + 0.3*recency"""
        results = self._deduplicate(results)
        today = datetime.now().date()

        for r in results:
            url = (r.get("url") or "").lower()
            snippet = r.get("snippet") or ""

            recency = 0.5
            if r.get("published"):
                try:
                    pub = datetime.fromisoformat(r["published"]).date()
                    days_old = (today - pub).days
                    recency = max(0.0, 1.0 - days_old / 730)
                except Exception:
                    pass

            credibility = 0.6
            if "arxiv" in url:
                credibility = 0.9
            elif "github" in url:
                credibility = 0.8

            depth = min(1.0, len(snippet) / 300)
            r["final_score"] = 0.4 * depth + 0.3 * credibility + 0.3 * recency

        return sorted(results, key=lambda x: x.get("final_score", 0.0), reverse=True)

    def _extract_findings(self, topic: str, ranked_results: list) -> dict:
        """Backward-compatible extraction API expected by tests and callers."""
        result = self._extract(ranked_results, topic)
        if isinstance(result, dict) and result.get("key_findings"):
            return result
        return {
            "key_findings": ["Could not parse LLM response — check Ollama connection"],
            "metrics": [],
            "datasets": [],
            "limitations": [],
            "recommended_models": [],
            "problem_type": "unknown",
        }

    def _extract(self, sources: list, topic: str) -> dict:
        context = "\n\n".join([
            f"{i+1}. {s.get('title', '')}\n{s.get('snippet', '')}"
            for i, s in enumerate(sources)
        ])
        prompt = (
            "You are a technical research analyst.\n\n"
            f"Topic: {topic}\n\n"
            f"Sources:\n{context}\n\n"
            "Extract structured findings.\n\n"
            "STRICT RULES:\n"
            "- Include REAL numbers if present\n"
            "- No vague statements\n"
            "- Each finding must mention model/dataset/metric\n\n"
            "Return ONLY JSON with this structure:\n"
            '{\n'
            '  "key_findings": [],\n'
            '  "metrics": [],\n'
            '  "limitations": [],\n'
            '  "datasets": [],\n'
            '  "recommended_models": [],\n'
            '  "problem_type": "classification|regression|nlp|graph|unknown"\n'
            "}\n\n"
            "Return ONLY valid JSON."
        )
        return self._safe_json(self._ask_llm(prompt), fallback={})

    def _compose(self, topic: str, extracted: dict, sources: list) -> dict:
        findings = extracted.get("key_findings", [])
        metrics = extracted.get("metrics", [])
        return {
            "topic": topic,
            "overview": findings[:3],
            "current_state": findings[3:6],
            "key_findings": findings,
            "limitations": extracted.get("limitations", []),
            "metrics": metrics,
            "datasets": extracted.get("datasets", []),
            "recommended_models": extracted.get("recommended_models", []),
            "problem_type": extracted.get("problem_type", "unknown"),
            "sources": sources[:10],
        }

    def _detect_contradictions(self, findings: dict) -> list:
        """Numeric metric variance > 15% signals a contradiction."""
        from collections import defaultdict
        metrics = findings.get("metrics", [])
        contradictions = []
        metric_groups: dict = defaultdict(list)

        for m in metrics:
            try:
                val = float(str(m["value"]).replace("%", ""))
                metric_groups[m["name"]].append(val)
            except Exception:
                pass

        for name, vals in metric_groups.items():
            if len(vals) > 1:
                avg = sum(vals) / len(vals)
                variance = (max(vals) - min(vals)) / avg if avg > 0 else 0
                if variance > 0.15:
                    contradictions.append(
                        f"Contradicting numeric values for '{name}': "
                        f"{vals} — variance {variance:.0%}"
                    )
        return contradictions

    def _detect_semantic_contradictions(self, findings: dict) -> list:
        """
        PRIORITY 5 — Ask Ollama to compare key findings pairwise and identify
        semantic contradictions (e.g. conflicting directional claims).
        """
        key_findings = findings.get("key_findings", [])
        if len(key_findings) < 2:
            return []

        numbered = "\n".join(
            f"[{i}] {f}" for i, f in enumerate(key_findings)
        )
        prompt = (
            "You are a scientific fact-checker. "
            "Given these research findings, identify any pairs that make "
            "contradictory or conflicting claims:\n\n"
            f"{numbered}\n\n"
            "Return ONLY a JSON array. Each element should be:\n"
            '{"finding_a": <index>, "finding_b": <index>, "contradiction": "<brief explanation>"}\n'
            "If there are no contradictions, return an empty array [].\n"
            "Return ONLY valid JSON. No explanation."
        )
        response = self._ask_llm(prompt)
        try:
            clean = response.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            pairs = json.loads(clean)
            if not isinstance(pairs, list):
                return []
            results = []
            for pair in pairs:
                a = pair.get("finding_a", "?")
                b = pair.get("finding_b", "?")
                desc = pair.get("contradiction", "")
                if desc:
                    results.append(
                        f"Semantic contradiction between findings [{a}] and [{b}]: {desc}"
                    )
            return results
        except Exception:
            return []

    def _ask_llm(self, prompt: str) -> str:
        """Send prompt to Ollama — the user's local LLM."""
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=60
            )
            return resp.json().get("response", "")
        except Exception as e:
            return f'{{"error": "Ollama not reachable: {str(e)}"}}'

    def _deduplicate(self, results: list) -> list:
        seen = set()
        unique = []
        for r in results:
            url = r.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            unique.append(r)
        return unique

    def _safe_json(self, text: str, fallback):
        try:
            clean = text.strip()
            clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
            clean = re.sub(r"\s*```$", "", clean)
            return json.loads(clean)
        except Exception:
            return fallback
