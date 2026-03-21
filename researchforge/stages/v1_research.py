"""
V1 Research Pipeline
--------------------
1. Rewrite topic into 4 query variants
2. Search web (DuckDuckGo) + arXiv in parallel
3. Rerank by recency + credibility + depth
4. Extract structured findings via Ollama LLM
5. Flag numeric contradictions
6. Flag semantic contradictions via Ollama pairwise comparison
"""

import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor
from researchforge.config.settings import Settings


class V1Research:
    def __init__(self):
        self.settings = Settings()
        self.ollama_url = self.settings.ollama_url
        self.model = self.settings.llm_model

    def run(self, topic: str) -> dict:
        # Step 1: rewrite topic into 4 query variants
        queries = self._rewrite_queries(topic)

        # Step 2: parallel search
        with ThreadPoolExecutor(max_workers=4) as executor:
            web_futures   = [executor.submit(self._web_search,   q) for q in queries[:2]]
            arxiv_futures = [executor.submit(self._arxiv_search, q) for q in queries[2:]]
            all_results = []
            for f in web_futures + arxiv_futures:
                all_results.extend(f.result())

        # Step 3: deduplicate + rerank
        ranked = self._rerank(all_results)

        # Step 4: extract findings via LLM
        findings = self._extract_findings(topic, ranked)
        findings["topic"] = topic

        # Step 5: numeric contradiction detection
        numeric_contradictions = self._detect_contradictions(findings)

        # Step 6: semantic contradiction detection (NEW — Priority 5)
        semantic_contradictions = self._detect_semantic_contradictions(findings)

        findings["contradictions"] = numeric_contradictions + semantic_contradictions
        findings["sources"] = ranked[:10]

        return findings

    # ──────────────────────────────────────────────────────────────
    def _rewrite_queries(self, topic: str) -> list:
        prompt = (
            f'Given this research topic: "{topic}"\n'
            "Generate exactly 4 search query variants as a JSON array:\n"
            "1. General natural language query\n"
            "2. Keyword-focused query (nouns + technical terms only)\n"
            "3. Pseudo-answer query (what would an expert paper title look like)\n"
            "4. Core content query (most specific technical formulation)\n\n"
            "Return ONLY a JSON array of 4 strings. No explanation."
        )
        response = self._ask_llm(prompt)
        try:
            clean = response.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            queries = json.loads(clean)
            if isinstance(queries, list) and len(queries) >= 4:
                return queries[:4]
        except Exception:
            pass
        # fallback
        return [
            topic,
            topic + " benchmark",
            topic + " survey 2024",
            topic + " neural network",
        ]

    def _web_search(self, query: str) -> list:
        """DuckDuckGo instant answer API — free, no key required."""
        try:
            resp = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1},
                timeout=8
            )
            data = resp.json()
            results = []
            for item in data.get("RelatedTopics", [])[:5]:
                if "Text" in item and item.get("FirstURL"):
                    results.append({
                        "title":   item.get("Text", "")[:80],
                        "url":     item.get("FirstURL", ""),
                        "snippet": item.get("Text", ""),
                        "source":  "web",
                        "score":   0.5,
                    })
            return results
        except Exception:
            return []

    def _arxiv_search(self, query: str) -> list:
        """arXiv API — free, no auth required."""
        try:
            import urllib.parse
            import xml.etree.ElementTree as ET

            params = {
                "search_query": f"all:{urllib.parse.quote(query)}",
                "start": 0,
                "max_results": 5,
                "sortBy": "relevance",
            }
            resp = requests.get(
                "http://export.arxiv.org/api/query",
                params=params, timeout=10
            )
            results = []
            root = ET.fromstring(resp.text)
            ns = "{http://www.w3.org/2005/Atom}"
            for entry in root.findall(f"{ns}entry"):
                title     = entry.find(f"{ns}title").text.strip()
                summary   = entry.find(f"{ns}summary").text.strip()[:300]
                link      = entry.find(f"{ns}id").text.strip()
                published = entry.find(f"{ns}published").text[:10]
                results.append({
                    "title":     title,
                    "url":       link,
                    "snippet":   summary,
                    "source":    "arxiv",
                    "published": published,
                    "score":     0.8,
                })
            return results
        except Exception:
            return []

    def _rerank(self, results: list) -> list:
        """Score = 0.3*recency + 0.3*credibility + 0.4*depth"""
        import datetime
        today = datetime.date.today()

        for r in results:
            recency = 0.5
            if "published" in r:
                try:
                    pub = datetime.date.fromisoformat(r["published"])
                    days_old = (today - pub).days
                    recency = max(0.0, 1.0 - days_old / 730)  # decay over 2 years
                except Exception:
                    pass

            credibility = 0.8 if r.get("source") == "arxiv" else 0.5
            depth = min(1.0, len(r.get("snippet", "")) / 300)
            r["final_score"] = 0.3 * recency + 0.3 * credibility + 0.4 * depth

        # deduplicate by URL
        seen, unique = set(), []
        for r in results:
            if r["url"] not in seen:
                seen.add(r["url"])
                unique.append(r)

        return sorted(unique, key=lambda x: x["final_score"], reverse=True)

    def _extract_findings(self, topic: str, ranked_results: list) -> dict:
        context = "\n\n".join([
            f"[{i+1}] {r['title']}\n{r['snippet']}"
            for i, r in enumerate(ranked_results[:8])
        ])
        prompt = (
            f'You are a research analyst. Given this topic: "{topic}"\n\n'
            f"And these source snippets:\n{context}\n\n"
            "Extract and return ONLY a JSON object with this exact structure:\n"
            '{\n'
            '  "key_findings": ["finding 1 with citation [1]", "finding 2 with citation [3]"],\n'
            '  "metrics": [{"name": "metric_name", "value": "value", "unit": "unit", "source": 1}],\n'
            '  "datasets": [{"name": "dataset_name", "size": "Xk rows", "task": "task_type"}],\n'
            '  "limitations": ["limitation 1", "limitation 2"],\n'
            '  "recommended_models": ["model1", "model2"],\n'
            '  "problem_type": "classification|regression|nlp|graph"\n'
            "}\n\n"
            "Be specific and cite source numbers. Return ONLY valid JSON."
        )
        response = self._ask_llm(prompt)
        try:
            clean = response.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(clean)
        except Exception:
            return {
                "key_findings": ["Could not parse LLM response — check Ollama connection"],
                "metrics": [],
                "datasets": [],
                "limitations": [],
                "recommended_models": [],
                "problem_type": "unknown",
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
