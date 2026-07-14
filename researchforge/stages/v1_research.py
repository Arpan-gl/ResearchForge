"""
V1 Research Pipeline
--------------------
1. Rewrite topic into 4 query variants
2. Multi-source retrieval (Tavily + optional academic sources)
3. Deduplicate + relevance filter + rerank
4. Extract structured findings via Ollama LLM
5. Build research memory (chunking + embedding rerank)
6. Flag numeric contradictions
7. Flag semantic contradictions via Ollama pairwise comparison
"""

import requests
import json
import os
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from researchforge.config.settings import Settings
from researchforge.agents.planner.llm import LLMRouter


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
        self.semantic_scholar_key = self.settings.semantic_scholar_key
        self.github_token = self.settings.github_token
        self.hf_token = self.settings.huggingface_token
        self.multi_source = self.settings.enable_multi_source
        self.llm_router = LLMRouter(self.settings)
        self._research_llm_error = ""

    def run(self, topic: str) -> dict:
        queries = self._rewrite_queries(topic)

        # Step 2: parallel multi-source retrieval (gated)
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(self._collect_sources, q) for q in queries]
            results = []
            for f in futures:
                results.extend(f.result())

        # Step 3: deduplicate + relevance filter + rerank
        results = self._deduplicate(results)
        results = self._filter_relevance(results, topic)
        ranked = self._rerank(results)
        top_sources = ranked[:10]
        retrieved_at = datetime.now(timezone.utc).isoformat()
        for source in top_sources:
            source.setdefault(
                "provenance",
                {
                    "source": source.get("url", "unknown"),
                    "retrieved_at": retrieved_at,
                    "agent": "research",
                },
            )

        # Step 4: extract and compose
        extracted = self._extract(top_sources, topic)
        findings = self._compose(topic, extracted, top_sources)

        # Step 5: build research memory (lightweight chunk + embed)
        memory = self._build_research_memory(topic, top_sources)
        findings["research_memory"] = memory

        # Step 6: structured research blueprint (optional LLM pass)
        findings["research_blueprint"] = self._extract_research_blueprint(topic, top_sources)

        # Step 7/8: contradiction detection
        numeric_contradictions = self._detect_contradictions(findings)
        semantic_contradictions = self._detect_semantic_contradictions(findings)
        findings["contradictions"] = numeric_contradictions + semantic_contradictions
        findings["provenance"] = {
            "source": "research:multi-source",
            "retrieved_at": retrieved_at,
            "agent": "research",
        }
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
        lowered = topic.lower()
        if "ipl" in lowered or "indian premier league" in lowered or "cricket" in lowered:
            return [
                "IPL cricket match winner prediction",
                "Indian Premier League match outcome prediction",
                "IPL match results dataset benchmark",
                "cricket match outcome prediction academic paper",
            ]
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
        if not self.multi_source:
            return []
        return self._search_arxiv(query)

    def _collect_sources(self, query: str) -> list:
        sources = []
        sources.extend(self._web_search(query))
        if getattr(self, "multi_source", False):
            sources.extend(self._search_arxiv(query))
            sources.extend(self._search_semantic_scholar(query))
            sources.extend(self._search_hf_papers(query))
            sources.extend(self._search_github(query))
        return sources

    def _search_arxiv(self, query: str, max_results: int = 4) -> list:
        try:
            response = requests.get(
                "https://export.arxiv.org/api/query",
                params={
                    "search_query": f"all:{query}",
                    "start": 0,
                    "max_results": max_results,
                    "sortBy": "relevance",
                },
                timeout=10,
            )
            response.raise_for_status()
            root = ET.fromstring(response.text)
            namespace = {"atom": "http://www.w3.org/2005/Atom"}
            results = []
            for entry in root.findall("atom:entry", namespace):
                get_text = lambda name: (entry.findtext(f"atom:{name}", default="", namespaces=namespace) or "").strip()
                published = get_text("published")
                results.append({
                    "title": get_text("title"),
                    "url": get_text("id"),
                    "snippet": get_text("summary")[:800],
                    "source": "arxiv",
                    "published": published[:10] if published else None,
                })
            return results
        except Exception:
            return []

    def _search_semantic_scholar(self, query: str, max_results: int = 4) -> list:
        if not getattr(self, "semantic_scholar_key", ""):
            return []
        try:
            resp = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": query,
                    "limit": max_results,
                    "fields": "title,url,abstract,year,citationCount,venue",
                },
                headers={"x-api-key": getattr(self, "semantic_scholar_key", "")},
                timeout=10,
            )
            data = resp.json()
            results = []
            for item in data.get("data", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": (item.get("abstract") or "")[:800],
                    "source": "semantic_scholar",
                    "published": str(item.get("year") or ""),
                })
            return results
        except Exception:
            return []

    def _search_hf_papers(self, query: str, max_results: int = 4) -> list:
        if not getattr(self, "hf_token", ""):
            return []
        try:
            resp = requests.get(
                "https://huggingface.co/api/papers",
                params={"search": query, "limit": max_results},
                headers={"Authorization": f"Bearer {getattr(self, 'hf_token', '')}"},
                timeout=10,
            )
            results = []
            for item in resp.json():
                title = item.get("title") or item.get("paper", {}).get("title", "")
                url = item.get("url") or item.get("paper", {}).get("url", "")
                summary = item.get("summary") or item.get("abstract", "")
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": (summary or "")[:800],
                    "source": "hf_papers",
                })
            return results
        except Exception:
            return []

    def _search_github(self, query: str, max_results: int = 4) -> list:
        if not getattr(self, "github_token", ""):
            return []
        try:
            resp = requests.get(
                "https://api.github.com/search/repositories",
                params={"q": query, "sort": "stars", "order": "desc", "per_page": max_results},
                headers={"Authorization": f"Bearer {getattr(self, 'github_token', '')}"},
                timeout=10,
            )
            results = []
            for item in resp.json().get("items", []):
                results.append({
                    "title": item.get("full_name", ""),
                    "url": item.get("html_url", ""),
                    "snippet": (item.get("description") or "")[:400],
                    "source": "github",
                })
            return results
        except Exception:
            return []

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
        return filtered

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
        parsed = self._safe_json(self._ask_llm(prompt), fallback={})
        if isinstance(parsed, dict) and parsed.get("key_findings"):
            return parsed
        return self._deterministic_extraction(topic, sources)

    def _extract_research_blueprint(self, topic: str, sources: list) -> dict:
        if not sources:
            return {}
        context = "\n\n".join([
            f"{i+1}. {s.get('title', '')}\n{s.get('snippet', '')}"
            for i, s in enumerate(sources)
        ])
        prompt = (
            "You are a research analyst. Build a structured research memory.\n\n"
            f"Topic: {topic}\n\n"
            f"Sources:\n{context}\n\n"
            "Return ONLY JSON with this structure:\n"
            "{\n"
            "  \"problem_definition\": \"\",\n"
            "  \"datasets_used\": [],\n"
            "  \"architectures\": [],\n"
            "  \"metrics\": [],\n"
            "  \"limitations\": [],\n"
            "  \"contradictions\": [],\n"
            "  \"future_work\": [],\n"
            "  \"reproducibility_score\": 0.0\n"
            "}\n\n"
            "Return ONLY valid JSON."
        )
        parsed = self._safe_json(self._ask_llm(prompt), fallback={})
        if parsed:
            return parsed
        return {
            "problem_definition": topic,
            "datasets_used": [],
            "architectures": [],
            "metrics": [],
            "limitations": [self._research_llm_error] if self._research_llm_error else [],
            "contradictions": [],
            "future_work": [],
            "reproducibility_score": 0.0,
        }

    def _deterministic_extraction(self, topic: str, sources: list) -> dict:
        """Keep retrieved evidence usable when the summarizer is unavailable."""
        findings = []
        for index, source in enumerate(sources[:5], start=1):
            title = (source.get("title") or "Untitled source").strip()
            snippet = " ".join((source.get("snippet") or "").split())
            if snippet:
                snippet = snippet[:300]
                findings.append(f"[{index}] {title}: {snippet}")
            else:
                findings.append(f"[{index}] Retrieved source: {title}")

        lowered = topic.lower()
        if any(token in lowered for token in ("winner", "classify", "classification", "detect")):
            problem_type = "classification"
        elif any(token in lowered for token in ("forecast", "predict", "regression")):
            problem_type = "regression"
        else:
            problem_type = "unknown"

        return {
            "key_findings": findings,
            "metrics": [],
            "limitations": [
                "LLM evidence synthesis was unavailable; findings above are direct retrieved excerpts."
            ] if getattr(self, "_research_llm_error", "") else [],
            "datasets": [],
            "recommended_models": [],
            "problem_type": problem_type,
        }

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

    def _build_research_memory(self, topic: str, sources: list) -> dict:
        chunks = []
        for idx, src in enumerate(sources):
            snippet = (src.get("snippet") or "").strip()
            if not snippet:
                continue
            chunks.append({
                "source_index": idx,
                "title": src.get("title", ""),
                "url": src.get("url", ""),
                "text": snippet,
            })

        if not chunks:
            return {"query": topic, "chunks": []}

        texts = [c["text"] for c in chunks]
        scores, method = self._score_chunks(topic, texts)
        for chunk, score in zip(chunks, scores):
            chunk["score"] = float(score)

        chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return {
            "query": topic,
            "method": method,
            "chunks": chunks[:12],
        }

    def _score_chunks(self, query: str, texts: list) -> tuple[list, str]:
        if not texts:
            return [], "none"
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode([query] + texts, normalize_embeddings=True)
            query_vec = embeddings[0]
            doc_vecs = embeddings[1:]
            scores = [float((query_vec @ vec)) for vec in doc_vecs]
            return scores, "sentence_transformers"
        except Exception:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity

                vectorizer = TfidfVectorizer(stop_words="english")
                matrix = vectorizer.fit_transform([query] + texts)
                scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten().tolist()
                return scores, "tfidf"
            except Exception:
                return [0.0 for _ in texts], "none"

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
        """Use Gemini for research queries, summaries, and evidence reasoning."""
        if self._research_llm_error:
            return json.dumps({"error": self._research_llm_error})
        try:
            response, _provider = self.llm_router.generate_research(prompt)
            return response
        except Exception as exc:
            self._research_llm_error = str(exc)
            return json.dumps({"error": str(exc)})

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
