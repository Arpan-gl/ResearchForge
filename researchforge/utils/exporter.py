"""
Exporter — generate HTML and PDF reports from ResearchForge pipeline results
Priority 6 from ResearchForge build plan.

HTML report contains:
  - V1: key findings table + contradictions
  - V2: dataset score gauge (matplotlib → base64 inline)
  - V3: notebook section previews
  - Autoresearch: improvement chart (matplotlib → base64 inline)

PDF: convert HTML to PDF via weasyprint (optional dep).
     If weasyprint not installed, prints instructions.
"""

import base64
import io
import json
import os
from pathlib import Path
from datetime import datetime


class Exporter:

    # ── Public interface ──────────────────────────────────────────

    def to_html(self, results: dict, output_path: str) -> str:
        """Generate a clean HTML report and write to output_path."""
        v1   = results.get("v1", {})
        v2   = results.get("v2", {})
        v3   = results.get("v3", {})
        auto = results.get("autoresearch", {})
        ts   = results.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        topic = v1.get("topic", "Unknown topic")

        score_chart_b64   = self._score_gauge_chart(v2.get("score", 0))
        auto_chart_b64    = self._autoresearch_chart(auto) if auto else None
        metric_name       = v3.get("metric_name", "metric")
        notebook_preview  = self._notebook_preview(v3.get("notebook_path"))

        html = self._render_html(
            topic=topic,
            timestamp=ts,
            v1=v1,
            v2=v2,
            v3=v3,
            auto=auto,
            score_chart_b64=score_chart_b64,
            auto_chart_b64=auto_chart_b64,
            metric_name=metric_name,
            notebook_preview=notebook_preview,
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return output_path

    def to_pdf(self, results: dict, output_path: str) -> str:
        """Convert to PDF via weasyprint. Falls back to HTML if not installed."""
        html_path = output_path.replace(".pdf", "_tmp.html")
        self.to_html(results, html_path)

        try:
            import weasyprint
            weasyprint.HTML(filename=html_path).write_pdf(output_path)
            os.remove(html_path)
            return output_path
        except ImportError:
            raise ImportError(
                "weasyprint is required for PDF export.\n"
                "Install with: pip install weasyprint\n"
                "Or on Windows: see https://doc.courtbouillon.org/weasyprint/stable/first_steps.html\n"
                f"HTML report saved at: {html_path}"
            )

    # ── Chart helpers ─────────────────────────────────────────────

    def _score_gauge_chart(self, score: float) -> str:
        """Dataset quality score — horizontal bar chart → base64 PNG."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 1.2))
            colors = ["#2ecc71" if score >= 0.7 else "#e67e22" if score >= 0.4 else "#e74c3c"]
            ax.barh(["Dataset quality"], [score], color=colors, height=0.5)
            ax.barh(["Dataset quality"], [1.0 - score], left=[score],
                    color="#ecf0f1", height=0.5)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Score (0–1)")
            ax.set_title(f"V2 Dataset Score: {score:.2f}", fontsize=11, fontweight="bold")
            ax.spines[["top", "right", "left"]].set_visible(False)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode()
        except Exception:
            return ""

    def _autoresearch_chart(self, auto: dict) -> str:
        """Autoresearch history — score over experiments → base64 PNG."""
        history = auto.get("history", [])
        if not history:
            return ""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            exp_nums = list(range(1, len(history) + 1))
            scores   = [h.get("score", 0) for h in history]
            accepted = [h.get("accepted", False) for h in history]

            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(exp_nums, scores, color="#3498db", linewidth=1.5, alpha=0.6)
            for i, (x, y, a) in enumerate(zip(exp_nums, scores, accepted)):
                ax.scatter(x, y,
                           color="#2ecc71" if a else "#e74c3c",
                           s=60, zorder=5)

            baseline = auto.get("baseline_score", 0)
            best     = auto.get("best_score", 0)
            ax.axhline(baseline, color="#e74c3c", linestyle="--", linewidth=1, label=f"Baseline {baseline:.3f}")
            ax.axhline(best, color="#2ecc71", linestyle="--", linewidth=1, label=f"Best {best:.3f}")

            ax.set_xlabel("Experiment #")
            ax.set_ylabel(auto.get("metric", "metric"))
            ax.set_title("Autoresearch Experiment History", fontweight="bold")
            ax.legend(fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode()
        except Exception:
            return ""

    def _notebook_preview(self, nb_path: str | None, max_cells: int = 3) -> str:
        """Return first N code cells as pre-formatted HTML snippets."""
        if not nb_path or not os.path.exists(nb_path):
            return "<p><em>Notebook not found.</em></p>"
        try:
            import nbformat
            with open(nb_path, encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)
            snippets = []
            count = 0
            for cell in nb.cells:
                if cell.cell_type == "code" and count < max_cells:
                    code = cell.source[:800].replace("<", "&lt;").replace(">", "&gt;")
                    snippets.append(f'<pre class="code-cell"><code>{code}</code></pre>')
                    count += 1
            return "\n".join(snippets) if snippets else "<p>No code cells found.</p>"
        except Exception as e:
            return f"<p>Could not read notebook: {e}</p>"

    # ── HTML template ─────────────────────────────────────────────

    def _render_html(self, **kw) -> str:
        v1, v2, v3    = kw["v1"], kw["v2"], kw["v3"]
        auto          = kw["auto"]
        topic         = kw["topic"]
        ts            = kw["timestamp"]
        metric_name   = kw["metric_name"]
        score_b64     = kw["score_chart_b64"]
        auto_b64      = kw.get("auto_chart_b64")
        nb_preview    = kw["notebook_preview"]

        # V1 findings table rows
        findings_rows = "".join(
            f"<tr><td>{i+1}</td><td>{f}</td></tr>"
            for i, f in enumerate(v1.get("key_findings", []))
        )
        contradictions_html = ""
        for c in v1.get("contradictions", []):
            contradictions_html += f'<li class="warn">⚠ {c}</li>'
        if contradictions_html:
            contradictions_html = f"<ul>{contradictions_html}</ul>"

        # V2 risks
        risks_html = "".join(
            f'<li class="warn">⚠ {r}</li>' for r in v2.get("risks", [])
        )

        # Auto section
        auto_html = ""
        if auto:
            accepted = sum(1 for h in auto.get("history", []) if h.get("accepted"))
            auto_html = f"""
            <h2>🤖 Stage 4 — Autoresearch Results</h2>
            <table>
              <tr><th>Baseline {metric_name}</th><td>{auto.get('baseline_score', 0):.4f}</td></tr>
              <tr><th>Best {metric_name}</th><td><strong>{auto.get('best_score', 0):.4f}</strong></td></tr>
              <tr><th>Improvement</th><td>+{auto.get('improvement_pct', 0):.2f}%</td></tr>
              <tr><th>Experiments run</th><td>{auto.get('experiments_run', 0)}</td></tr>
              <tr><th>Accepted changes</th><td>{accepted}</td></tr>
              <tr><th>Best commit</th><td><code>{auto.get('best_commit', '—')}</code></td></tr>
            </table>
            """
            if auto_b64:
                auto_html += f'<img src="data:image/png;base64,{auto_b64}" alt="Experiment history chart">'

        score_img = (
            f'<img src="data:image/png;base64,{score_b64}" alt="Dataset score">'
            if score_b64 else ""
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ResearchForge Report — {topic}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e2e8f0;
            line-height: 1.6; padding: 2rem; }}
    .container {{ max-width: 1000px; margin: 0 auto; }}
    h1 {{ font-size: 2rem; color: #63b3ed; margin-bottom: 0.25rem; }}
    h2 {{ font-size: 1.2rem; color: #90cdf4; margin: 2rem 0 0.75rem; border-left: 3px solid #3182ce;
          padding-left: 0.75rem; }}
    .meta {{ color: #718096; font-size: 0.9rem; margin-bottom: 2rem; }}
    table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
    th, td {{ padding: 0.6rem 0.8rem; border: 1px solid #2d3748; text-align: left; }}
    th {{ background: #1e2535; color: #90cdf4; font-weight: 600; }}
    tr:nth-child(even) {{ background: #1a202c; }}
    .warn {{ color: #f6ad55; }}
    ul {{ padding-left: 1.5rem; margin: 0.5rem 0; }}
    li {{ margin: 0.3rem 0; }}
    pre.code-cell {{ background: #1a202c; border: 1px solid #2d3748; border-radius: 6px;
                     padding: 1rem; overflow-x: auto; font-size: 0.82rem;
                     color: #a8d8ea; margin: 0.75rem 0; }}
    code {{ font-family: 'Fira Code', 'Consolas', monospace; }}
    img {{ max-width: 100%; margin: 1rem 0; border-radius: 8px; }}
    .badge {{ display: inline-block; background: #2d3748; border-radius: 4px;
              padding: 0.2rem 0.5rem; font-size: 0.8rem; margin-right: 0.5rem; color: #90cdf4; }}
    footer {{ margin-top: 3rem; color: #4a5568; font-size: 0.8rem; text-align: center; }}
  </style>
</head>
<body>
<div class="container">
  <h1>🔬 ResearchForge Report</h1>
  <p class="meta">
    <span class="badge">Topic</span>{topic}
    &nbsp;|&nbsp;
    <span class="badge">Generated</span>{ts}
    &nbsp;|&nbsp;
    <span class="badge">Model</span>{v3.get('model', '—')}
  </p>

  <h2>📚 Stage 1 — V1 Research Findings</h2>
  <table>
    <thead><tr><th>#</th><th>Key Finding</th></tr></thead>
    <tbody>{findings_rows}</tbody>
  </table>
  {contradictions_html if contradictions_html else ""}

  <h2>📊 Stage 2 — V2 Dataset Audit</h2>
  {score_img}
  <table>
    <tr><th>Dataset</th><td>{v2.get('name', '—')}</td></tr>
    <tr><th>Shape</th><td>{v2.get('shape', '—')}</td></tr>
    <tr><th>Source</th><td>{v2.get('source', '—')}</td></tr>
    <tr><th>Label column</th><td><code>{v2.get('label_column', '—')}</code></td></tr>
    <tr><th>Problem type</th><td>{v2.get('problem_type', '—')}</td></tr>
    <tr><th>Score</th><td><strong>{v2.get('score', 0):.2f}</strong></td></tr>
  </table>
  {f'<ul>{risks_html}</ul>' if risks_html else ''}

  <h2>📓 Stage 3 — V3 Generated Notebook Preview</h2>
  <p>Notebook: <code>{v3.get('notebook_path', '—')}</code>
     &nbsp;|&nbsp; Expected {metric_name}: {v3.get('expected_range', '—')}</p>
  {nb_preview}

  {auto_html}

  <footer>Generated by ResearchForge · {ts}</footer>
</div>
</body>
</html>
"""
