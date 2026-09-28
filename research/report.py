"""ResearchReport: assembles, saves, and renders the final action report.

:class:`ResearchReport` takes ranked opportunities, avoid-lists, clipping
guidance, and run metadata, then produces one canonical report dict that
answers the six questions that matter:

1. TOP 5 safest revenue streams to pursue.
2. Which YouTube clipping strategies are safe.
3. What to absolutely AVOID (with reasons).
4. The first 3 projects to build with the BuilderAgent.
5. Realistic revenue expectations.
6. Mandatory legal precautions.

The report renders to rich console output (:meth:`render_console`) and to a
timestamped JSON file (:meth:`save_json`, default
``research/reports/research_TIMESTAMP.json``). All file I/O is
pathlib + UTF-8 (Windows-safe); console output avoids glyphs outside the
Windows cp1252 set.

Example:
    ```python
    from research.report import ResearchReport

    report = ResearchReport.build(
        project_id="research_001", niche="productivity tools",
        ranked=ranked, clipping=clipping_result,
        api_calls=12, phase_status={"reddit": "ok"},
    )
    ResearchReport.render_console(report)
    path = ResearchReport.save_json(report)
    ```
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from research.legal_analyzer import DISCLAIMER

__all__ = ["ResearchReport", "REPORTS_DIR", "COST_PER_API_CALL_USD"]

#: Directory holding timestamped JSON reports.
REPORTS_DIR: str = "research/reports"

#: Documented cost assumption per model API call (clearly labelled estimate).
COST_PER_API_CALL_USD: float = 0.002

_console = Console()


class ResearchReport:
    """Build, render, and persist the final actionable research report."""

    @staticmethod
    def build(
        project_id: str,
        niche: Optional[str],
        ranked: List[Dict[str, Any]],
        clipping: Optional[Dict[str, Any]] = None,
        recommendations: Optional[List[str]] = None,
        avoid_extra: Optional[List[str]] = None,
        revenue_notes: Optional[List[str]] = None,
        legal_notes: Optional[List[str]] = None,
        first_projects: Optional[List[Dict[str, Any]]] = None,
        api_calls: int = 0,
        phase_status: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Assemble the canonical report dict.

        Args:
            project_id: Research session identifier.
            niche: Focus niche, or None for a broad scan.
            ranked: Ranked opportunity dicts (best first).
            clipping: Optional YouTubeClippingResearch result dict.
            recommendations: Extra recommendation strings.
            avoid_extra: Extra avoid strings (with reasons).
            revenue_notes: Realistic-expectation strings.
            legal_notes: Precaution strings.
            first_projects: First-to-build picks (each with ``name`` and
                ``build_reason``); defaults to top low-risk ranked entries.
            api_calls: Total model API calls observed for the session.
            phase_status: Mapping of phase name → ``"ok"`` / error text.

        Returns:
            JSON-serialisable report dict. Never raises for well-formed
            inputs; empty rankings yield an explicit "no data" report.
        """
        ranked = list(ranked or [])
        top5 = ranked[:5]
        safe = [o for o in ranked if o.get("risk_level") == "low"][:5]
        avoid_ranked = [o for o in reversed(ranked) if o.get("risk_level") == "high"]
        avoid = [
            f"{o.get('name', '?')} — "
            f"{(o.get('risk_factors') or ['high risk'])[0]}"
            for o in avoid_ranked
        ]
        avoid.extend([str(a) for a in (avoid_extra or []) if str(a).strip()])

        if first_projects is None:
            first_projects = [
                {
                    "name": o.get("name", "?"),
                    "build_reason": o.get(
                        "build_reason",
                        f"Rank #{o.get('rank', '?')}, "
                        f"{o.get('risk_level', '?')} risk.",
                    ),
                }
                for o in ([x for x in ranked if x.get("risk_level") == "low"]
                          + [x for x in ranked if x.get("risk_level") != "high"])[:3]
            ]

        clipping = clipping or {}
        report = {
            "project_id": project_id,
            "niche": niche,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "executive_summary": ResearchReport._executive_summary(
                niche, ranked, safe, phase_status or {}
            ),
            "top_5": [
                {
                    "rank": o.get("rank", i + 1),
                    "name": o.get("name", "?"),
                    "category": o.get("category", "?"),
                    "score": o.get("score"),
                    "risk_level": o.get("risk_level", "?"),
                    "revenue_potential": o.get("revenue_potential", ""),
                    "automation_level": o.get("automation_level", "?"),
                    "sources": o.get("sources", [])[:5],
                }
                for i, o in enumerate(top5)
            ],
            "safe_bets": [
                {"name": o.get("name", "?"), "why": _first(o.get("risk_factors", []),
                      "Low-risk profile with original, sellable output.")}
                for o in safe
            ],
            "avoid": avoid,
            "clipping_safe": [
                {"strategy": s.get("strategy", "?"), "why": s.get("why", "")}
                for s in clipping.get("safe_strategies", [])
            ],
            "clipping_dangerous": [
                {"strategy": d.get("strategy", "?"),
                 "risk_level": d.get("risk_level", "high"),
                 "why": d.get("why", "")}
                for d in clipping.get("dangerous_strategies", [])
            ],
            "clipping_recommendations": list(clipping.get("recommendations", [])),
            "first_3_projects": first_projects[:3],
            "revenue_expectations": list(revenue_notes or []) or [
                "Micro-products (€5–€20) typically earn $0–$500/mo each in "
                "year one; portfolios of several products compound."
            ],
            "legal_precautions": list(legal_notes or []) or [
                "Use only original or properly licensed assets.",
                "Honour every platform's Terms of Service.",
            ],
            "recommendations": list(recommendations or []),
            "api_calls": api_calls,
            "estimated_cost_usd": round(api_calls * COST_PER_API_CALL_USD, 4),
            "cost_note": (
                f"Estimate assumes ${COST_PER_API_CALL_USD}/call; "
                "actual OpenRouter billing varies by model and tokens."
            ),
            "phase_status": dict(phase_status or {}),
            "legal_disclaimer": DISCLAIMER,
        }
        if not ranked:
            report["executive_summary"] = (
                "No opportunities could be ranked — every research phase "
                "failed. See phase_status, fix the failures, and re-run."
            )
        return report

    @staticmethod
    def _executive_summary(
        niche: Optional[str],
        ranked: List[Dict[str, Any]],
        safe: List[Dict[str, Any]],
        phase_status: Dict[str, str],
    ) -> str:
        """One-paragraph summary naming the single best bet."""
        failed = [k for k, v in phase_status.items() if v != "ok"]
        scope = f"the '{niche}' niche" if niche else "the scanned categories"
        if not ranked:
            return f"Research for {scope} produced no rankable opportunities."
        best = ranked[0]
        summary = (
            f"Researched {scope}: {len(ranked)} opportunities ranked. "
            f"Best bet: '{best.get('name', '?')}' (score {best.get('score')}, "
            f"{best.get('risk_level', '?')} risk). "
            f"{len(safe)} low-risk safe bet(s) identified."
        )
        if failed:
            summary += f" Phases needing attention: {', '.join(failed)}."
        return summary

    # -- rendering ------------------------------------------------------------

    @staticmethod
    def render_console(report: Dict[str, Any]) -> None:
        """Print the full actionable report with rich panels and tables."""
        _console.print(Panel(
            report.get("executive_summary", ""),
            title="Executive Summary",
        ))

        top_table = Table(title="Top 5 Safest Revenue Streams")
        top_table.add_column("#", justify="right")
        top_table.add_column("Opportunity")
        top_table.add_column("Risk")
        top_table.add_column("Revenue")
        top_table.add_column("Automation")
        for entry in report.get("top_5", []):
            top_table.add_row(
                str(entry.get("rank", "?")),
                str(entry.get("name", "?"))[:50],
                str(entry.get("risk_level", "?")),
                str(entry.get("revenue_potential", "-"))[:40],
                str(entry.get("automation_level", "?")),
            )
        _console.print(top_table)

        if report.get("safe_bets"):
            safe_table = Table(title="Safe Bets")
            safe_table.add_column("Product")
            safe_table.add_column("Why It Is Safe")
            for entry in report["safe_bets"]:
                safe_table.add_row(
                    str(entry.get("name", "?"))[:50],
                    str(entry.get("why", ""))[:100],
                )
            _console.print(safe_table)

        if report.get("avoid"):
            avoid_table = Table(title="Absolutely Avoid")
            avoid_table.add_column("What + Reason")
            for line in report["avoid"][:10]:
                avoid_table.add_row(str(line)[:140])
            _console.print(avoid_table)

        if report.get("clipping_safe") or report.get("clipping_dangerous"):
            clip_table = Table(title="YouTube Clipping: Safe vs Dangerous")
            clip_table.add_column("Verdict")
            clip_table.add_column("Strategy")
            for entry in report.get("clipping_safe", []):
                clip_table.add_row("SAFE", str(entry.get("strategy", "?"))[:60])
            for entry in report.get("clipping_dangerous", []):
                clip_table.add_row(
                    f"AVOID ({entry.get('risk_level', 'high')})",
                    str(entry.get("strategy", "?"))[:60],
                )
            _console.print(clip_table)
            for rec in report.get("clipping_recommendations", [])[:5]:
                _console.print(f"  - {rec}"[:160])

        first_table = Table(title="First 3 Projects to Build")
        first_table.add_column("#", justify="right")
        first_table.add_column("Project")
        first_table.add_column("Why First")
        for i, entry in enumerate(report.get("first_3_projects", [])[:3], 1):
            first_table.add_row(
                str(i),
                str(entry.get("name", "?"))[:50],
                str(entry.get("build_reason", ""))[:100],
            )
        _console.print(first_table)

        _console.print(Panel(
            "\n".join(f"- {line}" for line in report.get("revenue_expectations", [])),
            title="Realistic Revenue Expectations",
        ))
        _console.print(Panel(
            "\n".join(f"- {line}" for line in report.get("legal_precautions", [])),
            title="Legal Precautions",
        ))
        _console.print(Panel(
            f"Total API calls: {report.get('api_calls', 0)}\n"
            f"Estimated cost: ${report.get('estimated_cost_usd', 0)} "
            f"({report.get('cost_note', '')})\n"
            f"Phase status: {report.get('phase_status', {})}",
            title="Run Telemetry",
        ))

    # -- persistence ------------------------------------------------------------

    @staticmethod
    def save_json(
        report: Dict[str, Any], reports_dir: str = REPORTS_DIR
    ) -> Path:
        """Save the report to ``reports_dir/research_TIMESTAMP.json`` (UTF-8).

        Args:
            report: The built report dict.
            reports_dir: Destination directory (created if needed).

        Returns:
            Path of the written JSON file.

        Raises:
            IOError: If the directory or file cannot be written.
        """
        try:
            directory = Path(reports_dir)
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise IOError(
                f"Could not create reports directory '{reports_dir}': {exc}"
            ) from exc
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        target = directory / f"research_{stamp}.json"
        try:
            target.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise IOError(f"Could not write report file '{target}': {exc}") from exc
        _console.print(f"[green]Report saved to '{target}'.[/green]")
        return target


def _first(values: List[Any], default: str) -> str:
    """First non-empty string in ``values`` or ``default``."""
    for value in values:
        if str(value).strip():
            return str(value).strip()[:200]
    return default
