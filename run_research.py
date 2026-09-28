"""Research runner: discovers WHAT to build next, end to end.

This script runs the complete ResearchAgent pipeline — Reddit demand
signals, marketplace competition, YouTube clipping safety, legal screening,
deterministic ranking — and produces a final actionable report telling you
exactly what to build next with the BuilderAgent.

Example usage:
    ```bash
    python run_research.py --project-id research_001 --niche "productivity tools"
    python run_research.py --project-id research_002 --skip-scraping
    python run_research.py --project-id research_003 --categories digital_products,automation_tools
    ```

Full flow, research to action:
    ```bash
    python run_research.py --project-id r1 --niche "productivity tools"
    # read research/reports/research_<timestamp>.json, take a name from
    # "first_3_projects", then:
    python main.py --project-id build1 --niche "<chosen product niche>"
    # or: python main.py --mode research --project-id r2
    ```

Execution flow: initialise → Phase 1 digital products → Phase 2 content
creation (YouTube clipping) → Phase 3 services & automation → Phase 4 legal
screening → Phase 5 ranking & synthesis → console/JSON output → state save.

Resilience: a failing phase is logged and the pipeline continues with the
rest; scraping failures fall back to AI-only analysis; model calls are
retried once by the Orchestrator, then treated as partial data. The script
never crashes without producing output — the report always states which
phases succeeded and which failed. Exit code 0 unless nothing usable was
produced (or arguments are invalid).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel

from core.orchestrator import Orchestrator
from core.state_manager import StateManager
from research.legal_analyzer import LegalAnalyzer
from research.opportunity_ranker import OpportunityRanker
from research.report import ResearchReport
from research.research_agent import ResearchAgent
from research.youtube_clipping import YouTubeClippingResearch

__all__ = [
    "parse_args",
    "parse_categories",
    "run_research",
    "main",
    "CATEGORIES",
]

#: Research categories selectable via --categories (default: all).
CATEGORIES: tuple = (
    "digital_products",
    "content_creation",
    "services",
    "automation_tools",
)

#: Cap on per-opportunity legal screenings (bounds API cost).
MAX_LEGAL_SCREENS: int = 10

_console = Console()


def parse_categories(text: Optional[str]) -> List[str]:
    """Parse a comma-separated category list (None/empty → all).

    Raises:
        ValueError: On unknown category names.
    """
    if not text or not text.strip():
        return list(CATEGORIES)
    chosen = [c.strip().lower() for c in text.split(",") if c.strip()]
    unknown = [c for c in chosen if c not in CATEGORIES]
    if unknown:
        raise ValueError(
            f"Unknown categories: {unknown}. Choose from {list(CATEGORIES)}."
        )
    if not chosen:
        return list(CATEGORIES)
    return chosen


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Define and parse the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Research runner: discover safe micro-revenue streams.",
    )
    parser.add_argument(
        "--project-id", required=True, help="Unique ID for this research session."
    )
    parser.add_argument(
        "--niche",
        default=None,
        help='Optional focus niche, e.g. "productivity tools".',
    )
    parser.add_argument(
        "--categories",
        default=None,
        help="Comma-separated subset of: digital_products, content_creation, "
             "services, automation_tools (default: all).",
    )
    parser.add_argument(
        "--skip-scraping",
        action="store_true",
        help="AI-only analysis, no web scraping (useful for testing).",
    )
    parser.add_argument(
        "--output-format",
        choices=("json", "console", "both"),
        default="both",
        help="Where the final report goes (default: both).",
    )
    return parser.parse_args(argv)


def run_research(
    project_id: str,
    niche: Optional[str] = None,
    categories: Optional[List[str]] = None,
    skip_scraping: bool = False,
    output_format: str = "both",
    orchestrator: Optional[Orchestrator] = None,
) -> Dict[str, Any]:
    """Run the full research pipeline and return a summary dict.

    Args:
        project_id: Unique research session identifier.
        niche: Optional focus niche.
        categories: Subset of :data:`CATEGORIES` (default: all).
        skip_scraping: AI-only mode, no HTTP scraping.
        output_format: ``"json"``, ``"console"``, or ``"both"``.
        orchestrator: Optional pre-built Orchestrator (tests inject a stub).

    Returns:
        Summary with ``success``, ``project_id``, ``report`` (or None),
        ``report_path`` (or None), ``api_calls``, ``phase_status`` and
        ``errors``. Always returns — never raises for phase failures.
    """
    categories = list(categories or CATEGORIES)
    phase_status: Dict[str, str] = {}
    errors: List[str] = []
    _console.print(Panel(
        f"[bold]Project:[/bold] {project_id}\n"
        f"[bold]Niche:[/bold] {niche or '(broad scan)'}\n"
        f"[bold]Categories:[/bold] {', '.join(categories)}\n"
        f"[bold]Mode:[/bold] {'AI-only (no scraping)' if skip_scraping else 'full (scrape + AI)'}\n"
        f"[bold]Output:[/bold] {output_format}",
        title="Research Runner",
    ))

    # -- Step 1: initialise ---------------------------------------------------
    try:
        state_manager = StateManager(project_id=project_id)
        if orchestrator is None:
            orchestrator = Orchestrator()
        agent = ResearchAgent(orchestrator, state_manager)
        legal = LegalAnalyzer(orchestrator, state_manager)
        ranker = OpportunityRanker()
        clipping = YouTubeClippingResearch(
            orchestrator=orchestrator, state_manager=state_manager
        )
    except Exception as exc:
        _console.print(Panel(f"[red]Initialisation failed: {exc}[/red]"))
        return {
            "success": False, "project_id": project_id, "report": None,
            "report_path": None, "api_calls": -1,
            "phase_status": {"init": f"failed: {exc}"},
            "errors": [f"init: {exc}"],
        }
    if skip_scraping:
        _disable_scraping(agent, clipping)

    candidates: List[Dict[str, Any]] = []
    clipping_result: Dict[str, Any] = {}
    legal_assessments: List[Dict[str, Any]] = []

    # -- Phase 1: digital products --------------------------------------------
    if "digital_products" in categories:
        _console.rule("[bold]Phase 1 — Digital Products[/bold]")
        try:
            agent.research_reddit(
                ["SideProject", "Entrepreneur", "smallbusiness"],
                [niche or "side hustle", "digital product", "passive income"],
            )
            for platform, category in (("gumroad", niche or "digital template"),
                                       ("etsy", niche or "planner template")):
                try:
                    agent.research_marketplace(platform, category, (5, 20))
                except Exception as exc:
                    _console.print(
                        f"[yellow]{platform} scan issue ({exc}); continuing.[/yellow]"
                    )
            phase_status["digital_products"] = "ok"
        except Exception as exc:
            phase_status["digital_products"] = _degraded(
                "digital_products", exc, errors, agent, niche, candidates
            )

    # -- Phase 2: content creation + clipping ----------------------------------
    if "content_creation" in categories:
        _console.rule("[bold]Phase 2 — Content Creation & Clipping[/bold]")
        try:
            agent.analyze_youtube_opportunities(
                f"faceless {niche} channels" if niche else "faceless channels"
            )
            clipping_result = clipping.research(
                niche or "general", include_videos=not skip_scraping
            )
            phase_status["content_creation"] = "ok"
        except Exception as exc:
            phase_status["content_creation"] = _degraded(
                "content_creation", exc, errors, agent, niche, candidates,
                clipping_fallback=clipping,
            )

    # -- Phase 3: services & automation -----------------------------------------
    if "services" in categories or "automation_tools" in categories:
        _console.rule("[bold]Phase 3 — Services & Automation[/bold]")
        try:
            topics = []
            if "services" in categories:
                topics += [niche or "freelance services", "one-time service"]
            if "automation_tools" in categories:
                topics += ["automation", "python scripts", "no-code tools"]
            agent.research_reddit(["freelance", "automation", "Entrepreneur"], topics)
            phase_status["services_automation"] = "ok"
        except Exception as exc:
            phase_status["services_automation"] = _degraded(
                "services_automation", exc, errors, agent, niche, candidates
            )

    # -- Synthesise candidates ---------------------------------------------------
    _console.rule("[bold]Phase 4 — Legal Screening[/bold]")
    try:
        generated = agent.generate_opportunity_report()
        candidates.extend(generated.get("opportunities", []))
        phase_status["synthesis"] = "ok"
    except Exception as exc:
        message = f"AI synthesis failed ({exc}); using fallback candidates."
        _console.print(f"[yellow]{message}[/yellow]")
        errors.append(f"synthesis: {exc}")
        phase_status["synthesis"] = f"degraded: {exc}"
    if not candidates:
        _console.print("[yellow]No candidates yet; AI-brainstorming fallback.[/yellow]")
        candidates.extend(_ai_brainstorm(agent, niche, categories))

    # -- Phase 4 (cont.): legal screening per opportunity -------------------------
    screened = candidates[:MAX_LEGAL_SCREENS]
    if len(candidates) > MAX_LEGAL_SCREENS:
        _console.print(
            f"[yellow]Screening top {MAX_LEGAL_SCREENS} of {len(candidates)} "
            "candidates (cost cap).[/yellow]"
        )
    legal_ok = True
    for opp in screened:
        label = f"{opp.get('name', '?')}: {opp.get('description', '')}"
        try:
            assessment = legal.analyze(label)
            legal_assessments.append(assessment)
            if assessment.get("risk_level") == "high":
                opp["needs_human_review"] = True
        except Exception as exc:
            legal_ok = False
            errors.append(f"legal({opp.get('name', '?')}): {exc}")
            _console.print(
                f"[yellow]Legal screening failed for '{opp.get('name', '?')}': "
                f"{exc}[/yellow]"
            )
    phase_status["legal"] = "ok" if legal_ok else "degraded: some screenings failed"
    for assessment in legal_assessments:
        if assessment.get("risk_level") == "high":
            _console.print(
                f"[yellow]Flagged for human review (high risk): "
                f"{assessment.get('revenue_stream', '?')[:80]}[/yellow]"
            )

    # -- Phase 5: ranking & synthesis ----------------------------------------------
    _console.rule("[bold]Phase 5 — Ranking & Report[/bold]")
    try:
        ranked = ranker.rank(candidates, legal_assessments)
    except ValueError as exc:
        return _finish(
            state_manager, project_id, None, None, phase_status,
            errors + [f"ranking: {exc}"], success=False,
            note="No rankable candidates.",
        )
    first_3 = ranker.first_projects(ranked, n=3)
    revenue_notes = _revenue_notes(ranked)
    legal_notes = _legal_notes(legal_assessments, clipping_result)
    api_calls = _safe_api_calls(state_manager)
    report = ResearchReport.build(
        project_id=project_id,
        niche=niche,
        ranked=ranked,
        clipping=clipping_result or None,
        recommendations=[
            f"Pursue '{first_3[0]['name']}' first." if first_3 else "No recommendation."
        ],
        revenue_notes=revenue_notes,
        legal_notes=legal_notes,
        first_projects=[
            {"name": p.get("name", "?"), "build_reason": p.get("build_reason", "")}
            for p in first_3
        ],
        api_calls=api_calls if api_calls >= 0 else 0,
        phase_status=phase_status,
    )

    # -- Step 3: output -------------------------------------------------------------
    report_path: Optional[str] = None
    if output_format in ("json", "both"):
        try:
            report_path = str(ResearchReport.save_json(report))
        except IOError as exc:
            errors.append(f"report-save: {exc}")
            _console.print(f"[yellow]Could not save JSON report: {exc}[/yellow]")
    if output_format in ("console", "both"):
        ResearchReport.render_console(report)

    return _finish(
        state_manager, project_id, report, report_path, phase_status, errors,
        success=True,
    )


def _finish(
    state_manager: StateManager,
    project_id: str,
    report: Optional[Dict[str, Any]],
    report_path: Optional[str],
    phase_status: Dict[str, str],
    errors: List[str],
    success: bool,
    note: str = "",
) -> Dict[str, Any]:
    """Persist findings + phase, then return the run summary (never raises)."""
    api_calls = _safe_api_calls(state_manager)
    try:
        state = state_manager.get_state()
        findings = state.research_findings
        if not isinstance(findings, dict):
            findings = {}
        findings["final_report"] = report
        findings["run"] = {
            "project_id": project_id,
            "report_path": report_path,
            "phase_status": dict(phase_status),
            "note": note,
        }
        state.research_findings = findings
        state_manager.save(state)
        state_manager.update_phase("research_complete" if success else "research")
    except Exception as exc:
        errors.append(f"state-save: {exc}")
        _console.print(f"[yellow]Could not persist final state: {exc}[/yellow]")
    _console.print(Panel(
        f"Phases: {phase_status}\n"
        f"API calls: {api_calls}\n"
        f"Report: {report_path or '(console only)'}",
        title="Research Run Complete" if success else "Research Run Partial",
    ))
    return {
        "success": success,
        "project_id": project_id,
        "report": report,
        "report_path": report_path,
        "api_calls": api_calls,
        "phase_status": dict(phase_status),
        "errors": errors,
    }


def _degraded(
    phase: str,
    exc: Exception,
    errors: List[str],
    agent: ResearchAgent,
    niche: Optional[str],
    candidates: List[Dict[str, Any]],
    clipping_fallback: Optional[YouTubeClippingResearch] = None,
) -> str:
    """Log a phase failure, AI-backfill candidates, return status string."""
    _console.print(f"[yellow]Phase '{phase}' issue ({exc}); AI fallback engaged.[/yellow]")
    errors.append(f"{phase}: {exc}")
    try:
        if clipping_fallback is not None and phase == "content_creation":
            fallback = clipping_fallback.research(
                niche or "general", include_videos=False
            )
            _console.print(
                f"[yellow]Clipping legal screening continued "
                f"({len(fallback.get('assessments', []))} assessments).[/yellow]"
            )
        candidates.extend(_ai_brainstorm(agent, niche, [phase]))
    except Exception as fallback_exc:
        errors.append(f"{phase}-fallback: {fallback_exc}")
    return f"degraded: {exc}"


def _ai_brainstorm(
    agent: ResearchAgent, niche: Optional[str], categories: List[str]
) -> List[Dict[str, Any]]:
    """AI-only candidate generation (no scraping) for a category hint."""
    scope = niche or "micro-revenue streams"
    try:
        raw = agent.orchestrator.ask_agent(
            system_prompt=(
                "You are a market researcher for digital micro-products "
                "($5-$20). Suggest 3 original, low-risk, automatable "
                "opportunities. Respond with VALID JSON ONLY: "
                '{"opportunities": [{"name": "...", "category": "...", '
                '"description": "...", "price_range": [5, 20], '
                '"feasibility_score": 7, "risk_level": "low", '
                '"risk_factors": [], "revenue_potential": "...", '
                '"technical_requirements": [], "legal_considerations": [], '
                '"automation_level": "partial", "sources": []}]}'
            ),
            user_prompt=f"Suggest opportunities for {scope} "
                        f"(categories: {', '.join(categories)}).",
            state_manager=agent.state_manager,
        )
        payload = json.loads(_strip_fences(raw))
        items = payload.get("opportunities", []) if isinstance(payload, dict) else []
    except Exception as exc:
        _console.print(f"[yellow]AI brainstorm failed: {exc}[/yellow]")
        return []
    return [o for o in items if isinstance(o, dict) and str(o.get("name", "")).strip()]


def _strip_fences(text: str) -> str:
    """Remove a surrounding markdown code fence, if present."""
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped or "{}"


def _disable_scraping(
    agent: ResearchAgent, clipping: YouTubeClippingResearch
) -> None:
    """Neutralise network fetching for --skip-scraping (AI-only mode)."""
    for scraper in (agent.reddit, agent.marketplaces, agent.youtube,
                    clipping.youtube):
        original = scraper.fetch

        def _blocked(url: str, params: Optional[dict] = None,
                     _o: Any = original) -> None:
            scraper._journal(url, None, "check-skipped",
                             "Skipped (--skip-scraping AI-only mode).")
            return None

        scraper.fetch = _blocked  # type: ignore[method-assign]


def _revenue_notes(ranked: List[Dict[str, Any]]) -> List[str]:
    """Build realistic-expectation lines from top ranked entries."""
    notes = []
    for opp in ranked[:3]:
        potential = str(opp.get("revenue_potential", "")).strip()
        if potential:
            notes.append(f"{opp.get('name', '?')}: {potential}")
    notes.append(
        "Expect $0–$500/mo per micro-product in year one; treat figures as "
        "rough niche estimates, not guarantees."
    )
    return notes


def _legal_notes(
    assessments: List[Dict[str, Any]], clipping: Dict[str, Any]
) -> List[str]:
    """Build precaution lines from screenings and clipping guidance."""
    notes = [
        "Use only original or properly licensed assets (fonts, images, music).",
        "Honour every platform's Terms of Service; read the copyright policy first.",
    ]
    for assessment in assessments:
        if assessment.get("risk_level") == "high":
            notes.append(
                f"Human review required: {assessment.get('revenue_stream', '?')[:80]}"
            )
    for rec in (clipping.get("recommendations", []) or [])[:3]:
        notes.append(f"Clipping: {rec}")
    return notes


def _safe_api_calls(state_manager: StateManager) -> int:
    """Read api_calls_count without ever raising."""
    try:
        return state_manager.get_state().api_calls_count
    except Exception:
        return -1


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: parse args, run research, return exit code."""
    args = parse_args(argv)
    try:
        categories = parse_categories(args.categories)
    except ValueError as exc:
        _console.print(Panel(f"[red]{exc}[/red]", title="Bad arguments"))
        return 1
    if args.output_format not in ("json", "console", "both"):
        _console.print(Panel("[red]Bad --output-format.[/red]", title="Bad arguments"))
        return 1
    summary = run_research(
        project_id=args.project_id,
        niche=args.niche,
        categories=categories,
        skip_scraping=args.skip_scraping,
        output_format=args.output_format,
    )
    if summary["report"] is None:
        return 1
    # A produced report is a successful run even with degraded phases;
    # phase_status inside the report (and state) says what needs attention.
    return 0


if __name__ == "__main__":
    sys.exit(main())
