"""Product Factory: research new revenue streams or build products.

Two modes (see ``--mode``):

* ``build`` (default) turns a market niche into a tested, sellable ZIP:

    python main.py --niche "Notion templates for students" --project-id demo

    python main.py --niche "Python automation scripts" --project-id scripts1 \\
        --price-range "5-15" --max-build-attempts 5 --select-index 0

* ``research`` discovers WHAT to build next via the research pipeline
  (same as ``run_research.py``):

    python main.py --mode research --project-id research_001 --niche "productivity tools"

    python main.py --mode research --project-id research_002 --skip-scraping

Research-first flow: run ``--mode research``, take a name from the report's
``first_3_projects``, then run the default build mode on that niche.

Pipeline stages (each logs to the :class:`StateManager` single source of
truth, every model call goes through the :class:`Orchestrator`):

1. **Research** — :class:`ScoutAgent` finds 3 opportunities in the niche.
2. **Build + test** — :class:`BuilderAgent.build_with_testing` builds the
   chosen opportunity, validates it with :class:`TestRunner`, and rebuilds
   with fix feedback until tests pass (or attempts run out).
3. **Package** — :class:`ProductPackager` zips the tested product into
   ``packages/{product}.zip`` (already done inside the builder on success;
   this stage verifies the artefact and only re-packages if it is missing).
4. **Report** — prints a summary (package path, test results, total API
   calls) and marks the project ``"completed"`` (or ``"failed"``).

Windows-compatible throughout (:mod:`pathlib`, UTF-8). Exit code is 0 on
success, 1 when any stage fails (the failure is recorded on the state).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agents.builder import BuilderAgent
from agents.scout import ScoutAgent
from core.orchestrator import Orchestrator
from core.product_packager import ProductPackager
from core.state_manager import StateManager
from core.test_runner import TestRunner

__all__ = ["parse_args", "parse_price_range", "run_pipeline", "main"]

_console = Console()


def parse_price_range(text: str) -> Tuple[float, float]:
    """Parse a ``"MIN-MAX"`` price range string into ``(low, high)`` floats.

    Args:
        text: E.g. ``"5-20"``. Whitespace and ``€`` symbols are tolerated.

    Raises:
        ValueError: If the format is invalid or low > high.
    """
    cleaned = (text or "").replace("€", "").strip()
    parts = cleaned.split("-")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid --price-range {text!r}; expected format 'MIN-MAX' (e.g. '5-20')."
        )
    try:
        low, high = float(parts[0].strip()), float(parts[1].strip())
    except ValueError as exc:
        raise ValueError(
            f"Invalid --price-range {text!r}; bounds must be numbers."
        ) from exc
    if low > high:
        raise ValueError(
            f"Invalid --price-range {text!r}; MIN must not exceed MAX."
        )
    return (low, high)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Define and parse the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Autonomous product factory: research revenue streams or build, test, package.",
    )
    parser.add_argument(
        "--mode",
        choices=("build", "research"),
        default="build",
        help="Pipeline to run: 'build' a product or 'research' what to build (default: build).",
    )
    parser.add_argument(
        "--niche",
        default=None,
        help="Market niche (required for build mode; optional focus for research mode).",
    )
    parser.add_argument(
        "--project-id", required=True, help="Unique identifier for this project."
    )
    parser.add_argument(
        "--price-range",
        default="5-20",
        help="Target price range as 'MIN-MAX' (default: '5-20').",
    )
    parser.add_argument(
        "--max-build-attempts",
        type=int,
        default=3,
        help="Max build+test iterations (default: 3).",
    )
    parser.add_argument(
        "--select-index",
        type=int,
        default=0,
        help="Which researched opportunity to build (0-based, default: 0).",
    )
    parser.add_argument(
        "--categories",
        default=None,
        help="Research mode only: comma-separated subset of digital_products, "
             "content_creation, services, automation_tools (default: all).",
    )
    parser.add_argument(
        "--skip-scraping",
        action="store_true",
        help="Research mode only: AI-only analysis without web scraping.",
    )
    parser.add_argument(
        "--output-format",
        choices=("json", "console", "both"),
        default="both",
        help="Research mode only: where the report goes (default: both).",
    )
    return parser.parse_args(argv)


def _fail(
    state_manager: StateManager, stage: str, message: str
) -> Dict[str, Any]:
    """Record a stage failure on the state and return a failure summary."""
    _console.print(Panel(f"[red]{stage} failed:[/red] {message}", title="Error"))
    try:
        state_manager.update_phase("failed")
    except Exception as exc:  # State write must not hide the real error.
        _console.print(f"[yellow]Could not record failure phase: {exc}[/yellow]")
    return {
        "success": False,
        "stage": stage,
        "package_path": None,
        "test_results": {},
        "api_calls": _safe_api_calls(state_manager),
        "errors": [f"{stage}: {message}"],
    }


def _safe_api_calls(state_manager: StateManager) -> int:
    """Read api_calls_count without ever raising."""
    try:
        return state_manager.get_state().api_calls_count
    except Exception:
        return -1


def run_pipeline(
    niche: str,
    project_id: str,
    price_range: Tuple[float, float] = (5.0, 20.0),
    max_build_attempts: int = 3,
    select_index: int = 0,
    orchestrator: Optional[Orchestrator] = None,
) -> Dict[str, Any]:
    """Run the full research → build → test → package pipeline.

    Args:
        niche: Market niche to research.
        project_id: Unique project identifier (state file stem).
        price_range: ``(min_price, max_price)`` for Scout validation.
        max_build_attempts: Build+test iterations for the Builder.
        select_index: Which opportunity to build (0-based).
        orchestrator: Optional pre-built Orchestrator (used by tests to
            inject a stub; a real one is created from the environment when
            omitted).

    Returns:
        Summary dict with ``success``, ``project_id``, ``niche``,
        ``opportunity``, ``package_path``, ``test_results``, ``api_calls``
        and ``errors``.
    """
    low, high = price_range
    _console.print(
        Panel(
            f"[bold]Niche:[/bold] {niche}\n"
            f"[bold]Project:[/bold] {project_id}\n"
            f"[bold]Price range:[/bold] €{low:g}–€{high:g}  "
            f"[bold]Max attempts:[/bold] {max_build_attempts}",
            title="Product Factory",
        )
    )

    # -- Stage 0: initialise ------------------------------------------------
    try:
        state_manager = StateManager(project_id=project_id)
        if orchestrator is None:
            orchestrator = Orchestrator()
        scout = ScoutAgent(orchestrator, state_manager)
        builder = BuilderAgent(orchestrator, state_manager)
        tester = TestRunner(state_manager)
        packager = ProductPackager(state_manager)
    except Exception as exc:
        _console.print(Panel(f"[red]Initialisation failed: {exc}[/red]"))
        return {
            "success": False,
            "stage": "init",
            "package_path": None,
            "test_results": {},
            "api_calls": -1,
            "errors": [f"init: {exc}"],
        }

    # -- Stage 1: research --------------------------------------------------
    _console.rule("[bold]Stage 1/3 — Research[/bold]")
    try:
        findings = scout.research_opportunities(
            niche, target_price_range=(low, high)
        )
    except (ValueError, RuntimeError) as exc:
        return _fail(state_manager, "research", str(exc))
    opportunities = findings["opportunities"]
    _show_opportunities(opportunities)
    if not 0 <= select_index < len(opportunities):
        return _fail(
            state_manager,
            "research",
            f"--select-index {select_index} out of range "
            f"(0–{len(opportunities) - 1}).",
        )
    chosen = opportunities[select_index]
    _console.print(
        f"[green]Selected opportunity {select_index}:[/green] "
        f"[bold]{chosen['product_name']}[/bold]"
    )

    # -- Stage 2: build + test ----------------------------------------------
    _console.rule("[bold]Stage 2/3 — Build & Test[/bold]")
    try:
        build_result = builder.build_with_testing(
            opportunity=chosen, max_attempts=max_build_attempts
        )
    except ValueError as exc:
        return _fail(state_manager, "build", str(exc))
    except RuntimeError as exc:
        return _fail(state_manager, "build", str(exc))
    _show_test_results(build_result["test_results"])
    if not build_result["success"]:
        return _fail(
            state_manager,
            "build",
            f"all {build_result['attempts']} attempt(s) failed: "
            + "; ".join(build_result["errors"][:3]),
        )

    # -- Stage 3: package ----------------------------------------------------
    _console.rule("[bold]Stage 3/3 — Package[/bold]")
    package_path = build_result["package_path"]
    try:
        if package_path and Path(package_path).exists():
            _console.print(
                f"[green]Package already built by builder:[/green] "
                f"'{package_path}'"
            )
        else:
            # Fallback: package the tested product explicitly.
            product_type = (
                "python"
                if _compiles(build_result["final_product_code"])
                else "template"
            )
            with _console.status("[cyan]Packaging tested product…[/cyan]"):
                zip_path = packager.create_package(
                    chosen["product_name"],
                    build_result["final_product_code"],
                    product_type=product_type,
                )
            package_path = str(zip_path)
    except (ValueError, IOError, RuntimeError) as exc:
        return _fail(state_manager, "package", str(exc))
    _ = tester  # Instantiated for API symmetry; testing ran inside the builder.

    # -- Done -----------------------------------------------------------------
    api_calls = _safe_api_calls(state_manager)
    try:
        state_manager.update_phase("completed")
    except Exception as exc:
        _console.print(f"[yellow]Could not mark project completed: {exc}[/yellow]")
    summary = {
        "success": True,
        "project_id": project_id,
        "niche": niche,
        "opportunity": chosen["product_name"],
        "package_path": package_path,
        "test_results": build_result["test_results"],
        "api_calls": api_calls,
        "errors": [],
    }
    _show_summary(summary)
    return summary


def _show_opportunities(opportunities: List[Dict[str, Any]]) -> None:
    """Render researched opportunities as a rich table."""
    table = Table(title="Market Opportunities")
    table.add_column("#", justify="right")
    table.add_column("Product")
    table.add_column("Audience")
    table.add_column("Price", justify="right")
    for i, opp in enumerate(opportunities):
        table.add_row(
            str(i),
            str(opp.get("product_name", "?"))[:60],
            str(opp.get("target_audience", "?"))[:40],
            f"€{float(opp.get('estimated_price', 0)):g}",
        )
    _console.print(table)


def _show_test_results(report: Dict[str, Any]) -> None:
    """Render the TestRunner report compactly."""
    if report.get("success"):
        _console.print("[green]All product tests passed.[/green]")
        return
    table = Table(title="Test Failures")
    table.add_column("Error")
    for error in report.get("errors", [])[:10]:
        table.add_row(str(error)[:160])
    _console.print(table)


def _show_summary(summary: Dict[str, Any]) -> None:
    """Print the final creation summary panel."""
    _console.print(
        Panel(
            f"[bold]Package:[/bold] {summary['package_path']}\n"
            f"[bold]Product:[/bold] {summary['opportunity']}\n"
            f"[bold]Tests:[/bold] "
            f"{'passed' if summary['test_results'].get('success') else 'see above'}\n"
            f"[bold]Total API calls:[/bold] {summary['api_calls']}",
            title="Product ready for sale",
        )
    )


def _compiles(code: Optional[str]) -> bool:
    """Return True if ``code`` compiles as Python."""
    if not isinstance(code, str) or not code.strip():
        return False
    try:
        compile(code, "<product>", "exec")
        return True
    except (SyntaxError, ValueError):
        return False


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: parse args, run the selected mode, return exit code."""
    args = parse_args(argv)
    if args.mode == "research":
        # Lazy import: research stack is only needed in research mode.
        from run_research import main as research_main

        research_argv: List[str] = ["--project-id", args.project_id]
        if args.niche:
            research_argv += ["--niche", args.niche]
        if args.categories:
            research_argv += ["--categories", args.categories]
        if args.skip_scraping:
            research_argv.append("--skip-scraping")
        research_argv += ["--output-format", args.output_format]
        return research_main(research_argv)
    if not args.niche or not args.niche.strip():
        _console.print(
            Panel("[red]--niche is required in build mode.[/red]",
                  title="Bad arguments")
        )
        return 1
    try:
        price_range = parse_price_range(args.price_range)
    except ValueError as exc:
        _console.print(Panel(f"[red]{exc}[/red]", title="Bad arguments"))
        return 1
    if args.max_build_attempts < 1:
        _console.print(
            Panel("[red]--max-build-attempts must be >= 1[/red]",
                  title="Bad arguments")
        )
        return 1
    summary = run_pipeline(
        niche=args.niche,
        project_id=args.project_id,
        price_range=price_range,
        max_build_attempts=args.max_build_attempts,
        select_index=args.select_index,
    )
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
