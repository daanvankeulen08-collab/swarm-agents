"""SubNicheFinder: generates hyper-specific sub-niches and ranks them by competition.

The finder asks Space Bunny Alpha (via the :class:`Orchestrator`) for 5
highly specific sub-niches inside a main category (e.g. ``'weekly planner'``
→ ``'ADHD student weekly planner'``), measures real competition per
sub-niche, estimates the addressable market per sub-niche, drops anything
too small to be viable, ranks the survivors with a weighted score (low
competition 40% + large market 40% + specificity 20%), persists everything
to ``state.research_findings["subniches"]``, and prints the top 3
as a rich table.

Competition measurement has two paths:

* **Tavily (preferred):** when ``TAVILY_API_KEY`` is set and the
  ``tavily-python`` package is installed, each sub-niche is searched as
  ``'<name> notion template'`` and the returned result count is used as the
  estimated competitor count (real data).
* **Orchestrator fallback:** otherwise the model estimates competition from
  its training data (clearly labelled as an estimate).

Example:
    ```python
    from core.orchestrator import Orchestrator
    from core.state_manager import StateManager
    from research.subniche_finder import SubNicheFinder

    finder = SubNicheFinder(Orchestrator(), StateManager(project_id="demo"))
    ranked = finder.find_subniches(main_category="weekly planner")
    print(ranked[0]["name"])
    ```
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from core.orchestrator import Orchestrator
from core.state_manager import StateManager

try:
    from tavily import TavilyClient

    _TAVILY_INSTALLED = True
except ImportError:  # tavily-python optional; fallback covers its absence.
    TavilyClient = None  # type: ignore[assignment]
    _TAVILY_INSTALLED = False

__all__ = [
    "SubNicheFinder",
    "COMPETITION_LEVELS",
    "TAVILY_AVAILABLE",
    "MIN_VIABLE_MARKET",
]

#: Ordered competition levels, lowest to highest.
COMPETITION_LEVELS: tuple = ("low", "medium", "high")

#: Markets below this many people score viability < 5 and are filtered out.
MIN_VIABLE_MARKET: int = 5000

#: True when the tavily-python package is importable.
TAVILY_AVAILABLE: bool = _TAVILY_INSTALLED

_console = Console()


def _utcnow_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


class SubNicheFinder:
    """Generate, competition-check, and rank sub-niches for a category.

    Args:
        orchestrator: The swarm's :class:`Orchestrator` for sub-niche
            generation (and competition estimates when Tavily is absent).
        state_manager: The project's :class:`StateManager`; results are
            saved under ``state.research_findings["subniches"]``.

    Example:
        ```python
        finder = SubNicheFinder(orchestrator, state_manager)
        top = finder.find_subniches("weekly planner")
        ```
    """

    def __init__(
        self, orchestrator: Orchestrator, state_manager: StateManager
    ) -> None:
        # Duck-typed checks so test doubles work (swarm-wide convention).
        for name in ("ask_agent", "build_context_prompt", "get_model_info"):
            if not hasattr(orchestrator, name) or not callable(
                getattr(orchestrator, name, None)
            ):
                raise TypeError(
                    "orchestrator must expose the Orchestrator interface "
                    f"(missing callable {name!r}); "
                    f"got {type(orchestrator).__name__}."
                )
        for name in ("update_phase", "get_state", "save", "log_api_call"):
            if not hasattr(state_manager, name) or not callable(
                getattr(state_manager, name, None)
            ):
                raise TypeError(
                    "state_manager must expose the StateManager interface "
                    f"(missing callable {name!r}); "
                    f"got {type(state_manager).__name__}."
                )
        self.orchestrator: Orchestrator = orchestrator
        self.state_manager: StateManager = state_manager

    def find_subniches(
        self, main_category: str = "weekly planner", max_competitors: int = 100
    ) -> List[Dict[str, Any]]:
        """Generate sub-niches, score competition + market, rank, save, display.

        Flow: generate 5 candidates → measure competition (Tavily or model
        estimate) → estimate addressable market per sub-niche → drop
        anything with ``viability_score < 5`` (market below
        ``MIN_VIABLE_MARKET``) → re-rank survivors by weighted composite
        (low competition 40% + large market 40% + specificity 20%).

        Args:
            main_category: Broad category to specialise, e.g.
                ``"weekly planner"``.
            max_competitors: Upper bound on results considered per
                sub-niche search (Tavily returns at most 10 per query;
                larger values simply keep the full result set).

        Returns:
            Ranked list of viable sub-niche dicts (weighted best first),
            each with ``name``, ``target_audience``, ``specificity``
            (1–10), ``description``, ``estimated_competitors`` (int),
            ``competition_level`` (low/medium/high), ``competition_source``
            (``"tavily"`` or ``"model-estimate"``), ``market_size`` (int),
            ``market_reasoning`` (str), ``viability_score`` (1–10),
            ``composite_score`` (float), and ``rank``.

        Raises:
            ValueError: On invalid inputs, an unparseable model response,
                or when no sub-niche passes the viability filter.
            RuntimeError: If a model call or persistence fails.
        """
        cleaned = main_category.strip() if isinstance(main_category, str) else ""
        if not cleaned:
            raise ValueError("main_category must be a non-empty string.")
        if (
            not isinstance(max_competitors, int)
            or isinstance(max_competitors, bool)
            or max_competitors < 1
        ):
            raise ValueError(
                f"max_competitors must be an integer >= 1, got {max_competitors!r}."
            )
        _console.print(
            f"[cyan]Finding sub-niches inside[/cyan] [bold]{cleaned}[/bold]…"
        )

        subniches = self._generate_subniches(cleaned)
        _console.print(
            f"[green]Generated {len(subniches)} candidate sub-niches.[/green]"
        )

        use_tavily = self._tavily_key_present() and TAVILY_AVAILABLE
        if use_tavily:
            _console.print("[cyan]Measuring real competition via Tavily…[/cyan]")
            self._measure_with_tavily(subniches, max_competitors)
        else:
            reason = (
                "tavily-python not installed"
                if not TAVILY_AVAILABLE
                else "TAVILY_API_KEY not set"
            )
            _console.print(
                f"[yellow]Tavily unavailable ({reason}); estimating "
                "competition via model knowledge.[/yellow]"
            )
            self._measure_with_model(subniches)

        _console.print("[cyan]Estimating addressable market per sub-niche…[/cyan]")
        for sub in subniches:
            sizing = self.estimate_market_size(sub)
            sub["market_size"] = sizing["estimated_market_size"]
            sub["market_reasoning"] = sizing["reasoning"]
            sub["viability_score"] = sizing["viability_score"]

        viable = [s for s in subniches if s["viability_score"] >= 5]
        dropped = len(subniches) - len(viable)
        if dropped:
            _console.print(
                f"[yellow]Filtered out {dropped} sub-niche(s) below viability "
                f"threshold (market < {MIN_VIABLE_MARKET:,} people).[/yellow]"
            )
        if not viable:
            raise ValueError(
                f"No viable sub-niches in '{cleaned}': all 5 candidates scored "
                f"viability < 5 (addressable market under {MIN_VIABLE_MARKET:,} "
                "people). Try a broader main_category."
            )

        ranked = self._rank(viable)
        self._save_results(cleaned, ranked, use_tavily)
        self._print_top3(ranked)
        return ranked

    # -- generation -----------------------------------------------------------

    def _generate_subniches(self, main_category: str) -> List[Dict[str, Any]]:
        """Ask the model for 5 specific sub-niches (raises on failure)."""
        try:
            raw = self.orchestrator.ask_agent(
                system_prompt=(
                    "You are a micro-niche strategist for digital products. "
                    "You specialise in hyper-specific, underserved audiences."
                ),
                user_prompt=(
                    f"Generate 5 highly specific, underserved sub-niches inside "
                    f"the category '{main_category}' (e.g. 'ADHD student weekly "
                    "planner', 'freelance designer project planner'). Each must "
                    "name a concrete audience + use case. Respond with VALID "
                    "JSON ONLY, no commentary: "
                    '{"subniches": [{"name": "...", '
                    '"target_audience": "...", '
                    '"specificity": 8, '
                    '"description": "..."}]} '
                    "where specificity is 1 (broad) to 10 (hyper-specific)."
                ),
                state_manager=self.state_manager,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Sub-niche generation failed for {main_category!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        try:
            payload = _parse_json(raw)
        except ValueError as exc:
            raise ValueError(
                f"Could not parse sub-niche list as JSON: {exc}"
            ) from exc
        items = payload.get("subniches", []) if isinstance(payload, dict) else []
        if len(items) != 5:
            raise ValueError(
                f"Expected 5 sub-niches from the model, got {len(items)}."
            )
        normalised = []
        for i, item in enumerate(items):
            if not isinstance(item, dict) or not str(item.get("name", "")).strip():
                raise ValueError(f"Sub-niche #{i + 1} is missing a name.")
            normalised.append({
                "name": str(item["name"]).strip()[:120],
                "target_audience": str(item.get("target_audience", "")).strip()[:160],
                "specificity": _clamp_int(item.get("specificity", 5), 1, 10),
                "description": str(item.get("description", "")).strip()[:500],
            })
        return normalised

    # -- competition measurement ------------------------------------------------

    @staticmethod
    def _tavily_key_present() -> bool:
        """True when TAVILY_API_KEY is set (loads .env first)."""
        load_dotenv()
        return bool(os.getenv("TAVILY_API_KEY", "").strip())

    def _measure_with_tavily(
        self, subniches: List[Dict[str, Any]], max_competitors: int
    ) -> None:
        """Fill estimated_competitors from live Tavily result counts."""
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", "").strip())
        per_query = max(1, min(int(max_competitors), 10))  # Tavily caps at 10.
        for sub in subniches:
            query = f"{sub['name']} notion template"
            try:
                response = client.search(query, max_results=per_query)
                results = response.get("results", []) if isinstance(response, dict) else []
                count = len(results)
                source = "tavily"
            except Exception as exc:
                _console.print(
                    f"[yellow]Tavily search failed for '{sub['name']}': "
                    f"{exc}; falling back to model estimate.[/yellow]"
                )
                count, source = self._estimate_one(sub["name"]), "model-estimate"
            sub["estimated_competitors"] = int(count)
            sub["competition_level"] = _level_from_count(int(count), per_query)
            sub["competition_source"] = source

    def _measure_with_model(self, subniches: List[Dict[str, Any]]) -> None:
        """Fill estimated_competitors via a single model estimation call."""
        digest = "\n".join(
            f"- {s['name']} (audience: {s['target_audience'] or 'unspecified'})"
            for s in subniches
        )
        try:
            raw = self.orchestrator.ask_agent(
                system_prompt=(
                    "You are a market analyst estimating competition for "
                    "digital-product niches from your training data."
                ),
                user_prompt=(
                    "For EACH sub-niche below, estimate how many competing "
                    "products exist and rate competition as low, medium, or "
                    "high. Respond with VALID JSON ONLY: "
                    '{"estimates": [{"name": "...", '
                    '"estimated_competitors": 25, '
                    '"competition_level": "medium"}]}'
                    f"\n\nSub-niches:\n{digest}"
                ),
                state_manager=self.state_manager,
            )
            payload = _parse_json(raw)
        except Exception as exc:
            raise RuntimeError(
                f"Competition estimation failed: {type(exc).__name__}: {exc}"
            ) from exc
        by_name = {
            str(e.get("name", "")).strip().lower(): e
            for e in (payload.get("estimates", []) if isinstance(payload, dict) else [])
            if isinstance(e, dict)
        }
        for sub in subniches:
            hit = by_name.get(sub["name"].strip().lower(), {})
            try:
                count = max(0, int(hit.get("estimated_competitors", 50)))
            except (TypeError, ValueError):
                count = 50
            level = str(hit.get("competition_level", "medium")).strip().lower()
            if level not in COMPETITION_LEVELS:
                level = "medium"
            sub["estimated_competitors"] = count
            sub["competition_level"] = level
            sub["competition_source"] = "model-estimate"

    def _estimate_one(self, name: str) -> int:
        """Last-resort single-niche estimate used when Tavily errors mid-run."""
        return 50

    # -- market sizing ------------------------------------------------------------

    def estimate_market_size(self, subniche: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate the US/EU addressable market for one sub-niche.

        Asks Space Bunny Alpha for a conservative headcount plus reasoning,
        then derives ``viability_score`` (1–10) from size tiers anchored at
        ``MIN_VIABLE_MARKET`` (5000 people → score 5, the keep/drop cutoff).

        Args:
            subniche: Candidate dict with at least a ``name`` key.

        Returns:
            Dict with ``estimated_market_size`` (int headcount),
            ``reasoning`` (str), and ``viability_score`` (int 1–10).

        Raises:
            ValueError: If the sub-niche has no name or the model response
                is unparseable.
            RuntimeError: If the model call fails.
        """
        name = subniche.get("name", "") if isinstance(subniche, dict) else ""
        name = str(name).strip()
        if not name:
            raise ValueError("subniche must be a dict with a non-empty 'name'.")
        audience = str(subniche.get("target_audience", "")).strip() or "unspecified"
        _console.print(f"[cyan]Sizing market for[/cyan] [bold]{name}[/bold]…")
        base_prompt = (
            f"Estimate the total addressable market size for "
            f"'{name}' (target audience: {audience}). How many people "
            "in the US/EU fit this description? Give a conservative "
            "estimate and explain your reasoning. Respond with VALID "
            "JSON ONLY, no commentary: "
            '{"estimated_market_size": 12000, '
            '"reasoning": "..."}'
        )
        last_error: Optional[str] = None
        for attempt in (1, 2):  # One retry with a stricter format nudge.
            try:
                raw = self.orchestrator.ask_agent(
                    system_prompt=(
                        "You are a market-sizing analyst. Give conservative, "
                        "defensible headcount estimates — never hype numbers."
                    ),
                    user_prompt=(
                        base_prompt
                        if attempt == 1
                        else base_prompt
                        + " Reply with ONLY the JSON object and a NUMERIC "
                        "estimated_market_size (no null, no ranges)."
                    ),
                    state_manager=self.state_manager,
                )
                payload = _parse_json(raw)
                size = _extract_market_size(payload, name)
                reasoning = _extract_reasoning(payload)
                score = _viability_from_size(size)
                _console.print(
                    f"  Market ~[bold]{size:,}[/bold] people -> viability "
                    f"[bold]{score}/10[/bold]"
                )
                return {
                    "estimated_market_size": size,
                    "reasoning": reasoning,
                    "viability_score": score,
                }
            except (ValueError, RuntimeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                _console.print(
                    f"[yellow]Market sizing attempt {attempt}/2 failed for "
                    f"'{name}': {last_error}[/yellow]"
                )
        # Both attempts failed: stay honest and conservative — mark unviable
        # so the filter drops it, and record why in the reasoning.
        _console.print(
            f"[yellow]Could not size '{name}'; marking unviable.[/yellow]"
        )
        return {
            "estimated_market_size": 0,
            "reasoning": (
                "Model did not return a numeric estimate after 2 attempts "
                f"({last_error}); treated as unviable rather than guessing."
            ),
            "viability_score": 1,
        }

    # -- ranking / persistence / display -----------------------------------------

    @staticmethod
    def _rank(subniches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank by weighted composite and assign ranks (best first).

        Composite (0–10): low competition 40% + large market 40% +
        specificity 20%. Competition maps to points (low 10 / medium 5 /
        high 0); market reuses ``viability_score`` (already 1–10 and
        size-based); specificity is 1–10 as generated.
        """
        points = {"low": 10.0, "medium": 5.0, "high": 0.0}
        for sub in subniches:
            competition_part = points.get(sub.get("competition_level", "medium"), 5.0)
            market_part = float(sub.get("viability_score", 5))
            specificity_part = float(sub.get("specificity", 5))
            composite = round(
                0.4 * competition_part + 0.4 * market_part + 0.2 * specificity_part,
                2,
            )
            sub["composite_score"] = composite
            sub["score_breakdown"] = {
                "competition_40": round(0.4 * competition_part, 2),
                "market_40": round(0.4 * market_part, 2),
                "specificity_20": round(0.2 * specificity_part, 2),
            }
        ranked = sorted(
            subniches,
            key=lambda s: (
                -s.get("composite_score", 0.0),
                s.get("estimated_competitors", 10**9),
                s.get("name", ""),
            ),
        )
        for i, sub in enumerate(ranked, start=1):
            sub["rank"] = i
        return ranked

    def _save_results(
        self, main_category: str, ranked: List[Dict[str, Any]], used_tavily: bool
    ) -> None:
        """Persist results under state.research_findings['subniches']."""
        try:
            self.state_manager.update_phase("research")
        except Exception as exc:
            _console.print(f"[yellow]Could not set research phase: {exc}[/yellow]")
        try:
            state = self.state_manager.get_state()
            findings = state.research_findings
            if not isinstance(findings, dict):
                findings = {}
            findings["subniches"] = {
                "main_category": main_category,
                "competition_source": "tavily" if used_tavily else "model-estimate",
                "ranking_method": "weighted composite: competition 40% + market 40% + specificity 20%",
                "min_viable_market": MIN_VIABLE_MARKET,
                "researched_at": _utcnow_iso(),
                "subniches": ranked,
            }
            state.research_findings = findings
            self.state_manager.save(state)
            _console.print("[green]Sub-niche results saved to state.[/green]")
        except Exception as exc:
            raise RuntimeError(
                f"Could not persist sub-niche results: {type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _print_top3(ranked: List[Dict[str, Any]]) -> None:
        """Render the top 3 viable sub-niches as a rich table."""
        table = Table(title="Top 3 Validated Sub-Niches (weighted score)")
        table.add_column("Rank", justify="right")
        table.add_column("Sub-niche Name")
        table.add_column("Target Audience")
        table.add_column("Estimated Competitors", justify="right")
        table.add_column("Competition Level")
        table.add_column("Market Size", justify="right")
        table.add_column("Viability Score", justify="right")
        for sub in ranked[:3]:
            table.add_row(
                str(sub.get("rank", "?")),
                str(sub.get("name", "?"))[:50],
                str(sub.get("target_audience", "-") or "-")[:40],
                str(sub.get("estimated_competitors", "?")),
                str(sub.get("competition_level", "?")),
                f"{int(sub.get('market_size', 0)):,}",
                f"{sub.get('viability_score', '?')}/10",
            )
        _console.print(table)


def _extract_market_size(payload: Any, name: str) -> int:
    """Pull a non-negative int headcount from a sizing payload.

    Raises:
        ValueError: If the payload is not an object with a numeric
            ``estimated_market_size`` (null, ranges, and text all reject).
    """
    if not isinstance(payload, dict):
        raise ValueError(f"Sizing response for '{name}' was not a JSON object.")
    raw_value = payload.get("estimated_market_size")
    if isinstance(raw_value, bool) or raw_value is None:
        raise ValueError(
            f"Sizing response for '{name}' has no numeric "
            "estimated_market_size (got null/missing)."
        )
    if isinstance(raw_value, str):
        cleaned = raw_value.strip().replace(",", "")
        if not re.fullmatch(r"\d+(\.\d+)?", cleaned):
            raise ValueError(
                f"Sizing response for '{name}' has non-numeric "
                f"estimated_market_size: {raw_value!r}."
            )
        raw_value = cleaned
    try:
        return max(0, int(float(raw_value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Sizing response for '{name}' has non-numeric "
            f"estimated_market_size: {exc}"
        ) from exc


def _extract_reasoning(payload: Any) -> str:
    """Pull the reasoning string from a sizing payload (may be empty)."""
    if isinstance(payload, dict):
        return str(payload.get("reasoning", "")).strip()[:800]
    return ""


def _viability_from_size(size: int) -> int:
    """Map headcount to a 1–10 viability score (5000 people → 5, the cutoff)."""
    if size >= 100_000:
        return 10
    if size >= 50_000:
        return 9
    if size >= 20_000:
        return 8
    if size >= 10_000:
        return 7
    if size >= MIN_VIABLE_MARKET:
        return 5
    if size >= 2_000:
        return 3
    return 1


def _level_from_count(count: int, per_query: int) -> str:
    """Map a Tavily result count to low/medium/high (relative to depth)."""
    if count <= max(2, per_query // 3):
        return "low"
    if count <= max(5, (2 * per_query) // 3):
        return "medium"
    return "high"


def _clamp_int(value: Any, low: int, high: int) -> int:
    """Coerce to int and clamp into [low, high] (default: midpoint)."""
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        number = (low + high) // 2
    return max(low, min(high, number))


def _parse_json(raw: str) -> Any:
    """Parse model JSON tolerantly (strips code fences first)."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("Empty model response.")
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            return json.loads(text[start: end + 1])
        raise
