"""Scout Agent: market-opportunity researcher for digital micro-products.

The scout is the first agent in the swarm pipeline (phase ``"research"``).
Given a niche, it asks Space Bunny Alpha (via the
:class:`~core.orchestrator.Orchestrator`) for concrete product opportunities,
validates them, and persists the findings as the single source of truth in
the :class:`~core.state_manager.StateManager`.

Example usage:
    ```python
    from agents.scout import ScoutAgent
    from core.orchestrator import Orchestrator
    from core.state_manager import StateManager

    state_manager = StateManager(project_id="demo")
    orchestrator = Orchestrator()  # reads OPENROUTER_* from env / .env
    scout = ScoutAgent(orchestrator=orchestrator, state_manager=state_manager)

    findings = scout.research_opportunities(
        niche="Notion templates for students",
        target_price_range=(5, 20),
    )
    print(findings["opportunities"][0]["product_name"])

    # Findings are also persisted in state — any agent can read them back:
    state = state_manager.get_state()
    print(state.phase)  # -> "research"
    print(state.scout_findings["niche"])
    ```

All terminal output uses :mod:`rich` for colour. This module is
Windows-compatible: :class:`pathlib.Path` for paths, UTF-8 for file I/O.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console

from core.orchestrator import Orchestrator
from core.state_manager import StateManager

__all__ = ["ScoutAgent", "REQUIRED_OPPORTUNITY_FIELDS", "SCOUT_SYSTEM_PROMPT"]

#: Fields every validated opportunity dict must contain.
REQUIRED_OPPORTUNITY_FIELDS: Tuple[str, ...] = (
    "product_name",
    "target_audience",
    "pain_point",
    "estimated_price",
    "unique_value_proposition",
    "estimated_build_time",
)

#: Role instructions sent as the system prompt for every research call.
SCOUT_SYSTEM_PROMPT: str = (
    "You are an expert market researcher specializing in digital micro-products "
    "(templates, scripts, tools) that sell for €5-€20 on platforms like Gumroad."
)

_console = Console()


def _deep_enabled() -> bool:
    """Read the ``ENABLE_DEEP_RESEARCH`` flag from build_pipeline lazily.

    Imported inside the function to avoid any import cycle; any failure
    (missing module, missing flag) safely defaults to False.
    """
    try:
        from build_pipeline import ENABLE_DEEP_RESEARCH

        return bool(ENABLE_DEEP_RESEARCH)
    except Exception:
        return False


class ScoutAgent:
    """Research market opportunities for digital products in a given niche.

    Args:
        orchestrator: The swarm's :class:`Orchestrator` used for all model
            calls (usage is tracked automatically via the state manager).
        state_manager: The project's :class:`StateManager` where validated
            findings are persisted and the phase is set to ``"research"``.

    Example:
        ```python
        scout = ScoutAgent(orchestrator, state_manager)
        findings = scout.research_opportunities("Python automation scripts")
        ```
    """

    def __init__(
        self, orchestrator: Orchestrator, state_manager: StateManager
    ) -> None:
        # Duck-typed validation (rather than isinstance) so the agent accepts
        # the real Orchestrator/StateManager as well as test doubles/mocks
        # that expose the same interface.
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
        # Price range used by validate_opportunity(). Always reflects the
        # most recent research_opportunities() call (default €5-€20).
        self._target_price_range: Tuple[float, float] = (5.0, 20.0)

    def research_opportunities(
        self,
        niche: str,
        target_price_range: tuple = (5, 20),
        deep: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Research product opportunities in ``niche`` via Space Bunny Alpha.

        Args:
            niche: Market niche, e.g. ``"Notion templates for students"`` or
                ``"Python automation scripts"``. Must be a non-empty string.
            target_price_range: ``(min_price, max_price)`` tuple constraining
                ``estimated_price``. Defaults to ``(5, 20)``.
            deep: When True, run 3 angled research queries (base demand,
                competitor pricing, audience gaps) instead of 1, merge and
                dedupe opportunities, and attach ``competitor_analysis``.
                When None, uses the ``ENABLE_DEEP_RESEARCH`` flag from
                ``build_pipeline`` (defaults False).

        Returns:
            Findings dict persisted to state, with structure::

                {
                    "niche": niche,
                    "opportunities": [list of validated opportunity dicts],
                    "researched_at": ISO timestamp,
                }

        Raises:
            ValueError: If inputs are invalid, the model response is not
                valid JSON, or no valid opportunities were found.
            RuntimeError: If the underlying API call fails.

        Side effects:
            Sets the project phase to ``"research"``, stores
            ``scout_findings`` on the state, and saves (which also refreshes
            ``updated_at`` and bumps ``api_calls_count`` via the orchestrator).
        """
        cleaned_niche = niche.strip() if isinstance(niche, str) else ""
        if not cleaned_niche:
            raise ValueError("niche must be a non-empty string.")
        low, high = self._parse_price_range(target_price_range)
        self._target_price_range = (low, high)
        deep_mode = bool(deep) if deep is not None else _deep_enabled()

        base_prompt = self._build_user_prompt(cleaned_niche, low, high)
        if deep_mode:
            query_angles = [
                ("base demand", base_prompt),
                (
                    "competitor pricing",
                    base_prompt
                    + "\nAngle for this pass: focus especially on competitor "
                    "pricing — typical price points and what best sellers charge.",
                ),
                (
                    "audience gaps",
                    base_prompt
                    + "\nAngle for this pass: focus especially on the target "
                    "audience — who buys, which features they praise, and "
                    "which gaps they complain about.",
                ),
            ]
        else:
            query_angles = [("base demand", base_prompt)]

        _console.print(
            f"[cyan]Scout researching niche[/cyan] "
            f"[bold]{cleaned_niche}[/bold] (price €{low:g}–€{high:g})…"
            + (" [deep: 3 angled queries]" if deep_mode else "")
        )
        validated: List[Dict[str, Any]] = []
        seen_names = set()
        query_errors: List[str] = []
        for label, user_prompt in query_angles:
            try:
                raw_response: str = self.orchestrator.ask_agent(
                    system_prompt=SCOUT_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    state_manager=self.state_manager,
                )
            except Exception as exc:
                query_errors.append(f"{label}: {exc}")
                _console.print(
                    f"[yellow]Scout query '{label}' failed for niche "
                    f"{cleaned_niche!r}: {exc}[/yellow]"
                )
                continue
            try:
                candidates = self._parse_response(raw_response)
            except ValueError as exc:
                query_errors.append(f"{label} parse: {exc}")
                _console.print(
                    f"[yellow]Scout query '{label}' unparseable: {exc}[/yellow]"
                )
                continue
            for candidate in candidates:
                if isinstance(candidate, dict) and self.validate_opportunity(candidate):
                    name_key = str(candidate["product_name"]).strip().lower()
                    if name_key in seen_names:
                        _console.print(
                            f"[yellow]Duplicate opportunity {candidate['product_name']!r} "
                            f"from '{label}' pass; keeping first.[/yellow]"
                        )
                        continue
                    seen_names.add(name_key)
                    validated.append(candidate)
                else:
                    name = (
                        candidate.get("product_name", "(unnamed)")
                        if isinstance(candidate, dict)
                        else repr(candidate)
                    )
                    _console.print(
                        f"[yellow]Discarded invalid opportunity {name!r}: "
                        "missing fields, over-long name, or price out of range."
                        "[/yellow]"
                    )

        if not validated:
            raise ValueError(
                f"No valid opportunities found for niche {cleaned_niche!r} "
                f"across {len(query_angles)} quer"
                f"{'y' if len(query_angles) == 1 else 'ies'}; "
                f"query errors: {query_errors or 'none (candidates failed validation)'}; "
                f"price must be within €{low:g}–€{high:g}."
            )

        for opportunity in validated:
            _console.print(
                f"[green]Found opportunity:[/green] "
                f"[bold]{opportunity['product_name']}[/bold] "
                f"(€{float(opportunity['estimated_price']):g} — "
                f"{opportunity['target_audience']})"
            )

        findings: Dict[str, Any] = {
            "niche": cleaned_niche,
            "opportunities": validated,
            "researched_at": datetime.now(timezone.utc).isoformat(),
        }
        if deep_mode:
            findings["deep_research"] = True
            findings["query_angles"] = [label for label, _ in query_angles]
            findings["competitor_analysis"] = self._competitor_analysis(validated)

        try:
            self.state_manager.update_phase("research")
            state = self.state_manager.get_state()
            state.scout_findings = findings
            self.state_manager.save(state)
        except Exception as exc:
            raise RuntimeError(
                f"Scout research succeeded but persisting findings failed "
                f"for project '{self.state_manager.project_id}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        _console.print(
            f"[green]Scout saved {len(validated)} validated "
            f"opportunit{'y' if len(validated) == 1 else 'ies'} "
            f"for niche '{cleaned_niche}'.[/green]"
        )
        return findings

    @staticmethod
    def _competitor_analysis(
        opportunities: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Summarise competitor pricing and audiences (deep mode only).

        Extracts min/max/avg estimated prices plus the most common target
        audiences across validated opportunities. Never raises for
        well-formed input.
        """
        prices = [
            float(o["estimated_price"])
            for o in opportunities
            if isinstance(o, dict)
            and isinstance(o.get("estimated_price"), (int, float))
            and not isinstance(o.get("estimated_price"), bool)
        ]
        tally: Dict[str, int] = {}
        for item in opportunities:
            if not isinstance(item, dict):
                continue
            audience = str(item.get("target_audience", "")).strip()
            if audience:
                tally[audience] = tally.get(audience, 0) + 1
        return {
            "opportunity_count": len(opportunities),
            "min_price": min(prices) if prices else None,
            "max_price": max(prices) if prices else None,
            "avg_price": round(sum(prices) / len(prices), 2) if prices else None,
            "top_audiences": sorted(tally, key=lambda k: (-tally[k], k))[:5],
        }

    def validate_opportunity(self, opportunity: Dict[str, Any]) -> bool:
        """Check whether an opportunity dict is complete and in range.

        The price is checked against the ``target_price_range`` of the most
        recent :meth:`research_opportunities` call (default ``(5, 20)``).

        Args:
            opportunity: Candidate dict, expected to contain all
                ``REQUIRED_OPPORTUNITY_FIELDS``.

        Returns:
            True if every required field is present and valid (non-empty
            strings, ``product_name`` at most 60 chars, ``estimated_price``
            numeric and within range); False otherwise. Never raises for
            malformed input.
        """
        if not isinstance(opportunity, dict):
            return False
        for field in REQUIRED_OPPORTUNITY_FIELDS:
            if field not in opportunity:
                return False
        try:
            product_name = opportunity["product_name"]
            if not isinstance(product_name, str) or not product_name.strip():
                return False
            if len(product_name.strip()) > 60:
                return False
            for field in (
                "target_audience",
                "pain_point",
                "unique_value_proposition",
                "estimated_build_time",
            ):
                value = opportunity[field]
                if not isinstance(value, str) or not value.strip():
                    return False
            price = opportunity["estimated_price"]
            # bool is a subclass of int — exclude it explicitly.
            if isinstance(price, bool) or not isinstance(price, (int, float)):
                return False
            low, high = self._target_price_range
            if not (low <= float(price) <= high):
                return False
        except (TypeError, ValueError, ArithmeticError):
            return False
        return True

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _parse_price_range(target_price_range: tuple) -> Tuple[float, float]:
        """Validate the price-range tuple, returning ``(low, high)`` floats.

        Raises:
            ValueError: If the range is malformed or ``low > high``.
        """
        try:
            low_raw, high_raw = target_price_range
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "target_price_range must be a (min_price, max_price) tuple, "
                f"got {target_price_range!r}."
            ) from exc
        if isinstance(low_raw, bool) or isinstance(high_raw, bool):
            raise ValueError(
                "target_price_range bounds must be numbers, "
                f"got {target_price_range!r}."
            )
        try:
            low, high = float(low_raw), float(high_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "target_price_range bounds must be numbers, "
                f"got {target_price_range!r}."
            ) from exc
        if low > high:
            raise ValueError(
                f"Invalid target_price_range {target_price_range!r}: "
                "min_price must not exceed max_price."
            )
        return (low, high)

    @staticmethod
    def _build_user_prompt(niche: str, low: float, high: float) -> str:
        """Compose the user prompt requesting 3 opportunities as JSON."""
        return (
            f"Research 3 specific, sellable digital product opportunities "
            f"in the niche: {niche!r}.\n\n"
            f"For EACH product, provide ALL of these fields:\n"
            f"- product_name (string, max 60 chars, compelling)\n"
            f"- target_audience (string, specific persona)\n"
            f"- pain_point (string, what problem it solves)\n"
            f"- estimated_price (number, in euros, between {low:g} and {high:g})\n"
            f"- unique_value_proposition (string, why people would buy this)\n"
            f"- estimated_build_time (string, e.g. \"2 hours\")\n\n"
            f"Respond with VALID JSON ONLY — no markdown fences, no commentary, "
            f"no trailing text. Use exactly this shape:\n"
            f'{{\n  "opportunities": [\n'
            f'    {{\n'
            f'      "product_name": "...",\n'
            f'      "target_audience": "...",\n'
            f'      "pain_point": "...",\n'
            f'      "estimated_price": 12.0,\n'
            f'      "unique_value_proposition": "...",\n'
            f'      "estimated_build_time": "2 hours"\n'
            f"    }}\n"
            f"  ]\n"
            f"}}"
        )

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove Markdown code fences (```json … ```) if present."""
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            # Drop the opening fence (``` or ```json).
            lines = lines[1:]
            # Drop the closing fence if present.
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
        return stripped

    def _parse_response(self, raw_response: str) -> List[Any]:
        """Parse the model reply into a list of candidate opportunity dicts.

        Accepts either ``{"opportunities": [...]}`` (requested shape) or a
        bare ``[...]`` list. Common model quirks (code fences, surrounding
        prose) are tolerated where unambiguous.

        Raises:
            ValueError: If no valid JSON with opportunities can be extracted,
                with a log line and a snippet of the offending response.
        """
        if not raw_response or not raw_response.strip():
            _console.print("[red]Scout received an empty model response.[/red]")
            raise ValueError("Model returned an empty response; expected JSON.")
        text = self._strip_code_fences(raw_response)
        payload: Optional[Any] = None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            # Last resort: parse the largest {...} span in the reply.
            start, end = text.find("{"), text.rfind("}")
            if 0 <= start < end:
                try:
                    payload = json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    payload = None
        if payload is None:
            snippet = raw_response.strip()[:300]
            _console.print(
                f"[red]Scout could not parse model response as JSON: "
                f"{snippet!r}…[/red]"
            )
            raise ValueError(
                "Failed to parse model response as JSON. "
                f"Response started with: {snippet!r}. "
                "Expected valid JSON with an 'opportunities' list."
            )
        if isinstance(payload, dict):
            for key in ("opportunities", "products", "items"):
                if isinstance(payload.get(key), list):
                    return payload[key]
            _console.print(
                "[red]Scout response JSON has no opportunities list.[/red]"
            )
            raise ValueError(
                "Model JSON parsed but contained no 'opportunities' list. "
                f"Top-level keys were: {sorted(str(k) for k in payload.keys())}."
            )
        if isinstance(payload, list):
            return payload
        raise ValueError(
            "Model JSON must be an object with an 'opportunities' list or a "
            f"bare list; got {type(payload).__name__}."
        )
