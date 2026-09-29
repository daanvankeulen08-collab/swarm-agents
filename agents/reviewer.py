"""ReviewerAgent: quality gate between packaging and publishing.

The reviewer inspects a packaged ZIP (file inventory + content) with
deterministic structural checks, then asks Space Bunny Alpha (via the
:class:`Orchestrator`) for a qualitative scored review. The verdict is
``"approved"`` only when the structure passes AND the model quality score
reaches the threshold; otherwise ``"rejected"`` with actionable feedback
suitable for a builder revision loop.

Each check maps to the review contract:

* Completeness — every required section is present in the template text.
* Usability — clear headings/structure, logical flow (model-judged).
* Value proposition — pain points actually solved (model-judged).
* Professional quality — no placeholder text, proper formatting,
  reasonable size (deterministic + model-judged).

Example:
    ```python
    from agents.reviewer import ReviewerAgent
    from core.orchestrator import Orchestrator
    from core.state_manager import StateManager

    reviewer = ReviewerAgent(Orchestrator(), StateManager(project_id="demo"))
    result = reviewer.review_product("packages/My_Product.zip")
    print(result["verdict"], result["quality_score"])
    ```
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

from core.orchestrator import Orchestrator
from core.state_manager import StateManager

__all__ = ["ReviewerAgent", "APPROVAL_THRESHOLD", "PLACEHOLDER_PATTERNS"]

#: Minimum model quality score (1–10) required for approval.
APPROVAL_THRESHOLD: int = 7

#: Case-insensitive markers indicating unfinished/placeholder content.
PLACEHOLDER_PATTERNS: tuple = (
    "todo",
    "fixme",
    "xxx",
    "lorem ipsum",
    "[insert",
    "<insert",
    "tbd",
    "coming soon",
)

_console = Console()


class ReviewerAgent:
    """Review packaged products and return an approval verdict + feedback.

    Args:
        orchestrator: The swarm's :class:`Orchestrator` for qualitative
            review (usage tracked automatically via the state manager).
        state_manager: The project's :class:`StateManager`; each review
            writes ``review_status``/``review_feedback`` and sets the phase
            to ``"reviewing"``.

    Example:
        ```python
        reviewer = ReviewerAgent(orchestrator, state_manager)
        result = reviewer.review_product(zip_path, required_sections=[...])
        ```
    """

    def __init__(
        self, orchestrator: Orchestrator, state_manager: StateManager
    ) -> None:
        # Duck-typed validation (rather than isinstance) so the agent accepts
        # the real Orchestrator/StateManager as well as test doubles/mocks.
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

    def review_product(
        self,
        package_path: str | Path,
        required_sections: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Review a packaged ZIP and return verdict, score, and feedback.

        Section/placeholder/size checks run ONLY against the actual
        template content — ``state.product_code`` (the Builder output),
        falling back to the largest non-auxiliary Markdown file in the
        ZIP when state holds nothing. README, guides, and examples never
        count toward section coverage.

        Args:
            package_path: Path to the ``.zip`` produced by ProductPackager.
            required_sections: Section keywords that must appear in the
                template text (e.g. ``["appointment grid", "invoice"]``).
                Defaults to a generic completeness check.
            context: Optional background (product_name, target_audience,
                pain_point) forwarded to the qualitative model review.

        Returns:
            Dict with ``verdict`` (``"approved"``/``"rejected"``),
            ``quality_score`` (int 1–10), ``feedback`` (str, actionable),
            ``checks`` (dict of deterministic check name → bool), and
            ``details`` (list of human-readable findings).

        Raises:
            ValueError: If the path does not point at a readable ZIP.
            RuntimeError: If the qualitative model review fails.
        """
        zip_path = Path(package_path)
        if not zip_path.is_file() or zip_path.suffix.lower() != ".zip":
            raise ValueError(
                f"package_path must be an existing .zip file, got {str(package_path)!r}."
            )
        _console.print(
            f"[cyan]Reviewer inspecting package[/cyan] [bold]{zip_path.name}[/bold]…"
        )

        try:
            inventory, texts = self._read_package(zip_path)
        except zipfile.BadZipFile as exc:
            raise ValueError(f"File '{zip_path}' is not a valid ZIP: {exc}") from exc

        template_text, template_source = self._resolve_template_text(
            inventory, texts
        )
        required = [s for s in (required_sections or []) if str(s).strip()]
        checks, details = self._structural_checks(
            inventory, template_text, template_source, required
        )

        qualitative = self._model_review(template_text, required, context or {})
        model_score = qualitative["score"]

        structural_ok = all(checks.values())
        approved = structural_ok and model_score >= APPROVAL_THRESHOLD
        quality_score = (
            min(10, max(1, round((model_score + (9 if structural_ok else 4)) / 2)))
            if approved or structural_ok
            else min(model_score, 4)
        )
        verdict = "approved" if approved else "rejected"

        feedback_parts = list(details)
        feedback_parts.append(
            f"Model quality score: {model_score}/10 — {qualitative['summary']}"
        )
        if not approved:
            feedback_parts.append(
                "To earn approval: fix every FAILED check above and address "
                "the model notes, then resubmit the complete product."
            )
        feedback = "\n".join(f"- {part}" for part in feedback_parts)

        self._record_review(verdict, quality_score, feedback)
        _console.print(
            f"[{'green' if approved else 'red'}]Review verdict: "
            f"{verdict.upper()} (quality {quality_score}/10)[/]"
        )
        return {
            "verdict": verdict,
            "quality_score": quality_score,
            "feedback": feedback,
            "checks": checks,
            "details": details,
            "model_score": model_score,
        }

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _read_package(zip_path: Path) -> tuple[List[str], Dict[str, str]]:
        """Return (file list, {name: decoded text}) for text files in the ZIP."""
        with zipfile.ZipFile(zip_path, "r") as archive:
            names = archive.namelist()
            texts: Dict[str, str] = {}
            for name in names:
                if name.endswith("/") or not name.lower().endswith(
                    (".md", ".txt", ".py", ".csv", ".json")
                ):
                    continue
                try:
                    texts[name] = archive.read(name).decode("utf-8", errors="replace")
                except (KeyError, RuntimeError, UnicodeError):
                    continue
        _console.print(f"[cyan]  Package holds {len(names)} file(s).[/cyan]")
        return names, texts

    def _resolve_template_text(
        self, inventory: List[str], texts: Dict[str, str]
    ) -> tuple[str, str]:
        """Return (template text, source label) scoped to real product content.

        Prefers ``state.product_code`` (the Builder output); otherwise picks
        the largest Markdown file that is not an auxiliary doc (README,
        quick-start, features, filled examples).
        """
        try:
            state_code = self.state_manager.get_state().product_code
        except Exception:
            state_code = None
        if isinstance(state_code, str) and state_code.strip():
            _console.print("[cyan]  Template source: state.product_code.[/cyan]")
            return state_code, "state.product_code"
        aux_markers = ("readme", "quickstart", "quick_start", "features",
                       "filled_example", "example_usage")
        candidates = [
            (name, text) for name, text in texts.items()
            if name.lower().endswith(".md")
            and not any(m in name.lower() for m in aux_markers)
        ]
        if candidates:
            best = max(candidates, key=lambda kv: len(kv[1]))
            _console.print(f"[cyan]  Template source: {best[0]} (state empty).[/cyan]")
            return best[1], best[0]
        longest = max(texts.values(), key=len) if texts else ""
        _console.print("[cyan]  Template source: largest ZIP text (fallback).[/cyan]")
        return longest, "largest-zip-text"

    def _structural_checks(
        self,
        inventory: List[str],
        template_text: str,
        template_source: str,
        required: List[str],
    ) -> tuple[Dict[str, bool], List[str]]:
        """Run deterministic checks against the template text only."""
        checks: Dict[str, bool] = {}
        details: List[str] = [
            f"Template under review: {template_source} "
            f"({len(template_text):,} chars; README/guides/examples excluded)."
        ]
        lowered = template_text.lower()

        has_readme = any(n.lower().endswith("readme.md") for n in inventory)
        checks["has_readme"] = has_readme
        details.append(
            f"{'PASS' if has_readme else 'FAIL'}: README.md present in package."
        )

        has_template = bool(template_text.strip())
        checks["has_template_content"] = has_template
        details.append(
            f"{'PASS' if has_template else 'FAIL'}: template content present."
        )

        missing = [s for s in required if s.lower() not in lowered]
        checks["all_sections_present"] = not missing
        if missing:
            details.append(
                f"FAIL: missing required sections in template: {missing}."
            )
        else:
            details.append(
                f"PASS: all {len(required)} required sections present "
                "in template."
                if required
                else "PASS: no required sections configured (skipped)."
            )

        leftovers = sorted({m for m in PLACEHOLDER_PATTERNS if m in lowered})
        checks["no_placeholder_text"] = not leftovers
        details.append(
            "PASS: no placeholder text detected in template."
            if not leftovers
            else f"FAIL: placeholder text remains in template: {leftovers}."
        )

        size_ok = 500 <= len(template_text) <= 500_000
        checks["reasonable_size"] = size_ok
        details.append(
            f"{'PASS' if size_ok else 'FAIL'}: template size "
            f"{len(template_text):,} chars."
        )
        return checks, details

    def _model_review(
        self,
        template_text: str,
        required: List[str],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Score the template text for usability, value, quality (1–10)."""
        excerpt = template_text[:12000]
        background = "\n".join(
            f"- {key}: {context.get(key, '(not specified)')}"
            for key in ("product_name", "target_audience", "pain_point")
        )
        try:
            raw = self.orchestrator.ask_agent(
                system_prompt=(
                    "You are a strict quality reviewer for paid digital "
                    "products. Judge only what is in front of you."
                ),
                user_prompt=(
                    "Review this digital product template for a paying customer.\n\n"
                    f"Product background:\n{background}\n\n"
                    f"Required sections: {required or '(none specified)'}\n\n"
                    f"Template content:\n{excerpt}\n\n"
                    "Score completeness, usability, value-for-money, and "
                    "professional quality. Respond with VALID JSON ONLY, no "
                    "commentary: "
                    '{"score": 8, '
                    '"summary": "one-paragraph verdict", '
                    '"strengths": ["..."], '
                    '"issues": ["..."]}'
                ),
                state_manager=self.state_manager,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Reviewer model call failed: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            payload = _parse_json(raw)
        except ValueError as exc:
            raise RuntimeError(
                f"Reviewer model response was not valid JSON: {exc}"
            ) from exc
        try:
            score = max(1, min(10, int(float(payload.get("score", 1)))))
        except (TypeError, ValueError):
            score = 1
        _console.print(f"[cyan]  Model quality score: {score}/10.[/cyan]")
        return {
            "score": score,
            "summary": str(payload.get("summary", "")).strip()[:800],
            "strengths": payload.get("strengths", []),
            "issues": payload.get("issues", []),
        }

    def _record_review(
        self, verdict: str, quality_score: int, feedback: str
    ) -> None:
        """Persist verdict to review_status/review_feedback (phase reviewing)."""
        state = self.state_manager.get_state()
        state.review_status = verdict
        state.review_feedback = f"Quality {quality_score}/10. {feedback}"[:2000]
        self.state_manager.save(state)
        self.state_manager.update_phase("reviewing")


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
