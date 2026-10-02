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
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from rich.console import Console

from core.orchestrator import Orchestrator
from core.placeholders import detect_abandoned_content
from core.sections import (
    SECTION_DEFINITIONS,
    normalize_required_sections,
    section_requirement_lines,
)
from core.state_manager import StateManager
from core.vision import VisionClient

__all__ = [
    "ReviewerAgent",
    "APPROVAL_THRESHOLD",
    "PLACEHOLDER_PATTERNS",
    "VISUAL_PASS_SCORE",
    "VISUAL_FAIL_SCORE",
    "VISUAL_FLAGS",
    "MIN_OVERALL_SCORE",
    "MIN_SINGLE_CRITERION",
    "REVIEWER_TIMEOUT",
    "REVIEWER_MODEL",
    "REVIEWER_BASE_URL",
    "REVIEWER_TEMPERATURE",
    "REVIEWER_MAX_TOKENS",
]

#: Minimum model quality score (1–10) required for approval.
APPROVAL_THRESHOLD: int = 7

#: Overall average a per-section review must reach for approval. Mirrors the
#: pipeline's MIN_OVERALL_SCORE without importing build_pipeline (which would
#: be circular: build_pipeline imports this module).
MIN_OVERALL_SCORE: float = 7.0

#: No single section may score below this. Mirrors MIN_SINGLE_CRITERION.
MIN_SINGLE_CRITERION: float = 6.0

#: Case-insensitive markers indicating unfinished/placeholder content.
#:
#: ``tbd`` was removed deliberately: empty fields are legitimate in a
#: planner template, and the blanket match produced a false rejection during
#: Maiden Voyage 3. Abandoned content is detected structurally via
#: :func:`core.placeholders.detect_abandoned_content`.
PLACEHOLDER_PATTERNS: tuple = (
    "todo",
    "fixme",
    "xxx",
    "[insert",
    "<insert",
)

#: Visual-review thresholds: >= PASS is good, <= FAIL is a hard layout problem.
VISUAL_PASS_SCORE: float = 7.0
VISUAL_FAIL_SCORE: float = 4.0

#: Dedicated reviewer model (NVIDIA Nemotron 3 Ultra on NIM).
#: Chosen for judging because of its 1M-token context window — a full product
#: plus per-section reviews fit without any truncation, which is what blinded
#: the cascade-based review in Maiden Voyage 011. Proven working on this
#: account. Override via ``NVIDIA_REVIEWER_MODEL``.
REVIEWER_MODEL: str = "nvidia/nemotron-3-ultra-550b-a55b"

#: NIM endpoint for the reviewer client.
REVIEWER_BASE_URL: str = "https://integrate.api.nvidia.com/v1"

#: Low temperature for consistent, repeatable judging.
REVIEWER_TEMPERATURE: float = 0.3

#: Cap per reviewer call; a section verdict needs hundreds of tokens, not
#: thousands.
REVIEWER_MAX_TOKENS: int = 4000

#: Timeout per reviewer call. Deliberately generous (10 minutes): the
#: dedicated NIM judge is free and stable, and timing out a section review
#: mid-run kills the pipeline for no benefit. The 60s stall guard stays
#: ONLY on the Zen/OpenRouter cascade hops — never on dedicated NIM judges.
REVIEWER_TIMEOUT: float = 600.0

#: The four layout defects the vision pass checks for.
VISUAL_FLAGS: tuple = (
    "truncated_tables",
    "broken_layout",
    "missing_content",
    "readability_issues",
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
        for name in ("ask_agent", "ask_agent_with_fallback",
                     "build_context_prompt", "get_model_info"):
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
        # Vision reviewer for template screenshots; disabled gracefully
        # without a NIM key (visual review then reports unavailability).
        self.vision_client: VisionClient = VisionClient()
        # Dedicated reviewer model (Nemotron 3 Ultra on NIM). This client
        # deliberately bypasses the Zen/OpenRouter/NIM provider cascade:
        # judging must be deterministic and is not subject to whichever
        # provider happens to be healthy. Accepts either NIM key variable;
        # disabled gracefully (None) when neither is set.
        reviewer_key = (
            os.getenv("NVIDIA_NIM_API_KEY")
            or os.getenv("NVIDIA_API_KEY")
            or ""
        ).strip()
        self.reviewer_model: str = (
            os.getenv("NVIDIA_REVIEWER_MODEL") or REVIEWER_MODEL
        ).strip() or REVIEWER_MODEL
        self.reviewer_client: Optional[OpenAI] = None
        if reviewer_key:
            self.reviewer_client = OpenAI(
                api_key=reviewer_key,
                base_url=REVIEWER_BASE_URL,
                timeout=REVIEWER_TIMEOUT,
            )
            _console.print(
                "[green]Reviewer model ready[/green] — "
                f"[bold]{self.reviewer_model}[/bold] (dedicated NIM client, "
                "bypasses provider cascade)"
            )
        else:
            _console.print(
                "[yellow]No NIM key — dedicated reviewer disabled. "
                "Per-section review will fall back to the cascade.[/yellow]"
            )

    def _ask_reviewer(self, system_prompt: str, user_prompt: str) -> str:
        """Ask the dedicated reviewer model one judging question.

        Uses the NIM client directly (never the provider cascade) at low
        temperature for consistent verdicts.

        Args:
            system_prompt: Judging instructions (JSON contract, criteria).
            user_prompt: The section content to judge.

        Returns:
            The model's raw text, or ``""`` on any failure (callers treat
            empty as "judge unavailable" and degrade gracefully instead of
            crashing the review).
        """
        if self.reviewer_client is None:
            _console.print(
                "[yellow]Reviewer client unavailable (no NIM key).[/yellow]"
            )
            return ""
        try:
            response = self.reviewer_client.chat.completions.create(
                model=self.reviewer_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=REVIEWER_TEMPERATURE,
                max_tokens=REVIEWER_MAX_TOKENS,
                timeout=REVIEWER_TIMEOUT,
            )
            choices = getattr(response, "choices", None) or []
            message = getattr(choices[0], "message", None) if choices else None
            return (getattr(message, "content", None) or "").strip()
        except Exception as exc:
            _console.print(
                f"[yellow]Reviewer model call failed "
                f"({type(exc).__name__}); treating as unavailable.[/yellow]"
            )
            return ""

    def review_product(
        self,
        package_path: str | Path,
        required_sections: Optional[Any] = None,
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
            required_sections: Requirement specification, normalised by
                :func:`core.sections.normalize_required_sections`. Accepts
                ``None`` (use the canonical
                :data:`core.sections.SECTION_DEFINITIONS`), a mapping of
                ``{section_id: keyword(s)}``, or a sequence of keywords /
                headings. Each section passes when **any** of its accepted
                spellings appears, so a template writing "Assignment Notes"
                satisfies the ``protocol`` requirement.
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
        # FIX 1: requirements come from the shared SECTION_DEFINITIONS, so
        # the reviewer can never drift from the headings the builder writes.
        required = normalize_required_sections(required_sections)
        checks, details = self._structural_checks(
            inventory, template_text, template_source, required
        )

        qualitative = self._model_review(
            template_text, required, context or {},
            sections_dir=self._sections_dir(),
        )
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

    def review_visually(self, image_path: str) -> Dict[str, Any]:
        """Analyze a template screenshot with the Vision API.

        Checks for truncated tables, broken layouts, missing content, and
        readability issues.

        Args:
            image_path: Path to the template screenshot (PNG).

        Returns:
            Dict with ``visual_score`` (float 1–10), ``issues``
            (list of strings), ``recommendation`` (``PASS`` /
            ``NEEDS_WORK`` / ``FAIL``), ``flags`` (per-defect booleans),
            ``summary``, plus ``model`` and raw ``text``. The shape is stable
            on failure. Vision outages yield ``visual_score`` 0.0, the error
            in ``issues``, and recommendation ``FAIL`` — callers must treat
            that as "vision unavailable", not as a failed product.
        """
        prompt = """You are a visual quality assurance specialist reviewing a screenshot of a digital template.

Inspect the image carefully, especially the RIGHT and BOTTOM edges, and check:
1. Truncated tables: rows cut off mid-way, or columns/headers clipped by the image edge. Check whether the last row and the last column are fully visible and complete.
2. Broken layouts: overlapping text, columns running off the edge, misaligned or overflowing sections.
3. Missing or blank content where content is expected.
4. Readability problems: text too small, poor contrast, unreadable fonts.

Be strict: a high score requires that every row and column is fully visible. If any text is cut off at an edge, truncated_tables must be true and the score must be 6 or lower.

Reply with ONLY a JSON object, no prose, in exactly this shape:
{"visual_score": <integer 1-10>,
 "truncated_tables": <true|false>,
 "broken_layout": <true|false>,
 "missing_content": <true|false>,
 "readability_issues": <true|false>,
 "issues": ["<specific issue, or empty if none>"],
 "summary": "<one short sentence>"}

Scoring guidance: 9-10 flawless and complete; 7-8 minor cosmetic issues;
5-6 one real problem such as clipping; 1-4 broken or unusable. If the
screenshot looks clean, report a high score and an empty issues list.
"""
        _console.print(
            f"[cyan]Visual review of[/cyan] [bold]{image_path}[/bold]…"
        )
        clip = _edge_clipping_check(image_path)
        outcome = self.vision_client.analyze_image(image_path, prompt)
        if not outcome.get("ok"):
            _console.print(
                f"[yellow]Visual review unavailable: "
                f"{outcome.get('error')}[/yellow]"
            )
            return {
                "visual_score": 0.0,
                "issues": [str(outcome.get("error", "unknown vision error"))],
                "recommendation": "FAIL",
                "summary": "vision unavailable",
                "flags": {name: False for name in VISUAL_FLAGS},
                "model": getattr(self.vision_client, "model", "?"),
                "text": "",
                "ok": False,
            }
        text = str(outcome.get("text", ""))
        parsed = _parse_visual_review(text)
        # The smaller vision models sometimes answer in prose despite the
        # format request. One strict re-ask converts most of those replies
        # into parseable JSON instead of failing the template on formatting.
        if not parsed["parsed"]:
            _console.print(
                "[yellow]Vision reply was not JSON; re-asking for strict "
                "JSON output…[/yellow]"
            )
            retry = self.vision_client.analyze_image(
                image_path,
                prompt + "\n\nCRITICAL FORMAT RULE: your entire reply must be "
                "a single JSON object whose first character is '{' and whose "
                "last character is '}'. No preamble, no explanation, no "
                "markdown fences. Output only that JSON object.\n",
            )
            if retry.get("ok"):
                retry_parsed = _parse_visual_review(str(retry.get("text", "")))
                if retry_parsed["parsed"]:
                    parsed = retry_parsed
                    text = str(retry.get("text", ""))
        score = parsed["visual_score"]
        # Deterministic guard: the vision models are inconsistent about
        # subtle clipping, so ink touching the outer edge of a full-page
        # render is treated as proof of truncation and caps the score. This
        # runs regardless of what the model claimed.
        issues = list(parsed["issues"])
        if clip["clipped"]:
            issues.insert(0, clip["detail"])
            score = min(score, 5.0)
            parsed["flags"]["truncated_tables"] = True
        recommendation = (
            "PASS" if score >= VISUAL_PASS_SCORE
            else ("NEEDS_WORK" if score >= VISUAL_FAIL_SCORE else "FAIL")
        )
        colour = {"PASS": "green", "NEEDS_WORK": "yellow", "FAIL": "red"}[
            recommendation
        ]
        _console.print(
            f"[{colour}]Visual review: {recommendation} "
            f"({score}/10, {len(issues)} issue(s)).[/{colour}]"
        )
        return {
            "visual_score": score,
            "issues": issues,
            "recommendation": recommendation,
            "summary": parsed["summary"],
            "flags": parsed["flags"],
            "parsed": parsed["parsed"],
            "edge_clipping": clip,
            "model": outcome.get("model"),
            "text": text[:2000],
            "ok": True,
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
        required: Dict[str, Tuple[str, ...]],
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

        # A section passes when ANY accepted spelling is present, so the
        # builder's heading and the reviewer's keyword both count.
        missing = [
            section_id
            for section_id, keywords in required.items()
            if not any(kw in lowered for kw in keywords)
        ]
        checks["all_sections_present"] = not missing
        if missing:
            details.append(
                f"FAIL: missing required sections in template: {missing} "
                f"(expected any of: "
                f"{ {k: required[k] for k in missing} })."
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

        # Structural abandonment check. Fillable TBD fields are the product
        # working correctly; a section that stops at TBD is a defect. Shares
        # one implementation with the test gate so they cannot disagree.
        abandoned = detect_abandoned_content(template_text)
        checks["no_abandoned_content"] = not abandoned
        if abandoned:
            for defect in abandoned[:5]:
                details.append(f"FAIL: {defect}")
        else:
            details.append(
                "PASS: no abandoned content (fillable TBD fields accepted)."
            )

        size_ok = 500 <= len(template_text) <= 500_000
        checks["reasonable_size"] = size_ok
        details.append(
            f"{'PASS' if size_ok else 'FAIL'}: template size "
            f"{len(template_text):,} chars."
        )
        return checks, details

    def _sections_dir(self) -> Path:
        """Per-project directory holding the builder's section files."""
        project_id = getattr(self.state_manager, "project_id", None) or "default"
        return Path("state") / "output" / str(project_id) / "sections"

    def _review_per_section(
        self, sections_dir: Path, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Judge every section file individually with the dedicated model.

        Each section is read whole — no truncation, so the reviewer is never
        blind to the back half of the product again. Every call goes to the
        dedicated NIM reviewer (never the provider cascade).

        Args:
            sections_dir: Directory holding ``section_{i}.md`` files.
            context: Product background forwarded to each section prompt.

        Returns:
            Dict with ``sections`` (per-section ``{score, issues, verdict}``
            keyed by file stem), ``average_score`` (mean, 1 decimal),
            ``weak_sections`` (names scoring below 6.0), ``model``,
            ``reviewed`` (count) and ``available`` (False when the judge was
            unreachable — callers must not treat that as a pass).
        """
        section_files = sorted(
            sections_dir.glob("section_*.md"),
            key=lambda p: p.name,
        ) if sections_dir.is_dir() else []
        background = "\n".join(
            f"- {key}: {context.get(key, '(not specified)')}"
            for key in ("product_name", "target_audience", "pain_point")
        )
        system_prompt = (
            "You are a strict technical quality reviewer for Notion template "
            "content. Review ONLY the provided section. Evaluate: "
            "completeness, accuracy, structure, usability. Return a JSON with: "
            'score (1-10), issues (list of strings), verdict (PASS/FAIL). '
            "Reply with ONLY the JSON object, no prose."
        )
        results: Dict[str, Dict[str, Any]] = {}
        judge_reached = False
        for path in section_files:
            try:
                content = path.read_text(encoding="utf-8")
            except OSError:
                continue
            title = path.stem.replace("_", " ")
            raw = self._ask_reviewer(
                system_prompt,
                f"Review this section: {title}\n\nContent:\n{content}",
            )
            if not raw:
                continue
            judge_reached = True
            try:
                payload = _parse_json(raw)
            except ValueError:
                continue
            if not isinstance(payload, dict):
                continue
            try:
                score = min(10.0, max(1.0, float(payload.get("score", 1))))
            except (TypeError, ValueError):
                score = 1.0
            issues = [str(i)[:300] for i in (payload.get("issues") or [])
                      if str(i).strip()][:10]
            verdict = str(payload.get("verdict", "")).strip().upper()
            if verdict not in ("PASS", "FAIL"):
                verdict = "PASS" if score >= 6.0 else "FAIL"
            results[path.stem] = {
                "title": title,
                "score": round(score, 1),
                "issues": issues,
                "verdict": verdict,
            }
            _console.print(
                f"[cyan]  Section {path.stem}: {score:.1f}/10 "
                f"{verdict} ({len(issues)} issue(s)).[/cyan]"
            )
        scores = [r["score"] for r in results.values()]
        average = round(sum(scores) / len(scores), 1) if scores else 0.0
        weak = [r["title"] for r in results.values() if r["score"] < 6.0]
        return {
            "sections": results,
            "average_score": average,
            "weak_sections": weak,
            "model": self.reviewer_model,
            "reviewed": len(results),
            "available": judge_reached,
        }

    def _model_review(
        self,
        template_text: str,
        required: Dict[str, Tuple[str, ...]],
        context: Dict[str, Any],
        sections_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Score the template for usability, value, quality (1–10).

        Prefers per-section review of the builder's section files (full
        content, no truncation). Falls back to the legacy whole-template
        cascade call only when no section files exist or the dedicated judge
        is unreachable.

        Args:
            template_text: Full template text (fallback path only).
            required: Normalised requirement mapping.
            context: Product background.
            sections_dir: Builder section workspace, or None.

        Returns:
            Dict with ``score``, ``summary``, ``strengths``, ``issues`` and
            ``per_section`` (the per-section report, or None on fallback).
        """
        if sections_dir is not None:
            per_section = self._review_per_section(sections_dir, context)
            if per_section["available"] and per_section["reviewed"]:
                weak = per_section["weak_sections"]
                approved = (
                    per_section["average_score"] >= MIN_OVERALL_SCORE
                    and not weak
                )
                summary = (
                    f"Per-section review ({per_section['reviewed']} sections, "
                    f"avg {per_section['average_score']}/10"
                    + (f"; weak: {', '.join(weak)}" if weak else "; all PASS")
                    + ")."
                )
                issues: List[str] = []
                for stem, entry in per_section["sections"].items():
                    issues.extend(
                        f"[{entry['title']}] {i}" for i in entry["issues"]
                    )
                _console.print(
                    f"[cyan]  Per-section average: "
                    f"{per_section['average_score']}/10.[/cyan]"
                )
                return {
                    "score": per_section["average_score"],
                    "summary": summary,
                    "strengths": [],
                    "issues": issues[:12],
                    "per_section": per_section,
                }
            _console.print(
                "[yellow]Per-section review unavailable "
                "(no section files or judge unreachable); using legacy "
                "whole-template review.[/yellow]"
            )
        return self._legacy_whole_template_review(
            template_text, required, context
        )

    def _legacy_whole_template_review(
        self,
        template_text: str,
        required: Dict[str, Tuple[str, ...]],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Original whole-template review via the provider cascade.

        Kept ONLY as a fallback when per-section review cannot run. Still
        carries the 12,000-char excerpt limit — the reason it exists is
        precisely why per-section review is preferred.
        """
        excerpt = template_text[:12000]
        background = "\n".join(
            f"- {key}: {context.get(key, '(not specified)')}"
            for key in ("product_name", "target_audience", "pain_point")
        )
        try:
            review_result = self.orchestrator.ask_agent_with_fallback(
                system_prompt=(
                    "You are a strict quality reviewer for paid digital "
                    "products. Judge only what is in front of you."
                ),
                user_prompt=(
                    "Review this digital product template for a paying customer.\n\n"
                    f"Product background:\n{background}\n\n"
                    f"Required sections (a section is satisfied if any of "
                    f"its accepted spellings appears):\n"
                    f"{_format_requirements(required)}\n\n"
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
                expected_sections=required,
            )
            raw = review_result["content"]
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
            "per_section": None,
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

    def generate_improvement_prompt(
        self,
        product_content: str,
        review_feedback: dict,
        product_name: str,
    ) -> str:
        """Generate a surgical improvement prompt for the Builder's revision.

        Only call this when the review failed (verdict below the quality
        threshold). The returned prompt names exact sections/tables/content
        to fix, prioritised with the most critical issues first, capped at
        ~300 words.

        Args:
            product_content: The reviewed template/code (excerpted for the
                model so it can reference concrete sections).
            review_feedback: Dict with ``verdict``, ``scores`` (criterion →
                score), and ``issues`` (list of specific problems).
            product_name: Human-readable product name.

        Returns:
            The improvement prompt string for Builder injection.

        Raises:
            ValueError: If inputs are malformed, the review actually passed,
                or the model returns an empty prompt.
            RuntimeError: If the model call fails.
        """
        if not isinstance(product_content, str) or not product_content.strip():
            raise ValueError("product_content must be a non-empty string.")
        cleaned_name = product_name.strip() if isinstance(product_name, str) else ""
        if not cleaned_name:
            raise ValueError("product_name must be a non-empty string.")
        if not isinstance(review_feedback, dict):
            raise ValueError("review_feedback must be a dict.")
        verdict = str(review_feedback.get("verdict", "")).strip().lower()
        if verdict == "approved":
            raise ValueError(
                "generate_improvement_prompt is only for failed reviews "
                "(verdict below the quality threshold)."
            )
        scores = review_feedback.get("scores", {})
        score_lines = "\n".join(
            f"- {key}: {value}/10"
            for key, value in (scores.items() if isinstance(scores, dict) else [])
        ) or "- (no per-criterion scores recorded)"
        issues = review_feedback.get("issues", [])
        issue_lines = "\n".join(
            f"- {item}" for item in issues if str(item).strip()
        ) or "- (no specific issues recorded)"
        excerpt = product_content.strip()[:6000]
        _console.print(
            "[cyan]Generating improvement prompt for[/cyan] "
            f"[bold]{cleaned_name}[/bold]…"
        )
        try:
            result = self.orchestrator.ask_agent_with_fallback(
                system_prompt=(
                    "You are an expert product improvement specialist. Based "
                    "on the review feedback, write a SPECIFIC, DETAILED "
                    "improvement prompt that will be given to a Builder AI "
                    "to fix the identified issues. The prompt must be:\n"
                    "1. Specific - mention exact sections, tables, or content "
                    "that needs fixing\n"
                    "2. Actionable - clear instructions on WHAT to change and HOW\n"
                    "3. Concise - maximum 300 words\n"
                    "4. Prioritized - most critical issues first\n"
                    "Do NOT write generic advice like 'improve quality'. "
                    "Be surgical and precise. Respond with the prompt text "
                    "ONLY, no preamble."
                ),
                user_prompt=(
                    f"Product name: {cleaned_name}\n\n"
                    f"Review scores (failed criteria need the most attention):\n"
                    f"{score_lines}\n\n"
                    f"Specific issues identified:\n{issue_lines}\n\n"
                    f"Relevant product excerpt:\n{excerpt}"
                ),
                state_manager=self.state_manager,
            )
            prompt = result["content"].strip()
        except Exception as exc:
            raise RuntimeError(
                f"Improvement-prompt generation failed for {cleaned_name!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not prompt:
            raise ValueError("Model returned an empty improvement prompt.")
        if len(prompt.split()) > 350:
            _console.print(
                "[yellow]Improvement prompt exceeds ~300 words; trimming.[/yellow]"
            )
            prompt = " ".join(prompt.split()[:300])
        _console.print(
            f"[green]Improvement prompt ready ({len(prompt.split())} words).[/green]"
        )
        return prompt


def _format_requirements(
    required: Dict[str, Tuple[str, ...]],
) -> str:
    """Render requirement id -> accepted keywords for the review prompt.

    The model sees the *heading* the builder writes plus every tolerated
    spelling, so it does not flag a template that used a legitimate variant.
    """
    if not required:
        return "- (none specified)"
    lines = []
    for section_id, keywords in required.items():
        heading = next(
            (str(d["heading"]) for d in SECTION_DEFINITIONS
             if d["id"] == section_id),
            section_id,
        )
        lines.append(f"- {heading}: any of {list(keywords)}")
    return "\n".join(lines)


def _edge_clipping_check(image_path: str, band: int = 2) -> Dict[str, Any]:
    """Detect content clipped by the image edge (deterministic).

    Renders of a full template keep a margin, so ink sitting in the outermost
    pixel band means a table or paragraph ran off the canvas and was cut —
    the exact defect the vision model is unreliable about spotting.

    Args:
        image_path: Screenshot to inspect.
        band: Width of the edge band in pixels to test for non-white ink.

    Returns:
        ``{"clipped": bool, "sides": [...], "detail": str}``. On any failure
        to read the image, ``clipped`` is False (never blocks a good
        template on a technicality).
    """
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            grey = img.convert("L")
            width, height = grey.size
            if width < 8 or height < 8:
                return {"clipped": False, "sides": [], "detail": ""}
            pixels = grey.load()
            threshold = 235
            sides = []
            for x in range(width - band, width):
                if any(pixels[x, y] < threshold for y in range(0, height, 2)):
                    sides.append("right")
                    break
            for y in range(height - band, height):
                if any(pixels[x, y] < threshold for x in range(0, width, 2)):
                    sides.append("bottom")
                    break
    except Exception:
        return {"clipped": False, "sides": [], "detail": ""}
    if not sides:
        return {"clipped": False, "sides": [], "detail": ""}
    return {
        "clipped": True,
        "sides": sides,
        "detail": (
            "Content is cut off at the "
            + " and ".join(sides)
            + " edge of the render (truncated table/section)."
        ),
    }


def _parse_visual_review(text: str) -> Dict[str, Any]:
    """Parse the vision reply into score/issues/flags.

    Prefers the requested JSON object; falls back to a conservative score
    (5.0) and explicit "no issues detected" when the model ignores the
    format, so a malformed reply never silently reads as a 1/10 failure.
    """
    summary = ""
    flags: Dict[str, bool] = {name: False for name in VISUAL_FLAGS}
    try:
        data = _parse_json(text) if text else None
    except (ValueError, TypeError):
        data = None
    if isinstance(data, dict):
        score = _clamp_visual_score(data.get("visual_score"))
        issues = [str(i)[:300] for i in (data.get("issues") or [])
                  if str(i).strip()][:10]
        for name in VISUAL_FLAGS:
            flags[name] = bool(data.get(name))
        if not issues:
            issues = [f"{name.replace('_', ' ').title()} detected."
                      for name in VISUAL_FLAGS if flags[name]]
        summary = str(data.get("summary", "")).strip()[:300]
        return {
            "visual_score": score,
            "issues": issues,
            "flags": flags,
            "summary": summary,
            "parsed": True,
        }
    return {
        "visual_score": 5.0,
        "issues": [f"Vision model reply was not parseable as JSON; "
                   f"treated as unverified. Raw: {str(text)[:200]}"],
        "flags": flags,
        "summary": "unparsed vision reply",
        "parsed": False,
    }


def _clamp_visual_score(value: Any) -> float:
    """Coerce any model-supplied score into the 1.0-10.0 range."""
    try:
        match = re.search(r"\d+(?:\.\d+)?", str(value))
        if match:
            return min(10.0, max(1.0, float(match.group(0))))
    except (TypeError, ValueError):
        pass
    return 5.0


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
