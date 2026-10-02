"""Builder Agent: turns Scout research into complete digital products.

The builder is the second agent in the swarm pipeline (phase ``"building"``).
It takes a validated opportunity dict — either passed directly or read from
``state_manager.scout_findings`` (the Scout agent's output) — asks Space Bunny
Alpha (via the :class:`~core.orchestrator.Orchestrator`) for a complete,
working product, validates the result, and persists it both in state
(``product_code``) and on disk under ``products/``.

For production use, prefer :meth:`BuilderAgent.build_with_testing`: it builds
the product, validates it with :class:`~core.test_runner.TestRunner`,
rebuilds with the test errors as fix feedback (up to ``max_attempts`` times),
and packages successes with
:class:`~core.product_packager.ProductPackager` into ``packages/*.zip``.

Example usage (building from Scout findings):
    ```python
    from agents.builder import BuilderAgent
    from agents.scout import ScoutAgent
    from core.orchestrator import Orchestrator
    from core.state_manager import StateManager

    state_manager = StateManager(project_id="demo")
    orchestrator = Orchestrator()  # reads OPENROUTER_* from env / .env

    scout = ScoutAgent(orchestrator, state_manager)
    scout.research_opportunities("Notion templates for students")

    builder = BuilderAgent(orchestrator, state_manager)
    result = builder.build_with_testing(max_attempts=3)  # uses 1st finding
    print(result["success"], result["package_path"])

    # Or build a specific opportunity explicitly:
    # product_code = builder.build_product(opportunity={...})
    ```

Sampling temperature (0.7, balancing creativity and reliability) is enforced
centrally by :meth:`Orchestrator.ask_agent`. All terminal output uses
:mod:`rich`. This module is Windows-compatible: :class:`pathlib.Path` for
every file operation and UTF-8 for all reads/writes.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

from core.orchestrator import Orchestrator
from core.placeholders import detect_abandoned_content
from core.product_packager import ProductPackager
from core.sections import SECTION_DEFINITIONS, section_guidance_by_heading
from core.state_manager import StateManager
from core.test_runner import TestRunner
from core.usda_client import USDAClient

__all__ = [
    "BuilderAgent",
    "BUILDER_SYSTEM_PROMPT",
    "NUTRITION_DATA_RULES",
    "PRODUCTS_DIR",
    "FALLBACK_SECTIONS",
    "MAX_WORDS_PER_SECTION",
    "SECTION_WORKSPACE",
    "WORD_CAP_TOLERANCE",
    "WORD_HARD_CAP",
    "MIN_SECTION_LINES",
    "MIN_SECTION_CHARS",
    "MAX_AVG_CHARS_PER_LINE",
    "MAX_TABLE_CELLS_PER_ROW",
    "MAX_COLLAPSED_LINE_LEN",
]

#: Role instructions sent as the system prompt for every build call.
BUILDER_SYSTEM_PROMPT: str = (
    "You are an expert developer and product creator. You build complete, "
    "production-ready digital products (Python scripts, templates, tools) "
    "that deliver exactly what was promised."
)

#: Section titles that trigger nutrition-data instructions. These instructions
#: are advisory: the language model cannot invoke Python methods, so exact
#: USDA compliance is still independently checked after generation.
_NUTRITION_SECTION_KEYWORDS: tuple = (
    "nutrition",
    "nutrient",
    "kcal",
    "calorie",
    "macros",
    "dietary",
)

#: Nutrition-data instructions appended to relevant section prompts.
NUTRITION_DATA_RULES: str = (
    "NUTRITION DATA RULES (CRITICAL):\n"
    "1. State every nutritional energy value per 100 g.\n"
    "2. Do NOT invent or estimate kcal, protein, carbohydrate, or fat values.\n"
    "3. Use the exact USDA value supplied in these requirements when one is "
    "present; otherwise use only values already verified in the supplied "
    "source material.\n"
    "4. If verified nutrition data is unavailable, write "
    "'Nutrition data not available' instead of guessing.\n"
    "5. Preserve table structure, units, and serving wording; do not alter "
    "executable or template semantics to accommodate a nutrition correction."
)


def _with_nutrition_rules(title: str, user_prompt: str) -> str:
    """Append nutrition-data rules to relevant section prompts."""
    if any(
        keyword in (title or "").lower()
        for keyword in _NUTRITION_SECTION_KEYWORDS
    ):
        return f"{user_prompt.strip()}\n\n{NUTRITION_DATA_RULES}"
    return user_prompt

#: Directory (relative to the working directory) holding built product files.
PRODUCTS_DIR: str = "products"

#: Section definitions — the single source of truth shared with the reviewer
#: and the pipeline (see :mod:`core.sections`). Each entry is
#: ``(heading, accepted keyword variants)`` and is derived mechanically from
#: the canonical structure so the two agents can never drift apart again.
FALLBACK_SECTIONS: tuple = tuple(
    (
        str(d["heading"]),
        tuple([str(d["heading"]).lower(), *d["keywords"]]),
    )
    for d in SECTION_DEFINITIONS
)

#: Hard per-section word ceiling (Vector 2). The section prompt states this
#: limit and :meth:`BuilderAgent._sanitize_section` enforces it, so an
#: over-long section is truncated at a sentence boundary instead of silently
#: bloating the assembled product.
MAX_WORDS_PER_SECTION: int = 800

#: Buffer above :data:`MAX_WORDS_PER_SECTION` before truncation kicks in, so
#: a section a few words over the target survives untouched.
WORD_CAP_TOLERANCE: int = 50

#: Absolute ceiling a section may reach before being truncated to
#: :data:`MAX_WORDS_PER_SECTION`.
WORD_HARD_CAP: int = MAX_WORDS_PER_SECTION + WORD_CAP_TOLERANCE

#: Degeneracy floor for generated sections (15% of the word ceiling). A
#: draft below this is a stub, not a section: it is vetoed before scoring
#: so a clean-but-empty draft can never outscore real content. Maiden
#: Voyage 015 shipped a 40-word stub that self-scored 8.0/10 because the
#: ungrounded scorer rewards short, clean text over long, flawed text.
MIN_SECTION_WORDS: int = int(MAX_WORDS_PER_SECTION * 0.15)

#: Renderability floor (newline-collapse gate). A section with fewer than
#: this many lines AND more than :data:`MIN_SECTION_CHARS` characters is
#: almost certainly a newline-collapsed blob that renders as a wall of text.
MIN_SECTION_LINES: int = 15

#: Character count above which a short section is suspicious. Below this a
#: compact-but-valid section (e.g. a short checklist) passes untouched.
MIN_SECTION_CHARS: int = 500

#: Average characters per line above which short content is declared
#: collapsed. Normal structured markdown averages well under 120/line;
#: observed collapsed blobs averaged ~1000/line.
MAX_AVG_CHARS_PER_LINE: float = 200.0

#: Maximum pipe-delimited cells allowed on one line. No legitimate template
#: table exceeds ~14 columns; 50 cells on one line means rows were never
#: separated. (Divider rows count too — they share the same structure.)
MAX_TABLE_CELLS_PER_ROW: int = 16

#: A pipe-bearing line that does NOT start a table row (no leading ``|``,
#: ``#``, ``-``, digit, ``>``) and exceeds this length is collapsed table
#: content glued onto prose. Prose essentially never contains a bare ``|``.
MAX_COLLAPSED_LINE_LEN: int = 300

#: Matches a markdown table divider row: ``|---|---|`` (with optional
#: alignment colons and spacing). Used to anchor table reflow — the divider
#: reveals the column count — and to spot dividers glued mid-line, which
#: proves rows were concatenated without line breaks.
_DIVIDER_RE = re.compile(r"\|(?:\s*:?-{1,}:?\s*\|)+")

#: Lines matching these patterns leak internal swarm state into the customer
#: -facing product and are stripped before a section is written to disk.
INTERNAL_STATE_PATTERNS: tuple = (
    re.compile(r"^\s*[-*>|]*\s*\**\s*planning status\b", re.IGNORECASE),
    # Strip only an *unpopulated* product/project/build name. A named product
    # is customer-facing content (e.g. "Product name: Acme Widgets") and must
    # survive sanitization; a missing name is internal build state.
    re.compile(r"^\s*[-*>|]*\s*\**\s*(?:product|project|build)\s+name\s+"
               r"not\s+set\b", re.IGNORECASE),
    re.compile(r"^\s*[-*>|]*\s*\**\s*(?:product|project|build)\s+name\s*[:=]"
               r"\s*\**\s*not\s+set\s*\**\s*$", re.IGNORECASE),
    re.compile(r"^\s*[-*>|]*\s*\**\s*phase\s*[:=]\s*\w+", re.IGNORECASE),
    re.compile(r"^\s*[-*>|]*\s*\**\s*api[_ ]?calls?\s*[:=]", re.IGNORECASE),
    re.compile(r"^\s*[-*>|]*\s*\**\s*project[_ ]?id\s*[:=]", re.IGNORECASE),
    re.compile(r"^\s*[-*>|]*\s*\**\s*state\s+(file|manager)\b", re.IGNORECASE),
    re.compile(r"^\s*[-*>|]*\s*\**\s*(orchestrator|vision\s?client|"
               r"state_manager|build_pipeline)\b", re.IGNORECASE),
    re.compile(r"\bproduct\s+name\s+not\s+set\b", re.IGNORECASE),
    re.compile(r"\bphase\s*[:=]\s*(research|building|reviewing|publishing|"
               r"completed|failed)\b", re.IGNORECASE),
    re.compile(r"state\.(product_code|scout_findings|review_status)\b"),
    # A build-status *report* ("**Project status:** Research phase; …"). The
    # status label is what makes this a leak, not the word "phase": a project
    # management template may legitimately have a "Research phase" step, so the
    # pattern stays anchored to the label rather than the phase word.
    re.compile(r"^\s*[-*>|]*\s*\**\s*(?:project|build|swarm)?\s*status\s*[:=]",
               re.IGNORECASE),
    # "Scout findings" is internal scout terminology and never legitimate
    # customer copy, in any casing or separator style.
    re.compile(r"\bscout[\s_-]*findings?\b", re.IGNORECASE),
    # The template narrating its own unset identity. Shipped on the Blog Post
    # Outline Generator as "the product name is not set, product code does not
    # exist, and no scout findings are available", which the customer reviewer
    # rejected as making the package look unfinished.
    re.compile(r"\b(?:this\s+)?(?:product|template|package)\b[^.\n]{0,30}?"
               r"\b(?:name|code|title)\b[^.\n]{0,20}?"
               r"\b(?:is|are|was|were|does|do|has|have)\s+"
               r"(?:not\s+)?(?:set|exist|defined|specified|provided|"
               r"available|established)\b",
               re.IGNORECASE),
)

#: Trailing ``- Iteration 2`` / ``(iteration 3)`` marker the model appends to
#: a heading when it echoes the refine prompt ("REFINE the 'Monthly Summary'
#: section (iteration 2)"). It is a build-loop artifact that must never ship.
#:
#: Deliberately anchored to the *end of a heading line* rather than matched
#: anywhere: an unanchored pattern also eats legitimate product copy ("Draft 1
#: initial proposal", "repeat iteration 3 times"), which is worse than the
#: leak it prevents.
_ITERATION_MARKER = re.compile(
    r"^(\s*#{1,6}[ \t]+)(.*?)[ \t]*(?:[—\-–:(][ \t]*)?"
    r"(?:iteration|iter)[ \t]*#?[ \t]*\d+[ \t]*\)?[.:]?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

#: Where per-section Markdown files are persisted during loose assembly:
#: ``state/output/{project_id}/sections/section_{i}.md``.
SECTION_WORKSPACE: str = "state/output"

_console = Console()


def _iteration_limit() -> int:
    """Max iterative refinement cycles per section.

    Reads ``MAX_ITERATIONS_PER_SECTION`` from ``build_pipeline`` lazily
    (avoids a circular import); defaults to 5 when unavailable.
    """
    try:
        from build_pipeline import MAX_ITERATIONS_PER_SECTION

        return max(1, int(MAX_ITERATIONS_PER_SECTION))
    except Exception:
        return 5


def _quality_target() -> float:
    """Target section quality score.

    Reads ``TARGET_SECTION_QUALITY`` from ``build_pipeline`` lazily;
    defaults to 9.0 when unavailable.
    """
    try:
        from build_pipeline import TARGET_SECTION_QUALITY

        return float(TARGET_SECTION_QUALITY)
    except Exception:
        return 9.0


#: Rendered body width in CSS pixels. The screenshot viewport is set to
#: exactly this, so content is laid out to the page rather than overflowing
#: it (see :data:`_BODY_MAX_WIDTH`).
_BODY_MAX_WIDTH: int = 900

#: Stylesheet for exported screenshots/PDFs.
#:
#: The table rules fix the "tables run off the render edge" defect. With the
#: default ``table-layout: auto``, a wide table sizes each column to its
#: *content* (min-content width) and refuses to shrink, so a 9-column table
#: overflows the 900px body and is clipped at the image border. The vision
#: model correctly reads that clipping as "truncated table".
#:
#: ``table-layout: fixed`` makes columns share the container width evenly and
#: never exceed it, while ``overflow-wrap: anywhere`` lets long unbroken cell
#: text (URLs, long facility names) wrap instead of forcing the column wider.
_EXPORT_CSS: str = (
    "*{box-sizing:border-box}"
    "html,body{max-width:100%;overflow-x:hidden}"
    "body{font-family:Arial,sans-serif;"
    f"max-width:{_BODY_MAX_WIDTH}px;"
    "margin:0 auto;padding:24px;color:#111;"
    "font-size:14px;line-height:1.45}"
    "table{border-collapse:collapse;width:100%;max-width:100%;"
    "table-layout:fixed;margin:12px 0}"
    "th,td{border:1px solid #888;padding:6px 8px;text-align:left;"
    "vertical-align:top;overflow-wrap:anywhere;word-break:break-word;"
    "word-wrap:break-word;hyphens:auto}"
    "th{background:#f0f0f0;font-weight:bold}"
    "h1{border-bottom:2px solid #333;padding-bottom:6px}"
    "h2{color:#1a4d8f;margin-top:28px}"
    "h3{margin-top:20px}"
    "code{background:#f4f4f4;padding:1px 4px;overflow-wrap:anywhere}"
    "pre{background:#f7f7f7;padding:10px;overflow-x:auto;white-space:pre-wrap}"
    "ul,ol{padding-left:22px}"
    "img{max-width:100%;height:auto}"
)


def _min_section_quality() -> float:
    """Bar a section must clear to avoid being marked DEGRADED.

    Reads ``MIN_SECTION_QUALITY`` from ``build_pipeline`` lazily (avoids a
    circular import); defaults to 7.0 when unavailable.
    """
    try:
        from build_pipeline import MIN_SECTION_QUALITY

        return float(MIN_SECTION_QUALITY)
    except Exception:
        return 7.0


def _markdown_to_html(content: str) -> str:
    """Convert Markdown to a styled standalone HTML document."""
    try:
        import markdown as _md

        body = _md.markdown(content, extensions=["tables", "fenced_code"])
    except Exception:
        import html as _html

        body = "<pre>" + _html.escape(content) + "</pre>"
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<style>{_EXPORT_CSS}</style>"
        "</head><body>" + body + "</body></html>"
    )


#: Hard ceiling on the capture viewport height, in CSS pixels.
#:
#: Chromium cannot allocate a surface taller than this, and a capture that
#: exceeds it comes back truncated or mis-composited. Section assembly keeps
#: products well below the ceiling, so hitting it means a runaway render.
MAX_CAPTURE_HEIGHT: int = 30000


def _playwright_screenshot(document: str, target: Path) -> bool:
    """Full-page PNG screenshot via headless Chromium. Returns success.

    The page is captured by growing the viewport to the measured document
    height rather than by passing ``full_page=True``. On tall products
    ``full_page`` mis-composites the lower portion of the page: measured on a
    9,136px product it first diverged from a correct render at row 7,934 and
    smeared 84 extra rows of ink into the last 1,200px, driving content into
    the bottom image edge. The pixel gate correctly read that as
    "content is cut off at the bottom edge", so the defect was invisible
    upstream and only surfaced as a spurious visual FAIL.

    Growing the viewport renders the same page in one pass and leaves the
    body's own bottom padding intact, so the final text line keeps its margin
    instead of touching the image border.
    """
    from playwright.sync_api import sync_playwright

    playwright = sync_playwright().start()
    try:
        browser = playwright.chromium.launch(headless=True)
        try:
            # Viewport width matches the CSS body width, so the screenshot is
            # exactly as wide as the layout: a table that still overflowed
            # would be clipped here and flagged by the vision gate.
            page = browser.new_page(
                viewport={"width": _BODY_MAX_WIDTH, "height": 900}
            )
            page.set_content(document, wait_until="networkidle")

            height = page.evaluate(
                "() => Math.max("
                "document.documentElement.scrollHeight,"
                "document.body.scrollHeight)"
            )
            height = min(int(height) + 1, MAX_CAPTURE_HEIGHT)
            page.set_viewport_size({"width": _BODY_MAX_WIDTH, "height": height})
            # The resize reflows the page; capture only once the measured
            # height has settled, or the shot can be a pixel short.
            page.wait_for_function(
                "h => Math.max(document.documentElement.scrollHeight,"
                "document.body.scrollHeight) <= h",
                arg=height,
            )
            page.screenshot(path=str(target))
        finally:
            browser.close()
    finally:
        playwright.stop()
    return target.is_file()


def _playwright_pdf(document: str, target: Path) -> bool:
    """Render HTML to PDF via headless Chromium. Returns success."""
    from playwright.sync_api import sync_playwright

    playwright = sync_playwright().start()
    try:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page(
                viewport={"width": _BODY_MAX_WIDTH, "height": 900}
            )
            page.set_content(document, wait_until="networkidle")
            page.pdf(
                path=str(target),
                format="A4",
                margin={"top": "16mm", "bottom": "16mm",
                        "left": "14mm", "right": "14mm"},
                print_background=True,
            )
        finally:
            browser.close()
    finally:
        playwright.stop()
    return target.is_file()


def _pil_render_text(content: str, target: Path) -> bool:
    """Simple Pillow text rendering fallback (always dependency-light)."""
    from PIL import Image, ImageDraw

    lines: List[str] = []
    for raw_line in content.splitlines():
        while len(raw_line) > 100:
            lines.append(raw_line[:100])
            raw_line = raw_line[100:]
        lines.append(raw_line)
    width, line_height, margin = 1200, 15, 20
    height = margin * 2 + line_height * max(len(lines), 1)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    y = margin
    for line in lines:
        draw.text((margin, y), line, fill="black")
        y += line_height
    if target.suffix.lower() == ".pdf":
        image.save(str(target), "PDF")
    else:
        image.save(str(target), "PNG")
    return target.is_file()


class BuilderAgent:
    """Build complete digital products from Scout opportunities.

    Args:
        orchestrator: The swarm's :class:`Orchestrator` used for all model
            calls (usage is tracked automatically via the state manager).
        state_manager: The project's :class:`StateManager` where the built
            product (``product_code`` / ``product_name``) is persisted and
            the phase is set to ``"building"``.

    Example:
        ```python
        builder = BuilderAgent(orchestrator, state_manager)
        code = builder.build_product()
        ```
    """
    def __init__(
        self,
        orchestrator: Optional[Orchestrator] = None,
        state_manager: Optional[StateManager] = None,
    ) -> None:
        # Duck-typed validation (rather than isinstance) so the agent accepts
        # the real Orchestrator/StateManager as well as test doubles/mocks
        # that expose the same interface. Both are optional so lightweight
        # helpers like export_to_image() work without a full swarm.
        if orchestrator is not None:
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
        if state_manager is not None:
            for name in ("update_phase", "get_state", "save", "log_api_call"):
                if not hasattr(state_manager, name) or not callable(
                    getattr(state_manager, name, None)
                ):
                    raise TypeError(
                        "state_manager must expose the StateManager interface "
                        f"(missing callable {name!r}); "
                        f"got {type(state_manager).__name__}."
                    )
        self.orchestrator: Optional[Orchestrator] = orchestrator
        self.state_manager: Optional[StateManager] = state_manager
        self.products_dir: Path = Path(PRODUCTS_DIR)
        # Chronological per-attempt records from build_with_testing().
        # Each entry: attempt, product_name, product_type, build_ok,
        # tests_passed, package_path, error, timestamp.
        self._build_history: List[Dict[str, Any]] = []
        # Section titles that failed completeness validation twice.
        self._degraded_sections: List[str] = []
        # Latest per-section refinement feedback, keyed by section title.
        self._section_feedback: Dict[str, str] = {}

    def _lookup_nutrition(self, food_name: str) -> Optional[Dict[str, Any]]:
        """Look up authoritative USDA nutrition data during generation.

        This helper gives orchestrated Python code direct access to USDA
        values. It does not give the language model itself a callable tool:
        prompt text can request exact values, but only post-generation
        validation can prove that the model complied.

        Args:
            food_name: Food wording, for example ``"apple"``.

        Returns:
            Normalized USDA nutrition record, or None when USDA has no key,
            no reliable match, no energy value, or a transport failure. Callers
            must treat None as "nutrition data not available," not as
            permission to invent values.
        """
        cleaned = food_name.strip() if isinstance(food_name, str) else ""
        if not cleaned:
            return None
        try:
            lookup = USDAClient().lookup_food(cleaned, page_size=10)
        except (EnvironmentError, LookupError, RuntimeError, ValueError) as exc:
            _console.print(
                f"[yellow]USDA lookup unavailable for {cleaned!r}: "
                f"{type(exc).__name__}[/yellow]"
            )
            return None
        details = lookup.get("details", {})
        return {
            "food_name": details.get("description") or cleaned,
            "query": lookup.get("query", cleaned),
            "kcal_per_100g": details.get("energy_kcal_per_100g"),
            "protein_g_per_100g": details.get("protein_g_per_100g"),
            "carbs_g_per_100g": details.get("carbohydrate_g_per_100g"),
            "fat_g_per_100g": details.get("fat_g_per_100g"),
            "data_type": details.get("data_type"),
            "match_score": lookup.get("match_score"),
            "fdc_id": details.get("fdc_id"),
            "url": details.get("food_url"),
        }

    def build_product(
        self,
        opportunity: Dict[str, Any] = None,
        feedback: Optional[str] = None,
        improvement_prompt: Optional[str] = None,
    ) -> str:
        """Build a complete product for ``opportunity``.

        .. deprecated::
            **Vector 2 removed the whole-product generation path.** A single
            unbounded request is what caused truncated products and lost
            section boundaries, so this method now delegates to strict
            section-by-section assembly
            (:meth:`build_product_with_fallback`). It is kept only so old
            callers keep working — prefer calling
            :meth:`build_product_with_fallback` directly.

        Args:
            opportunity: Validated opportunity dict as produced by
                :meth:`ScoutAgent.research_opportunities` (keys such as
                ``product_name``, ``target_audience``, ``pain_point``,
                ``estimated_price``, ``unique_value_proposition``,
                ``estimated_build_time``). If ``None``, the first opportunity
                in ``state_manager.scout_findings["opportunities"]`` is used.
            feedback: Optional fix feedback from a previous failed attempt
                (e.g. test errors). When given, it is appended to the user
                prompt so the model returns the complete corrected product.
            improvement_prompt: Optional reviewer-generated revision brief.
                When given, it is prepended to the build instructions as an
                IMPORTANT REVISION INSTRUCTIONS block so the model fixes
                the reviewed defects first. Omitted on first attempts.

        Returns:
            The generated product content (markdown fences stripped) as a
            string. It is also persisted via :meth:`save_product`.

        Raises:
            ValueError: If no opportunity is available, the opportunity is
                malformed, or the generated product fails validation (with
                the reason included).
            RuntimeError: If the underlying API call fails.

        Side effects:
            Validates the output, then saves it to ``product_code`` + a file
            under ``products/``, sets ``product_name`` and the phase to
            ``"building"``, and persists state.
        """
        if opportunity is None:
            opportunity = self._load_first_opportunity()
        else:
            opportunity = self._checked_opportunity(opportunity)

        # Vector 2: whole-product generation is disabled. Route straight to
        # the bounded, file-backed section assembly path.
        return self.build_product_with_fallback(
            opportunity=opportunity,
            feedback=feedback,
            improvement_prompt=improvement_prompt,
        )

    def build_product_with_fallback(
        self,
        opportunity: Dict[str, Any] = None,
        feedback: Optional[str] = None,
        improvement_prompt: Optional[str] = None,
        sections: Optional[tuple] = None,
    ) -> str:
        """Build a complete template via strict section-by-section assembly.

        **Vector 2 — there is no full-generation path.** Whole-product
        generation is architecturally disabled: a single unbounded request is
        exactly what produced truncated products, so the builder only ever
        asks for one bounded section at a time and assembles the result from
        files on disk.

        Flow:

        1. Clear any stale section files for this project.
        2. For each of :data:`FALLBACK_SECTIONS`, generate that one section
           with a hard ``MAXIMUM {MAX_WORDS_PER_SECTION} WORDS`` limit,
           validate it, and persist it to
           ``state/output/{project_id}/sections/section_{i}.md``.
        3. Read the section files back in index order and concatenate them
           into the final template.

        Because sections are persisted as they succeed, a failure in section
        3 leaves sections 1, 2 (and any later ones already written) intact
        on disk and resumable.

        Args:
            opportunity: Opportunity dict (same rule as :meth:`build_product`;
                ``None`` uses the first Scout finding in state).
            feedback: Optional extra instructions applied to every section
                (e.g. reviewer notes).
            improvement_prompt: Optional reviewer-generated revision brief,
                used as refinement context in every section's iteration 1.
            sections: Optional ``((title, keywords), ...)`` section list.
                Defaults to :data:`FALLBACK_SECTIONS`. Lets capability tests
                and future niches assemble different section sets through
                the same file-backed pipeline.

        Returns:
            The assembled template (saved via :meth:`save_product`).

        Raises:
            ValueError: If the opportunity is malformed or a section fails
                validation after its retry budget.
            RuntimeError: If an API call fails.
        """
        if opportunity is None:
            opportunity = self._load_first_opportunity()
        else:
            opportunity = self._checked_opportunity(opportunity)
        _console.print(
            "[cyan]Strict section-by-section assembly "
            "(full generation disabled).[/cyan]"
        )
        return self._build_section_by_section(
            opportunity, improvement_prompt=improvement_prompt,
            extra_feedback=feedback, sections=sections,
        )

    def _validate_section_complete(self, content: str, section_name: str) -> bool:
        """Check if a generated section ends cleanly (not truncated).

        Rules: non-empty; final character is proper closing punctuation
        (``.!?)|``); no half-written markdown table row in the last 5
        lines (starts with ``|`` but doesn't end with one); within
        1.5x :data:`MAX_WORDS_PER_SECTION` (headroom above the 800-word
        prompt cap so a slightly-long section still validates).
        """
        if not content or not content.strip():
            return False
        stripped = content.strip()

        # 1. Must end with proper sentence punctuation, not mid-word/mid-sentence
        if stripped[-1] not in ".!?)|":
            return False

        # 2. Check for incomplete markdown tables in the last 5 lines
        #    (a line starting with | but not ending with | = truncated row)
        for line in stripped.split("\n")[-5:]:
            row = line.strip()
            if row.startswith("|") and not row.endswith("|"):
                return False

        # 3. Word count sanity (Vector 2 hard cap plus validation headroom)
        if len(stripped.split()) > MAX_WORDS_PER_SECTION * 3 // 2:
            return False

        return True

    def _call_section_model(self, title: str, user_prompt: str) -> str:
        """Send one section prompt via the fallback-capable orchestrator.

        Returns the fence-stripped, non-empty section body.

        Raises:
            RuntimeError: If the API call fails.
            ValueError: If the section comes back empty.
        """
        user_prompt = _with_nutrition_rules(title, user_prompt)
        try:
            section_result = self.orchestrator.ask_agent_with_fallback(
                system_prompt=(
                    "You are building ONE section of a template. "
                    "Generate ONLY this section, complete and detailed."
                ),
                user_prompt=user_prompt,
                state_manager=self.state_manager,
                expected_sections=[title],
            )
            raw = section_result["content"]
        except Exception as exc:
            raise RuntimeError(
                f"Section build failed for '{title}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        body = self._extract_product_content(raw).strip()
        if not body:
            raise ValueError(
                f"Section '{title}' came back empty; aborting fallback."
            )
        return body

    def _generate_section(
        self,
        title: str,
        product_name: str,
        audience: str,
        concise_retry: bool = False,
    ) -> str:
        """Generate one section body via the fallback-capable orchestrator.

        Args:
            concise_retry: When True, use the tightened retry prompt
                (complete but concise, under 600 words, closed tables,
                complete final sentence).

        Raises:
            RuntimeError: If the API call fails.
            ValueError: If the section comes back empty.
        """
        if concise_retry:
            user_prompt = (
                f"Your previous attempt was truncated. Generate a COMPLETE "
                f"but CONCISE version of '{title}' in under 600 words for a "
                f"{product_name} ({audience}). Every table must be fully "
                "closed. End with a complete sentence. Output ONLY the "
                "section content."
            )
        else:
            user_prompt = (
                f"Generate the '{title}' section for a "
                f"{product_name} ({audience}). Use Markdown with full "
                "tables/checklists and realistic example rows. "
                f"MAXIMUM {MAX_WORDS_PER_SECTION} WORDS. DO NOT EXCEED. "
                "Complete and detailed — no truncation, no '...' "
                "placeholders. Output ONLY the section content."
            )
        return self._call_section_model(title, user_prompt)

    def _get_previous_feedback(self, section_name: str) -> str:
        """Return stored refinement feedback for a section (or empty)."""
        return self._section_feedback.get(section_name, "")

    @staticmethod
    def _degeneracy_veto(content: str, section_name: str) -> Optional[str]:
        """Return a veto reason when a draft is a stub, else None.

        A draft is degenerate when it holds fewer than
        :data:`MIN_SECTION_WORDS` words or when
        :func:`core.placeholders.detect_abandoned_content` flags it.
        Vetoed drafts score 0.0 implicitly (they never reach the scorer)
        and therefore can never be selected as the best version.
        """
        words = len((content or "").split())
        if words < MIN_SECTION_WORDS:
            return (
                f"section '{section_name}' is a {words}-word stub "
                f"(minimum {MIN_SECTION_WORDS} words)"
            )
        abandoned = detect_abandoned_content(content or "")
        if abandoned:
            return (
                f"section '{section_name}' contains abandoned content: "
                f"{abandoned[0][:120]}"
            )
        return None

    def _iterative_section_build(
        self,
        section_name: str,
        section_requirements: str,
        improvement_context: str = "",
        max_iterations: int = 5,
        target_quality: float = 9.0,
    ) -> dict:
        """Iteratively build and refine one section until it meets quality.

        Each iteration generates (or refines), completeness-validates with
        a strict retry, scores via the model, and banks improvement feedback
        for the next pass. Stops early at ``target_quality``.

        Args:
            section_name: Section title, e.g. ``"Invoice Tracking"``.
            section_requirements: What the section must contain.
            improvement_context: Revision brief carried into iteration 1
                (e.g. reviewer notes); per-iteration feedback thereafter.
            max_iterations: Refinement cycle cap.
            target_quality: Score (0–10) that stops iteration early.

        Returns:
            Dict with ``content`` (best version, may be None if every
            attempt came back empty), ``final_score``, ``iterations_used``,
            and ``quality_history`` (per-iteration score + word count).

        Raises:
            RuntimeError: If an API call fails.
        """
        best_content = None
        best_score = 0.0
        quality_history = []

        for iteration in range(max_iterations):
            # Step 1: Generate or refine the section
            if iteration == 0:
                prompt = (
                    f"Generate the '{section_name}' section based on these "
                    f"requirements:\n{section_requirements}"
                )
            else:
                previous_feedback = (
                    improvement_context or self._get_previous_feedback(section_name)
                )
                prompt = f"""
REFINE the '{section_name}' section (iteration {iteration + 1}).

Previous version had these issues:
{previous_feedback}

Requirements:
{section_requirements}

IMPORTANT:
- Address ALL feedback from the previous version
- MAXIMUM {MAX_WORDS_PER_SECTION} WORDS. DO NOT EXCEED.
- Every table must be complete
- End with a clear conclusion
"""
            content = self._call_section_model(section_name, prompt)

            # Step 2: Validate completeness
            if not self._validate_section_complete(content, section_name):
                # Force a retry with stricter constraints
                content = self._call_section_model(
                    section_name,
                    f"Generate '{section_name}' in under 600 words. "
                    "ALL tables must be complete. End with a sentence.",
                )

            # Step 2b: Degeneracy veto — a stub must never reach scoring.
            # The model scorer rewards short, clean text, so without this a
            # 40-word stub can outscore a 1,000-word draft (MV015 proved it).
            veto_reason = self._degeneracy_veto(content, section_name)
            if veto_reason is not None:
                _console.print(
                    f"[yellow]  '{section_name}' iteration {iteration + 1}: "
                    f"VETOED ({veto_reason}); regenerating.[/yellow]"
                )
                quality_history.append({
                    "iteration": iteration + 1,
                    "score": 0.0,
                    "word_count": len(content.split()),
                })
                improvement_context = (
                    f"{veto_reason} Regenerate the complete section; "
                    "do not return a stub."
                )
                continue

            # Step 3: Score this version
            score = self._score_section_quality(content, section_name)
            quality_history.append({
                "iteration": iteration + 1,
                "score": score,
                "word_count": len(content.split()),
            })
            _console.print(
                f"[cyan]  '{section_name}' iteration {iteration + 1}: "
                f"score {score}/10.[/cyan]"
            )

            # Step 4: Track the best version
            if score > best_score:
                best_score = score
                best_content = content

            # Step 5: Check if we've reached target quality
            if score >= target_quality:
                break

            # Step 6: Get improvement feedback for next iteration
            improvement_context = self._generate_improvement_feedback(
                content, section_name, score
            )
            self._section_feedback[section_name] = improvement_context

        return {
            "content": best_content,
            "final_score": best_score,
            "iterations_used": len(quality_history),
            "quality_history": quality_history,
        }

    def _score_section_quality(self, content: str, section_name: str) -> float:
        """Score a section 0–10 via the model (default 5.0 on parse failure).

        Criteria: completeness, clarity, usefulness, formatting.
        """
        prompt = f"""
Score this '{section_name}' section on a scale of 0-10 based on:
1. Completeness (does it cover all necessary aspects?)
2. Clarity (is it easy to understand?)
3. Usefulness (would this actually help someone?)
4. Formatting (are tables complete, headings clear?)

Content to evaluate:
{content[:2000]}

Respond with ONLY a number between 0 and 10.
"""
        response = self.orchestrator.ask_primary_agent(
            "You are a quality assessor.", prompt, self.state_manager
        )

        try:
            match = re.search(r"\b(\d+(?:\.\d+)?)\b", response)
            if match:
                score = float(match.group(1))
                return min(10.0, max(0.0, score))
        except Exception:
            pass

        return 5.0  # Default middle score if parsing fails

    def _generate_improvement_feedback(
        self, content: str, section_name: str, current_score: float
    ) -> str:
        """Generate top-3 actionable fixes to lift a section toward 9/10."""
        prompt = f"""
This '{section_name}' section scored {current_score}/10.

Content:
{content[:2000]}

What are the TOP 3 specific improvements needed to reach 9/10?
Be concise and actionable. Format as:
1. [Specific issue] - [Specific fix]
2. [Specific issue] - [Specific fix]
3. [Specific issue] - [Specific fix]
"""
        try:
            return self.orchestrator.ask_primary_agent(
                "You are a quality improvement specialist.",
                prompt,
                self.state_manager,
            )
        except Exception as exc:
            _console.print(
                f"[yellow]Feedback generation failed for '{section_name}' "
                f"({exc}); using generic guidance.[/yellow]"
            )
            return (
                "Improve completeness, clarity, and formatting; "
                "ensure tables are closed."
            )

    def _sections_dir(self) -> Path:
        """Per-project directory holding assembled section files."""
        project_id = getattr(self.state_manager, "project_id", None) or "default"
        return Path(SECTION_WORKSPACE) / str(project_id) / "sections"

    def _clear_section_files(self) -> None:
        """Delete stale section files so a retry never mixes old and new."""
        try:
            sections_dir = self._sections_dir()
            if sections_dir.is_dir():
                for stale in sections_dir.glob("section_*.md"):
                    stale.unlink()
        except OSError as exc:
            _console.print(
                f"[yellow]Could not clear stale section files: {exc}[/yellow]"
            )

    def _write_section_file(
        self, index: int, title: str, body: str
    ) -> Path:
        """Persist one validated section to its own file (loose assembly).

        Writing each section immediately is what makes a mid-build failure
        cheap: everything generated so far survives on disk, so a retry only
        has to redo the section that failed.

        Args:
            index: 1-based section position.
            title: Section display title.
            body: Section body (heading already stripped).

        Returns:
            The :class:`pathlib.Path` of the written ``section_{index}.md``.

        Raises:
            IOError: If the file cannot be written.
        """
        sections_dir = self._sections_dir()
        try:
            sections_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise IOError(
                f"Could not create sections directory '{sections_dir}': {exc}"
            ) from exc
        target = sections_dir / f"section_{index}.md"
        payload = f"<!-- section: {title} -->\n\n## {title}\n\n{body.strip()}\n"
        try:
            target.write_text(payload, encoding="utf-8")
        except OSError as exc:
            raise IOError(
                f"Could not write section file '{target}': {exc}"
            ) from exc
        return target

    def _sanitize_section(self, content: str) -> str:
        """Strip internal artifacts and enforce the hard word cap.

        Runs on every section before it is written to disk. Four passes:

        1. **HTML comments** — removes ``<!-- ... -->`` blocks, including the
           builder's own ``REVIEW NOTE`` markers, which are internal
           annotations and must never reach a customer.
        2. **Internal state lines** — drops lines that expose swarm state
           (planning status, phase, API call counts, project ids). These
           leaked into the Maiden Voyage 2.0 product as customer-facing text.
        3. **Iteration markers** — removes the ``— Iteration 2`` suffix the
           model appends to a heading when echoing the refine prompt. It is a
           build-loop artifact, and it also defeats the echoed-heading dedup
           in :meth:`_build_section_by_section`, which shipped a duplicated
           ``## Monthly Summary`` heading on the budget tracker.
        4. **Word cap** — sections above :data:`WORD_HARD_CAP` are truncated
           to :data:`MAX_WORDS_PER_SECTION` at the last complete paragraph or
           sentence, never mid-line inside a table. Newlines are preserved
           so truncation cannot itself collapse the layout.
        5. **Renderability** — rejects newline-collapsed blobs (a 5,000-char
           single line renders as a wall of text no gate otherwise catches).
           A reflow is attempted once; if the content still fails, a
           ``ValueError`` is raised so the section is regenerated instead of
           shipped broken.

        Args:
            content: Raw section body from the model.

        Returns:
            The cleaned section body. Empty input returns an empty string.

        Raises:
            ValueError: If the content is unrenderable even after reflow.
        """
        if not content or not content.strip():
            return ""

        # 1. HTML comments (DOTALL: the model spans them across lines).
        cleaned = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)

        # 2. Internal state lines.
        kept_lines = []
        for line in cleaned.splitlines():
            if any(pattern.search(line) for pattern in INTERNAL_STATE_PATTERNS):
                continue
            kept_lines.append(line)
        cleaned = "\n".join(kept_lines)

        # 3. Iteration markers leaked from the refine prompt into headings.
        cleaned = _ITERATION_MARKER.sub(r"\1\2", cleaned)

        # 4. Hard word cap (structure-preserving: keeps original newlines).
        words = cleaned.split()
        if len(words) > WORD_HARD_CAP:
            cleaned = self._truncate_to_word_cap(cleaned)
            _console.print(
                f"[yellow]  Word cap enforced: truncated to "
                f"{len(cleaned.split())} words (limit "
                f"{MAX_WORDS_PER_SECTION}).[/yellow]"
            )

        # 5. Renderability: detect collapse, reflow once, re-check.
        renderable, reason = self._check_renderability(cleaned)
        if not renderable:
            _console.print(
                f"[yellow]  Renderability check failed ({reason}); "
                f"attempting reflow…[/yellow]"
            )
            cleaned = self._reflow_content(cleaned)
            renderable, reason = self._check_renderability(cleaned)
            if renderable:
                _console.print(
                    f"[green]  Reflow succeeded "
                    f"({len(cleaned.splitlines())} lines).[/green]"
                )
            else:
                raise ValueError(
                    f"Renderability failed after reflow: {reason}"
                )
        return cleaned.strip()

    @staticmethod
    def _truncate_to_word_cap(text: str, limit: int = MAX_WORDS_PER_SECTION) -> str:
        """Trim ``text`` to ``limit`` words at a safe boundary.

        Prefers the last blank-line (paragraph) break, then the last sentence
        terminator, then a hard cut — so tables and sentences are never cut in
        half. Original newlines are preserved throughout: an earlier version
        joined words with spaces and turned every truncated section into a
        single-line blob (the very collapse this module now guards against).

        Args:
            text: Text to trim.
            limit: Maximum words to keep.

        Returns:
            The trimmed text.
        """
        words = text.split()
        if len(words) <= limit:
            return text
        # Walk whitespace-preserving tokens so newlines survive the cut.
        kept: List[str] = []
        count = 0
        for token in re.findall(r"\S+\s*", text):
            if token.strip():
                count += 1
                if count > limit:
                    break
            kept.append(token)
        head = "".join(kept)

        # 1. Last paragraph break, but only in the latter half of the
        #    window — tables use single newlines, so the last "\n\n" in a
        #    table-heavy section is the one after the heading, and cutting
        #    there would gut the section to its title.
        paragraph = head.rfind("\n\n", max(0, int(len(head) * 0.5)))
        if paragraph > 0:
            head = head[:paragraph]
        else:
            # 2. Last complete table row (a "|" line closed on both ends).
            #    This is the structurally safe cut for table-dense sections.
            rows = [
                match.end()
                for match in re.finditer(r"(?m)^\|.*\|\s*$", head)
            ]
            if rows and rows[-1] > len(head) * 0.5:
                head = head[:rows[-1]]
            else:
                # 3. Last sentence terminator clear of table pipes.
                sentence = max(
                    head.rfind(". "), head.rfind("! "), head.rfind("? "),
                    head.rfind(".\n"), head.rfind("!\n"), head.rfind("?\n"),
                )
                if sentence > len(head) * 0.5 and "|" not in head[
                        sentence:sentence + 3]:
                    head = head[: sentence + 1]

        # Strip a DANGLING partial row only: a last line that opens a table
        # row without closing it. A complete `|...|` row keeps its pipes.
        stripped = head.rstrip()
        last_line = stripped.split("\n")[-1].strip()
        if last_line.startswith("|") and not last_line.endswith("|"):
            stripped = stripped.rstrip("|").rstrip()
        return stripped + "\n"

    @staticmethod
    def _check_renderability(content: str) -> tuple:
        """Check content renders as readable structure, not a wall of text.

        Three structural tests (thresholds are module constants so a failing
        section can be tuned without touching logic):

        1. **Density** — fewer than :data:`MIN_SECTION_LINES` lines holding
           more than :data:`MIN_SECTION_CHARS` characters at over
           :data:`MAX_AVG_CHARS_PER_LINE` chars/line is a collapsed blob.
        2. **Row width** — any line with more than
           :data:`MAX_TABLE_CELLS_PER_ROW` pipe-delimited cells was never
           split into rows (no legitimate table is that wide).
        3. **Glued structure** — headings (``## ``), table dividers
           (``|---|``), or checklist items (``- [ ]``) appearing mid-line, or
           a 300+ char pipe-bearing line that starts as prose, means rows
           were concatenated without newlines.

        Args:
            content: Section body to inspect.

        Returns:
            ``(True, "Renderable")`` or ``(False, reason)``.
        """
        if not content or not content.strip():
            return False, "Empty content"
        stripped = content.strip()
        lines = stripped.split("\n")
        total_chars = len(stripped)

        # 1. Density: too much content on too few lines.
        if (len(lines) < MIN_SECTION_LINES and total_chars > MIN_SECTION_CHARS
                and total_chars / len(lines) > MAX_AVG_CHARS_PER_LINE):
            return False, (
                f"Content collapsed: {total_chars} chars on {len(lines)} "
                f"lines (avg {total_chars / len(lines):.0f}/line)"
            )

        for line in lines:
            text = line.strip()
            if not text:
                continue
            # 2. Row width: more cells than any legitimate table.
            cells = text.count("|") - 1
            if cells > MAX_TABLE_CELLS_PER_ROW:
                return False, (
                    f"Table row collapsed: {len(text)} chars, "
                    f"{cells} cells on one line"
                )
            # 3a. Heading glued mid-line ("prose ## Heading ..." or a cell
            #     run ending then a heading starting without a break).
            if re.search(r"\S\s+(#{1,4}\s+\S)", text):
                return False, (
                    f"Heading glued mid-line: {text[:80]!r}…"
                )
            # 3b. Table divider mid-line (a second table, or rows, glued
            #     onto preceding content without a line break).
            for match in _DIVIDER_RE.finditer(text):
                if match.start() > 0:
                    return False, (
                        f"Table divider glued mid-line: {text[:80]!r}…"
                    )
            # 3c. Checklist item glued mid-line.
            check = re.search(r"-\s*\[[ xX]\]", text)
            if check and check.start() > 0 and len(text[:check.start()].strip()) > 3:
                return False, (
                    f"Checklist item glued mid-line: {text[:80]!r}…"
                )
            # 3d. Long pipe-bearing line that starts as prose (collapsed
            #     table content glued onto a paragraph or heading line).
            if ("|" in text and len(text) > MAX_COLLAPSED_LINE_LEN
                    and not re.match(r"^[|#\-0-9*>\s]", text)):
                return False, (
                    f"Table content glued onto prose "
                    f"({len(text)} chars): {text[:80]!r}…"
                )
        return True, "Renderable"

    def _reflow_content(self, content: str) -> str:
        """Re-flow newline-collapsed content into readable structure.

        Splits headings, checklist items, and table rows onto their own
        lines, then breaks very long prose lines at safe sentence
        boundaries. Idempotent: already-structured content passes through
        unchanged (verified by re-running :meth:`_check_renderability`).

        Args:
            content: Collapsed section body.

        Returns:
            The re-flowed body.
        """
        text = content
        # 1. Headings onto their own lines.
        text = re.sub(r"[ \t]*(#{1,4}\s+)", r"\n\1", text)
        # 2. Checklist items onto their own lines.
        text = re.sub(r"[ \t]*(-\s*\[[ xX]\]\s*)", r"\n\1", text)
        # 3. Table runs split into rows (divider-anchored, column-aware).
        lines_out: List[str] = []
        for line in text.split("\n"):
            lines_out.extend(self._split_table_runs(line))
        # 4. Very long prose lines split at safe sentence boundaries.
        final: List[str] = []
        for line in lines_out:
            final.extend(self._split_long_prose(line))
        text = re.sub(r"\n{3,}", "\n\n", "\n".join(final))
        return text.strip()

    @classmethod
    def _split_table_runs(cls, line: str) -> List[str]:
        """Split one line's collapsed table runs into separate rows.

        A divider row (``|---|``) anchors the split: the column count comes
        from the divider, the trailing ``|cell|…|`` run before it becomes
        the header row, and the leading run after it is chunked into rows
        of that width. Surrounding prose/headings are emitted untouched and
        any remainder is processed recursively (a blob may hold several
        tables). Lines without dividers pass through unchanged.

        Args:
            line: One (possibly collapsed) line.

        Returns:
            List of lines with table rows separated.
        """
        match = _DIVIDER_RE.search(line)
        if not match:
            return [line]
        cols = max(1, match.group(0).count("|") - 1)
        pre, divider, post = (
            line[:match.start()], match.group(0).strip(), line[match.end():]
        )
        out: List[str] = []
        # Head side: a trailing cell-run is the header row; anything before
        # it (prose, a heading) is emitted first (recursively, in case it
        # holds an earlier table).
        header_run = re.search(r"((?:\|[^|]*)+\|?)\s*$", pre)
        head_before = pre[:header_run.start()] if header_run else pre
        if head_before.strip():
            out.extend(cls._split_table_runs(head_before.strip()))
        if header_run and header_run.group(1).strip().strip("|").strip():
            out.append(header_run.group(1).strip())
        out.append(divider)
        # Data side: leading cell-run chunked into rows of `cols` cells.
        data_run = re.match(r"\s*((?:\|[^|]*)+\|?)", post)
        rest = post
        if data_run and data_run.group(1).strip().strip("|").strip():
            inner = data_run.group(1).strip().strip("|")
            cells = [cell.strip() for cell in inner.split("|")]
            for start in range(0, len(cells), cols):
                chunk = [c for c in cells[start:start + cols]]
                if any(chunk):
                    out.append("| " + " | ".join(chunk) + " |")
            rest = post[data_run.end():]
        if rest.strip():
            # Recurse for further tables glued after this one. Guarded: the
            # remainder is strictly shorter, so this always terminates.
            out.extend(cls._split_table_runs(rest.strip()))
        return out or [line]

    @staticmethod
    def _split_long_prose(line: str, limit: int = MAX_COLLAPSED_LINE_LEN) -> List[str]:
        """Split an overlong prose line at safe sentence boundaries.

        Only sentence ends (``. `` + capital) following an 80+ char segment
        qualify, so abbreviations (``St. Marys``) and short fragments never
        split. Table rows, headings, list items, and code are returned
        untouched.

        Args:
            line: One line that may be too long.
            limit: Length above which splitting is attempted.

        Returns:
            One or more lines.
        """
        if (len(line) <= limit or "|" in line
                or re.match(r"^\s*(#{1,6}|[-*>]|\d+[.)]|```|\|)", line)):
            return [line]
        parts: List[str] = []
        start = 0
        for match in re.finditer(r"\.\s+(?=[A-Z0-9\"'])", line):
            end = match.start() + 1  # keep the period
            if end - start >= 80:
                parts.append(line[start:end].strip())
                start = match.end()
        tail = line[start:].strip()
        if tail:
            parts.append(tail)
        return parts or [line]

    def _read_section_files(self) -> List[Path]:
        """Return the section files that exist, ordered by index."""
        sections_dir = self._sections_dir()
        if not sections_dir.is_dir():
            return []
        return sorted(
            sections_dir.glob("section_*.md"),
            key=lambda p: int(re.search(r"section_(\d+)\.md", p.name).group(1))
            if re.search(r"section_(\d+)\.md", p.name) else 0,
        )

    def _assemble_from_files(
        self, product_name: str, sections: Optional[tuple] = None
    ) -> str:
        """Concatenate persisted section files into the final template.

        Args:
            product_name: H1 title for the assembled document.
            sections: Optional ``((title, keywords), ...)`` section list
                used for the completeness check. Defaults to
                :data:`FALLBACK_SECTIONS`.

        Returns:
            The assembled Markdown text.

        Raises:
            ValueError: If no section files are present, a file cannot be
                read, or the assembled result fails validation.
        """
        section_files = self._read_section_files()
        if not section_files:
            raise ValueError(
                f"No section files found in '{self._sections_dir()}'; "
                "nothing to assemble."
            )
        _console.print(
            f"[cyan]Assembling {len(section_files)} section file(s) from "
            f"{self._sections_dir()}[/cyan]"
        )
        chunks: List[str] = []
        for path in section_files:
            try:
                text = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise ValueError(
                    f"Could not read section file '{path}': {exc}"
                ) from exc
            # Strip the internal `<!-- section: ... -->` bookkeeping marker:
            # it belongs in the workspace file, never in the product.
            text = re.sub(r"^<!--.*?-->\s*", "", text, count=1)
            chunk = text.strip()
            # Final renderability gate: a broken file must fail the build
            # HERE, never assemble into the product. (Files are checked at
            # write time too; this catches stale or hand-edited files.)
            renderable, reason = self._check_renderability(chunk)
            if not renderable:
                raise ValueError(
                    f"Section file '{path.name}' is unrenderable "
                    f"({reason}); refusing to assemble broken content."
                )
            chunks.append(chunk)
        combined = f"# {product_name}\n\n" + "\n\n".join(chunks) + "\n"
        reason = self._validation_error(combined)
        if reason is not None:
            raise ValueError(
                f"Assembled section template failed validation: {reason}"
            )
        problem = self._completeness_error(
            combined, sections=sections)
        if problem is not None:
            raise ValueError(
                f"Assembled template still incomplete: {problem}"
            )
        return combined

    def _build_section_by_section(
        self,
        opportunity: Dict[str, Any] = None,
        improvement_prompt: Optional[str] = None,
        extra_feedback: Optional[str] = None,
        resume_existing: bool = True,
        sections: Optional[tuple] = None,
    ) -> str:
        """Build each section to quality, persist it, then assemble from disk.

        Every section goes through :meth:`_iterative_section_build`
        (generate → validate → score → refine, up to 5 passes stopping at
        the quality target) and is then written to its own file under
        ``state/output/{project_id}/sections/section_{i}.md``. Nothing is
        kept only in memory: the final template is assembled by reading
        those files back in order, so a failure part-way through leaves all
        completed sections intact and resumable.

        A section still below the quality bar afterwards is kept but marked
        degraded (visible to the Reviewer) instead of crashing the build.
        Per-section quality is logged to state via
        :meth:`StateManager.log_section_quality`.

        Args:
            opportunity: Opportunity dict (same rule as :meth:`build_product`).
            improvement_prompt: Optional reviewer revision brief, used as
                refinement context in every section's iteration 1.
            extra_feedback: Optional extra instructions applied to every
                section (e.g. reviewer notes from the pipeline).
            resume_existing: When True (default), section files already on
                disk are reused instead of regenerated. Set False to force
                a clean rebuild of every section.
            sections: Optional ``((title, keywords), ...)`` section list.
                Defaults to :data:`FALLBACK_SECTIONS`.

        Raises:
            ValueError: If a section produces no usable content, or the
                assembled template fails validation.
            IOError: If a section file cannot be written.
            RuntimeError: If an API call fails.
        """
        if opportunity is None:
            opportunity = self._load_first_opportunity()
        else:
            opportunity = self._checked_opportunity(opportunity)
        product_name = str(opportunity["product_name"]).strip()
        audience = str(opportunity.get("target_audience", "")).strip() or "user"
        active_sections = tuple(sections) if sections else FALLBACK_SECTIONS

        # FIX 2: a quality revision must never reuse the sections it was
        # criticising. Resume exists for fault tolerance (a crashed run), not
        # for "the reviewer rejected this" — in Maiden Voyage 2.0 the resume
        # path made the self-improving loop a byte-identical no-op.
        is_revision = bool(improvement_prompt and improvement_prompt.strip())
        if is_revision and resume_existing:
            discarded = len(self._read_section_files())
            self._clear_section_files()
            resume_existing = False
            _console.print(
                "[yellow]Improvement prompt supplied: forcing a full rebuild "
                f"(discarded {discarded} section file(s)) so the revision is "
                "actually applied.[/yellow]"
            )

        if resume_existing:
            # Keep section files from a previous attempt: sections already on
            # disk are reused instead of regenerated (Vector 2's main win).
            resumed = self._read_section_files()
            if resumed:
                _console.print(
                    f"[cyan]Resuming: {len(resumed)} existing section "
                    f"file(s) in {self._sections_dir()} will be reused.[/cyan]"
                )
        else:
            self._clear_section_files()
        self._degraded_sections: List[str] = []
        failed_sections: List[tuple] = []

        for index, (title, _) in enumerate(active_sections, start=1):
            existing = self._load_existing_section(index, title)
            if existing is not None:
                _console.print(
                    f"[cyan]Section {index}/{len(active_sections)}:[/cyan] "
                    f"[bold]{title}[/bold] [green]reused from disk "
                    f"({len(existing.split())} words)[/green]"
                )
                continue
            _console.print(
                f"[cyan]Building section {index}/{len(active_sections)}:[/cyan] "
                f"[bold]{title}[/bold]"
            )
            requirements = (
                f"Complete '{title}' section for a {product_name} ({audience}) "
                "with full tables/checklists and realistic example rows. "
                f"MAXIMUM {MAX_WORDS_PER_SECTION} WORDS. DO NOT EXCEED. "
                "Every table must be complete. End with a clear conclusion."
            )
            guidance = section_guidance_by_heading(title)
            if guidance:
                requirements += f" Section guidance: {guidance}"
            context_parts = [p for p in (improvement_prompt, extra_feedback)
                             if p and p.strip()]
            # Any API-level failure (timeout, 404, rate limit) aborts the
            # build, but only after reporting which section files already
            # survived on disk — those are never regenerated.
            # Per-section failure isolation: a timeout or 404 on section 3
            # must not discard the sections already on disk, nor prevent the
            # remaining sections from being attempted. The failure is
            # recorded and the loop continues.
            try:
                result = self._iterative_section_build(
                    section_name=title,
                    section_requirements=requirements,
                    improvement_context="\n\n".join(context_parts),
                    max_iterations=_iteration_limit(),
                    target_quality=_quality_target(),
                )
            except (RuntimeError, ValueError, IOError) as exc:
                _console.print(
                    f"[red]Section {index} '{title}' failed: "
                    f"{type(exc).__name__}: {exc}[/red]"
                )
                failed_sections.append((index, title, str(exc)))
                continue
            body = result["content"]
            if not body:
                _console.print(
                    f"[red]Section {index} '{title}' produced no usable "
                    f"content after {result['iterations_used']} "
                    f"iteration(s).[/red]"
                )
                failed_sections.append((index, title, "no usable content"))
                continue
            for entry in result["quality_history"]:
                _console.print(
                    f"[cyan]  '{title}' iteration {entry['iteration']}: "
                    f"score {entry['score']}/10, "
                    f"{entry['word_count']} words.[/cyan]"
                )
            try:
                self.state_manager.log_section_quality(
                    title, result["final_score"], result["iterations_used"]
                )
            except Exception as exc:
                _console.print(
                    f"[yellow]Could not log quality for '{title}': {exc}[/yellow]"
                )
            # Bar for a clean section: completeness (hard) plus the shared
            # MIN_SECTION_QUALITY. Reads the pipeline constant so the pass bar
            # can never drift from the iteration target (both are 7.0 now —
            # an 8.0 bar marked all 6 sections DEGRADED in production).
            passed = (
                self._validate_section_complete(body, title)
                and result["final_score"] >= _min_section_quality()
            )
            if not passed:
                self._degraded_sections.append(title)
            # Avoid a duplicated heading when the model echoes the title.
            # The echoed heading may carry the refine prompt's iteration
            # suffix ("## Monthly Summary - Iteration 2"), so normalise it
            # away before comparing: otherwise the comparison fails, the
            # duplicate ships, and _sanitize_section's marker strip leaves the
            # section with two headings for one section.
            first_line = body.splitlines()[0].strip()
            first_line = _ITERATION_MARKER.sub(r"\1\2", first_line)
            first_line = first_line.lstrip("#").strip().lower()
            if first_line == title.lower():
                body = "\n".join(body.splitlines()[1:]).strip()
            if title in self._degraded_sections:
                _console.print(
                    f"[yellow]  Warning: section '{title}' marked DEGRADED "
                    "for reviewer attention.[/yellow]"
                )
                # Record the degradation in state (internal), never in the
                # product: an HTML comment here leaked to customers in
                # Maiden Voyage 2.0. _sanitize_section strips comments anyway.
                self._note_degraded_section(title)

            # FIX 3 + 4 + renderability: sanitize (strip comments/state
            # leaks, enforce the hard word cap, reject unrenderable blobs)
            # BEFORE the section is written to disk. A section that cannot
            # be rendered is recorded as a failure like any other — the
            # loop continues and the final gate refuses to assemble without
            # it — instead of crashing the whole build.
            try:
                body = self._sanitize_section(body)
            except (ValueError, IOError) as exc:
                _console.print(
                    f"[red]Section {index} '{title}' failed sanitization: "
                    f"{type(exc).__name__}: {exc}[/red]"
                )
                failed_sections.append((index, title, str(exc)))
                continue
            try:
                path = self._write_section_file(index, title, body)
            except IOError as exc:
                _console.print(
                    f"[red]Section {index} '{title}' could not be written: "
                    f"{exc}[/red]"
                )
                failed_sections.append((index, title, str(exc)))
                continue
            _console.print(
                f"[cyan]  Section {index} '{title}': "
                f"{len(body.split())} words, "
                f"final score {result['final_score']}/10, "
                f"iterations {result['iterations_used']}, "
                f"validation={'PASS' if passed else 'DEGRADED'}, "
                f"persisted to {path}.[/cyan]"
            )

        # Every section has now been attempted: report which files survived
        # before deciding whether assembly can proceed.
        if failed_sections:
            self._report_partial_assembly(
                failed_sections[0][0], product_name,
                extra_failed=failed_sections[1:],
                total_sections=len(active_sections),
            )
            names = ", ".join(
                f"{i}('{t}')" for i, t, _ in failed_sections
            )
            raise ValueError(
                f"{len(failed_sections)} of {len(active_sections)} sections "
                f"failed and were not written: {names}. The "
                f"{len(self._read_section_files())} completed section file(s) "
                f"are preserved in {self._sections_dir()} and will be reused "
                "on the next run (they are never regenerated)."
            )

        combined = self._assemble_from_files(product_name, active_sections)
        self.save_product(combined, product_name)
        _console.print(
            f"[bold green]Section assembly complete "
            f"({len(combined)} characters from "
            f"{len(self._read_section_files())} section files).[/bold green]"
        )
        return combined

    def _note_degraded_section(self, title: str) -> None:
        """Record a degraded section in state, not in the product text.

        Degradation used to be annotated with an HTML comment inside the
        section body, which shipped to customers. It is now tracked here so
        the Reviewer can still see it via ``state.section_quality``.
        """
        try:
            self.state_manager.log_section_quality(
                f"{title} (DEGRADED)", 0.0, 0
            )
        except Exception as exc:
            _console.print(
                f"[yellow]Could not record degradation for '{title}': "
                f"{exc}[/yellow]"
            )

    def _load_existing_section(self, index: int, title: str) -> Optional[str]:
        """Return a previously persisted section body, or None.

        A section file counts as reusable only when it carries this
        section's marker and holds non-empty content, so a truncated or
        stale file from a different build is regenerated rather than
        silently assembled into the product.
        """
        path = self._sections_dir() / f"section_{index}.md"
        if not path.is_file():
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None
        if f"<!-- section: {title} -->" not in text:
            return None
        body = re.sub(r"^<!--.*?-->\s*", "", text, count=1)
        body = re.sub(r"^##\s+.*$", "", body, count=1, flags=re.MULTILINE)
        body = body.strip()
        if not body:
            return None
        # A stale collapsed file must never be reused: it would pass the
        # marker check above, sail through assembly, and re-poison every
        # retry. Delete it so this section regenerates instead.
        renderable, reason = self._check_renderability(body)
        if not renderable:
            _console.print(
                f"[yellow]Discarding stale unrenderable section file "
                f"'{path.name}' ({reason}); regenerating.[/yellow]"
            )
            try:
                path.unlink()
            except OSError:
                pass
            return None
        return body

    def _report_partial_assembly(
        self,
        failed_index: int,
        product_name: str,
        extra_failed: Optional[List[tuple]] = None,
        total_sections: Optional[int] = None,
    ) -> None:
        """Log which section files survived a mid-build failure.

        Args:
            failed_index: 1-based index of the first failing section.
            product_name: Product being built (for the log line).
            extra_failed: Additional ``(index, title, error)`` tuples.
            total_sections: Total section count for display. Defaults to
                ``len(FALLBACK_SECTIONS)``.
        """
        surviving = self._read_section_files()
        others = (
            f" (+{len(extra_failed)} more)" if extra_failed else ""
        )
        total = total_sections if total_sections else len(FALLBACK_SECTIONS)
        _console.print(
            f"[yellow]Assembly incomplete for '{product_name}': first "
            f"failure at section {failed_index}/{total}"
            f"{others}. {len(surviving)} completed section file(s) preserved "
            f"in {self._sections_dir()}: "
            f"{[p.name for p in surviving]}[/yellow]"
        )

    @staticmethod
    def _completeness_error(
        content: Any, sections: Optional[tuple] = None
    ) -> Optional[str]:
        """Return None when all sections are present and text looks whole.

        Detects: missing sections, ``...``/ellipsis placeholders,
        continuation markers, and abrupt endings (trailing ``...``,
        ``:``, ``,``, ``-`` with no closing content).

        Args:
            content: Assembled template text.
            sections: Optional ``((title, variants), ...)`` section list.
                Defaults to :data:`FALLBACK_SECTIONS`.
        """
        if not isinstance(content, str) or not content.strip():
            return "content is empty."
        lowered = re.sub(r"\s+", " ", content.lower())
        active = tuple(sections) if sections else FALLBACK_SECTIONS
        missing = [
            title
            for title, variants in active
            if not any(v in lowered for v in variants)
        ]
        if missing:
            return f"missing sections: {missing}."
        for marker in ("...", "…", "to be continued", "[truncated",
                       "(continued", "more to come"):
            if marker in lowered:
                return f"truncation marker detected: {marker!r}."
        tail = content.strip()[-120:]
        if re.search(r"(\.\.\.|…|:|,|-|—)\s*$", tail):
            return "text ends abruptly (trailing punctuation, no closing)."
        return None

    def build_with_testing(
        self, opportunity: Dict[str, Any] = None, max_attempts: int = 3
    ) -> Dict[str, Any]:
        """Build, test, fix, and package a product (up to ``max_attempts``).

        Each attempt builds the product with :meth:`build_product`, runs it
        through :class:`TestRunner`, and — on success — packages it with
        :class:`ProductPackager`. Failed attempts produce a fix prompt from
        the test errors that is fed back into the next build. Every attempt
        is recorded (see :meth:`get_build_history`) and mirrored to
        ``state.build_history`` / ``state.last_test_results``.

        Args:
            opportunity: Opportunity dict, or ``None`` to use the first
                Scout finding in state (same rule as :meth:`build_product`).
            max_attempts: Maximum build attempts (must be >= 1).

        Returns:
            Dict with exactly these keys:

            * ``success`` (bool): True if a tested package was produced.
            * ``attempts`` (int): Number of build attempts performed.
            * ``final_product_code`` (str | None): Last built product text.
            * ``test_results`` (dict): Last :class:`TestRunner` report
              (``{"success": ..., "errors": ..., "test_results": ...}``).
            * ``package_path`` (str | None): ZIP path when packaging
              succeeded, else None.
            * ``errors`` (list): Human-readable failure descriptions.

        Raises:
            ValueError: If inputs are invalid (bad opportunity, or
                ``max_attempts < 1``).
        """
        if opportunity is None:
            opportunity = self._load_first_opportunity()
        else:
            opportunity = self._checked_opportunity(opportunity)
        if (
            not isinstance(max_attempts, int)
            or isinstance(max_attempts, bool)
            or max_attempts < 1
        ):
            raise ValueError(
                f"max_attempts must be an integer >= 1, got {max_attempts!r}."
            )
        product_name = str(opportunity["product_name"]).strip()

        tester = TestRunner(self.state_manager)
        packager = ProductPackager(self.state_manager)

        errors: List[str] = []
        attempts = 0
        final_code: Optional[str] = None
        package_path: Optional[str] = None
        last_report: Dict[str, Any] = {
            "success": False,
            "output": "",
            "errors": ["No attempts performed yet."],
            "test_results": {},
        }
        feedback: Optional[str] = None

        for attempt in range(1, max_attempts + 1):
            attempts = attempt
            _console.print(
                f"[bold cyan]Build attempt {attempt}/{max_attempts}[/bold cyan] "
                f"for [bold]{product_name}[/bold]"
                + (" (retry with fix feedback)" if feedback else "")
            )
            entry: Dict[str, Any] = {
                "attempt": attempt,
                "product_name": product_name,
                "product_type": None,
                "build_ok": False,
                "tests_passed": False,
                "package_path": None,
                "error": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            try:
                code = self.build_product(
                    opportunity=opportunity, feedback=feedback
                )
            except (RuntimeError, ValueError) as exc:
                message = f"Attempt {attempt}: build failed: {exc}"
                _console.print(f"[red]{message}[/red]")
                errors.append(message)
                entry["error"] = str(exc)
                self._build_history.append(entry)
                feedback = (
                    "The previous build attempt failed before testing: "
                    f"{exc}\nReturn the COMPLETE corrected product."
                )
                continue

            entry["build_ok"] = True
            final_code = code
            product_type = self._detect_product_type(code)
            entry["product_type"] = product_type

            _console.print(
                f"[cyan]Attempt {attempt}: running tests "
                f"({product_type})…[/cyan]"
            )
            report = tester.run_tests(code, product_type=product_type)
            last_report = {
                "success": report["success"],
                "output": report["output"],
                "errors": list(report["errors"]),
                "test_results": report["test_results"],
            }
            if report["success"]:
                entry["tests_passed"] = True
                _console.print(
                    f"[green]Attempt {attempt}: tests passed. "
                    "Packaging…[/green]"
                )
                try:
                    zip_path = packager.create_package(
                        product_name, code, product_type=product_type
                    )
                except (ValueError, IOError, RuntimeError) as exc:
                    message = f"Attempt {attempt}: packaging failed: {exc}"
                    _console.print(f"[red]{message}[/red]")
                    errors.append(message)
                    entry["error"] = str(exc)
                    self._build_history.append(entry)
                    break
                package_path = str(zip_path)
                entry["package_path"] = package_path
                self._build_history.append(entry)
                _console.print(
                    f"[bold green]Build succeeded on attempt {attempt}: "
                    f"'{zip_path}'[/bold green]"
                )
                break

            message = (
                f"Attempt {attempt}: tests failed "
                f"({len(report['errors'])} error(s))."
            )
            _console.print(f"[yellow]{message}[/yellow]")
            errors.append(message + " " + " ".join(report["errors"][:3]))
            entry["error"] = "; ".join(report["errors"][:5])
            self._build_history.append(entry)
            feedback = self._build_fix_prompt(code, report)

        success = package_path is not None
        self._persist_history(last_report)
        if not success:
            _console.print(
                f"[red]Build failed after {attempts} attempt(s). "
                "See errors for details.[/red]"
            )
        return {
            "success": success,
            "attempts": attempts,
            "final_product_code": final_code,
            "test_results": last_report,
            "package_path": package_path,
            "errors": errors,
        }

    def get_build_history(self) -> List[Dict[str, Any]]:
        """Return all recorded build attempts and their outcomes.

        Returns:
            A list of per-attempt dicts (attempt number, product name/type,
            build/test/package outcomes, error text, timestamp), oldest
            first. Returns an empty list if :meth:`build_with_testing` has
            not run yet in this process. A copy is returned, so callers
            cannot mutate internal records.
        """
        return [dict(entry) for entry in self._build_history]

    def save_product(self, product_content: str, product_name: str) -> Path:
        """Persist a validated product to state and to ``products/``.

        Args:
            product_content: The complete product text (code or template).
            product_name: Human-readable name; sanitised for the filename
                (special chars removed, spaces become underscores).

        Returns:
            The :class:`pathlib.Path` of the written product file
            (``products/{sanitised}.py`` or ``.md`` for templates).

        Raises:
            ValueError: If either argument is empty.
            IOError: If the product file cannot be written, with details.
            RuntimeError: If persisting state fails.
        """
        if not isinstance(product_content, str) or not product_content.strip():
            raise ValueError("product_content must be a non-empty string.")
        cleaned_name = product_name.strip() if isinstance(product_name, str) else ""
        if not cleaned_name:
            raise ValueError("product_name must be a non-empty string.")

        filename = self._sanitise_filename(cleaned_name)
        extension = self._guess_extension(product_content)
        try:
            self.products_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise IOError(
                f"Could not create products directory '{self.products_dir}': {exc}"
            ) from exc
        target = self.products_dir / f"{filename}{extension}"
        try:
            target.write_text(product_content, encoding="utf-8")
        except OSError as exc:
            raise IOError(
                f"Could not write product file '{target}': {exc}"
            ) from exc

        try:
            self.state_manager.update_phase("building")
            state = self.state_manager.get_state()
            state.product_code = product_content
            state.product_name = cleaned_name
            self.state_manager.save(state)
        except Exception as exc:
            raise RuntimeError(
                f"Product file '{target}' written but persisting state failed "
                f"for project '{self.state_manager.project_id}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        _console.print(
            f"[green]Saved product '{cleaned_name}' to '{target}' "
            "and updated state (phase=building).[/green]"
        )
        return target

    def export_to_image(self, content: str, output_path: str) -> bool:
        """Render a Markdown template to a PNG/PDF image file.

        Strategy: Markdown → styled HTML, then a real Chromium screenshot
        via Playwright when available, else a simple Pillow text rendering.
        Works without orchestrator/state (standalone-friendly).

        Args:
            content: Markdown template text.
            output_path: Destination file (``.png`` screenshot or ``.pdf``).

        Returns:
            True when the file was written, False on any failure (never
            raises for rendering problems).
        """
        if not isinstance(content, str) or not content.strip():
            return False
        target = Path(output_path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            document = _markdown_to_html(content)
            suffix = target.suffix.lower()
            try:
                if suffix == ".pdf":
                    rendered = _playwright_pdf(document, target)
                else:
                    rendered = _playwright_screenshot(document, target)
            except Exception as exc:
                _console.print(
                    f"[yellow]Browser render unavailable ({exc}); "
                    "falling back to simple text rendering.[/yellow]"
                )
                rendered = False
            if rendered:
                _console.print(f"[green]Image saved to '{target}'.[/green]")
                return True
            _console.print(
                "[yellow]Falling back to simple text rendering.[/yellow]"
            )
            return _pil_render_text(content, target)
        except Exception as exc:
            _console.print(f"[yellow]Image export failed: {exc}[/yellow]")
            return False

    def validate_product(self, product_content: str) -> bool:
        """Check whether generated content is a real, usable product.

        Rules:

        * Empty/whitespace-only content is invalid.
        * Content that compiles as Python (``compile()``) is valid, provided
          it has minimal substance (length and real code lines, not just
          comments).
        * Otherwise the content is treated as a template: it is valid if it
          shows template structure (headings, sections, placeholders, or
          example data) with sufficient length.

        Args:
            product_content: Candidate product text.

        Returns:
            True if valid, False otherwise. Never raises for malformed input.
        """
        return self._validation_error(product_content) is None

    # -- internals ----------------------------------------------------------

    def _load_first_opportunity(self) -> Dict[str, Any]:
        """Return the first Scout opportunity from state, or raise ValueError."""
        try:
            state = self.state_manager.get_state()
        except Exception as exc:
            raise ValueError(
                "No opportunity given and loading state failed: "
                f"{type(exc).__name__}: {exc}. "
                "Run the Scout agent first or pass opportunity={...}."
            ) from exc
        findings = state.scout_findings
        if not isinstance(findings, dict):
            raise ValueError(
                "No opportunity given and state holds no scout_findings. "
                "Run the Scout agent first or pass opportunity={...}."
            )
        opportunities = findings.get("opportunities")
        if not isinstance(opportunities, list) or not opportunities:
            raise ValueError(
                "No opportunity given and scout_findings contains no "
                "opportunities list. Run the Scout agent first or pass "
                "opportunity={...}."
            )
        first = opportunities[0]
        _console.print(
            f"[cyan]Using Scout opportunity 1/{len(opportunities)}:[/cyan] "
            f"[bold]{first.get('product_name', '(unnamed)') if isinstance(first, dict) else first!r}[/bold]"
        )
        return self._checked_opportunity(first)

    @staticmethod
    def _checked_opportunity(opportunity: Any) -> Dict[str, Any]:
        """Ensure the opportunity is a dict with at least a product_name."""
        if not isinstance(opportunity, dict):
            raise ValueError(
                "opportunity must be a dict like Scout's findings entries, "
                f"got {type(opportunity).__name__}."
            )
        name = opportunity.get("product_name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "opportunity must contain a non-empty 'product_name' string."
            )
        return opportunity

    def _build_user_prompt(
        self,
        opportunity: Dict[str, Any],
        feedback: Optional[str] = None,
        improvement_prompt: Optional[str] = None,
    ) -> str:
        """Compose the user prompt carrying the full opportunity details."""
        details = "\n".join(
            f"- {key}: {opportunity.get(key, '(not specified)')}"
            for key in (
                "product_name",
                "target_audience",
                "pain_point",
                "estimated_price",
                "unique_value_proposition",
                "estimated_build_time",
            )
        )
        prompt = ""
        if improvement_prompt and improvement_prompt.strip():
            prompt += (
                "IMPORTANT REVISION INSTRUCTIONS:\n"
                "The previous version of this product was reviewed and found "
                "lacking. Here are the SPECIFIC improvements you must make:\n\n"
                f"{improvement_prompt.strip()}\n\n"
                "Apply ALL of these improvements in your new version. "
                "Do not ignore any.\n\n"
            )
        prompt += (
            "Build the complete, production-ready digital product for this "
            "validated market opportunity:\n\n"
            f"{details}\n\n"
            "Requirements:\n"
            "- Create the ACTUAL product — complete, working, requiring NO "
            "modifications before use.\n"
            "- Include the main functionality, clear inline comments, concise "
            "usage instructions (as comments at the top), and a dependencies "
            "list (as comments; standard library preferred).\n"
            "- For a Python script: return ONE complete, runnable file with an "
            '`if __name__ == "__main__":` block demonstrating real usage.\n'
            "- For a template: return the COMPLETE structure with realistic "
            "example data filled in, not placeholders alone.\n"
            "- Output ONLY the product content (fenced in a single code block "
            "is fine). No marketing pitch, no explanations outside the product."
        )
        if feedback and feedback.strip():
            prompt += (
                "\n\nIMPORTANT — FIX REQUESTED:\n"
                f"{feedback.strip()}\n"
                "Return the ENTIRE corrected product, not a patch or diff."
            )
        return prompt

    @staticmethod
    def _detect_product_type(product_content: str) -> str:
        """Return ``"python"`` if the content compiles, else ``"template"``."""
        try:
            compile(product_content, "<product>", "exec")
            return "python"
        except (SyntaxError, ValueError):
            return "template"

    @staticmethod
    def _build_fix_prompt(failed_code: str, report: Dict[str, Any]) -> str:
        """Compose fix feedback from a failed TestRunner report."""
        failures = "\n".join(
            f"- {error}" for error in report.get("errors", [])[:10]
        ) or "- (no error details captured)"
        output_tail = (report.get("output") or "").strip()[-1500:]
        code_head = (failed_code or "")[:6000]
        return (
            "The previous attempt produced output that FAILED automated "
            "testing. Fix EVERY issue below and return the COMPLETE corrected "
            "product (never a patch or diff).\n\n"
            f"Test failures:\n{failures}\n\n"
            f"Captured output (tail):\n{output_tail or '(empty)'}\n\n"
            f"Previously generated code:\n{code_head}"
        )

    def _persist_history(self, last_report: Dict[str, Any]) -> None:
        """Mirror build history + latest test report onto the state."""
        try:
            state = self.state_manager.get_state()
            state.build_history = [dict(entry) for entry in self._build_history]
            state.last_test_results = {
                "success": last_report.get("success", False),
                "errors": list(last_report.get("errors", [])),
                "test_results": last_report.get("test_results", {}),
            }
            self.state_manager.save(state)
        except Exception as exc:  # Never mask the build outcome.
            _console.print(
                f"[yellow]Could not persist build history to state: {exc}[/yellow]"
            )

    @staticmethod
    def _extract_product_content(raw_response: str) -> str:
        """Strip a single surrounding markdown code fence, if present."""
        text = raw_response.strip() if isinstance(raw_response, str) else ""
        if text.startswith("```"):
            lines = text.splitlines()[1:]  # drop opening fence (``` or ```python)
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]  # drop closing fence
            text = "\n".join(lines).strip()
        return text

    @staticmethod
    def _sanitise_filename(product_name: str) -> str:
        """Sanitise a product name for use as a filename stem.

        Spaces become underscores; anything that is not a letter, digit,
        underscore, or hyphen is removed; repeats are collapsed. Falls back
        to ``"product"`` if nothing remains.
        """
        name = product_name.strip().replace(" ", "_")
        name = re.sub(r"[^A-Za-z0-9_-]", "", name)
        name = re.sub(r"_+", "_", name).strip("_-")
        return name or "product"

    @staticmethod
    def _guess_extension(product_content: str) -> str:
        """Pick ``.py`` for Python content, ``.md`` for templates."""
        try:
            compile(product_content, "<product>", "exec")
            return ".py"
        except (SyntaxError, ValueError):
            pass
        if re.search(r"^#{1,6}\s+\S", product_content, re.MULTILINE):
            return ".md"
        return ".py"

    @classmethod
    def _validation_error(cls, product_content: Any) -> Optional[str]:
        """Return None when valid, else a human-readable reason."""
        if not isinstance(product_content, str) or not product_content.strip():
            return "content is empty."
        text = product_content.strip()
        try:
            compile(text, "<product>", "exec")
        except (SyntaxError, ValueError) as exc:
            return cls._template_error(text, f"invalid Python syntax ({exc})")
        code_lines = [
            line
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if len(text) < 50:
            return "Python content is too short to be a complete product."
        if len(code_lines) < 3:
            return "Python content has no substantive code (only comments/blank lines)."
        return None

    @staticmethod
    def _template_error(text: str, python_reason: str) -> Optional[str]:
        """Validate non-Python content as a template; None when valid."""
        if len(text) < 100:
            return (
                f"content is not valid Python ({python_reason}) and is too "
                "short to be a usable template."
            )
        has_heading = re.search(r"^#{1,6}\s+\S", text, re.MULTILINE) is not None
        has_section = (
            re.search(r"^(.+)\n([=-])\1*\s*$", text, re.MULTILINE) is not None
            or "---" in text
        )
        has_placeholder_or_example = (
            re.search(r"[\{\}\[\]]", text) is not None
            or "example" in text.lower()
        )
        if has_heading or (has_section and has_placeholder_or_example):
            return None
        return (
            f"content is not valid Python ({python_reason}) and shows no "
            "template structure (headings, sections, placeholders, examples)."
        )
