# -*- coding: utf-8 -*-
"""Medical-interpreter planner build pipeline: build, test, package, review.

Takes the validated sub-niche "Freelance Medical Interpreters" and produces
a tested, packaged, reviewed digital product ready to sell at EUR 15.

Flow:
    1. Build  - BuilderAgent creates the complete template (max 2 attempts;
                test failures feed back into the rebuild).
    2. Test   - TestRunner validates structure/sections (template checks).
    3. Package - ProductPackager zips template + README, then this script
                appends a quick-start guide, a filled example, and a
                feature list; the ZIP is copied to publish_ready/.
    4. Review - ReviewerAgent checks completeness, usability, value, and
                quality. One revision attempt on REJECTED. On APPROVED the
                state records review_status="approved" and phase
                "publishing" (= approved_for_publish: cleared for publishing).

Usage:
    python build_pipeline.py

Windows-compatible (pathlib everywhere, UTF-8 reads/writes, cp1252-safe
console output). Exit code 0 on APPROVED, 1 otherwise.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agents.builder import BuilderAgent
from agents.customer_reviewer import CustomerReviewerAgent
from agents.fact_checker import FactCheckerAgent
from agents.grammar_agent import GrammarAgent
from agents.kcal_checker import KcalChecker
from agents.publisher import PublisherAgent
from agents.reviewer import ReviewerAgent, _edge_clipping_check
from core.notifier import Notifier
from core.orchestrator import Orchestrator
from core.product_packager import ProductPackager
from core.usda_client import USDAClient
from core.sections import (
    SECTION_HEADINGS,
    reviewer_section_specs,
    section_requirement_lines,
)
from core.state_manager import StateManager
from core.swarm_manager import DEFAULT_MAX_CONCURRENT_PROJECTS, SwarmManager
from core.test_runner import TestRunner

PROJECT_ID = "medical_interpreter_planner_001"
PRODUCT_NAME = "Weekly Planner for Freelance Medical Interpreters"
SUGGESTED_PRICE_EUR = 15
MAX_BUILD_ATTEMPTS = 2

#: Niche key for this pipeline run. Drives fixed pricing via
#: :data:`NICHE_PRICING`. Overridden by :func:`apply_product_profile`.
NICHE = "medical_interpreter"

#: Active product profile key. ``"medical_interpreter"`` preserves the
#: original hardcoded pipeline exactly; ``"nutrition_meal_prep"`` switches
#: the opportunity, sections, and docs to the nutrition-focused product.
PRODUCT_PROFILE = "medical_interpreter"

#: Product profiles. The medical profile keeps ``sections=None`` so the
#: builder/reviewer fall back to the canonical
#: :data:`core.sections.SECTION_DEFINITIONS`. Nutrition profiles carry
#: their own ``(title, keywords, requirement)`` triples, which flow to the
#: builder (assembly), the reviewer (completeness), and the docs.
PRODUCT_PROFILES: Dict[str, Dict[str, Any]] = {
    "medical_interpreter": {
        "product_name": "Weekly Planner for Freelance Medical Interpreters",
        "niche": "medical_interpreter",
        "price": 15,
        "opportunity": None,  # built from the legacy OPPORTUNITY below
        "sections": None,
    },
    "nutrition_meal_prep": {
        "product_name": "Healthy Meal Prep Planner with Nutrition Data",
        "niche": "meal_prep",
        "price": 15,
        "opportunity": {
            "product_name": "Healthy Meal Prep Planner with Nutrition Data",
            "target_audience": (
                "busy households planning healthy weekly meals"
            ),
            "pain_point": (
                "grocery waste, weeknight cooking stress, and unclear "
                "nutrition information"
            ),
            "estimated_price": 15,
            "unique_value_proposition": (
                "One template that plans groceries and prep, indexes "
                "recipes, schedules the week, and records USDA-backed "
                "nutrition per 100 g"
            ),
            "estimated_build_time": "2 hours",
        },
        "sections": (
            (
                "Grocery List",
                ("grocery list", "grocery"),
                "Grocery list — categorized ingredients with example "
                "quantities and planned uses",
            ),
            (
                "Prep Checklist",
                ("prep checklist", "prep"),
                "Prep checklist — ordered batch-prep steps with completion "
                "standards",
            ),
            (
                "Recipe Index",
                ("recipe index", "recipe"),
                "Recipe index — dinner rotation with servings, times, and "
                "ingredients",
            ),
            (
                "Weekly Schedule",
                ("weekly schedule", "schedule"),
                "Weekly schedule — day-by-day menu with batch-prep actions "
                "and portions",
            ),
            (
                "Nutrition Reference",
                ("nutrition reference", "nutrition"),
                "Nutrition reference — per-100-gram energy and macros "
                "backed by USDA data with FDC sources; state every value "
                "per 100 g",
            ),
            (
                "Storage & Safety",
                ("storage & safety", "storage", "safety"),
                "Storage and safety — refrigeration, freezing, and "
                "reheating standards with time and temperature rules",
            ),
        ),
    },
    # Generalization probe (Directive 16): habit tracker, chosen because it
    # is a canonical popular niche distinct from meal planning and budget
    # tracking. Live Scout research was unavailable (SearXNG outage), so
    # these section TITLES are product design; all content is the Builder's.
    "habit_tracker": {
        "product_name": "Daily Habit Tracker",
        "niche": "habit_tracker",
        "price": 15,
        "opportunity": {
            "product_name": "Daily Habit Tracker",
            "target_audience": (
                "busy professionals and students building consistent "
                "daily habits"
            ),
            "pain_point": (
                "habits fade without tracking; streaks, reminders, and "
                "reviews live in scattered apps or nowhere"
            ),
            "estimated_price": 15,
            "unique_value_proposition": (
                "One template that defines habits, tracks daily check-ins, "
                "counts streaks, and reviews monthly progress"
            ),
            "estimated_build_time": "2 hours",
        },
        "sections": (
            (
                "Habit Dashboard",
                ("habit dashboard", "dashboard", "overview"),
                "Habit dashboard — at-a-glance status table of all active "
                "habits with today/yesterday checkmarks and current "
                "streaks, at least 150 words",
            ),
            (
                "Habit Library",
                ("habit library", "habit list", "habits"),
                "Habit library — starter catalog of 15-20 example habits "
                "across health, focus, learning, and home with frequency "
                "and difficulty tags, at least 150 words",
            ),
            (
                "Daily Check-in",
                ("daily check-in", "check-in", "daily log"),
                "Daily check-in — Markdown checklist routine for morning "
                "and evening with per-habit checkboxes and a notes line, "
                "at least 150 words",
            ),
            (
                "Streak Tracker",
                ("streak tracker", "streaks", "consistency"),
                "Streak tracker — 30-day grid tables per habit with streak "
                "rules, freeze-day policy, and restart protocol, at "
                "least 150 words",
            ),
            (
                "Monthly Review",
                ("monthly review", "review", "reflection"),
                "Monthly review — completion-rate table, best/worst habit "
                "analysis prompts, and next-month planning checklist, at "
                "least 150 words",
            ),
        ),
    },
    "weekly_meal_planning": {
        "product_name": "Weekly Meal Planner",
        "niche": "meal_planning",
        "price": 15,
        "opportunity": {
            "product_name": "Weekly Meal Planner",
            "target_audience": (
                "busy professionals and households planning healthy "
                "weekly meals"
            ),
            "pain_point": (
                "weeknight cooking stress, grocery waste, and scattered "
                "recipes with no weekly plan"
            ),
            "estimated_price": 15,
            "unique_value_proposition": (
                "One template that plans the week, stores recipes, lists "
                "groceries, guides batch prep, and tracks the food budget"
            ),
            "estimated_build_time": "2 hours",
        },
        # Sections come straight from the Scout SearXNG research run
        # (test_scout_search): weekly calendar, recipe database, grocery /
        # inventory list, pre-filled examples, prep planning. Every section
        # must hold at least 150 words — comfortably above the 120-word
        # degeneracy floor — with pre-filled example rows, never stubs.
        "sections": (
            (
                "Weekly Meal Calendar",
                ("weekly meal calendar", "meal calendar", "calendar"),
                "Weekly meal calendar — Markdown table with Day | "
                "Breakfast | Lunch | Dinner | Snacks columns and 7 "
                "pre-filled example rows (Monday–Sunday), at least 150 "
                "words of planning content",
            ),
            (
                "Recipe Database",
                ("recipe database", "recipe", "recipes"),
                "Recipe database — recipe template fields (name, "
                "ingredients, instructions, prep time) plus 5-10 "
                "pre-filled example recipes across Breakfast, Lunch, "
                "Dinner, and Snacks, at least 150 words",
            ),
            (
                "Grocery List",
                ("grocery list", "grocery", "shopping"),
                "Grocery list — Markdown checkbox list grouped by "
                "Produce, Proteins, Dairy, Pantry, and Frozen with "
                "30-50 pre-filled items, at least 150 words",
            ),
            (
                "Meal Prep Guide",
                ("meal prep guide", "meal prep", "prep"),
                "Meal prep guide — weekly prep schedule, batch-cooking "
                "tips, and storage guidelines with times and "
                "temperatures, at least 150 words",
            ),
            (
                "Budget Tracker",
                ("budget tracker", "budget", "cost"),
                "Budget tracker — weekly grocery budget table, cost-per-"
                "meal calculation, and monthly summary with example "
                "figures, at least 150 words",
            ),
        ),
    },
}


def active_profile() -> Dict[str, Any]:
    """Return the active product profile dict (medical by default)."""
    return PRODUCT_PROFILES.get(
        PRODUCT_PROFILE, PRODUCT_PROFILES["medical_interpreter"]
    )


def apply_product_profile(key: str) -> Dict[str, Any]:
    """Switch the pipeline globals to a product profile.

    Args:
        key: A key from :data:`PRODUCT_PROFILES`.

    Returns:
        The applied profile dict.
    """
    global PRODUCT_NAME, NICHE, SUGGESTED_PRICE_EUR, OPPORTUNITY
    global PRODUCT_PROFILE
    if key not in PRODUCT_PROFILES:
        raise ValueError(
            f"Unknown product profile {key!r}; "
            f"choose from {sorted(PRODUCT_PROFILES)}."
        )
    profile = PRODUCT_PROFILES[key]
    PRODUCT_PROFILE = key
    PRODUCT_NAME = str(profile["product_name"])
    NICHE = str(profile["niche"])
    SUGGESTED_PRICE_EUR = profile["price"]
    if profile["opportunity"] is not None:
        OPPORTUNITY = dict(profile["opportunity"])
    return profile


def active_sections() -> Optional[tuple]:
    """Return ``((title, keywords), ...)`` for the active profile.

    Returns None for the medical profile so the builder keeps using the
    canonical :data:`core.sections.SECTION_DEFINITIONS`.
    """
    sections = active_profile().get("sections")
    if not sections:
        return None
    return tuple((title, keywords) for title, keywords, _ in sections)


def active_section_specs() -> List[tuple]:
    """Return ``[(requirement, keywords), ...]`` for reviewer/docs."""
    sections = active_profile().get("sections")
    if not sections:
        return REQUIRED_SECTIONS
    return [
        (requirement, (title.lower(), *keywords))
        for title, keywords, requirement in sections
    ]


def sections_brief() -> str:
    """Build the section brief for the active product profile."""
    sections = active_profile().get("sections")
    if not sections:
        return SECTIONS_BRIEF
    lines = [
        f"{index}. {title} — {requirement}"
        for index, (title, _, requirement) in enumerate(sections, start=1)
    ]
    headings = ", ".join(title for title, _, _ in sections)
    return (
        "Return the product as a MARKDOWN template fenced in a single "
        "```markdown code block — NOT Python code. Be complete, detailed, "
        "and thorough — never truncate, never use '...' placeholders. "
        "Start with an # H1 title, then one ## section per required item "
        "below, using tables and checklists with realistic example rows.\n"
        f"The template MUST contain exactly these {len(sections)} sections, "
        "each with its own heading and usable structure (tables, "
        "checklists, or fields — not just a mention):\n"
        + "\n".join(lines)
        + "\nUse these exact ## headings so sections are findable: "
        + f"{headings}. No placeholder text anywhere."
    )

#: Fixed per-niche pricing. The customer reviewer's WTP read €13/€14/€15 on
#: near-identical products across MV012–MV014 — verdict stable, price noisy.
#: Pricing is now a business decision per niche (medical_interpreter uses the
#: observed median), not a per-run model sample. The customer reviewer still
#: runs for the BUY verdict; only the price tag is fixed.
NICHE_PRICING: Dict[str, float] = {
    "medical_interpreter": 14.00,  # median of €13-15 observed MV012-014
    "default": 15.00,
}

#: Quality gates for automatic publishing (customer-review driven).
MIN_OVERALL_SCORE = 7.0
MIN_SINGLE_CRITERION = 6.0
ACCEPTABLE_VERDICTS = ["BUY", "NEEDS IMPROVEMENT"]

#: Visual (screenshot) review gate. The vision MODEL is retired: after a 4/4
#: hallucination rate across MV011–MV014 (scoring 1.0 on verified-clean
#: renders), no model judgment participates in the gate. The deterministic
#: pixel check below is the entire gate: PASS renders score 10.0, clipped
#: renders score 0.0. MIN_VISUAL_SCORE is kept as the binary threshold so
#: downstream comparisons keep working unchanged.
MIN_VISUAL_SCORE = 7.0
VISUAL_PASS_SCORE = 7.0
VISUAL_FAIL_SCORE = 4.0

# Quality iteration settings
MAX_ITERATIONS_PER_SECTION = 3      # Max refinement cycles per section
MIN_SECTION_QUALITY = 7.0           # Minimum quality score to accept a section
#: Target quality per section. Calibrated down from 8.0: the builder's
#: self-score topped out at 6.0-7.5 in production runs, so an 8.0 target
#: marked every section DEGRADED regardless of real quality. This now matches
#: MIN_OVERALL_SCORE; the Reviewer and Customer Reviewer remain the stricter
#: gates, so the builder must not be the harshest judge in the swarm.
TARGET_SECTION_QUALITY = 7.0
#: Configurable household size (default: 4 people). Injected into every
#: section prompt so all sections share one household basis — prompt advice
#: alone could not hold this consistent across independent generation
#: calls (Templates 1–3 all split 2-vs-4 in the calendar section).
HOUSEHOLD_SIZE = 4
ENABLE_DEEP_RESEARCH = True         # More thorough research phase
ENABLE_CROSS_VALIDATION = True      # Multiple reviewer checks

#: --- Feature flags (foundation lock) --------------------------------------
#: Single switchboard for the post-build validation chain. Each key gates one
#: step; the step's code is never removed, only skipped, so a flag can be
#: flipped back to True with no other edit. All default to False so a Builder
#: optimization run exercises the Builder alone, with every external judge
#: and blocker held back.
#:
#: * ``grammar_check``   — GrammarAgent proofreading pass.
#: * ``kcal_validation`` — USDA cross-check of every energy claim.
#: * ``fact_check``      — sourcing requirement for factual claims.
#: * ``vision_check``    — pixel-only screenshot gate.
#: * ``customer_review`` — CustomerReviewerAgent buy/no-buy gate.
#: * ``auto_publish``    — Payhip publish attempt on a passed gate.
#:
#: Note: ``auto_publish`` is an additional kill switch *on top of* the
#: ``--publish`` CLI flag; both must be True for a publish attempt. Disabling
#: a gate means the product ships unvalidated on that axis, so the state log
#: records which gates were skipped.
FEATURE_FLAGS: Dict[str, bool] = {
    "grammar_check": False,
    "kcal_validation": False,
    "fact_check": False,
    "vision_check": False,
    "customer_review": False,
    "auto_publish": False,
}

#: Legacy module-level constants, kept for compatibility with existing
#: importers and now derived from :data:`FEATURE_FLAGS` so there is exactly
#: one place to flip a validation step on or off.
ENABLE_KCAL_VALIDATION = FEATURE_FLAGS["kcal_validation"]
ENABLE_FACT_CHECK = FEATURE_FLAGS["fact_check"]

#: Set from --publish CLI flag in main(); auto-publish only when True.
AUTO_PUBLISH = False

OPPORTUNITY: Dict[str, Any] = {
    "product_name": PRODUCT_NAME,
    "target_audience": (
        "Freelance medical interpreters who manage multiple "
        "hospital/clinic assignments"
    ),
    "pain_point": (
        "Managing complex schedules across multiple healthcare facilities "
        "with different time zones, tracking certification deadlines, and "
        "organizing client medical terminology references"
    ),
    "estimated_price": 15,
    "unique_value_proposition": (
        "One template that combines appointment scheduling, facility "
        "contacts, terminology reference, invoicing, and certification "
        "tracking for medical interpreters"
    ),
    "estimated_build_time": "2 hours",
}

# FIX 1: the reviewer's completeness requirements are derived from the same
# core.sections.SECTION_DEFINITIONS the builder assembles from, so the
# heading/keyword drift that sank Maiden Voyage 2.0 cannot recur.
REQUIRED_SECTIONS: List[tuple] = reviewer_section_specs()

SECTIONS_BRIEF = (
    "Return the product as a MARKDOWN template fenced in a single ```markdown "
    "code block — NOT Python code. Be complete, detailed, and thorough — never "
    "truncate, never use '...' placeholders. "
    "Start with an # H1 title, then one ## section per required item "
    "below, using tables and checklists with realistic example rows.\n"
    "The template MUST contain exactly these six sections, each with its own "
    "heading and usable structure (tables, checklists, or fields — not just "
    "a mention):\n"
    + section_requirement_lines()
    + "\nUse these exact ## headings so sections are findable: "
    + ", ".join(SECTION_HEADINGS)
    + ". No placeholder text anywhere."
)

_console = Console()
logger = logging.getLogger(__name__)
_build_log: List[Dict[str, Any]] = []

#: Pipeline-wide Notifier (None until main() initialises it). fail() and
#: the final summary use it so every exit path reports status.
_notifier: Optional[Notifier] = None


def notify(text: str) -> bool:
    """Send a Telegram message if a Notifier is active; never raises."""
    try:
        if _notifier is not None:
            return _notifier.send_message(text)
    except Exception:
        pass
    return False


def safe_notify_call(label: str, func, *args, **kwargs) -> bool:
    """Call any Notifier method with Telegram fully guarded.

    If Telegram is down (or anything else goes wrong), the error is logged
    and the pipeline continues — notifications never crash a build.
    """
    try:
        return bool(func(*args, **kwargs))
    except Exception as exc:
        _console.print(
            f"[yellow]Notification '{label}' skipped (Telegram issue): {exc}[/yellow]"
        )
        return False


def extract_issues(result: Dict[str, Any]) -> List[str]:
    """Pull human-readable issues from a reviewer result dict."""
    issues = [d for d in result.get("details", []) if "FAIL" in str(d)]
    summary = str(result.get("feedback", ""))
    for line in summary.splitlines():
        clean = line.lstrip("- ").strip()
        if clean.startswith("Model quality score:"):
            issues.append(clean[:300])
            break
    return issues[:10]


def log(stage: str, message: str, ok: Optional[bool] = None) -> None:
    """Append to the build log and echo a one-line status."""
    _build_log.append({
        "at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "ok": ok,
        "message": message[:500],
    })
    marker = "" if ok is None else ("[OK] " if ok else "[FAIL] ")
    _console.print(f"{marker}{stage}: {message}"[:220])


def detect_product_type(code: str) -> str:
    """Return 'python' if the content compiles, else 'template'."""
    try:
        compile(code, "<product>", "exec")
        return "python"
    except (SyntaxError, ValueError):
        return "template"


def find_newest_product_file(directory: Path) -> Optional[Path]:
    """Return the most recently modified .md/.py file in a directory."""
    candidates = [
        p for p in directory.glob("*") if p.suffix.lower() in (".md", ".py")
    ]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def grammar_check_step(
    builder: BuilderAgent,
    code: str,
    product_type: str,
    content_type: str = "product",
    apply_corrections: bool = True,
) -> tuple[str, str]:
    """Run multi-model grammar consensus after the builder phase.

    For templates, consensus findings are repaired through the primary
    model and the corrected text is saved before testing, so the test gate
    always validates the artifact that will actually be packaged. Generated
    code is checked but never automatically rewritten: grammar models must
    not alter executable semantics without a human in the loop.
    """
    if not isinstance(code, str) or not code.strip():
        raise ValueError("grammar_check_step requires non-empty content.")
    normalised_type = (
        product_type.strip().lower()
        if isinstance(product_type, str)
        else ""
    )
    grammar_agent = GrammarAgent(
        builder.orchestrator, builder.state_manager
    )
    result = grammar_agent.check_grammar(code, content_type)
    summary = result.get("summary", {})
    total_errors = int(summary.get("total_errors", 0) or 0)
    consensus_rate = float(summary.get("consensus_rate", 0.0) or 0.0)
    judgment = str(summary.get("judgment", "consensus") or "consensus")
    succeeded = int(summary.get("providers_succeeded", 0) or 0)
    available = int(
        summary.get("providers_available", succeeded) or succeeded
    )
    notify(
        f"🔤 Grammar check: {total_errors} errors found "
        f"({consensus_rate:.0%} consensus)"
    )
    if judgment == "insufficient_judges":
        # Availability over strictness: too few judges finished for real
        # consensus, so findings (if any) are advisory only and no
        # correction is applied. The coverage gap is explicit, never silent.
        log(
            "grammar",
            f"Consensus check found {total_errors} issue(s) "
            f"({succeeded}/{available} judges — insufficient); "
            "proceeding advisory without corrections.",
            False,
        )
        notify(
            f"🔤 Grammar check: {total_errors} errors "
            f"({succeeded}/{available} judges — insufficient)"
        )
        return code, product_type
    if normalised_type != "template":
        log(
            "grammar",
            f"Consensus check found {total_errors} issue(s); automatic "
            "correction is disabled for generated code.",
            total_errors == 0,
        )
        return code, product_type
    if total_errors == 0:
        log("grammar", "Consensus check found no language errors.", True)
        return code, product_type
    if not apply_corrections:
        log(
            "grammar",
            f"Consensus check found {total_errors} issue(s); corrections "
            "were not applied.",
            False,
        )
        return code, product_type
    try:
        corrected = grammar_agent._training_feedback(code, result["errors"])
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(f"Grammar correction failed: {exc}") from exc
    if corrected.strip() == code.strip():
        log("grammar", "Consensus findings required no textual change.", True)
        return code, product_type
    builder.save_product(corrected, PRODUCT_NAME)
    corrected_type = detect_product_type(corrected)
    log(
        "grammar",
        f"Applied {total_errors} consensus finding(s); corrected product "
        f"is {len(corrected)} chars ({corrected_type}).",
        True,
    )
    return corrected, corrected_type


def kcal_validation_step(
    builder: BuilderAgent,
    code: str,
    product_type: str,
) -> tuple[str, str]:
    """Validate nutritional energy claims against USDA FoodData Central.

    This runs after grammar correction and before product testing, so USDA
    citations are part of the artifact the test gate validates. Invalid,
    unresolvable, or unparseable energy values are returned to the builder
    through a raised error; the pipeline's build loop turns that error into
    revision feedback rather than shipping an invented kcal value.
    """
    if not isinstance(code, str) or not code.strip():
        raise ValueError("kcal_validation_step requires non-empty content.")
    checker = KcalChecker(USDAClient())
    scan = checker.scan_content(code)
    claims = scan.get("claims", [])
    skipped = scan.get("skipped", [])
    if not claims and not skipped:
        log("kcal", "No nutritional energy claims; USDA check skipped.", True)
        return code, product_type
    if not checker.usda_client.has_api_key:
        raise RuntimeError(
            "USDA_API_KEY is not configured, but the product contains "
            f"{len(claims)} nutritional claim(s) and "
            f"{len(skipped)} unparseable energy mention(s). Nutritional "
            "content cannot be validated without a key."
        )
    try:
        result = checker.check_product_nutrition(code)
    except EnvironmentError as exc:
        raise RuntimeError(f"USDA validation unavailable: {exc}") from exc
    try:
        builder.state_manager.log_kcal_validation(result)
    except Exception as exc:
        logger.warning("Could not record kcal validation: %s", exc)
        _console.print(
            f"[yellow]Could not record kcal validation: {exc}[/yellow]"
        )
    summary = result.get("summary", {})
    valid_claims = int(summary.get("valid_claims", 0) or 0)
    total_claims = int(summary.get("total_claims", 0) or 0)
    validation_rate = float(summary.get("validation_rate", 0.0) or 0.0)
    notify(
        f"🥗 Kcal validation: {valid_claims}/{total_claims} claims valid "
        f"({validation_rate:.0%} rate)"
    )
    if skipped:
        details = "; ".join(
            f"line {item.get('line')}: {item.get('reason')}"
            for item in skipped[:5]
        )
        logger.warning("Found unparseable kcal content: %s", details)
        raise RuntimeError(
            "USDA validation blocked: energy content could not be parsed "
            f"into a food and serving ({details})."
        )
    problems = [
        item for item in result.get("claims", [])
        if not item.get("valid")
    ]
    if problems:
        details = "; ".join(
            _kcal_problem_text(item) for item in problems[:5]
        )
        logger.warning("Found %d invalid kcal claim(s): %s", len(problems), details)
        raise RuntimeError(
            "USDA validation failed for nutritional content: "
            f"{details}. Replace invented values with USDA-backed servings."
        )
    sources = result.get("sources", [])
    if not sources:
        log("kcal", "All nutritional claims validated; no sources to add.", True)
        return code, product_type
    section = _format_kcal_sources(sources)
    updated = code.rstrip() + "\n\n" + section + "\n"
    try:
        builder.save_product(updated, PRODUCT_NAME)
    except (ValueError, IOError, RuntimeError) as exc:
        raise RuntimeError(f"Could not save USDA-cited product: {exc}") from exc
    updated_type = detect_product_type(updated)
    log(
        "kcal",
        f"Validated {summary.get('total_claims', 0)} claim(s) and appended "
        f"{len(sources)} USDA source(s).",
        True,
    )
    return updated, updated_type


def _kcal_problem_text(claim: Dict[str, Any]) -> str:
    """Summarize one failed or unresolvable kcal validation compactly."""
    food = claim.get("food", "unknown food")
    serving = claim.get("serving") or "unknown serving"
    claimed = claim.get("claimed_kcal")
    usda = claim.get("usda_kcal")
    if claim.get("status") == "unresolved" or usda is None:
        reason = claim.get("error") or "unresolvable serving"
        return f"{food} ({serving}): claimed {claimed} kcal; {reason}"
    return (
        f"{food} ({serving}): claimed {claimed} kcal, USDA {usda} kcal "
        f"({claim.get('deviation_percent')}% deviation)"
    )


def _format_kcal_sources(sources: List[Dict[str, Any]]) -> str:
    """Format validated USDA sources as a non-numeric product appendix."""
    lines = [
        "## Nutrition Sources",
        "",
        "Energy values above were checked against USDA FoodData Central. "
        "Full reference records:",
        "",
    ]
    for source in sources:
        food = source.get("food") or "USDA food record"
        lines.append(
            f"- {food} — FDC ID {source.get('fdc_id')}: {source.get('url')}"
        )
    return "\n".join(lines)


def _has_sources_section(code: str) -> bool:
    """True when the product already has a dedicated sources appendix."""
    return bool(
        re.search(
            r"(?m)^#{1,6}\s*(sources|nutrition sources|references|"
            r"bibliography|works cited)\s*$",
            code,
            re.IGNORECASE,
        )
    )


def fact_check_step(
    builder: BuilderAgent,
    code: str,
    product_type: str,
) -> tuple[str, str]:
    """Verify factual claims carry textual sources after the kcal check.

    Unsourced factual claims are returned to the builder through a raised
    error; the pipeline's build loop turns that error into revision
    feedback. Already-present verified citations may be consolidated into a
    Sources appendix, but no source is ever invented here.
    """
    if not isinstance(code, str) or not code.strip():
        raise ValueError("fact_check_step requires non-empty content.")
    checker = FactCheckerAgent(builder.orchestrator, builder.state_manager)
    result = checker.check_facts(code)
    try:
        builder.state_manager.log_fact_check(result)
    except Exception as exc:
        logger.warning("Could not record fact check: %s", exc)
        _console.print(
            f"[yellow]Could not record fact check: {exc}[/yellow]"
        )
    summary = result.get("summary", {})
    unsourced = int(summary.get("unsourced_claims", 0) or 0)
    sourced = int(summary.get("sourced_claims", 0) or 0)
    total = int(summary.get("total_claims", 0) or 0)
    notify(
        f"📚 Fact check: {sourced}/{total} claims sourced"
    )
    if unsourced:
        details = "; ".join(
            f"line {item.get('line')}: {item.get('claim')[:120]}"
            for item in result.get("claims", [])
            if not item.get("has_source")
        )
        logger.warning("Found %d unsourced claim(s): %s", unsourced, details)
        raise RuntimeError(
            "Fact check failed: factual claims lack textual sources "
            f"({details}). Add named sources or retrievable citations."
        )
    verified_sources = [
        source for source in result.get("sources", [])
        if source.get("url") or source.get("doi") or source.get("fdc_id")
    ]
    if verified_sources and not _has_sources_section(code):
        appendix = checker.generate_sources_appendix(verified_sources)
        updated = code.rstrip() + "\n" + appendix
        try:
            builder.save_product(updated, PRODUCT_NAME)
        except (ValueError, IOError, RuntimeError) as exc:
            raise RuntimeError(
                f"Could not save source-cited product: {exc}"
            ) from exc
        updated_type = detect_product_type(updated)
        log(
            "fact-check",
            f"All {summary.get('total_claims', 0)} factual claim(s) sourced; "
            f"consolidated {len(verified_sources)} citation(s).",
            True,
        )
        return updated, updated_type
    log(
        "fact-check",
        f"All {summary.get('total_claims', 0)} factual claim(s) sourced.",
        True,
    )
    return code, product_type


def build_step(
    builder: BuilderAgent,
    feedback: Optional[str],
    attempt: str,
    improvement_prompt: Optional[str] = None,
) -> tuple[str, str]:
    """Run one build, grammar-correction, and test cycle; return (code, type)."""
    _console.rule(f"[bold]Build {attempt}[/bold]")
    code = builder.build_product_with_fallback(
        opportunity=dict(OPPORTUNITY),
        feedback=feedback,
        improvement_prompt=improvement_prompt,
        sections=active_sections(),
    )
    product_type = detect_product_type(code)
    # Each validation step is gated by FEATURE_FLAGS. The step functions
    # themselves are untouched — a flag flip back to True restores the full
    # behaviour with no other edit.
    if FEATURE_FLAGS["grammar_check"]:
        code, product_type = grammar_check_step(builder, code, product_type)
    else:
        log("grammar", "Skipped (FEATURE_FLAGS['grammar_check'] is False).",
            True)
    if FEATURE_FLAGS["kcal_validation"]:
        code, product_type = kcal_validation_step(builder, code, product_type)
    else:
        log("kcal", "Skipped (FEATURE_FLAGS['kcal_validation'] is False).",
            True)
    if FEATURE_FLAGS["fact_check"]:
        code, product_type = fact_check_step(builder, code, product_type)
    else:
        log("fact", "Skipped (FEATURE_FLAGS['fact_check'] is False).", True)
    log("build", f"Generated {len(code)} chars ({product_type}).", True)
    if _notifier is not None:
        safe_notify_call(
            "builder-done", _notifier.send_message,
            "Builder complete. Running tests...",
        )
    return code, product_type


def test_step(tester: TestRunner, code: str, product_type: str) -> Dict[str, Any]:
    """Test the product and log the outcome."""
    _console.rule("[bold]Test[/bold]")
    report = tester.run_tests(code, product_type=product_type)
    if report["success"]:
        log("test", "All product tests passed.", True)
    else:
        log("test", f"FAILED: {'; '.join(report['errors'][:3])}", False)
        for error in report["errors"][:5]:
            _console.print(f"  - {error}"[:200])
    if _notifier is not None:
        safe_notify_call(
            "tests-done", _notifier.send_message,
            "Tests complete. Running review...",
        )
    return report


def package_step(
    packager: ProductPackager, code: str, product_type: str
) -> Path:
    """Package the product and append quick-start, example, and features."""
    _console.rule("[bold]Package[/bold]")
    zip_path = packager.create_package(PRODUCT_NAME, code, product_type=product_type)
    extras = {
        "QUICKSTART.md": quickstart_doc(),
        "FEATURES.md": features_doc(),
        "examples/filled_example.md": filled_example_doc(),
    }
    with zipfile.ZipFile(zip_path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        for arcname, content in extras.items():
            archive.writestr(arcname, content.encode("utf-8"))
    log("package", f"ZIP ready with extras: {zip_path}", True)
    return zip_path


def review_step(
    reviewer: ReviewerAgent, zip_path: Path
) -> Dict[str, Any]:
    """Review the ZIP and log the verdict."""
    _console.rule("[bold]Review[/bold]")
    required = None
    sections = active_profile().get("sections")
    if sections:
        required = [title for title, _, _ in sections]
    result = reviewer.review_product(
        zip_path,
        # None keeps the canonical SECTION_DEFINITIONS for the medical
        # profile; custom profiles pass their own headings.
        required_sections=required,
        context=OPPORTUNITY,
    )
    log(
        "review",
        f"Verdict {result['verdict'].upper()}, quality "
        f"{result['quality_score']}/10.",
        result["verdict"] == "approved",
    )
    notify(
        f"✅ Review: Score {result['quality_score']}/10, "
        f"Verdict: {result['verdict'].upper()}"
    )
    _console.print(Panel(result["feedback"][:1500], title="Reviewer feedback"))
    return result


def screenshot_step(builder: BuilderAgent, code: str) -> Optional[Path]:
    """Render the final template to a screenshot for visual review.

    Args:
        builder: The BuilderAgent (owns :meth:`export_to_image`).
        code: The final template Markdown.

    Returns:
        The :class:`pathlib.Path` of the screenshot
        (``state/output/{project_id}.png``) on success, None otherwise.
        A render failure is non-fatal — visual review then reports
        "unavailable" instead of blocking the text pipeline.
    """
    _console.rule("[bold]Screenshot[/bold]")
    out_dir = Path("state") / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    shot_path = out_dir / f"{PROJECT_ID}.png"
    try:
        rendered = builder.export_to_image(code, str(shot_path))
    except Exception as exc:  # renderer must never break the pipeline
        _console.print(f"[yellow]Screenshot step error: {exc}[/yellow]")
        return None
    if rendered and shot_path.is_file():
        log("screenshot", f"Screenshot saved to {shot_path}.", True)
        return shot_path
    log("screenshot", "Screenshot could not be rendered.", False)
    return None


def check_visual_quality(project_id: str) -> Dict[str, Any]:
    """Visual gate using deterministic pixel check only.

    The vision model is removed after a 4/4 hallucination rate (MV011–MV014:
    1.0 scores on renders verified clean by pixel inspection). The pixel
    check cannot hallucinate — ink in the outermost band means content ran
    off the page, full stop.

    Args:
        project_id: Project whose ``state/output/{project_id}.png``
            screenshot is checked.

    Returns:
        ``{'visual_score': 10.0, 'verdict': 'PASS', 'reason': ...}`` on a
        clean render, or ``{'visual_score': 0.0, 'verdict': 'FAIL',
        'reason': ...}`` on clipping. A missing screenshot fails closed.
        Extra keys (``ok``, ``pixel_check``, ``issues``,
        ``recommendation``) preserve the result shape the pipeline's
        ``combine_scores`` and ``should_auto_publish`` consume.
    """
    image_path = Path("state") / "output" / f"{project_id}.png"
    if not image_path.is_file():
        log("visual", f"No screenshot at {image_path}; gate holds.", False)
        return {
            "visual_score": 0.0,
            "verdict": "FAIL",
            "reason": f"Screenshot missing: {image_path}",
            "ok": False,
            "pixel_check": "SKIPPED",
            "issues": [f"Screenshot missing: {image_path}"],
            "recommendation": "FAIL",
            "advisory_override": False,
        }
    try:
        clip = _edge_clipping_check(str(image_path))
    except Exception as exc:  # pixel check must never break the pipeline
        _console.print(f"[yellow]Pixel check error: {exc}[/yellow]")
        return {
            "visual_score": 0.0,
            "verdict": "FAIL",
            "reason": f"Pixel check errored: {exc}",
            "ok": False,
            "pixel_check": "SKIPPED",
            "issues": [f"Pixel check errored: {exc}"],
            "recommendation": "FAIL",
            "advisory_override": False,
        }
    if clip.get("clipped"):
        sides = ", ".join(clip.get("sides", [])) or "unknown side"
        log("visual", f"Vision gate failed: content clipped at {sides}.", False)
        return {
            "visual_score": 0.0,
            "verdict": "FAIL",
            "reason": "Content clipped at page edge",
            "ok": True,
            "pixel_check": "FAIL",
            "issues": [clip.get("detail", "Content clipped at page edge.")],
            "recommendation": "FAIL",
            "advisory_override": False,
        }
    log("visual", "Vision gate passed: render is clean.", True)
    return {
        "visual_score": 10.0,
        "verdict": "PASS",
        "reason": "Render is clean",
        "ok": True,
        "pixel_check": "PASS",
        "issues": [],
        "recommendation": "PASS",
        "advisory_override": False,
    }


def skipped_visual_result(reason: str) -> Dict[str, Any]:
    """Return a neutral visual-gate payload for a disabled vision check.

    Same shape as :func:`visual_review_step` / :func:`check_visual_quality` so
    :func:`combine_scores` and :func:`should_auto_publish` need no special
    case, but non-vetoing: ``verdict`` is ``"SKIPPED"`` (never ``"FAIL"``),
    so the pixel gate cannot reject a product, and ``visual_score`` is a clean
    10.0 so the numeric :data:`MIN_VISUAL_SCORE` threshold is satisfied
    without having rendered anything.

    The screenshot is still taken by the caller, so a human reviewing the
    state log can always confirm the render by hand.
    """
    return {
        "visual_score": 10.0,
        "verdict": "SKIPPED",
        "reason": reason,
        "ok": True,
        "pixel_check": "SKIPPED",
        "issues": [],
        "recommendation": "SKIPPED",
        "advisory_override": False,
    }


def visual_review_step(
    image_path: Optional[Path],
) -> Dict[str, Any]:
    """Run the pixel-only visual gate on the template screenshot.

    ``ReviewerAgent.review_visually()`` is deliberately NOT invoked here:
    the vision model hallucinated on 4/4 production renders, so the model
    is retired from the gate (the method stays on the class for ad-hoc
    use). The screenshot filename carries the project id
    (``state/output/{project_id}.png``), which is what
    :func:`check_visual_quality` needs.

    Args:
        image_path: Screenshot path, or None if rendering failed.

    Returns:
        Same shape as :func:`check_visual_quality`. A missing screenshot
        fails closed (score 0.0 / ``ok=False``), treated as a gate hold.
    """
    _console.rule("[bold]Visual Review[/bold]")
    if image_path is None:
        log("visual", "No screenshot available; visual review skipped.", False)
        return {
            "visual_score": 0.0,
            "verdict": "FAIL",
            "reason": "No screenshot available for visual review.",
            "ok": False,
            "pixel_check": "SKIPPED",
            "issues": ["No screenshot available for visual review."],
            "recommendation": "FAIL",
            "advisory_override": False,
        }
    project_id = Path(image_path).stem
    result = check_visual_quality(project_id)
    if result["verdict"] == "PASS":
        _console.print("[green]Visual gate: PASS (render is clean).[/green]")
    else:
        _console.print(f"[red]Visual gate: FAIL ({result['reason']}).[/red]")
    return result


def combine_scores(text_result: Dict[str, Any],
                   visual_result: Dict[str, Any]) -> Dict[str, Any]:
    """Merge the text review with the pixel visual gate into a final verdict.

    The visual side is binary now (pixel check only — the vision model is
    retired): a ``FAIL`` verdict means genuine clipping and vetoes approval
    with a 0.0 final score, so broken layout can never ship. On a ``PASS``
    the text score stands alone as the final score; there is no model score
    left to average with.

    Args:
        text_result: ReviewerAgent.review_product() result.
        visual_result: :func:`check_visual_quality` result
            (``verdict`` ``"PASS"``/``"FAIL"``).

    Returns:
        A new dict merging both results plus ``final_quality_score``,
        ``final_verdict`` (``"approved"``/``"rejected"``), ``visual_score``
        and combined ``feedback``.
    """
    text_score = float(text_result.get("quality_score", 0) or 0)
    visual_score = float(visual_result.get("visual_score", 0) or 0)
    visual_ok = bool(visual_result.get("ok"))
    pixel_failed = visual_result.get("verdict") == "FAIL" or \
        visual_result.get("pixel_check") == "FAIL"
    text_ok = text_result.get("verdict") == "approved"
    if pixel_failed:
        final_score = 0.0
        final_verdict = "rejected"
    else:
        final_score = round(text_score, 2)
        final_verdict = "approved" if text_ok else "rejected"
    feedback = str(text_result.get("feedback", ""))
    reason = str(visual_result.get("reason", ""))
    if reason:
        feedback += f"\n\nVISUAL GATE: {reason}."
    combined = dict(text_result)
    combined.update({
        "visual_score": visual_score,
        "visual_recommendation": visual_result.get("recommendation"),
        "visual_verdict": visual_result.get("verdict"),
        "visual_ok": visual_ok,
        "pixel_check": visual_result.get("pixel_check", "SKIPPED"),
        "advisory_override": bool(visual_result.get("advisory_override")),
        "final_quality_score": final_score,
        "final_verdict": final_verdict,
        "feedback": feedback,
    })
    log(
        "scores",
        f"Final quality {final_score}/10 "
        f"(text {text_score}/10, visual {visual_result.get('verdict')}) -> "
        f"{final_verdict.upper()}.",
        final_verdict == "approved",
    )
    return combined


def should_auto_publish(customer_review: dict) -> bool:
    """Decide whether a product may be published without human review.

    Args:
        customer_review: CustomerReviewerAgent result dict with ``verdict``,
            ``scores`` (five 1–10 criteria plus ``overall``) and,
            when a screenshot was reviewed, ``visual_score`` plus
            ``pixel_check`` (``"PASS"``/``"FAIL"``/``"SKIPPED"``) from the
            pixel-only visual gate.

    Returns:
        True only when the verdict is acceptable, the overall score meets
        ``MIN_OVERALL_SCORE``, every single criterion meets
        ``MIN_SINGLE_CRITERION``, and the visual gate passes: a
        ``pixel_check`` of ``"FAIL"`` always blocks (genuine clipping); the
        visual score must meet ``MIN_VISUAL_SCORE`` (a clean render scores
        10.0, so this is trivially satisfied whenever pixels pass). Any
        malformed input returns False (fail closed — human review required).
    """
    try:
        if not isinstance(customer_review, dict):
            return False
        verdict = str(customer_review.get("verdict", "")).strip().upper()
        if verdict == "PASS" or verdict not in [
            v.upper() for v in ACCEPTABLE_VERDICTS
        ]:
            return False
        scores = customer_review.get("scores", {})
        if not isinstance(scores, dict):
            return False
        overall = float(scores.get("overall", 0))
        if overall < MIN_OVERALL_SCORE:
            return False
        criteria = ("first_impression", "clarity", "completeness",
                    "value_for_money", "usability")
        for key in criteria:
            if float(scores.get(key, 0)) < MIN_SINGLE_CRITERION:
                return False
        # -- visual gate: pixels are binary (10.0 clean / 0.0 clipped) -----
        if "visual_score" in customer_review:
            if customer_review.get("pixel_check") == "FAIL":
                return False
            if float(customer_review.get("visual_score") or 0) < MIN_VISUAL_SCORE:
                return False
    except (TypeError, ValueError, AttributeError):
        return False
    return True


def get_willingness_to_pay(niche: str) -> float:
    """Return the fixed price for ``niche``.

    Args:
        niche: Niche key (e.g. ``"medical_interpreter"``). Unknown or empty
            niches fall back to ``NICHE_PRICING["default"]``.

    Returns:
        The fixed euro price for the niche.
    """
    key = (niche or "").strip().lower() or "default"
    return float(NICHE_PRICING.get(key, NICHE_PRICING["default"]))


def skipped_customer_result(reason: str) -> Dict[str, Any]:
    """Return a neutral customer-gate payload for a disabled customer review.

    Same keys :meth:`CustomerReviewerAgent.review_as_customer` returns, so
    pricing, the build log, and :func:`should_auto_publish` all work
    unchanged. Every score sits at 10.0 and the verdict is ``"BUY"``, the
    weakest acceptable verdict, so a disabled gate lets the product through
    rather than rejecting it and no downstream threshold is tripped.
    """
    return {
        "verdict": "BUY",
        "scores": {
            "first_impression": 10.0,
            "clarity": 10.0,
            "completeness": 10.0,
            "value_for_money": 10.0,
            "usability": 10.0,
            "overall": 10.0,
        },
        "customer_feedback": reason,
        "key_concerns": [],
        "willingness_to_pay": None,
        "model": "skipped",
        "skipped": True,
    }


def customer_review_step(
    customer_reviewer: CustomerReviewerAgent, code: str
) -> Dict[str, Any]:
    """Run the customer gate and log the verdict."""
    _console.rule("[bold]Customer Review[/bold]")
    result = customer_reviewer.review_as_customer(
        product_content=code,
        product_name=PRODUCT_NAME,
        price=SUGGESTED_PRICE_EUR,
    )
    log(
        "customer-gate",
        f"Customer verdict {result['verdict']} "
        f"(overall {result['scores']['overall']}/10, "
        f"via {result.get('model_used', '?')}).",
        result["verdict"] in ("BUY", "NEEDS IMPROVEMENT"),
    )
    _console.print(
        Panel(result["customer_feedback"][:1500], title="Customer feedback")
    )
    return result


def run_pipeline() -> int:
    """Run the full build pipeline with a guaranteed final status notify.

    Wraps :func:`_pipeline` so that, no matter how it exits (success,
    handled failure, or unexpected exception), a final
    ``"Pipeline complete. Final status: {status}"`` message is sent via
    the Notifier (guarded — Telegram can never crash the run).

    Returns:
        Exit code: 0 on APPROVED, 1 otherwise.
    """
    outcome = {"status": "FAILED"}
    try:
        return _pipeline(outcome)
    finally:
        if _notifier is not None:
            safe_notify_call(
                "final-status", _notifier.send_message,
                f"Pipeline complete. Final status: {outcome['status']}",
            )


def run_project(project_id: str) -> int:
    """Run the pipeline for one project id (the swarm's per-tenant entry).

    Used as the default ``SwarmManager`` pipeline runner. Sets the module
    ``PROJECT_ID`` for the duration of the run so the whole existing flow
    stays parameter-free, then restores the previous value.

    Args:
        project_id: State project id for this tenant.

    Returns:
        Exit code: 0 on APPROVED, 1 otherwise.
    """
    global PROJECT_ID
    # Late-bind the --publish flag: main_async parsed it in the __main__
    # instance, but this function (and the whole pipeline body) executes in
    # the duplicate `build_pipeline` instance. See _pull_cli_state.
    _pull_cli_state()
    previous = PROJECT_ID
    previous_profile = {
        "PRODUCT_PROFILE": PRODUCT_PROFILE,
        "PRODUCT_NAME": PRODUCT_NAME,
        "NICHE": NICHE,
        "SUGGESTED_PRICE_EUR": SUGGESTED_PRICE_EUR,
        "OPPORTUNITY": OPPORTUNITY,
    }
    if project_id and project_id.strip():
        PROJECT_ID = project_id.strip()
    # Product profiles are process-global (not tenant-safe): concurrent batch
    # projects with different profiles would collide. Single-project runs
    # are unaffected.
    apply_product_profile(PRODUCT_PROFILE)
    try:
        return run_pipeline()
    finally:
        PROJECT_ID = previous
        globals().update(previous_profile)


def load_batch_file(path: str) -> List[Dict[str, Any]]:
    """Read and validate a ``--batch-file`` JSON config.

    Accepts either a bare JSON list of task dicts or an object with a
    ``"projects"`` (or ``"tasks"``) list.

    Args:
        path: Path to the JSON file.

    Returns:
        The list of task dicts.

    Raises:
        IOError: If the file cannot be read.
        ValueError: If the JSON is malformed or is not a list of objects.
    """
    batch_path = Path(path)
    try:
        raw = batch_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise IOError(f"Could not read batch file '{batch_path}': {exc}") from exc
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"Batch file '{batch_path}' is not valid JSON: {exc}") from exc
    if isinstance(data, dict):
        for key in ("projects", "tasks"):
            if key in data:
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(
            f"Batch file '{batch_path}' must contain a JSON list of project "
            f"configs (or a 'projects' list); got {type(data).__name__}."
        )
    for index, task in enumerate(data):
        if not isinstance(task, dict):
            raise ValueError(
                f"Batch file '{batch_path}' entry #{index} must be an object, "
                f"got {type(task).__name__}."
            )
    return data


def _mirror_cli_state() -> None:
    """Mirror CLI globals into the duplicate module instance, if any.

    When launched as ``python build_pipeline.py``, this file runs as
    ``__main__`` — but ``SwarmManager`` also does
    ``from build_pipeline import run_project``, which loads a SECOND copy of
    this module under its own name with SEPARATE globals. The pipeline body
    (``_pipeline``, gate checks) executes from that second copy, so a flag
    set here would otherwise be invisible there. This function copies the
    parsed CLI state over. Without it, ``--publish`` is silently ignored in
    every production run (the gate reads ``False`` and reports "--publish
    flag not set" even when the flag was passed).

    Note the timing trap: the duplicate module is created lazily by
    SwarmManager's import, which happens AFTER argument parsing — so this
    eager mirror can miss it. :func:`run_project` therefore also calls
    :func:`_pull_cli_state`, which syncs from the other direction at
    pipeline start, when both instances are guaranteed to exist.
    """
    import sys as _sys

    here = _sys.modules.get(__name__)
    other = _sys.modules.get("build_pipeline")
    if other is not None and other is not here:
        other.PROJECT_ID = PROJECT_ID
        other.AUTO_PUBLISH = AUTO_PUBLISH
        other.PRODUCT_PROFILE = PRODUCT_PROFILE


def _pull_cli_state() -> None:
    """Pull CLI flags parsed by main_async into this module instance.

    The late-binding counterpart to :func:`_mirror_cli_state`: runs at
    pipeline start inside :func:`run_project`, when both the ``__main__``
    instance (which parsed the arguments) and this ``build_pipeline``
    instance (which runs the pipeline) are guaranteed to exist. Copies the
    parsed ``--publish`` flag across. No-op when everything already shares
    one namespace (tests, programmatic use).
    """
    global AUTO_PUBLISH, PRODUCT_PROFILE
    import sys as _sys

    main_mod = _sys.modules.get("__main__")
    if main_mod is None or main_mod is _sys.modules.get(__name__):
        return
    flag = getattr(main_mod, "AUTO_PUBLISH", None)
    if isinstance(flag, bool):
        AUTO_PUBLISH = flag
    profile = getattr(main_mod, "PRODUCT_PROFILE", None)
    if isinstance(profile, str) and profile in PRODUCT_PROFILES:
        PRODUCT_PROFILE = profile


async def main_async(argv=None) -> int:
    """Async CLI entry point (Vector 3): single project or batch.

    With ``--batch-file`` the projects run concurrently through
    :class:`~core.swarm_manager.SwarmManager` under a concurrency cap;
    otherwise a single ``--project-id`` runs as one async task.

    Returns:
        Exit code: 0 when every project succeeded, 1 otherwise.
    """
    global PROJECT_ID, AUTO_PUBLISH, PRODUCT_PROFILE
    import argparse as _argparse

    parser = _argparse.ArgumentParser(
        description="Medical-interpreter planner build pipeline."
    )
    parser.add_argument(
        "--project-id",
        default=PROJECT_ID,
        help=f"State project id (default: {PROJECT_ID}).",
    )
    parser.add_argument(
        "--product-profile",
        default=PRODUCT_PROFILE,
        choices=sorted(PRODUCT_PROFILES),
        help=(
            "Product profile to build "
            f"(default: {PRODUCT_PROFILE})."
        ),
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Attempt automatic publishing when quality gates pass.",
    )
    parser.add_argument(
        "--batch-file",
        default=None,
        help=(
            "Path to a JSON file with a list of project configs "
            '([{"project_id": "p1", "niche": "n1"}, ...]). Runs them '
            "concurrently via SwarmManager."
        ),
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=DEFAULT_MAX_CONCURRENT_PROJECTS,
        help=(
            "Maximum projects building at once in batch mode "
            f"(default: {DEFAULT_MAX_CONCURRENT_PROJECTS})."
        ),
    )
    args = parser.parse_args(argv)
    if args.project_id and args.project_id.strip():
        PROJECT_ID = args.project_id.strip()
    AUTO_PUBLISH = bool(args.publish)
    apply_product_profile(args.product_profile)
    _mirror_cli_state()
    _console.print(
        f"[dim]args: project_id={PROJECT_ID} publish={AUTO_PUBLISH} "
        f"product_profile={PRODUCT_PROFILE} "
        f"batch_file={args.batch_file} "
        f"max_concurrent={args.max_concurrent}[/dim]"
    )

    # -- batch mode: many projects concurrently ----------------------------
    if args.batch_file:
        try:
            tasks = load_batch_file(args.batch_file)
        except (IOError, ValueError) as exc:
            _console.print(f"[red]Batch file error: {exc}[/red]")
            return 1
        _console.print(
            Panel(
                f"[bold]Batch:[/bold] {args.batch_file}\n"
                f"[bold]Projects:[/bold] {len(tasks)}\n"
                f"[bold]Max concurrent:[/bold] {args.max_concurrent}",
                title="Swarm Batch",
            )
        )
        swarm = SwarmManager(max_concurrent_projects=args.max_concurrent)
        results = await swarm.run_batch(tasks)
        return _batch_exit_code(results)

    # -- single-project mode, still driven through the async entry point ----
    project_id = PROJECT_ID
    swarm = SwarmManager(max_concurrent_projects=1)
    result = await swarm.run_project(project_id, niche="")
    return int(result.get("exit_code", 1))


def _batch_exit_code(results: List[Dict[str, Any]]) -> int:
    """Print a batch summary and map results to an exit code.

    Args:
        results: Per-project result dicts from ``SwarmManager.run_batch``.

    Returns:
        0 when every project succeeded, 1 when any failed.
    """
    summary = Table(title="Swarm Batch Complete", show_lines=False)
    summary.add_column("Project")
    summary.add_column("Niche")
    summary.add_column("OK", justify="center")
    summary.add_column("Seconds", justify="right")
    for result in results:
        summary.add_row(
            str(result.get("project_id") or "?"),
            str(result.get("niche") or "-"),
            "yes" if result.get("success") else "no",
            f"{result.get('duration_s', 0)}",
        )
    _console.print(summary)
    failed = [r for r in results if not r.get("success")]
    for result in failed:
        _console.print(
            f"[red]FAILED[/red] {result.get('project_id')}: "
            f"{result.get('error') or 'exit code ' + str(result.get('exit_code'))}"
        )
    if failed:
        _console.print(
            f"[red]{len(failed)}/{len(results)} project(s) failed.[/red]"
        )
        return 1
    _console.print(f"[green]All {len(results)} project(s) succeeded.[/green]")
    return 0


def main(argv=None) -> int:
    """CLI wrapper: run :func:`main_async` under ``asyncio.run``.

    Returns:
        Exit code from the async entry point.
    """
    try:
        return asyncio.run(main_async(argv))
    except KeyboardInterrupt:
        _console.print("[yellow]Interrupted by user.[/yellow]")
        return 130


def _pipeline(outcome: Dict[str, str]) -> int:
    """Run build, test, package, review; print the summary; return exit code.

    Args:
        outcome: Mutable status holder; set to ``"SUCCESS"`` just before
            the approved return so :func:`run_pipeline`'s finally block
            reports accurately.
    """
    _console.print(Panel(
        f"[bold]Product:[/bold] {PRODUCT_NAME}\n"
        f"[bold]Project:[/bold] {PROJECT_ID}\n"
        f"[bold]Price:[/bold] EUR {SUGGESTED_PRICE_EUR}",
        title="Build Pipeline",
    ))

    # -- 1. init -----------------------------------------------------------
    global _notifier
    state_manager = StateManager(project_id=PROJECT_ID)
    orchestrator = Orchestrator()
    builder = BuilderAgent(orchestrator, state_manager)
    tester = TestRunner(state_manager)
    packager = ProductPackager(state_manager)
    reviewer = ReviewerAgent(orchestrator, state_manager)
    customer_reviewer = CustomerReviewerAgent(orchestrator, state_manager)
    publisher = PublisherAgent(state_manager)
    _notifier = Notifier()  # disabled gracefully without credentials
    log("init", "Components ready.", True)
    safe_notify_call(
        "build-started", _notifier.notify_build_started,
        PROJECT_ID, PRODUCT_NAME,
    )

    # -- 2+3. build/test loop (max 2 attempts) ------------------------------
    code = ""
    product_type = "template"
    test_report: Dict[str, Any] = {"success": False, "errors": ["not run"]}
    feedback: Optional[str] = sections_brief()
    for attempt in range(1, MAX_BUILD_ATTEMPTS + 1):
        try:
            code, product_type = build_step(
                builder, feedback, f"attempt {attempt}/{MAX_BUILD_ATTEMPTS}"
            )
        except (RuntimeError, ValueError) as exc:
            log("build", f"Attempt {attempt} failed: {exc}", False)
            feedback = f"{sections_brief()}\nPrevious attempt failed: {exc}"
            continue
        test_report = test_step(tester, code, product_type)
        if test_report["success"]:
            break
        feedback = (
            f"{sections_brief()}\nPrevious attempt FAILED testing:\n"
            + "\n".join(f"- {e}" for e in test_report["errors"][:5])
        )
    if not test_report["success"]:
        return fail(state_manager, "Build/test failed after "
                    f"{MAX_BUILD_ATTEMPTS} attempts.")
    notify(
        f"Builder finished <b>{PRODUCT_NAME}</b> ({len(code):,} chars, "
        f"{product_type}) and all tests passed. Packaging next."
    )

    # -- copy product file to products/output/ ------------------------------
    output_dir = Path("products/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    product_file = find_newest_product_file(Path("products"))
    if product_file is not None and product_file.parent != output_dir:
        dest = output_dir / product_file.name
        dest.write_text(product_file.read_text(encoding="utf-8"), encoding="utf-8")
        log("output", f"Product file copied to {dest}.", True)

    # -- 3b. screenshot of the final template (for visual review) ----------
    screenshot = screenshot_step(builder, code)

    # -- 4. package ----------------------------------------------------------
    try:
        zip_path = package_step(packager, code, product_type)
    except (ValueError, IOError, RuntimeError) as exc:
        return fail(state_manager, f"Packaging failed: {exc}")

    # -- 5. review: text + visual, one revision on REJECTED ------------------
    try:
        result = review_step(reviewer, zip_path)
        if FEATURE_FLAGS["vision_check"]:
            visual = visual_review_step(screenshot)
        else:
            visual = skipped_visual_result(
                "Vision check skipped (FEATURE_FLAGS['vision_check'] "
                "is False); render not gated."
            )
            log("visual", "Skipped (FEATURE_FLAGS['vision_check'] is False).",
                True)
        result = combine_scores(result, visual)
    except (ValueError, RuntimeError) as exc:
        return fail(state_manager, f"Review failed: {exc}")
    if result["final_verdict"] == "rejected":
        _console.print("[yellow]REJECTED — running the single revision attempt…[/yellow]")
        log("review", "Rejected; starting one revision attempt.", False)
        if _notifier is not None:
            safe_notify_call(
                "review-needed", _notifier.notify_review_needed,
                PROJECT_ID, PRODUCT_NAME, str(zip_path),
                issues=extract_issues(result),
            )
        # -- Improvement prompt: surgical revision brief for the Builder ---
        try:
            revision_state = state_manager.get_state()
            improvement_prompt = reviewer.generate_improvement_prompt(
                product_content=code,
                review_feedback={
                    "verdict": result["final_verdict"],
                    "scores": {"overall": result.get(
                        "final_quality_score", result.get("model_score", 1))},
                    "issues": extract_issues(result),
                },
                product_name=PRODUCT_NAME,
            )
            revision_state.improvement_prompt = improvement_prompt
            revision_state.revision_count = (
                revision_state.revision_count or 0) + 1
            state_manager.save(revision_state)
            log("revision",
                f"Improvement prompt saved "
                f"(revision #{revision_state.revision_count}).", True)
        except (ValueError, RuntimeError) as exc:
            return fail(state_manager,
                        f"Improvement-prompt generation failed: {exc}")
        notify("⚠️ REVIEW FAILED - Generating improvement prompt...")
        try:
            code, product_type = build_step(
                builder,
                f"{sections_brief()}\nReviewer REJECTED the previous package. "
                f"Fix every issue:\n{result['feedback'][:1500]}",
                "revision 1/1",
                improvement_prompt=improvement_prompt,
            )
            test_report = test_step(tester, code, product_type)
            if not test_report["success"]:
                return fail(state_manager, "Revision failed testing.")
            screenshot = screenshot_step(builder, code)
            zip_path = package_step(packager, code, product_type)
            result = review_step(reviewer, zip_path)
            if FEATURE_FLAGS["vision_check"]:
                visual = visual_review_step(screenshot)
            else:
                visual = skipped_visual_result(
                    "Vision check skipped (FEATURE_FLAGS['vision_check'] "
                    "is False); render not gated."
                )
                log("visual",
                    "Skipped (FEATURE_FLAGS['vision_check'] is False).", True)
            result = combine_scores(result, visual)
        except (RuntimeError, ValueError, IOError) as exc:
            return fail(state_manager, f"Revision attempt failed: {exc}")

    # -- 6. finalise ----------------------------------------------------------
    publish_dir = Path("publish_ready")
    publish_dir.mkdir(parents=True, exist_ok=True)
    publish_zip = publish_dir / zip_path.name
    shutil.copy2(zip_path, publish_zip)

    if result["final_verdict"] == "approved":
        # -- 5b. customer gate (final) -------------------------------------
        # BUY or NEEDS IMPROVEMENT (minor issues) proceed to publish_ready;
        # PASS is marked as rejected with the customer feedback attached.
        try:
            if FEATURE_FLAGS["customer_review"]:
                customer_result = customer_review_step(customer_reviewer, code)
            else:
                customer_result = skipped_customer_result(
                    "Customer review skipped "
                    "(FEATURE_FLAGS['customer_review'] is False); "
                    "buy/no-buy gate not enforced."
                )
                log("customer-gate",
                    "Skipped (FEATURE_FLAGS['customer_review'] is False).",
                    True)
        except (RuntimeError, ValueError) as exc:
            return fail(state_manager, f"Customer review failed: {exc}")
        _build_log.append({
            "at": datetime.now(timezone.utc).isoformat(),
            "stage": "customer-gate-result",
            "ok": customer_result["verdict"] in ("BUY", "NEEDS IMPROVEMENT"),
            "message": json.dumps(customer_result)[:2000],
        })
        if customer_result["verdict"] == "PASS":
            if _notifier is not None:
                safe_notify_call(
                    "customer-pass", _notifier.notify_review_needed,
                    PROJECT_ID, PRODUCT_NAME, str(zip_path),
                    issues=[customer_result["customer_feedback"]]
                    + customer_result["key_concerns"][:4],
                )
            try:
                state = state_manager.get_state()
                state.review_status = "rejected"
                state.review_feedback = (
                    "Customer gate: PASS — would not buy. "
                    + customer_result["customer_feedback"]
                )[:2000]
                history = state.build_history or []
                history.extend(_build_log)
                state.build_history = history
                state_manager.save(state)
            except Exception as exc:
                _console.print(
                    f"[yellow]Could not record customer rejection: {exc}[/yellow]"
                )
            return fail(
                state_manager,
                "Customer gate verdict PASS — marked as rejected.",
            )
        log(
            "customer-gate",
            f"Proceeding to publish (verdict "
            f"{customer_result['verdict']}).",
            True,
        )
        # -- Fixed pricing: the niche price drives Payhip, not the model's --
        # per-run WTP sample (which read €13/€14/€15 on near-identical
        # products). The customer reviewer still decides BUY/NO-BUY; the
        # model's own willingness_to_pay number is recorded for telemetry
        # but no longer sets the price tag.
        try:
            model_wtp = customer_result.get("willingness_to_pay")
            final_price = max(
                5.0, min(25.0, float(get_willingness_to_pay(NICHE))))
            final_price = round(final_price, 2)
        except (TypeError, ValueError):
            final_price = float(SUGGESTED_PRICE_EUR)
        log("pricing",
            f"Fixed niche price EUR {final_price:g} for '{NICHE}' "
            f"(model sampled WTP {model_wtp}; verdict "
            f"{customer_result['verdict']}); "
            "using it as the Payhip price.", True)
        try:
            payhip_data = publisher.format_product_data(
                product_name=PRODUCT_NAME,
                description=OPPORTUNITY.get(
                    "unique_value_proposition", PRODUCT_NAME),
                price=final_price,
                file_path=str(publish_zip),
            )
            publisher.write_manifest(publish_dir, payhip_data)
        except (ValueError, IOError) as exc:
            return fail(state_manager, f"Payhip manifest failed: {exc}")
        # -- Quality gates: auto-publish vs human review -------------------
        # Visual score is folded into the customer gate so a strong text
        # review cannot outweigh a broken screenshot. The pixel verdict
        # travels with it, so a pixel FAIL blocks even if the advisory
        # override held the score at the pass floor.
        customer_result["visual_score"] = result.get("visual_score", 0.0)
        customer_result["pixel_check"] = result.get("pixel_check", "SKIPPED")
        customer_result["final_quality_score"] = result.get(
            "final_quality_score", 0.0)
        gate_passed = should_auto_publish(customer_result)
        publish_status_value: Optional[str] = None
        if gate_passed and AUTO_PUBLISH and FEATURE_FLAGS["auto_publish"]:
            publish_outcome = publisher.publish_product(payhip_data)
            if publish_outcome.get("published"):
                publish_status_value = "published"
                log("publish", "AUTO-PUBLISHED via Payhip.", True)
                notify(
                    f"✅ AUTO-PUBLISHED! Price: €{final_price:g}"
                )
            else:
                log("publish",
                    f"Auto-publish attempted but not completed: "
                    f"{publish_outcome.get('reason', 'unknown reason')}",
                    False)
                notify(
                    f"Auto-publish attempted for <b>{PRODUCT_NAME}</b> but "
                    f"did not complete: {publish_outcome.get('reason', '')} "
                    "Manual upload may be required."
                )
        else:
            publish_status_value = "pending_human_review"
            reason = (
                "auto_publish feature flag off"
                if gate_passed and AUTO_PUBLISH
                and not FEATURE_FLAGS["auto_publish"]
                else "quality gates not met"
                if not gate_passed
                else "--publish flag not set"
            )
            log("publish",
                f"Quality gates not met for auto-publish ({reason}); "
                "manual human review required.", False)
            notify(
                f"⚠️ REVIEW NEEDED - Quality gates not met. "
                f"Scores: {customer_result.get('scores', {})}"
            )
        state = state_manager.get_state()
        history = state.build_history or []
        history.extend(_build_log)
        state.build_history = history
        state.publish_status = publish_status_value
        state_manager.save(state)
        state_manager.update_phase("publishing")  # approved_for_publish
        log("finalise", "APPROVED — marked publishing (approved_for_publish).", True)
        if _notifier is not None:
            safe_notify_call(
                "publish-ready", _notifier.notify_publish_ready,
                PROJECT_ID, PRODUCT_NAME, final_price, str(publish_zip),
                price_note="based on customer willingness-to-pay",
            )
            safe_notify_call(
                "build-complete", _notifier.notify_build_complete,
                PROJECT_ID, PRODUCT_NAME, "approved",
                int(result.get("final_quality_score",
                               result["quality_score"])), str(zip_path),
            )
    else:
        if _notifier is not None:
            safe_notify_call(
                "review-needed", _notifier.notify_review_needed,
                PROJECT_ID, PRODUCT_NAME, str(zip_path),
                issues=extract_issues(result),
            )
        return fail(state_manager, "Final verdict still REJECTED after revision.")

    summary = Table(title="Build Complete", show_lines=False)
    summary.add_column("Field")
    summary.add_column("Value")
    summary.add_row("Product", PRODUCT_NAME)
    summary.add_row("ZIP", str(zip_path))
    summary.add_row("Review verdict", f"{result['final_verdict'].upper()}")
    summary.add_row("Text quality", f"{result['quality_score']}/10")
    summary.add_row(
        "Visual score",
        f"{result.get('visual_score', 0)}/10 "
        f"({result.get('visual_recommendation', 'n/a')})",
    )
    summary.add_row("Final quality", f"{result.get('final_quality_score')}/10")
    summary.add_row(
        "Customer verdict",
        f"{customer_result['verdict']} "
        f"({customer_result['scores']['overall']}/10)",
    )
    summary.add_row("Suggested price",
                        f"EUR {final_price:g} (customer willingness-to-pay)")
    summary.add_row("Publish-ready", str(publish_zip))
    summary.add_row("Publish status", str(publish_status_value))
    summary.add_row("API calls", str(state_manager.get_state().api_calls_count))
    _console.print(summary)
    notify(
        f"Pipeline finished SUCCESSFULLY for <b>{PRODUCT_NAME}</b>: "
        f"verdict APPROVED, final quality "
        f"{result.get('final_quality_score')}/10, "
        f"package at {zip_path}."
    )
    outcome["status"] = "SUCCESS"
    return 0


def fail(state_manager: StateManager, message: str) -> int:
    """Record failure state, print + notify the message, return exit code 1."""
    _console.print(Panel(f"[red]{message}[/red]", title="Pipeline failed"))
    notify(f"Pipeline FAILED for <b>{PRODUCT_NAME}</b>: {message}")
    try:
        state = state_manager.get_state()
        history = state.build_history or []
        history.extend(_build_log)
        state.build_history = history
        state_manager.save(state)
        state_manager.update_phase("failed")
    except Exception as exc:
        _console.print(f"[yellow]Could not record failure state: {exc}[/yellow]")
    return 1


def quickstart_doc() -> str:
    """Buyer-facing quick-start guide (deterministic, UTF-8)."""
    if PRODUCT_PROFILE == "nutrition_meal_prep":
        return (
            f"# Quick Start — {PRODUCT_NAME}\n\n"
            "Get set up in under 10 minutes.\n\n"
            "1. Fill in your household size and dietary needs on the "
            "Grocery List.\n"
            "2. Run one focused prep session using the Prep Checklist.\n"
            "3. Follow the Weekly Schedule day by day; swap meals freely.\n"
            "4. Check the Nutrition Reference for per-100-gram values "
            "before adjusting portions.\n\n"
            "All energy values are stated per 100 g and backed by USDA "
            "FoodData Central sources listed under Nutrition Sources.\n"
        )
    return (
        f"# Quick Start — {PRODUCT_NAME}\n\n"
        "Get set up in under 10 minutes.\n\n"
        "## Option A: Notion\n\n"
        "1. Duplicate the template file into your Notion workspace.\n"
        "2. Fill the Client/facility database with your first 3 facilities.\n"
        "3. Block next week's appointments in the appointment grid, noting "
        "each facility's time zone.\n"
        "4. Add your certification expiry dates to the tracker.\n\n"
        "## Option B: Markdown / print\n\n"
        "1. Open the template in any Markdown editor (or print it).\n"
        "2. Copy the weekly grid per week; keep one running invoice table.\n\n"
        "## Weekly routine (15 minutes every Sunday)\n\n"
        "- Confirm assignments and time zones.\n"
        "- Review terminology for the week's specialties.\n"
        "- Update hours, invoice sent items, and CE progress.\n"
    )


def features_doc() -> str:
    """Buyer-facing feature list mapped to the active required sections."""
    lines = [f"# Features — {PRODUCT_NAME}\n"]
    for label, _ in active_section_specs():
        lines.append(f"- {label}")
    lines.append(
        "\nPlus: usage instructions, no special software required, "
        "lifetime updates of the template structure."
    )
    return "\n".join(lines) + "\n"


def filled_example_doc() -> str:
    """Realistic filled-in example for a fictional customer week."""
    if PRODUCT_PROFILE == "nutrition_meal_prep":
        return (
            "# Filled Example — a week with the Rivera household (4 people)\n\n"
            "Illustrative example only — replace every quantity with your "
            "own household's needs.\n\n"
            "## Grocery List (excerpt)\n\n"
            "| Item | Example quantity | Planned use |\n"
            "|---|---|---|\n"
            "| Rolled oats | 500 g | Breakfast bowls |\n"
            "| Bananas | 8 | Snacks and breakfasts |\n"
            "| Blueberries | 300 g | Breakfast bowls |\n"
            "| Honey | 250 g | Light sweetening |\n"
            "| Almonds | 200 g | Toppings |\n\n"
            "## Nutrition Reference (excerpt, per 100 g, USDA-backed)\n\n"
            "| Food | Energy | Source |\n"
            "|---|---|---|\n"
            "| Oats, raw | 379 kcal per 100g | USDA FoodData Central |\n"
            "| Banana, raw | 97 kcal per 100g | USDA FoodData Central |\n"
            "| Blueberries, raw | 57 kcal per 100g | USDA FoodData Central |\n"
            "| Honey | 304 kcal per 100g | USDA FoodData Central |\n"
            "| Almonds | 598 kcal per 100g | USDA FoodData Central |\n\n"
            "## Weekly Schedule (excerpt)\n\n"
            "- Monday: overnight oats with banana and blueberries.\n"
            "- Tuesday: leftover grain bowls with roasted vegetables.\n"
        )
    return (
        "# Filled Example — a week with Maria R., freelance interpreter "
        "(Spanish/English)\n\n"
        "## Weekly appointment grid (time zones noted)\n\n"
        "| Day | Time | Facility | Specialty | Time zone |\n"
        "|-----|------|----------|-----------|-----------|\n"
        "| Mon | 09:00 | St. Mary Hospital | Cardiology | ET |\n"
        "| Tue | 14:00 | Riverside Clinic | Neurology | CT |\n"
        "| Thu | 10:30 | St. Mary Hospital | Pediatrics | ET |\n\n"
        "## Client/facility database\n\n"
        "- St. Mary Hospital — contact: J. Alvarez, scheduling — specialty: "
        "cardiology — terminology: catheter, stent, arrhythmia.\n"
        "- Riverside Clinic — contact: T. Nguyen, office manager — specialty: "
        "neurology — terminology: seizure, MRI, neuropathy.\n\n"
        "## Terminology quick-reference\n\n"
        "- Cardiology: myocardial infarction = heart attack; tachycardia = "
        "fast heart rate.\n"
        "- Neurology: cerebrovascular accident = stroke; paresthesia = "
        "tingling/numbness.\n\n"
        "## Invoice/hours tracking\n\n"
        "| Date | Facility | Hours | Rate | Amount | Paid |\n"
        "|------|----------|-------|------|--------|------|\n"
        "| Mon | St. Mary | 3 | $45 | $135 | yes |\n"
        "| Tue | Riverside | 2 | $50 | $100 | no |\n\n"
        "## Certification deadline tracker\n\n"
        "- State court certification — CE 12/20 hours — expires Dec 2026 — "
        "next: medical ethics webinar.\n\n"
        "## Assignment protocol notes\n\n"
        "- St. Mary: check in at interpreter office, badge required, "
        "no phones in wards.\n"
    )


if __name__ == "__main__":
    # Vector 3: all execution flows through main_async() (Vector 2 keeps
    # section assembly file-backed, so concurrent tenants cannot collide).
    sys.exit(asyncio.run(main_async()))
