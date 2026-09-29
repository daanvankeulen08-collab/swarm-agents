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
from agents.reviewer import ReviewerAgent
from core.orchestrator import Orchestrator
from core.product_packager import ProductPackager
from core.state_manager import StateManager
from core.test_runner import TestRunner

PROJECT_ID = "medical_interpreter_planner_001"
PRODUCT_NAME = "Weekly Planner for Freelance Medical Interpreters"
SUGGESTED_PRICE_EUR = 15
MAX_BUILD_ATTEMPTS = 2

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

# (human label, keyword the reviewer searches for in the template text)
REQUIRED_SECTIONS: List[tuple] = [
    ("Weekly appointment grid with time zone support", "appointment"),
    ("Client/facility database", "facility"),
    ("Medical terminology quick-reference by specialty", "terminology"),
    ("Invoice/hours tracking per assignment", "invoice"),
    ("Certification deadline tracker", "certification"),
    ("Assignment protocol notes", "protocol"),
]

SECTIONS_BRIEF = (
    "Return the product as a MARKDOWN template fenced in a single ```markdown "
    "code block — NOT Python code. There is NO word limit: be complete, "
    "detailed, and thorough — never truncate, never use '...' placeholders. "
    "Start with an # H1 title, then one ## section per required item "
    "below, using tables and checklists with realistic example rows.\n"
    "The template MUST contain exactly these six sections, each with its own "
    "heading and usable structure (tables, checklists, or fields — not just "
    "a mention):\n"
    "a) Weekly appointment grid with time zone support — day/time slots plus "
    "a time zone column and a conversion note.\n"
    "b) Client/facility database — hospital name, contact person, specialty, "
    "preferred terminology notes.\n"
    "c) Medical terminology quick-reference section organized by specialty "
    "(cardiology, neurology, orthopedics, pediatrics, oncology at minimum) "
    "with term + plain-language note columns.\n"
    "d) Invoice/hours tracking per assignment — date, facility, hours, rate, "
    "amount, paid status.\n"
    "e) Certification deadline tracker — credential name, continuing "
    "education hours required/earned, expiry date, renewal steps.\n"
    "f) Notes section for assignment-specific protocols — per-facility "
    "check-in, dress code, and confidentiality protocol notes.\n"
    "Use the literal words appointment, facility, terminology, invoice, "
    "certification, and protocol as headings/keywords so sections are "
    "findable. No placeholder text anywhere."
)

_console = Console()
_build_log: List[Dict[str, Any]] = []


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


def build_step(
    builder: BuilderAgent, feedback: Optional[str], attempt: str
) -> tuple[str, str]:
    """Run one detect-and-fallback build + test cycle; return (code, type)."""
    _console.rule(f"[bold]Build {attempt}[/bold]")
    code = builder.build_product_with_fallback(
        opportunity=dict(OPPORTUNITY), feedback=feedback
    )
    product_type = detect_product_type(code)
    log("build", f"Generated {len(code)} chars ({product_type}).", True)
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
    result = reviewer.review_product(
        zip_path,
        required_sections=[keyword for _, keyword in REQUIRED_SECTIONS],
        context=OPPORTUNITY,
    )
    log(
        "review",
        f"Verdict {result['verdict'].upper()}, quality "
        f"{result['quality_score']}/10.",
        result["verdict"] == "approved",
    )
    _console.print(Panel(result["feedback"][:1500], title="Reviewer feedback"))
    return result


def main() -> int:
    """Run build, test, package, review; print the summary; return exit code."""
    _console.print(Panel(
        f"[bold]Product:[/bold] {PRODUCT_NAME}\n"
        f"[bold]Project:[/bold] {PROJECT_ID}\n"
        f"[bold]Price:[/bold] EUR {SUGGESTED_PRICE_EUR}",
        title="Build Pipeline",
    ))

    # -- 1. init -----------------------------------------------------------
    state_manager = StateManager(project_id=PROJECT_ID)
    orchestrator = Orchestrator()
    builder = BuilderAgent(orchestrator, state_manager)
    tester = TestRunner(state_manager)
    packager = ProductPackager(state_manager)
    reviewer = ReviewerAgent(orchestrator, state_manager)
    log("init", "Components ready.", True)

    # -- 2+3. build/test loop (max 2 attempts) ------------------------------
    code = ""
    product_type = "template"
    test_report: Dict[str, Any] = {"success": False, "errors": ["not run"]}
    feedback: Optional[str] = SECTIONS_BRIEF
    for attempt in range(1, MAX_BUILD_ATTEMPTS + 1):
        try:
            code, product_type = build_step(
                builder, feedback, f"attempt {attempt}/{MAX_BUILD_ATTEMPTS}"
            )
        except (RuntimeError, ValueError) as exc:
            log("build", f"Attempt {attempt} failed: {exc}", False)
            feedback = f"{SECTIONS_BRIEF}\nPrevious attempt failed: {exc}"
            continue
        test_report = test_step(tester, code, product_type)
        if test_report["success"]:
            break
        feedback = (
            f"{SECTIONS_BRIEF}\nPrevious attempt FAILED testing:\n"
            + "\n".join(f"- {e}" for e in test_report["errors"][:5])
        )
    if not test_report["success"]:
        return fail(state_manager, "Build/test failed after "
                    f"{MAX_BUILD_ATTEMPTS} attempts.")

    # -- copy product file to products/output/ ------------------------------
    output_dir = Path("products/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    product_file = find_newest_product_file(Path("products"))
    if product_file is not None and product_file.parent != output_dir:
        dest = output_dir / product_file.name
        dest.write_text(product_file.read_text(encoding="utf-8"), encoding="utf-8")
        log("output", f"Product file copied to {dest}.", True)

    # -- 4. package ----------------------------------------------------------
    try:
        zip_path = package_step(packager, code, product_type)
    except (ValueError, IOError, RuntimeError) as exc:
        return fail(state_manager, f"Packaging failed: {exc}")

    # -- 5. review (+ one revision on REJECTED) -------------------------------
    try:
        result = review_step(reviewer, zip_path)
    except (ValueError, RuntimeError) as exc:
        return fail(state_manager, f"Review failed: {exc}")
    if result["verdict"] == "rejected":
        _console.print("[yellow]REJECTED — running the single revision attempt…[/yellow]")
        log("review", "Rejected; starting one revision attempt.", False)
        try:
            code, product_type = build_step(
                builder,
                f"{SECTIONS_BRIEF}\nReviewer REJECTED the previous package. "
                f"Fix every issue:\n{result['feedback'][:1500]}",
                "revision 1/1",
            )
            test_report = test_step(tester, code, product_type)
            if not test_report["success"]:
                return fail(state_manager, "Revision failed testing.")
            zip_path = package_step(packager, code, product_type)
            result = review_step(reviewer, zip_path)
        except (RuntimeError, ValueError, IOError) as exc:
            return fail(state_manager, f"Revision attempt failed: {exc}")

    # -- 6. finalise ----------------------------------------------------------
    publish_dir = Path("publish_ready")
    publish_dir.mkdir(parents=True, exist_ok=True)
    publish_zip = publish_dir / zip_path.name
    shutil.copy2(zip_path, publish_zip)

    if result["verdict"] == "approved":
        state = state_manager.get_state()
        history = state.build_history or []
        history.extend(_build_log)
        state.build_history = history
        state_manager.save(state)
        state_manager.update_phase("publishing")  # approved_for_publish
        log("finalise", "APPROVED — marked publishing (approved_for_publish).", True)
    else:
        return fail(state_manager, "Final verdict still REJECTED after revision.")

    summary = Table(title="Build Complete", show_lines=False)
    summary.add_column("Field")
    summary.add_column("Value")
    summary.add_row("Product", PRODUCT_NAME)
    summary.add_row("ZIP", str(zip_path))
    summary.add_row("Review verdict", f"{result['verdict'].upper()}")
    summary.add_row("Quality score", f"{result['quality_score']}/10")
    summary.add_row("Suggested price", f"EUR {SUGGESTED_PRICE_EUR}")
    summary.add_row("Publish-ready", str(publish_zip))
    summary.add_row("API calls", str(state_manager.get_state().api_calls_count))
    _console.print(summary)
    return 0


def fail(state_manager: StateManager, message: str) -> int:
    """Record failure state, print the message, return exit code 1."""
    _console.print(Panel(f"[red]{message}[/red]", title="Pipeline failed"))
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
    """Buyer-facing feature list mapped to the six required sections."""
    lines = [f"# Features — {PRODUCT_NAME}\n"]
    for label, _ in REQUIRED_SECTIONS:
        lines.append(f"- {label}")
    lines.append(
        "\nPlus: usage instructions, no special software required, "
        "lifetime updates of the template structure."
    )
    return "\n".join(lines) + "\n"


def filled_example_doc() -> str:
    """Realistic filled-in example for a fictional interpreter week."""
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
    sys.exit(main())
