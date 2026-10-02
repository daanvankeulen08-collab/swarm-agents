"""VECTOR 2 capability tests: controlled experiments per product type.

Drives the swarm agents directly (no Scout, no full pipeline) to measure
which content structures the swarm produces reliably:

* TEST A — budget_tracker: tables + calculations (Income, Expenses, ...).
* TEST B — meal_prep_planner: checklists + reference lists.
* TEST C — blog_outline_generator: long narrative + examples.

Per test the harness runs: section-by-section build → template tests →
per-section review (structural + Nemotron) → pixel-only visual gate →
customer verdict. Results land in ``capability_tests/<name>_results.json``.

Run from the project root::

    python scripts/capability_tests.py [--only budget_tracker]

Only the standard library is used besides the swarm's own modules.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.builder import BuilderAgent
from agents.customer_reviewer import CustomerReviewerAgent
from agents.reviewer import ReviewerAgent
from build_pipeline import check_visual_quality, get_willingness_to_pay
from core.orchestrator import Orchestrator
from core.product_packager import ProductPackager
from core.state_manager import StateManager
from core.test_runner import TestRunner

__all__ = ["TESTS", "run_test", "run_all", "build_matrix"]

RESULTS_DIR = ROOT / "capability_tests"

#: Section quality bar shared with the production pipeline.
SECTION_PASS_BAR = 7.0

TESTS: Dict[str, Dict[str, Any]] = {
    "budget_tracker": {
        "product_name": "Budget Tracker for Freelancers",
        "niche": "budget_tracker",
        "audience": "freelancers tracking irregular income and expenses",
        "pain_point": "irregular income, missed tax deadlines, unclear profit",
        "price_hint": 15,
        "unique_value_proposition": (
            "One template that logs income, categorizes expenses, "
            "reconciles monthly totals including tax set-asides, and "
            "checklists quarterly tax filings"
        ),
        "estimated_build_time": "1 hour",
        "focus": "tables + calculations (totals, tax set-asides)",
        "expected": ("High score if calculations reconcile, "
                     "low if totals fail"),
        "sections": (
            ("Income Log", ("income log", "income")),
            ("Expense Categories", ("expense categories", "expense")),
            ("Monthly Summary", ("monthly summary", "summary")),
            ("Tax Checklist", ("tax checklist", "tax")),
        ),
    },
    "meal_prep_planner": {
        "product_name": "Weekly Meal Prep Planner",
        "niche": "meal_prep_planner",
        "audience": "busy households planning a week of meals",
        "pain_point": "grocery waste, weeknight cooking stress, no plan",
        "price_hint": 12,
        "unique_value_proposition": (
            "One template that lists groceries, checklists prep steps, "
            "indexes recipes, and schedules the week"
        ),
        "estimated_build_time": "1 hour",
        "focus": "checklists + reference lists",
        "expected": "High score (checklists are reliable)",
        "sections": (
            ("Grocery List", ("grocery list", "grocery")),
            ("Prep Checklist", ("prep checklist", "prep")),
            ("Recipe Index", ("recipe index", "recipe")),
            ("Weekly Schedule", ("weekly schedule", "schedule")),
        ),
    },
    "blog_outline_generator": {
        "product_name": "Blog Post Outline Generator",
        "niche": "blog_outline_generator",
        "audience": "bloggers beating blank-page block",
        "pain_point": "slow drafting, weak structure, no SEO plan",
        "price_hint": 15,
        "unique_value_proposition": (
            "One template with outline frameworks, writing prompts, an SEO "
            "checklist, and worked examples"
        ),
        "estimated_build_time": "1 hour",
        "focus": "long explanatory narrative + examples",
        "expected": "Medium score (narrative is harder to validate)",
        "sections": (
            ("Outline Templates", ("outline templates", "outline")),
            ("Writing Prompts", ("writing prompts", "prompts")),
            ("SEO Checklist", ("seo checklist", "seo")),
            ("Examples", ("examples", "example")),
        ),
    },
}


def _utcnow() -> str:
    """Current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _section_self_scores(state_manager: StateManager) -> List[Dict[str, Any]]:
    """Latest builder self-score per section (excludes DEGRADED markers)."""
    seen: Dict[str, Dict[str, Any]] = {}
    for entry in state_manager.get_state().build_history or []:
        if entry.get("stage") != "section-quality":
            continue
        title = str(entry.get("section", ""))
        if "(degraded)" in title.lower():
            continue
        seen[title] = {
            "section": title,
            "score": entry.get("score"),
            "iterations": entry.get("iterations"),
        }
    return list(seen.values())


def run_test(name: str) -> Dict[str, Any]:
    """Run one capability test end to end; save + return the result dict.

    No revision loop by design: a REJECTED text verdict is recorded, not
    retried — the experiment measures first-pass capability, and the
    customer verdict is collected regardless so verdict/price data exists
    for every product type.
    """
    if name not in TESTS:
        raise ValueError(f"Unknown test {name!r}; choose from {list(TESTS)}.")
    cfg = TESTS[name]
    project_id = f"cap_{name}"
    started_wall = time.monotonic()
    result: Dict[str, Any] = {
        "test": name,
        "product_name": cfg["product_name"],
        "niche": cfg["niche"],
        "focus": cfg["focus"],
        "expected": cfg["expected"],
        "started_at": _utcnow(),
        "errors": [],
        "stages": {},
    }

    state_manager = StateManager(project_id=project_id)
    orchestrator = Orchestrator()
    builder = BuilderAgent(orchestrator, state_manager)
    tester = TestRunner(state_manager)
    packager = ProductPackager(state_manager)
    reviewer = ReviewerAgent(orchestrator, state_manager)
    customer = CustomerReviewerAgent(orchestrator, state_manager)

    # Fresh workspace so the measurement reflects a clean build.
    sections_dir = Path("state") / "output" / project_id / "sections"
    try:
        if sections_dir.is_dir():
            shutil.rmtree(sections_dir)
    except OSError as exc:
        result["errors"].append(f"workspace cleanup: {exc}")

    def api_calls() -> int:
        try:
            return int(state_manager.get_state().api_calls_count or 0)
        except Exception:
            return -1

    api_before = api_calls()
    opportunity = {
        "product_name": cfg["product_name"],
        "target_audience": cfg["audience"],
        "pain_point": cfg["pain_point"],
        "estimated_price": cfg["price_hint"],
        "unique_value_proposition": cfg["unique_value_proposition"],
        "estimated_build_time": cfg["estimated_build_time"],
    }

    # -- 1. build ---------------------------------------------------------
    code: Optional[str] = None
    try:
        code = builder.build_product_with_fallback(
            opportunity=dict(opportunity), sections=cfg["sections"])
        result["stages"]["build"] = {
            "ok": True, "chars": len(code),
            "sections_written": len(list(sections_dir.glob("section_*.md")))
            if sections_dir.is_dir() else 0,
        }
    except Exception as exc:
        result["errors"].append(f"build: {type(exc).__name__}: {exc}")
        result["stages"]["build"] = {"ok": False, "error": str(exc)[:300]}
        traceback.print_exc()

    result["section_self_scores"] = _section_self_scores(state_manager)

    # -- 2. template tests --------------------------------------------------
    if code is not None:
        try:
            report = tester.run_tests(code, product_type="template")
            result["stages"]["tests"] = {
                "ok": True, "success": bool(report.get("success")),
                "errors": list(report.get("errors", []))[:5],
            }
        except Exception as exc:
            result["errors"].append(f"tests: {type(exc).__name__}: {exc}")
            result["stages"]["tests"] = {"ok": False, "error": str(exc)[:300]}
    else:
        result["stages"]["tests"] = {"ok": False, "error": "no build output"}

    # -- 3. review (structural + per-section Nemotron) -----------------------
    if code is not None:
        try:
            zip_path = packager.create_package(
                cfg["product_name"], code, product_type="template")
            review = reviewer.review_product(
                zip_path,
                required_sections=[kw for _, kws in cfg["sections"]
                                   for kw in [kws[0]]],
                context={"product_name": cfg["product_name"],
                         "target_audience": cfg["audience"],
                         "pain_point": cfg["pain_point"]},
            )
            result["stages"]["review"] = {
                "ok": True,
                "verdict": review.get("verdict"),
                "quality_score": review.get("quality_score"),
                "model_score": review.get("model_score"),
                "checks": review.get("checks"),
            }
        except Exception as exc:
            result["errors"].append(f"review: {type(exc).__name__}: {exc}")
            result["stages"]["review"] = {"ok": False,
                                          "error": str(exc)[:300]}
            traceback.print_exc()
    else:
        result["stages"]["review"] = {"ok": False, "error": "no build output"}

    # -- 4. pixel-only visual gate -------------------------------------------
    if code is not None:
        try:
            shot = Path("state") / "output" / f"{project_id}.png"
            rendered = builder.export_to_image(code, str(shot))
            if rendered and shot.is_file():
                visual = check_visual_quality(project_id)
            else:
                visual = {"visual_score": 0.0, "verdict": "FAIL",
                          "reason": "screenshot render failed"}
            result["stages"]["visual"] = {
                "ok": True,
                "visual_score": visual.get("visual_score"),
                "verdict": visual.get("verdict"),
                "reason": visual.get("reason"),
            }
        except Exception as exc:
            result["errors"].append(f"visual: {type(exc).__name__}: {exc}")
            result["stages"]["visual"] = {"ok": False,
                                          "error": str(exc)[:300]}
    else:
        result["stages"]["visual"] = {"ok": False, "error": "no build output"}

    # -- 5. customer verdict (fixed niche pricing) -----------------------------
    if code is not None:
        try:
            price = float(get_willingness_to_pay(cfg["niche"]))
            review_out = customer.review_as_customer(
                product_content=code,
                product_name=cfg["product_name"],
                price=price,
            )
            result["stages"]["customer"] = {
                "ok": True,
                "verdict": review_out.get("verdict"),
                "overall": (review_out.get("scores") or {}).get("overall"),
                "scores": review_out.get("scores"),
                "model_used": review_out.get("model_used"),
                "price": price,
                "feedback": str(review_out.get("customer_feedback", ""))[:500],
            }
        except Exception as exc:
            result["errors"].append(f"customer: {type(exc).__name__}: {exc}")
            result["stages"]["customer"] = {"ok": False,
                                            "error": str(exc)[:300]}
            traceback.print_exc()
    else:
        result["stages"]["customer"] = {"ok": False,
                                        "error": "no build output"}

    # -- aggregate --------------------------------------------------------------
    result["finished_at"] = _utcnow()
    result["seconds"] = round(time.monotonic() - started_wall, 1)
    result["api_calls"] = api_calls() - api_before
    scores = [s["score"] for s in result["section_self_scores"]
              if isinstance(s.get("score"), (int, float))]
    result["avg_section_score"] = (
        round(sum(scores) / len(scores), 2) if scores else None)
    result["sections_passed"] = sum(1 for s in scores if s >= SECTION_PASS_BAR)
    result["sections_total"] = len(cfg["sections"])
    review_stage = result["stages"].get("review", {})
    result["final_verdict"] = review_stage.get("verdict")
    customer_stage = result["stages"].get("customer", {})
    result["customer_verdict"] = customer_stage.get("verdict")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{name}_results.json"
    out_path.write_text(json.dumps(result, indent=2, default=str),
                        encoding="utf-8")
    print(f"[capability] {name}: saved {out_path} "
          f"({result['seconds']}s, api={result['api_calls']})")
    return result


def run_all(only: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """Run the requested tests sequentially; return {name: result}."""
    names = [n for n in TESTS if not only or n in only]
    if only:
        unknown = [n for n in only if n not in TESTS]
        if unknown:
            raise ValueError(f"Unknown tests {unknown}; "
                             f"choose from {list(TESTS)}.")
    results = {}
    for name in names:
        print(f"\n{'=' * 70}\nCAPABILITY TEST: {name}\n{'=' * 70}")
        results[name] = run_test(name)
    return results


def build_matrix(
    results: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build the comparison matrix rows from saved/in-memory results."""
    rows = []
    for name in TESTS:
        result = results.get(name)
        if result is None:
            path = RESULTS_DIR / f"{name}_results.json"
            if not path.is_file():
                continue
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
        review = (result.get("stages") or {}).get("review", {})
        visual = (result.get("stages") or {}).get("visual", {})
        visual_result = visual.get("verdict")
        if visual_result and visual.get("visual_score") is not None:
            visual_result = f"{visual_result} ({visual.get('visual_score')})"
        rows.append({
            "Product Type": result.get("product_name", name),
            "Self Avg": result.get("avg_section_score"),
            "Review Avg": review.get("model_score"),
            "Pass Rate": (f"{result.get('sections_passed')}/"
                          f"{result.get('sections_total')}"),
            "Errors": len(result.get("errors", [])),
            "Time": f"{result.get('seconds', '?')}s",
            "Calls": result.get("api_calls"),
            "Visual": visual_result,
            "Verdict": result.get("customer_verdict"),
            "Text Verdict": result.get("final_verdict"),
        })
    return rows


def print_matrix(rows: List[Dict[str, Any]]) -> None:
    """Print the capability matrix as an aligned table."""
    if not rows:
        print("No results to display.")
        return
    headers = list(rows[0].keys())
    widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}
    header = " | ".join(h.ljust(widths[h]) for h in headers)
    print(header)
    print("-+-".join("-" * widths[h] for h in headers))
    for row in rows:
        print(" | ".join(str(row[h]).ljust(widths[h]) for h in headers))


def main(argv=None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="VECTOR 2 capability tests: controlled experiments.")
    parser.add_argument("--only", nargs="*", default=None,
                         help="Run only these tests "
                              f"(choices: {list(TESTS)}).")
    parser.add_argument("--full-suite", action="store_true",
                        help="Run every configured capability test.")
    parser.add_argument("--matrix-only", action="store_true",
                        help="Print the matrix from saved results; run nothing.")
    args = parser.parse_args(argv)
    if args.matrix_only:
        print_matrix(build_matrix({}))
        return 0
    if args.full_suite and args.only:
        parser.error("--full-suite cannot be combined with --only.")
    results = run_all(None if args.full_suite or args.only is None
                      else args.only)
    print(f"\n{'=' * 70}\nCAPABILITY MATRIX\n{'=' * 70}")
    print_matrix(build_matrix(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
