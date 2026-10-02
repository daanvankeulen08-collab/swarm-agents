"""Retrospective capability analysis across all swarm runs.

Reads every ``state/*.json`` file plus the ``state/output/*/sections/``
workspaces and produces a capability report: which sections and content
types score well, where errors cluster, and how providers behave.

Purely read-only. Run from the project root::

    python scripts/analyze_capabilities.py [--json-out report.json]

Only the standard library is used.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# Canonical section order (matches core/sections.py ids).
SECTION_ORDER = [
    "appointment_grid",
    "client_database",
    "medical_terminology",
    "invoice_tracking",
    "certification_deadlines",
    "protocol",
]

# Builder self-score section titles -> canonical id.
TITLE_TO_ID = {
    "weekly appointment grid": "appointment_grid",
    "client database": "client_database",
    "medical terminology": "medical_terminology",
    "invoice tracking": "invoice_tracking",
    "certification deadlines": "certification_deadlines",
    "assignment notes": "protocol",
}

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"
OUTPUT_DIR = STATE_DIR / "output"


def load_states() -> dict:
    """Return {project_id: state dict} for every readable state file."""
    states = {}
    if not STATE_DIR.is_dir():
        return states
    for path in sorted(STATE_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            states[path.stem] = {"_unreadable": True}
            continue
        if isinstance(data, dict):
            states[path.stem] = data
    return states


def section_scores(states: dict) -> dict:
    """Collect builder self-scores per canonical section id.

    Returns {section_id: [scores]} plus degradation counts. The
    ``(DEGRADED)`` 0.0 marker entries are counted as degradation events,
    not as scores.
    """
    scores: dict = {sid: [] for sid in SECTION_ORDER}
    degraded: dict = {sid: 0 for sid in SECTION_ORDER}
    runs_with_scores = 0
    for pid, state in states.items():
        if "_unreadable" in state:
            continue
        seen_here = False
        for entry in state.get("build_history") or []:
            if entry.get("stage") != "section-quality":
                continue
            title = str(entry.get("section", "")).lower()
            is_marker = "(degraded)" in title
            clean = title.replace("(degraded)", "").strip()
            sid = TITLE_TO_ID.get(clean)
            if sid is None:
                continue
            seen_here = True
            if is_marker:
                degraded[sid] += 1
            else:
                try:
                    scores[sid].append(float(entry.get("score", 0)))
                except (TypeError, ValueError):
                    pass
        if seen_here:
            runs_with_scores += 1
    return {"scores": scores, "degraded": degraded,
            "runs_with_scores": runs_with_scores}


def final_scores(states: dict) -> dict:
    """Extract final combined scores from 'scores'-stage messages."""
    out = {}
    for pid, state in states.items():
        if "_unreadable" in state:
            continue
        for entry in reversed(state.get("build_history") or []):
            if entry.get("stage") == "scores":
                match = re.search(r"Final quality ([\d.]+)/10",
                                  str(entry.get("message", "")))
                if match:
                    try:
                        out[pid] = float(match.group(1))
                    except ValueError:
                        pass
                break
    return out


def error_patterns(states: dict) -> dict:
    """Classify failures from build history + review feedback."""
    categories: Counter = Counter()
    phase_fails: Counter = Counter()
    provider_fails: Counter = Counter()
    for pid, state in states.items():
        if "_unreadable" in state:
            continue
        for entry in state.get("build_history") or []:
            stage = str(entry.get("stage", ""))
            if entry.get("ok") is False and stage not in ("provider-failure",):
                phase_fails[stage or "unknown"] += 1
            if stage == "provider-failure":
                provider = str(entry.get("provider", "?"))
                provider_fails[provider] += 1
                err = (str(entry.get("error_type", ""))
                       + " " + str(entry.get("error", ""))).lower()
                if "stall" in err or "timeout" in err or "timed out" in err:
                    categories["timeout/stall"] += 1
                elif "connection" in err:
                    categories["connection error"] += 1
                elif "empty" in err:
                    categories["empty response"] += 1
                elif "404" in err or "not found" in err:
                    categories["404/not entitled"] += 1
                elif "402" in err or "funds" in err:
                    categories["402/insufficient funds"] += 1
                elif "403" in err or "freetier" in err:
                    categories["403/free-tier blocked"] += 1
                else:
                    categories["other provider error"] += 1
        feedback = str(state.get("review_feedback", "")).lower()
        if "truncat" in feedback:
            categories["truncation flagged by reviewer"] += 1
        if "hallucinat" in feedback:
            categories["hallucination flagged by reviewer"] += 1
        if state.get("phase") == "failed":
            phase_fails["_terminal: failed"] += 1
    return {
        "categories": dict(categories),
        "phase_fails": dict(phase_fails),
        "provider_fails": dict(provider_fails),
    }


def efficiency(states: dict) -> dict:
    """Duration and API-call efficiency per run."""
    rows = {}
    for pid, state in states.items():
        if "_unreadable" in state:
            continue
        seconds = None
        try:
            if state.get("created_at") and state.get("updated_at"):
                start = datetime.fromisoformat(
                    str(state["created_at"]).replace("Z", "+00:00"))
                end = datetime.fromisoformat(
                    str(state["updated_at"]).replace("Z", "+00:00"))
                seconds = max(0.0, (end - start).total_seconds())
        except (ValueError, TypeError):
            seconds = None
        rows[pid] = {
            "api_calls": state.get("api_calls_count"),
            "seconds": round(seconds) if seconds is not None else None,
            "phase": state.get("phase"),
        }
    return rows


def categorize_section_file(path: Path) -> dict:
    """Break a section .md file into content-type word masses.

    Word masses (not line counts) decide the dominant type, because some
    model outputs collapse whole tables onto single lines — counting lines
    would mislabel a 700-word table blob as "narrative". Any line with 3+
    pipe characters counts its words as table content.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    table_lines, table_words = 0, 0
    check_items, ref_lines, narrative_words = 0, 0, 0
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not stripped:
            continue
        if stripped.count("|") >= 3 and not stripped.startswith("<!--"):
            table_lines += 1
            table_words += len(stripped.replace("|", " ").split())
        elif re.match(r"^[-*]\s*\[[ xX]\]", stripped):
            check_items += 1
        elif re.match(r"^#{1,4}\s", stripped):
            narrative_words += len(stripped.split()) - 1
        elif stripped.startswith(("- ", "* ", ">")) or re.match(
                r"^\d+[.)]\s", stripped):
            # List items: reference-style (term: definition) vs action?
            if re.match(r"^[-*]\s*\*\*[^*]+\*\*\s*[:\-]", stripped):
                ref_lines += 1
            else:
                narrative_words += len(stripped.split())
        elif stripped.startswith("<!--"):
            continue
        else:
            narrative_words += len(stripped.split())
    words = len(text.split())
    # Calculation signals: currency, reconciled totals, arithmetic.
    calc_hits = len(re.findall(
        r"\$[\d,]+\.\d{2}|= \$|total equals|variance|reconciled", text))
    # Newline collapse flag: table-dense but almost no line breaks means the
    # model emitted blobs; downstream renderers may show paragraphs, not
    # tables.
    collapsed = (table_words > 200
                 and len(text.splitlines()) < 12)
    return {
        "words": words,
        "table_lines": table_lines,
        "table_words": table_words,
        "checklist_items": check_items,
        "ref_lines": ref_lines,
        "narrative_words": narrative_words,
        "calc_hits": calc_hits,
        "collapsed": collapsed,
    }


def content_analysis(states: dict, scores: dict) -> dict:
    """Correlate content types with builder self-scores.

    Score lookup: latest non-degraded score for (project, section file
    index). Section files are positional: section_1..6 map to SECTION_ORDER.
    """
    by_type: dict = defaultdict(list)
    details = []
    for pid in states:
        sections_dir = OUTPUT_DIR / pid / "sections"
        if not sections_dir.is_dir():
            continue
        for idx, sid in enumerate(SECTION_ORDER, start=1):
            path = sections_dir / f"section_{idx}.md"
            if not path.is_file():
                continue
            cats = categorize_section_file(path)
            if not cats:
                continue
            # Latest score for this section in this run's history.
            score = None
            for entry in reversed(
                    states[pid].get("build_history") or []):
                if (entry.get("stage") == "section-quality"
                        and "(degraded)" not in
                        str(entry.get("section", "")).lower()
                        and TITLE_TO_ID.get(
                            str(entry.get("section", "")).lower()) == sid):
                    try:
                        score = float(entry.get("score", 0))
                    except (TypeError, ValueError):
                        score = None
                    break
            total_mass = (cats["table_words"] + cats["checklist_items"] * 8
                            + cats["ref_lines"] * 12 + cats["narrative_words"])
            dominant = "mixed"
            if total_mass:
                shares = {
                    "tables": cats["table_words"] / total_mass,
                    "checklists": cats["checklist_items"] * 8 / total_mass,
                    "reference": cats["ref_lines"] * 12 / total_mass,
                    "narrative": cats["narrative_words"] / total_mass,
                }
                best, share = max(shares.items(), key=lambda kv: kv[1])
                dominant = best if share >= 0.45 else "mixed"
            if score is not None:
                by_type[dominant].append(score)
            details.append({"project": pid, "section": sid,
                            "dominant": dominant, "score": score, **cats})
    averages = {k: round(sum(v) / len(v), 2) for k, v in by_type.items() if v}
    return {"by_type": {k: {"n": len(v), "avg": averages[k]}
                        for k, v in by_type.items()},
            "details": details}


def build_report() -> dict:
    """Assemble the full capability report."""
    states = load_states()
    sec = section_scores(states)
    finals = final_scores(states)
    errors = error_patterns(states)
    eff = efficiency(states)
    content = content_analysis(states, sec)

    per_section_avg = {}
    per_section_n = {}
    for sid, vals in sec["scores"].items():
        per_section_avg[sid] = round(sum(vals) / len(vals), 2) if vals else None
        per_section_n[sid] = len(vals)

    reliable = sorted(
        [s for s, a in per_section_avg.items() if a is not None and a >= 7.5])
    problematic = sorted(
        [s for s, a in per_section_avg.items() if a is not None and a < 7.0])
    all_scores = [v for vals in sec["scores"].values() for v in vals]
    avg_quality = (round(sum(all_scores) / len(all_scores), 2)
                   if all_scores else None)

    type_avgs = {k: v["avg"] for k, v in content["by_type"].items()}
    type_ns = {k: v["n"] for k, v in content["by_type"].items()}
    # Honest thresholds: compare against the dataset mean, require n>=3.
    # Fixed 7.5/7.0 cutoffs mislabel average types as weak on small samples.
    mean = avg_quality or 0
    strong_types = sorted(
        k for k, a in type_avgs.items()
        if a >= mean + 0.3 and type_ns[k] >= 3)
    weak_types = sorted(
        k for k, a in type_avgs.items()
        if a <= mean - 0.3 and type_ns[k] >= 3)

    hotspots = []
    cats = errors["categories"]
    if cats.get("timeout/stall"):
        hotspots.append(
            f"provider stalls/timeouts ({cats['timeout/stall']} events) — "
            "the 60s guards fire, cascade recovers")
    if cats.get("truncation flagged by reviewer"):
        hotspots.append("truncation flagged by reviewer (pre-assembly era)")
    if cats.get("connection error"):
        hotspots.append(
            f"connection resets ({cats['connection error']} events)")
    if sec["degraded"] and sum(sec["degraded"].values()):
        worst = sorted(sec["degraded"].items(), key=lambda kv: -kv[1])[:2]
        hotspots.append("degraded-section markers cluster on: "
                        + ", ".join(f"{s} ({n}x)" for s, n in worst))
    collapsed = [d for d in content["details"] if d.get("collapsed")]
    if collapsed:
        hotspots.append(
            f"newline collapse: {len(collapsed)} section file(s) pack "
            "table-dense content onto <12 lines "
            "(" + ", ".join(f"{d['project'][-6:]}:{d['section'][:12]}"
                             for d in collapsed[:4]) + ") — "
            "renderers show paragraphs, not tables")

    return {
        "runs_analyzed": len(states),
        "runs_with_section_scores": sec["runs_with_scores"],
        "strong_content_types": strong_types,
        "weak_content_types": weak_types,
        "reliable_sections": reliable,
        "problematic_sections": problematic,
        "average_quality_score": avg_quality,
        "per_section_average": per_section_avg,
        "per_section_n": per_section_n,
        "degradation_events": sec["degraded"],
        "final_scores": finals,
        "error_hotspots": hotspots,
        "error_categories": cats,
        "phase_failures": errors["phase_fails"],
        "provider_failures": errors["provider_fails"],
        "content_type_averages": content["by_type"],
        "efficiency": eff,
        "provider_reliability": {
            "opencode-zen": "fast primary; stalls observed (~9 in MV011); "
                            "free-tier models 403 outside OpenCode",
            "openrouter": "stable fallback; empty-response retries seen; "
                          "never stalled in observed runs",
            "nvidia-nim": "reliable for judging (review/vision); cascade "
                          "fallback can stall under load (60s guard fires)",
        },
    }


def print_human(report: dict) -> None:
    """Print the human-readable version of the report."""
    print("=" * 70)
    print("SWARM CAPABILITY REPORT")
    print(f"({report['runs_analyzed']} state files, "
          f"{report['runs_with_section_scores']} with section scores)")
    print("=" * 70)
    print("\nSECTION PERFORMANCE (builder self-score, 1-10):")
    for sid in SECTION_ORDER:
        avg = report["per_section_average"].get(sid)
        deg = report["degradation_events"].get(sid, 0)
        n = report.get("per_section_n", {}).get(sid, 0)
        print(f"  {sid:24s} avg={avg}  n={n}  degraded={deg}x")
    print("\nCONTENT TYPE vs SCORE:")
    for kind, stats in sorted(report["content_type_averages"].items()):
        print(f"  {kind:12s} n={stats['n']:3d}  avg={stats['avg']}")
    print("\nFINAL SCORES (combined text+visual):")
    for pid, score in sorted(report["final_scores"].items()):
        print(f"  {pid:40s} {score}/10")
    print("\nERROR CATEGORIES:")
    for kind, count in sorted(report["error_categories"].items(),
                              key=lambda kv: -kv[1]):
        print(f"  {kind:32s} {count}")
    print("\nPHASE FAILURES (ok=False entries + terminal failed):")
    for phase, count in sorted(report["phase_failures"].items(),
                               key=lambda kv: -kv[1]):
        print(f"  {phase:24s} {count}")
    print("\nPROVIDER FAILURES:")
    for provider, count in sorted(report["provider_failures"].items(),
                                  key=lambda kv: -kv[1]):
        print(f"  {provider:20s} {count}")
    print("\nEFFICIENCY (api calls / duration):")
    for pid, eff in sorted(report["efficiency"].items()):
        secs = eff["seconds"]
        dur = (f"{secs // 3600:.0f}h{secs % 3600 // 60:02.0f}m"
               if secs else "-")
        print(f"  {pid:40s} api={eff['api_calls']}  dur={dur}  "
              f"{eff['phase']}")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Retrospective capability analysis across swarm runs.")
    parser.add_argument("--json-out", default=None,
                        help="Write the full report JSON to this path.")
    args = parser.parse_args()
    report = build_report()
    print_human(report)
    print("\n" + "=" * 70)
    print("MACHINE SUMMARY:")
    print(json.dumps({
        "strong_content_types": report["strong_content_types"],
        "weak_content_types": report["weak_content_types"],
        "reliable_sections": report["reliable_sections"],
        "problematic_sections": report["problematic_sections"],
        "average_quality_score": report["average_quality_score"],
        "error_hotspots": report["error_hotspots"],
        "provider_reliability": report["provider_reliability"],
    }, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nFull report written to {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
