"""Swarm Command Center: live operational dashboard for the agent swarm.

Purely additive infrastructure — it only READS ``state/*.json`` and
``state/output/<project>/...png``. It never imports pipeline modules and
never writes to state files.

Launch:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

try:
    # Plain `streamlit run dashboard/app.py`: Streamlit puts the app's own
    # directory (dashboard/) on sys.path, so the sibling module imports
    # without a package prefix.
    from state_reader import StateReader
except ImportError:  # pragma: no cover - package-style import (tests, REPL)
    from dashboard.state_reader import StateReader

STATUS_DOT = {
    "idle": "\u26aa Idle",
    "running": "\U0001f7e1 Running",
    "done": "\U0001f7e2 Done",
    "error": "\U0001f534 Error",
    "unknown": "\u26aa Idle",
}

REFRESH_SECONDS = 5


def _entry_kind(entry: Dict[str, Any]) -> str:
    """Classify a build_history entry for color coding."""
    stage = str(entry.get("stage", ""))
    if entry.get("ok") is False or stage == "provider-failure":
        return "error"
    if stage in ("scores", "pricing", "customer-gate", "test", "review",
                 "visual", "package", "publish"):
        return "success"
    return "info"


def _format_entry(entry: Dict[str, Any]) -> str:
    """One-line human summary of a build_history entry."""
    at = str(entry.get("at", "?"))[11:19]  # HH:MM:SS from ISO timestamp
    stage = str(entry.get("stage", "?"))
    message = str(entry.get("message", entry.get("error", "")))[:160]
    if entry.get("stage") == "section-quality":
        message = (
            f"{entry.get('section', '?')}: "
            f"{entry.get('score', '?')}/10 "
            f"({entry.get('iterations', '?')} iters)"
        )
    if entry.get("stage") == "provider-failure":
        message = (
            f"{entry.get('provider', '?')} failed "
            f"({entry.get('error_type', '?')})"
        )
    return f"`{at}` **{stage}** — {message}"


def render_agents(col: Any, state: Dict[str, Any]) -> None:
    """Left column: per-agent status rows."""
    col.subheader("Agents")
    for row in StateReader.get_agent_statuses(state):
        dot = STATUS_DOT.get(row["status"], STATUS_DOT["unknown"])
        col.markdown(f"{row['icon']} **{row['agent']}** — {dot}")
    col.caption(f"phase: `{state.get('phase', '?')}` · "
                f"api calls: `{state.get('api_calls_count', 0)}` · "
                f"revisions: `{state.get('revision_count', 0)}`")


def render_log(col: Any, state: Dict[str, Any]) -> None:
    """Middle column: chronological build log with color coding."""
    col.subheader("Build Log")
    history: List[Dict[str, Any]] = state.get("build_history") or []
    if not history:
        col.info("No build history yet.")
        return
    for entry in history[-60:]:
        kind = _entry_kind(entry)
        text = _format_entry(entry)
        if kind == "error":
            col.error(text)
            traceback = str(entry.get("traceback", ""))
            if traceback:
                with col.expander("traceback"):
                    col.code(traceback[:1500])
        elif kind == "success":
            col.success(text)
        else:
            col.info(text)


def render_preview(col: Any, reader: StateReader, project_id: str,
                   state: Dict[str, Any]) -> None:
    """Right column: screenshot, section scores, metrics, review summary."""
    col.subheader("Preview")
    shot = reader.get_screenshot_path(project_id)
    if shot is not None:
        col.image(str(shot), caption=f"{project_id}.png",
                  use_container_width=True)
    else:
        col.info("No screenshot yet.")

    col.markdown("**Section quality**")
    sections = StateReader.get_section_scores(state)
    if sections:
        col.table([
            {
                "Section": r["section"].replace(" (DEGRADED)", ""),
                "Score": r["score"],
                "Status": "\u2705" if isinstance(r["score"], (int, float))
                and r["score"] >= 7.0 else "\u274c",
            }
            for r in sections
        ])
    else:
        col.caption("No section scores yet.")

    summary = StateReader.get_build_summary(state)
    m1, m2, m3 = col.columns(3)
    m1.metric("WTP", f"\u20ac{summary['wtp']:g}"
              if summary["wtp"] is not None else "-")
    m2.metric("Final score",
              f"{summary['final_score']}/10"
              if summary["final_score"] is not None else "-")
    m3.metric("Publish", str(state.get("publish_status") or "-"))

    feedback = str(state.get("review_feedback", ""))
    if feedback:
        with col.expander("Review report"):
            col.write(feedback[:2000])
    prompt = str(state.get("improvement_prompt", ""))
    if prompt:
        with col.expander("Improvement prompt"):
            col.write(prompt[:2000])


def render_history(reader: StateReader) -> None:
    """Bottom section: all runs with a summary row."""
    st.subheader("History")
    projects = reader.list_all_projects()
    if not projects:
        st.info("No projects found in state/.")
        return
    rows = []
    for pid in projects:
        s = StateReader.get_build_summary(reader.load_state(pid))
        icon = {"published": "\U0001f7e2 Published",
                "pending": "\U0001f7e1 Pending",
                "failed": "\U0001f534 Failed"}.get(s["status"], s["status"])
        rows.append({
            "Project": s["project_id"],
            "Status": icon,
            "Final score": s["final_score"] if s["final_score"] is not None
            else "-",
            "WTP": f"\u20ac{s['wtp']:g}" if s["wtp"] is not None else "-",
            "Errors": s["errors"],
            "Duration": s["duration"],
            "Updated": (s["timestamp"] or "-")[:19].replace("T", " "),
        })
    st.table(rows)

    scored = [reader.load_state(p) for p in projects]
    summaries = [StateReader.get_build_summary(s) for s in scored]
    published = sum(1 for s in summaries if s["status"] == "published")
    scores = [s["final_score"] for s in summaries
              if s["final_score"] is not None]
    avg = sum(scores) / len(scores) if scores else 0.0
    total = len(summaries)
    pct = (100.0 * published / total) if total else 0.0
    st.markdown(
        f"**Success rate: {published}/{total} ({pct:.0f}%)** · "
        f"**Average score: {avg:.1f}/10** · "
        f"**Total runs: {total}**"
    )


def main() -> None:
    """Dashboard entry point."""
    st.set_page_config(page_title="Swarm Command Center", layout="wide")
    reader = StateReader(
        str(Path(__file__).resolve().parent.parent / "state"))

    projects = reader.list_all_projects()
    st.sidebar.header("Projects")
    selected = st.sidebar.selectbox(
        "Project",
        options=projects or ["(none)"],
        index=0,
    ) if projects else "(none)"
    auto = st.sidebar.checkbox("Live refresh (5s)", value=True)

    st.title("\U0001f41d Swarm Command Center")
    if selected != "(none)":
        if hasattr(st, "badge"):
            st.badge(selected)
        else:
            st.caption(f"\U0001f4c1 {selected}")

    if selected == "(none)":
        st.warning("No state files found. Run the pipeline first.")
        return

    state = reader.load_state(selected)
    summary = StateReader.get_build_summary(state)

    col_agents, col_log, col_preview = st.columns([20, 40, 40])
    render_agents(col_agents, state)
    render_log(col_log, state)
    render_preview(col_preview, reader, selected, state)

    st.divider()
    render_history(reader)

    # Live updates: rerun every REFRESH_SECONDS while a run is active.
    active = summary["status"] == "pending" and state.get("phase") not in (
        "failed", "done", "completed", "publishing")
    if auto and active:
        time.sleep(REFRESH_SECONDS)
        st.rerun()


if __name__ == "__main__":
    main()
