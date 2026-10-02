"""StateReader: safe, read-only access to swarm state files.

The dashboard never touches the pipeline — it only reads
``state/*.json`` plus the ``state/output/<project_id>.png`` screenshots.
Every method degrades gracefully: missing files, corrupted JSON, or
unexpected shapes return empty defaults instead of raising.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["StateReader", "DEFAULT_STATE_DIR", "EMPTY_STATE"]

#: Default directory holding ``<project_id>.json`` state files.
DEFAULT_STATE_DIR: str = "state"

#: Returned when a state file is missing or unreadable.
EMPTY_STATE: Dict[str, Any] = {
    "project_id": "?",
    "phase": "unknown",
    "review_status": None,
    "review_feedback": "",
    "product_name": "",
    "api_calls_count": 0,
    "package_path": None,
    "publish_status": None,
    "build_history": [],
    "revision_count": 0,
    "created_at": None,
    "updated_at": None,
}


class StateReader:
    """Read-only accessor for swarm state.

    Args:
        state_dir: Directory holding ``<project_id>.json`` files.
    """

    def __init__(self, state_dir: str = DEFAULT_STATE_DIR) -> None:
        self.state_dir = Path(state_dir)

    # -- project discovery --------------------------------------------------

    def list_all_projects(self) -> List[str]:
        """Return project ids sorted newest-first by file mtime.

        Skips unreadable files. Returns an empty list when the directory
        does not exist.
        """
        if not self.state_dir.is_dir():
            return []
        files = []
        for path in self.state_dir.glob("*.json"):
            try:
                files.append((path.stat().st_mtime, path.stem))
            except OSError:
                continue
        files.sort(reverse=True)
        return [stem for _, stem in files]

    def get_current_project(self) -> Optional[str]:
        """Return the most recently modified project id, or None."""
        projects = self.list_all_projects()
        return projects[0] if projects else None

    # -- single-project reads -------------------------------------------------

    def load_state(self, project_id: str) -> Dict[str, Any]:
        """Return the parsed state dict for ``project_id``.

        Merges over :data:`EMPTY_STATE` so callers always see every key.
        Returns a copy of :data:`EMPTY_STATE` (with the requested id) when
        the file is missing or corrupt.
        """
        fallback = dict(EMPTY_STATE)
        fallback["project_id"] = project_id
        path = self.state_dir / f"{project_id}.json"
        if not path.is_file():
            return fallback
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, ValueError):
            return fallback
        if not isinstance(data, dict):
            return fallback
        merged = dict(fallback)
        merged.update(data)
        if not isinstance(merged.get("build_history"), list):
            merged["build_history"] = []
        return merged

    def get_screenshot_path(self, project_id: str) -> Optional[Path]:
        """Return the screenshot path if it exists, else None."""
        candidate = (
            self.state_dir / "output" / project_id / f"{project_id}.png"
        )
        # Fallback: some runs write directly under state/output/.
        alt = self.state_dir / "output" / f"{project_id}.png"
        for path in (candidate, alt):
            try:
                if path.is_file():
                    return path
            except OSError:
                continue
        return None

    def get_state_mtime(self, project_id: str) -> Optional[float]:
        """Return the state file mtime, or None when unavailable."""
        try:
            return (self.state_dir / f"{project_id}.json").stat().st_mtime
        except OSError:
            return None

    # -- summaries --------------------------------------------------------------

    @staticmethod
    def get_build_summary(state: Dict[str, Any]) -> Dict[str, Any]:
        """Derive dashboard metrics from one state dict.

        Returns status badge, final score, WTP price, error count, duration
        and timestamp — every value defaulted so the history table never
        breaks on an old or partial state file.
        """
        history = state.get("build_history") or []
        phase = str(state.get("phase") or "unknown")
        review = str(state.get("review_status") or "")
        publish = str(state.get("publish_status") or "")

        if publish == "published" or phase in ("done", "completed"):
            status = "published"
        elif phase == "failed" or review == "rejected":
            status = "failed"
        else:
            status = "pending"

        final_score: Optional[float] = None
        for entry in reversed(history):
            if entry.get("stage") == "scores":
                message = str(entry.get("message", ""))
                import re

                match = re.search(r"Final quality ([\d.]+)/10", message)
                if match:
                    try:
                        final_score = float(match.group(1))
                    except ValueError:
                        final_score = None
                break

        wtp: Optional[float] = None
        for entry in reversed(history):
            if entry.get("stage") == "pricing":
                import re

                match = re.search(r"EUR ([\d.]+)", str(entry.get("message", "")))
                if match:
                    try:
                        wtp = float(match.group(1))
                    except ValueError:
                        wtp = None
                break

        errors = sum(
            1 for e in history
            if e.get("stage") == "provider-failure" or e.get("ok") is False
        )

        started = state.get("created_at")
        updated = state.get("updated_at")
        duration = _duration_str(started, updated)

        return {
            "project_id": state.get("project_id", "?"),
            "status": status,
            "phase": phase,
            "final_score": final_score,
            "wtp": wtp,
            "errors": errors,
            "duration": duration,
            "timestamp": updated or started,
            "api_calls": state.get("api_calls_count", 0),
            "revisions": state.get("revision_count", 0),
        }

    @staticmethod
    def get_agent_statuses(state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Map pipeline phases to per-agent Idle/Running/Done/Error rows.

        The state file tracks one global ``phase``; agent rows are inferred:
        agents upstream of the current phase are Done, the active one is
        Running, downstream ones are Idle. A ``failed`` phase marks the
        active agent as Error.
        """
        order = ["research", "building", "reviewing", "customer",
                 "publishing", "completed", "failed"]
        agents = [
            ("Scout", "research", "\U0001f50d"),
            ("Builder", "building", "\U0001f3d7\ufe0f"),
            ("Reviewer", "reviewing", "\U0001f52c"),
            ("Customer", "customer", "\U0001f464"),
            ("Publisher", "publishing", "\U0001f4e6"),
        ]
        phase = str(state.get("phase") or "research")
        try:
            phase_idx = order.index(phase)
        except ValueError:
            phase_idx = 0
        failed = phase == "failed"
        rows = []
        for name, agent_phase, icon in agents:
            try:
                agent_idx = order.index(agent_phase)
            except ValueError:
                agent_idx = 0
            if failed and agent_idx == phase_idx:
                status = "error"
            elif agent_idx < phase_idx or phase in ("completed", "done"):
                status = "done"
            elif agent_idx == phase_idx:
                status = "running"
            else:
                status = "idle"
            rows.append({
                "agent": name,
                "icon": icon,
                "phase": agent_phase,
                "status": status,
            })
        return rows

    @staticmethod
    def get_section_scores(
        state: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Extract per-section quality rows from build history."""
        rows = []
        for entry in state.get("build_history") or []:
            if entry.get("stage") == "section-quality" and "section" in entry:
                rows.append({
                    "section": str(entry.get("section", "?")),
                    "score": entry.get("score"),
                    "iterations": entry.get("iterations", "?"),
                })
        # De-duplicate: keep the last score per section (revision reruns).
        seen: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            seen[row["section"]] = row
        return list(seen.values())


def _duration_str(started: Any, updated: Any) -> str:
    """Format the created_at → updated_at span as H:MM:SS, or '-'."""
    try:
        if not started or not updated:
            return "-"
        start = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(updated).replace("Z", "+00:00"))
        delta = end - start
        if delta.total_seconds() < 0:
            return "-"
        hours, rem = divmod(int(delta.total_seconds()), 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    except (ValueError, TypeError):
        return "-"
