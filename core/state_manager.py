"""Bulletproof State Manager for an autonomous AI agent swarm.

This module is the single source of truth for swarm projects. It prevents
agent hallucinations by centralising all mutable project state in a validated
Pydantic model persisted as JSON, so every agent (scout, builder, reviewer,
publisher) reads and writes the same data.

Example usage:
    ```python
    from core.state_manager import StateManager

    # Create (or resume) a project's state file at ``state/demo.json``.
    manager = StateManager(project_id="demo")

    # Read the current state (creates defaults on first run).
    state = manager.get_state()
    print(state.phase)  # -> "research"

    # Advance the workflow.
    manager.update_phase("building")

    # Track LLM / tool usage.
    manager.log_api_call()

    # Persist structured research findings.
    state = manager.get_state()
    state.scout_findings = {"sources": ["https://example.com"], "summary": "..."}
    state.product_name = "Acme Widget"
    manager.save(state)

    # Move through review.
    manager.update_phase("reviewing")
    state = manager.get_state()
    state.review_status = "approved"
    state.review_feedback = "Looks good to ship."
    manager.save(state)
    manager.update_phase("publishing")
    ```

State files live in ``state/{project_id}.json`` with all datetimes stored as
ISO-8601 format strings. Terminal output uses :mod:`rich` for colour.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator
from rich.console import Console

__all__ = [
    "ALLOWED_PHASES",
    "ALLOWED_REVIEW_STATUSES",
    "ProjectState",
    "StateManager",
    "ZEN_CIRCUIT_BREAKER_TRIPS",
]

#: Consecutive Zen stalls after which Zen is skipped for the rest of the
#: run. Zen stalls in bursts and rarely recovers mid-run (MV015: ~10 stalls
#: in 2 hours); each stall costs a full 60s wall-clock wait, so after this
#: many consecutive stalls the provider cascade starts at OpenRouter instead.
#: Lives here (not in ``core.orchestrator``) because the counter is part of
#: the persisted project state and the orchestrator imports this module.
ZEN_CIRCUIT_BREAKER_TRIPS: int = 3

#: Valid lifecycle phases for a swarm project, in typical order.
ALLOWED_PHASES: frozenset[str] = frozenset(
    {
        "research",
        "research_complete",
        "building",
        "reviewing",
        "publishing",
        "completed",
        "failed",
    }
)

#: Valid review outcomes. ``None`` means "not reviewed yet".
ALLOWED_REVIEW_STATUSES: frozenset[str] = frozenset({"approved", "rejected"})

_console = Console()


def _utcnow() -> datetime:
    """Return the current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class ProjectState(BaseModel):
    """Validated single source of truth for one swarm project.

    Attributes:
        project_id: Unique project identifier. Also the JSON filename stem.
        phase: Current lifecycle phase. Must be one of ``ALLOWED_PHASES``.
        scout_findings: Structured research results as a dict, if available.
        product_code: Generated code / templates as a string, if available.
        review_status: ``"approved"``, ``"rejected"``, or ``None`` if pending.
        review_feedback: Free-text reviewer feedback, if available.
        product_name: Human-readable product name, if available.
        sales_description: Marketing / sales copy, if available.
        api_calls_count: Cumulative count of billable API calls. Never negative.
        package_path: Filesystem path of the packaged ZIP, if one was built.
        packaged_at: ISO-8601 timestamp of the last packaging run, if any.
        publish_status: Publish outcome: "published", "pending_human_review",
            or None if undecided.
        build_history: Per-attempt records from BuilderAgent.build_with_testing.
        last_test_results: Most recent TestRunner report dict, if any.
        research_findings: Raw + ranked revenue-stream research from ResearchAgent.
        improvement_prompt: Latest reviewer improvement prompt for revisions.
        revision_count: How many builder revision attempts have run.
        created_at: Timestamp of state creation (auto-set, timezone-aware).
        updated_at: Timestamp of last mutation (auto-refreshed on save).
    """

    project_id: str = Field(..., min_length=1, description="Unique project identifier.")
    phase: str = Field(
        default="research",
        description="Lifecycle phase: research|research_complete|building|reviewing|publishing|completed|failed.",
    )
    scout_findings: Optional[Dict[str, Any]] = Field(
        default=None, description="Research results as a structured dict."
    )
    product_code: Optional[str] = Field(
        default=None, description="Generated code/templates as a string."
    )
    review_status: Optional[str] = Field(
        default=None, description='Review outcome: "approved", "rejected", or None.'
    )
    review_feedback: Optional[str] = Field(
        default=None, description="Free-text feedback from the reviewer."
    )
    product_name: Optional[str] = Field(
        default=None, description="Human-readable product name."
    )
    sales_description: Optional[str] = Field(
        default=None, description="Marketing / sales description."
    )
    api_calls_count: int = Field(
        default=0, ge=0, description="Cumulative billable API call count."
    )
    package_path: Optional[str] = Field(
        default=None, description="Filesystem path of the packaged ZIP, if built."
    )
    packaged_at: Optional[str] = Field(
        default=None, description="ISO-8601 timestamp of the last packaging run."
    )
    publish_status: Optional[str] = Field(
        default=None,
        description='Publish outcome: "published", "pending_human_review", or None.',
    )
    build_history: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Per-attempt build/test records from the Builder."
    )
    last_test_results: Optional[Dict[str, Any]] = Field(
        default=None, description="Most recent TestRunner report dict."
    )
    research_findings: Optional[Dict[str, Any]] = Field(
        default=None, description="Revenue-stream research from the ResearchAgent."
    )
    improvement_prompt: str = Field(
        default="", description="Latest reviewer improvement prompt for revisions."
    )
    zen_stall_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Consecutive Zen stalls this run. Reset to 0 on any Zen success; "
            f"at {ZEN_CIRCUIT_BREAKER_TRIPS} the orchestrator skips Zen for "
            "the rest of the run."
        ),
    )
    revision_count: int = Field(
        default=0, ge=0, description="How many builder revision attempts have run."
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="Creation timestamp (auto-set)."
    )
    updated_at: datetime = Field(
        default_factory=_utcnow, description="Last-mutation timestamp (auto-updated)."
    )

    @field_validator("project_id")
    @classmethod
    def _validate_project_id(cls, value: str) -> str:
        """Reject empty IDs and path-traversal characters in the ID."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("project_id must be a non-empty string.")
        if any(sep in cleaned for sep in ("/", "\\", "\x00")) or ".." in cleaned:
            raise ValueError(
                f"project_id {cleaned!r} must not contain path separators or '..'."
            )
        return cleaned

    @field_validator("phase")
    @classmethod
    def _validate_phase(cls, value: str) -> str:
        """Ensure the phase is one of the allowed lifecycle values."""
        if value not in ALLOWED_PHASES:
            raise ValueError(
                f"Invalid phase {value!r}. Must be one of: {sorted(ALLOWED_PHASES)}."
            )
        return value

    @field_validator("review_status")
    @classmethod
    def _validate_review_status(cls, value: Optional[str]) -> Optional[str]:
        """Ensure review_status is 'approved', 'rejected', or None."""
        if value is not None and value not in ALLOWED_REVIEW_STATUSES:
            raise ValueError(
                f"Invalid review_status {value!r}. "
                f"Must be one of: {sorted(ALLOWED_REVIEW_STATUSES)} or None."
            )
        return value


class StateManager:
    """Persist and mutate a :class:`ProjectState` as ``state/{project_id}.json``.

    The manager guarantees:

    * The ``state/`` directory exists (created on demand).
    * A missing state file is initialised with sane defaults.
    * Corrupted JSON raises a clear error naming the file path.
    * Every :meth:`save` refreshes ``updated_at`` and writes datetimes as
      ISO-8601 strings, atomically (temp file + rename).
    * All mutations are guarded by a thread lock for swarm concurrency.

    Args:
        project_id: Unique project identifier; becomes ``state/{id}.json``.
        state_dir: Directory holding state files. Defaults to ``state/`` in
            the current working directory.

    Example:
        ```python
        manager = StateManager(project_id="demo")
        manager.update_phase("building")
        manager.log_api_call()
        ```
    """

    def __init__(self, project_id: str, state_dir: str | Path = "state") -> None:
        cleaned = project_id.strip() if isinstance(project_id, str) else project_id
        if not cleaned:
            raise ValueError("project_id must be a non-empty string.")
        # Reuse the same traversal guard as the model for early failure.
        if any(sep in cleaned for sep in ("/", "\\", "\x00")) or ".." in cleaned:
            raise ValueError(
                f"project_id {cleaned!r} must not contain path separators or '..'."
            )
        self.project_id: str = cleaned
        self.state_dir: Path = Path(state_dir)
        self.state_file: Path = self.state_dir / f"{self.project_id}.json"
        self._lock = threading.Lock()
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to create state directory '{self.state_dir}': {exc}"
            ) from exc
        if not self.state_file.exists():
            _console.print(
                f"[yellow]No state file at '{self.state_file}'. "
                "Creating one with default values.[/yellow]"
            )
            self.save(ProjectState(project_id=self.project_id))

    def load(self) -> ProjectState:
        """Load state from ``state/{project_id}.json``.

        Returns:
            The validated :class:`ProjectState`.

        Raises:
            ValueError: If the file holds corrupted JSON or fails validation.
                The message always includes the file path.
            OSError: If the file cannot be read for OS-level reasons.
        """
        with self._lock:
            if not self.state_file.exists():
                _console.print(
                    f"[yellow]State file '{self.state_file}' missing. "
                    "Re-creating with default values.[/yellow]"
                )
                default = ProjectState(project_id=self.project_id)
                self._write_atomically(default)
                _console.print(
                    f"[green]Initialised new state for project "
                    f"'{self.project_id}' at '{self.state_file}'.[/green]"
                )
                return default
            try:
                raw_text = self.state_file.read_text(encoding="utf-8")
            except OSError as exc:
                raise OSError(
                    f"Failed to read state file '{self.state_file}': {exc}"
                ) from exc
            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Corrupted state file '{self.state_file}': invalid JSON "
                    f"(line {exc.lineno}, column {exc.colno}: {exc.msg})."
                ) from exc
            try:
                state = ProjectState.model_validate(data)
            except ValidationError as exc:
                raise ValueError(
                    f"Corrupted state file '{self.state_file}': "
                    f"schema validation failed: {exc}"
                ) from exc
            _console.print(
                f"[cyan]Loaded state for project '{self.project_id}' "
                f"(phase={state.phase}) from '{self.state_file}'.[/cyan]"
            )
            return state

    def save(self, state: ProjectState) -> ProjectState:
        """Persist ``state`` to disk, refreshing ``updated_at`` first.

        Args:
            state: The state to persist. Its ``project_id`` must match this
                manager's project. ``updated_at`` is overwritten with the
                current UTC time; ``created_at`` is preserved.

        Returns:
            The same ``state`` object, with ``updated_at`` refreshed.

        Raises:
            ValueError: If ``state.project_id`` does not match this manager.
            OSError: If the file cannot be written.
        """
        with self._lock:
            if state.project_id != self.project_id:
                raise ValueError(
                    f"project_id mismatch: manager handles "
                    f"'{self.project_id}' but state has "
                    f"'{state.project_id}'."
                )
            state.updated_at = _utcnow()
            self._write_atomically(state)
            _console.print(
                f"[green]Saved state for project '{self.project_id}' "
                f"(phase={state.phase}) to '{self.state_file}'.[/green]"
            )
            return state

    def update_phase(self, phase: str) -> ProjectState:
        """Validate, apply, and persist a new lifecycle phase.

        Args:
            phase: One of ``ALLOWED_PHASES``.

        Returns:
            The updated :class:`ProjectState`.

        Raises:
            ValueError: If ``phase`` is not an allowed value.
        """
        if phase not in ALLOWED_PHASES:
            raise ValueError(
                f"Invalid phase {phase!r}. Must be one of: {sorted(ALLOWED_PHASES)}."
            )
        # Load-then-save outside a single lock acquisition would race, so do
        # the read/modify/write inline. We duplicate load/save internals
        # minimally to stay atomic without re-entering the non-reentrant flow.
        with self._lock:
            current = self._load_locked()
            current.phase = phase
            current.updated_at = _utcnow()
            self._write_atomically(current)
            _console.print(
                f"[magenta]Project '{self.project_id}' phase -> "
                f"'{phase}'. Saved to '{self.state_file}'.[/magenta]"
            )
            return current

    def log_api_call(self) -> int:
        """Increment ``api_calls_count`` by one and persist.

        Returns:
            The new cumulative API call count.
        """
        with self._lock:
            current = self._load_locked()
            current.api_calls_count += 1
            current.updated_at = _utcnow()
            self._write_atomically(current)
            _console.print(
                f"[blue]Project '{self.project_id}' API calls: "
                f"{current.api_calls_count}. Saved.[/blue]"
            )
            return current.api_calls_count

    def log_section_quality(
        self, section_name: str, score: float, iterations_used: int
    ) -> None:
        """Record one iterative section-build quality entry and persist.

        Appends ``{"stage": "section-quality", "section": ..., "score": ...,
        "iterations": ..., "at": ...}`` to ``build_history`` so per-section
        refinement is traceable in state. Never raises for bad input —
        invalid arguments are ignored with a warning instead of crashing
        a running build.
        """
        try:
            name = str(section_name).strip()
            points = max(0.0, min(10.0, float(score)))
            rounds = max(0, int(iterations_used))
        except (TypeError, ValueError) as exc:
            _console.print(
                f"[yellow]Ignoring invalid section-quality entry: {exc}[/yellow]"
            )
            return
        if not name:
            _console.print("[yellow]Ignoring section-quality entry: no name.[/yellow]")
            return
        with self._lock:
            current = self._load_locked()
            history = list(current.build_history or [])
            history.append({
                "stage": "section-quality",
                "section": name,
                "score": round(points, 2),
                "iterations": rounds,
                "at": _utcnow().isoformat(),
            })
            current.build_history = history
            current.updated_at = _utcnow()
            self._write_atomically(current)
            _console.print(
                f"[blue]Section quality logged: '{name}' "
                f"{points:g}/10 ({rounds} iterations).[/blue]"
            )

    def log_fact_check(self, fact_result: Dict[str, Any]) -> None:
        """Record one fact-check result in ``build_history``.

        Args:
            fact_result: JSON-serialisable summary produced by
                :class:`agents.fact_checker.FactCheckerAgent`.

        Raises:
            ValueError: If ``fact_result`` is not a mapping.
        """
        if not isinstance(fact_result, dict):
            raise ValueError(
                "fact_result must be a mapping, "
                f"got {type(fact_result).__name__}."
            )
        entry = dict(fact_result)
        entry["stage"] = "fact-check"
        entry["at"] = _utcnow().isoformat()
        with self._lock:
            current = self._load_locked()
            history = list(current.build_history or [])
            history.append(entry)
            current.build_history = history
            current.updated_at = _utcnow()
            self._write_atomically(current)
            _console.print(
                "[blue]Fact check logged for project "
                f"'{self.project_id}'.[/blue]"
            )

    def log_kcal_validation(self, kcal_result: Dict[str, Any]) -> None:
        """Record one USDA kcal-validation result in ``build_history``.

        Args:
            kcal_result: JSON-serialisable summary produced by
                :class:`agents.kcal_checker.KcalChecker`.

        Raises:
            ValueError: If ``kcal_result`` is not a mapping.
        """
        if not isinstance(kcal_result, dict):
            raise ValueError(
                "kcal_result must be a mapping, "
                f"got {type(kcal_result).__name__}."
            )
        entry = dict(kcal_result)
        entry["stage"] = "kcal-validation"
        entry["at"] = _utcnow().isoformat()
        with self._lock:
            current = self._load_locked()
            history = list(current.build_history or [])
            history.append(entry)
            current.build_history = history
            current.updated_at = _utcnow()
            self._write_atomically(current)
            _console.print(
                "[blue]Kcal validation logged for project "
                f"'{self.project_id}'.[/blue]"
            )

    def log_grammar_check(self, grammar_result: Dict[str, Any]) -> None:
        """Record one grammar-consensus result in ``build_history``.

        Args:
            grammar_result: JSON-serialisable summary produced by
                :class:`agents.grammar_agent.GrammarAgent`.

        Raises:
            ValueError: If ``grammar_result`` is not a mapping.
        """
        if not isinstance(grammar_result, dict):
            raise ValueError(
                "grammar_result must be a mapping, "
                f"got {type(grammar_result).__name__}."
            )
        entry = dict(grammar_result)
        entry["stage"] = "grammar-check"
        entry["at"] = _utcnow().isoformat()
        with self._lock:
            current = self._load_locked()
            history = list(current.build_history or [])
            history.append(entry)
            current.build_history = history
            current.updated_at = _utcnow()
            self._write_atomically(current)
            _console.print(
                f"[blue]Grammar check logged for project "
                f"'{self.project_id}'.[/blue]"
            )

    def get_state(self) -> ProjectState:
        """Convenience method to load and return the current state.

        Returns:
            The current :class:`ProjectState` from disk.
        """
        return self.load()

    # -- internals ---------------------------------------------------------

    def _load_locked(self) -> ProjectState:
        """Load state assuming ``self._lock`` is already held.

        Internal helper so :meth:`update_phase` and :meth:`log_api_call` can
        perform atomic read-modify-write cycles without nested locking noise.
        """
        if not self.state_file.exists():
            default = ProjectState(project_id=self.project_id)
            self._write_atomically(default)
            return default
        try:
            raw_text = self.state_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise OSError(
                f"Failed to read state file '{self.state_file}': {exc}"
            ) from exc
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Corrupted state file '{self.state_file}': invalid JSON "
                f"(line {exc.lineno}, column {exc.colno}: {exc.msg})."
            ) from exc
        try:
            return ProjectState.model_validate(data)
        except ValidationError as exc:
            raise ValueError(
                f"Corrupted state file '{self.state_file}': "
                f"schema validation failed: {exc}"
            ) from exc

    def _write_atomically(self, state: ProjectState) -> None:
        """Serialise ``state`` to JSON (ISO datetimes) via temp-file rename.

        Args:
            state: Already-timestamped state to write.

        Raises:
            OSError: If the write or rename fails.
        """
        payload = state.model_dump(mode="json")
        # model_dump(mode="json") renders datetimes as ISO-8601 strings;
        # assert the contract explicitly so regressions are loud.
        for key in ("created_at", "updated_at"):
            if not isinstance(payload.get(key), str):
                raise RuntimeError(
                    f"Internal error: datetime field '{key}' did not "
                    "serialise to an ISO-format string."
                )
        tmp_path = self.state_file.with_suffix(".tmp")
        try:
            tmp_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            tmp_path.replace(self.state_file)
        except OSError as exc:
            raise OSError(
                f"Failed to write state file '{self.state_file}': {exc}"
            ) from exc
