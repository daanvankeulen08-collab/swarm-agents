"""SwarmManager: async multi-tenant orchestration (Vector 3).

Runs several independent build projects concurrently under a shared
:class:`asyncio.Semaphore` cap, so throughput scales without blowing through
OpenRouter / NVIDIA NIM rate limits.

Tenancy model
-------------
Every project owns its own :class:`~core.state_manager.StateManager` and
therefore its own ``state/{project_id}.json`` plus its own
``state/output/{project_id}/`` section workspace. Two projects never share
mutable state, which is what makes concurrent execution safe. Each project
also gets its **own** :class:`~core.orchestrator.Orchestrator` and
:class:`~agents.reviewer.ReviewerAgent` (and therefore its own
:class:`~core.vision.VisionClient`): the underlying OpenAI SDK client is
synchronous and holds a connection pool that is not designed to be driven
from several event-loop tasks at once, so sharing one would interleave
requests unpredictably.

The pipeline body itself is synchronous (it uses blocking HTTP and a
synchronous Playwright screenshot). Each project therefore runs in its own
thread via :func:`asyncio.to_thread`, which keeps the event loop responsive
and lets the semaphore gate real parallelism.

Example:
    ```python
    import asyncio
    from core.swarm_manager import SwarmManager

    async def go():
        swarm = SwarmManager(max_concurrent_projects=3)
        results = await swarm.run_batch([
            {"project_id": "p1", "niche": "notion templates"},
            {"project_id": "p2", "niche": "budget spreadsheets"},
        ])
        return results

    print(asyncio.run(go()))
    ```
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from rich.console import Console

__all__ = ["SwarmManager", "DEFAULT_MAX_CONCURRENT_PROJECTS"]

#: Default number of projects allowed to build at the same time.
DEFAULT_MAX_CONCURRENT_PROJECTS: int = 3

_console = Console()


class SwarmManager:
    """Run many build projects concurrently with a bounded concurrency cap.

    Args:
        max_concurrent_projects: Maximum projects in flight at once
            (``<= 0`` is coerced to 1 so the swarm can never run unbounded).
        pipeline_runner: Callable invoked as
            ``pipeline_runner(project_id) -> int`` for each project. Defaults
            to :func:`build_pipeline.run_project` (imported lazily to avoid a
            circular import). Tests inject a stub here.

    Example:
        ```python
        swarm = SwarmManager(max_concurrent_projects=3)
        results = await swarm.run_batch([{"project_id": "p1",
                                          "niche": "notion templates"}])
        ```
    """

    def __init__(
        self,
        max_concurrent_projects: int = DEFAULT_MAX_CONCURRENT_PROJECTS,
        pipeline_runner: Optional[Callable[[str], Any]] = None,
    ) -> None:
        if not isinstance(max_concurrent_projects, int) or isinstance(
            max_concurrent_projects, bool
        ):
            raise TypeError(
                "max_concurrent_projects must be an int, got "
                f"{type(max_concurrent_projects).__name__}."
            )
        self.max_concurrent_projects: int = max(1, max_concurrent_projects)
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._pipeline_runner: Optional[Callable[[str], Any]] = pipeline_runner
        self._active: Dict[str, float] = {}
        self._lock: Optional[asyncio.Lock] = None
        _console.print(
            f"[cyan]SwarmManager ready[/cyan] — max_concurrent_projects="
            f"[bold]{self.max_concurrent_projects}[/bold]"
        )

    # -- internals ----------------------------------------------------------

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Lazily bind the semaphore to the running loop.

        An :class:`asyncio.Semaphore` created in one loop cannot be awaited
        in another, so the swarm is only safe when one event loop drives it
        for its whole lifetime. Created on first use and reused after.
        """
        loop = asyncio.get_event_loop()
        if self._loop is None or self._loop is not loop:
            self._loop = loop
            self._semaphore = asyncio.Semaphore(self.max_concurrent_projects)
            self._lock = asyncio.Lock()
        return self._semaphore

    def _resolve_runner(self) -> Callable[[str], Any]:
        """Return the per-project pipeline callable (lazy import)."""
        if self._pipeline_runner is not None:
            return self._pipeline_runner
        from build_pipeline import run_project

        return run_project

    # -- public API ---------------------------------------------------------

    async def run_project(self, project_id: str, niche: str = "") -> Dict[str, Any]:
        """Run one project end-to-end, respecting the concurrency cap.

        Args:
            project_id: Unique project id; also names the state file
                (``state/{project_id}.json``) and the section workspace
                (``state/output/{project_id}/``).
            niche: Optional free-text niche hint forwarded to the pipeline.

        Returns:
            A result dict with ``project_id``, ``niche``, ``success``,
            ``exit_code``, ``started_at``, ``finished_at``,
            ``duration_s``, and ``error`` (``None`` on success).

        Raises:
            ValueError: If ``project_id`` is empty or not a string.
        """
        if not isinstance(project_id, str) or not project_id.strip():
            raise ValueError(
                f"project_id must be a non-empty string, got {project_id!r}."
            )
        project_id = project_id.strip()
        semaphore = self._get_semaphore()
        runner = self._resolve_runner()

        async with semaphore:
            started = time.monotonic()
            started_at = datetime.now(timezone.utc).isoformat()
            async with self._lock:
                self._active[project_id] = started
                inflight = len(self._active)
            _console.print(
                f"[green]START[/green] {project_id} "
                f"(niche={niche or 'default'}) at {started_at} "
                f"— {inflight} project(s) in flight"
            )
            try:
                exit_code = await asyncio.to_thread(runner, project_id)
                success = exit_code == 0
                error = None
            except Exception as exc:
                exit_code = 1
                success = False
                error = f"{type(exc).__name__}: {exc}"
                _console.print(
                    f"[red]ERROR[/red] {project_id}: {error}"
                )
            finally:
                async with self._lock:
                    self._active.pop(project_id, None)

        duration = round(time.monotonic() - started, 2)
        result = {
            "project_id": project_id,
            "niche": niche,
            "success": success,
            "exit_code": exit_code,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_s": duration,
            "error": error,
        }
        _console.print(
            f"[{'green' if success else 'red'}]{'DONE' if success else 'FAIL'}"
            f"[/{'green' if success else 'red'}] {project_id} in {duration}s"
            + (f" — {error}" if error else "")
        )
        return result

    async def run_batch(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run a batch of project configs concurrently.

        Args:
            tasks: List of ``{"project_id": str, "niche": str}`` dicts.
                ``niche`` is optional and defaults to ``""``. A
                ``"project_id"`` key is required; ``"id"`` is accepted as an
                alias.

        Returns:
            One result dict per task, in the same order as the input. A task
            with a missing/invalid id yields an error result rather than
            aborting the whole batch.

        Raises:
            TypeError: If ``tasks`` is not a list.
        """
        if not isinstance(tasks, list):
            raise TypeError(f"tasks must be a list, got {type(tasks).__name__}.")
        if not tasks:
            _console.print("[yellow]Empty batch: nothing to run.[/yellow]")
            return []
        _console.print(
            f"[bold cyan]Swarm batch:[/bold cyan] {len(tasks)} project(s), "
            f"max {self.max_concurrent_projects} concurrent"
        )
        coros = []
        for index, task in enumerate(tasks):
            if not isinstance(task, dict):
                coros.append(self._bad_task(index, task, "task must be a dict"))
                continue
            project_id = task.get("project_id") or task.get("id")
            niche = str(task.get("niche", "") or "")
            if not isinstance(project_id, str) or not project_id.strip():
                coros.append(self._bad_task(
                    index, task, "missing or invalid 'project_id'"))
                continue
            coros.append(self.run_project(project_id.strip(), niche))
        return list(await asyncio.gather(*coros, return_exceptions=False))

    async def _bad_task(
        self, index: int, task: Any, reason: str
    ) -> Dict[str, Any]:
        """Return a failed result for a malformed batch entry."""
        now = datetime.now(timezone.utc).isoformat()
        _console.print(f"[red]Bad task #{index}:[/red] {reason}")
        return {
            "project_id": None,
            "niche": None,
            "success": False,
            "exit_code": 1,
            "started_at": now,
            "finished_at": now,
            "duration_s": 0.0,
            "error": f"invalid batch task #{index}: {reason}",
        }
