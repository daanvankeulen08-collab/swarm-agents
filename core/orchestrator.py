"""Orchestrator for the autonomous AI agent swarm.

This module is the engine of the swarm: the single place that talks to the
``Space Bunny Alpha`` model via OpenRouter (OpenAI-compatible API). Every
agent (scout, builder, reviewer, publisher) goes through
:meth:`Orchestrator.ask_agent`, which injects live project context from the
:class:`~core.state_manager.StateManager` before each call. That grounding is
what prevents hallucinations -- the model always sees the current phase,
recent findings, and usage counters.

Example usage (integration with StateManager):
    ```python
    from core.orchestrator import Orchestrator
    from core.state_manager import StateManager

    # Reads OPENROUTER_API_KEY / OPENROUTER_BASE_URL / OPENROUTER_MODEL
    # from the environment (a ``.env`` file is supported via python-dotenv).
    orchestrator = Orchestrator()
    state_manager = StateManager(project_id="demo")

    reply = orchestrator.ask_agent(
        system_prompt="You are the scout agent. Research profitable digital products.",
        user_prompt="Find 3 product ideas for busy freelancers.",
        state_manager=state_manager,
    )
    print(reply)

    print(orchestrator.get_model_info())
    ```

Environment variables:
    OPENROUTER_API_KEY:  API key for OpenRouter (required).
    OPENROUTER_BASE_URL: API base URL (default:
        ``"https://openrouter.ai/api/v1"``).
    OPENROUTER_MODEL:    Model name (default: ``"stealth/space-bunny-alpha"``).
    NVIDIA_NIM_API_KEY:  API key for the NVIDIA NIM fallback (optional).
    NVIDIA_NIM_BASE_URL: NIM base URL (default:
        ``"https://integrate.api.nvidia.com/v1"``).
    NVIDIA_NIM_MODEL:    Fallback model (default:
        ``"nvidia/llama-3.1-nemotron-ultra-550b"``).

Use :meth:`Orchestrator.ask_agent_with_fallback` when a truncated primary
response must trigger the fallback automatically.

All status output uses :mod:`rich` (coloured messages, spinner, token usage
and response-time reporting). This module is Windows-compatible: it uses
:class:`pathlib.Path` for paths and UTF-8 for every file operation.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import traceback
from datetime import datetime, timezone
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, OpenAI
from rich.console import Console

from .state_manager import (
    ZEN_CIRCUIT_BREAKER_TRIPS,
    ProjectState,
    StateManager,
)

__all__ = [
    "Orchestrator",
    "ZenClient",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_FALLBACK_BASE_URL",
    "DEFAULT_FALLBACK_MODEL",
    "DEFAULT_ZEN_BASE_URL",
    "DEFAULT_ZEN_MODEL",
    "ZEN_HARD_TIMEOUT",
    "FALLBACK_HARD_TIMEOUT",
    "ZEN_CIRCUIT_BREAKER_TRIPS",
    "PROVIDER_ZEN",
    "PROVIDER_OPENROUTER",
    "PROVIDER_NIM",
]

#: Default OpenRouter base URL when ``OPENROUTER_BASE_URL`` is unset.
DEFAULT_BASE_URL: str = "https://openrouter.ai/api/v1"

#: Default model when ``OPENROUTER_MODEL`` is unset.
DEFAULT_MODEL: str = "stealth/space-bunny-alpha"

#: Default NVIDIA NIM base URL when ``NVIDIA_NIM_BASE_URL`` is unset.
DEFAULT_FALLBACK_BASE_URL: str = "https://integrate.api.nvidia.com/v1"

#: Default fallback model when ``NVIDIA_NIM_MODEL`` is unset.
DEFAULT_FALLBACK_MODEL: str = "nvidia/llama-3.1-nemotron-ultra-550b"

#: OpenCode Zen base URL for the OpenAI SDK.
#:
#: Zen's documented endpoint is ``.../v1/responses``, but probing showed that
#: path answers ``ModelProtocolUnsupported`` (HTTP 400) for most models. The
#: OpenAI SDK appends ``/chat/completions`` itself, so pointing it at the
#: ``/v1`` root is the correct base — that route returns HTTP 200.
DEFAULT_ZEN_BASE_URL: str = "https://opencode.ai/zen/v1"

#: Default Zen model. ``muse-spark-1.3`` is a paid model and returns HTTP 402
#: ("Insufficient account funds") on an unfunded account; ``space-bunny-free``
#: is served without a balance. Override via ``OPENCODE_ZEN_MODEL``.
DEFAULT_ZEN_MODEL: str = "muse-spark-1.3"

#: Hard wall-clock limit (seconds) for one Zen call, mirroring the NIM stall
#: guard: the HTTP timeout alone does not bound a hung upstream.
ZEN_HARD_TIMEOUT: float = 60.0

#: Hard wall-clock limit (seconds) for one fallback API call. The free-tier
#: NIM servers currently hang indefinitely, so this daemon-thread guard —
#: not the HTTP client timeout — is what guarantees bounded latency.
FALLBACK_HARD_TIMEOUT: float = 60.0

#: Sampling temperature: balances creativity against consistency.
DEFAULT_TEMPERATURE: float = 0.7

#: Re-exported from :mod:`core.state_manager` (where the persisted counter
#: lives) so ``core.orchestrator.ZEN_CIRCUIT_BREAKER_TRIPS`` keeps working.
PROVIDER_ZEN: str = "opencode-zen"
PROVIDER_OPENROUTER: str = "openrouter"
PROVIDER_NIM: str = "nvidia-nim"

_console = Console()


class ZenClient:
    """OpenCode Zen client with a hard stall guard and streaming.

    Zen speaks the standard OpenAI **chat completions** protocol. Its
    documented ``/v1/responses`` path is *not* interoperable: probing showed
    it answers ``ModelProtocolUnsupported`` (HTTP 400) for most models, while
    ``/v1/chat/completions`` returns HTTP 200. The :class:`openai.OpenAI` SDK
    appends the ``/chat/completions`` suffix itself, so the client is pointed
    at the ``/v1`` root (option (a) in the integration notes).

    Every call runs inside a daemon thread with a
    :data:`ZEN_HARD_TIMEOUT` wall clock, because an unfunded or hung upstream
    can otherwise block a build indefinitely.

    Args:
        api_key: Zen key. Defaults to ``OPENCODE_ZEN_API_KEY``.
        base_url: Defaults to ``OPENCODE_ZEN_BASE_URL`` (the ``/responses``
            URL from the docs is accepted and normalised to its ``/v1`` root).
        model: Defaults to ``OPENCODE_ZEN_MODEL`` or :data:`DEFAULT_ZEN_MODEL`.
        timeout: Per-request HTTP timeout in seconds.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = ZEN_HARD_TIMEOUT,
    ) -> None:
        self.api_key: str = (
            api_key or os.getenv("OPENCODE_ZEN_API_KEY") or ""
        ).strip()
        raw_base = (
            base_url or os.getenv("OPENCODE_ZEN_BASE_URL")
            or DEFAULT_ZEN_BASE_URL
        ).strip()
        self.base_url: str = self._normalise_base_url(raw_base)
        self.requested_base_url: str = raw_base
        self.model: str = (
            (model or os.getenv("OPENCODE_ZEN_MODEL") or DEFAULT_ZEN_MODEL).strip()
            or DEFAULT_ZEN_MODEL
        )
        self.timeout: float = timeout
        self.client: Optional[OpenAI] = None
        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
            )

    @staticmethod
    def _normalise_base_url(raw: str) -> str:
        """Return a base URL the OpenAI SDK can extend to /chat/completions.

        The documented ``https://opencode.ai/zen/v1/responses`` value is
        reduced to ``https://opencode.ai/zen/v1``; passing the full
        ``/responses`` path to the SDK would make it request
        ``/responses/chat/completions`` and 404.
        """
        url = raw.rstrip("/")
        if url.endswith("/responses"):
            url = url[: -len("/responses")]
        return url or DEFAULT_ZEN_BASE_URL

    @property
    def enabled(self) -> bool:
        """True when a key is configured and the client is usable."""
        return self.client is not None and bool(self.api_key)

    def stream_once(self, messages: list, label: str = "") -> str:
        """Run one streaming Zen call under the hard stall guard.

        Args:
            messages: Chat messages in OpenAI format.
            label: Short label for the status line.

        Returns:
            The assembled response text (empty string when the provider
            returned nothing).

        Raises:
            RuntimeError: If the call exceeds :data:`ZEN_HARD_TIMEOUT` or the
                provider returns an error. Any failure is raised so the
                cascade can fall through to the next provider.
        """
        if not self.enabled:
            raise RuntimeError("OpenCode Zen is not configured (no API key).")

        result: Dict[str, Any] = {}

        def _run() -> None:
            stream = self.client.chat.completions.create(  # type: ignore[union-attr]
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                stream=True,
                temperature=DEFAULT_TEMPERATURE,
                timeout=self.timeout,
            )
            pieces: List[str] = []
            for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                if choices:
                    delta = getattr(choices[0], "delta", None)
                    piece = getattr(delta, "content", None)
                    if piece:
                        pieces.append(piece)
            result["content"] = "".join(pieces).strip()

        worker = threading.Thread(
            target=lambda: _capture(_run, result), daemon=True
        )
        started = time.perf_counter()
        worker.start()
        worker.join(self.timeout)
        if worker.is_alive():
            raise RuntimeError(
                f"OpenCode Zen stalled: no response within {self.timeout}s "
                f"(model='{self.model}')."
            )
        elapsed = time.perf_counter() - started
        if "error" in result:
            raise RuntimeError(str(result["error"]))
        _console.print(
            f"[dim]zen stream ok ({label}) in {elapsed:.2f}s[/dim]"
        )
        return str(result.get("content", ""))


def _capture(fn, result: Dict[str, Any]) -> None:
    """Run ``fn``, capturing any exception into ``result['error']``.

    Lets the daemon stall-guard thread fail cleanly instead of dying silently
    and leaving the caller waiting for the full timeout.
    """
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller
        result["error"] = f"{type(exc).__name__}: {exc}"


class Orchestrator:
    """Single gateway from the agent swarm to Space Bunny Alpha via OpenRouter.

    The orchestrator owns the OpenAI-compatible client, grounds every prompt
    with live :class:`ProjectState` context, retries transient network faults
    once, and records each successful call via
    :meth:`StateManager.log_api_call`.

    Args:
        api_key: OpenRouter API key. Defaults to ``OPENROUTER_API_KEY`` from
            the environment (``.env`` supported). Must be non-empty.
        base_url: API base URL. Defaults to ``OPENROUTER_BASE_URL`` or
            ``DEFAULT_BASE_URL``. Trailing slashes are stripped.
        model: Model identifier. Defaults to ``OPENROUTER_MODEL`` or
            ``DEFAULT_MODEL``.
        timeout: Per-request timeout in seconds for the OpenAI client.
        fallback_api_key: NVIDIA NIM API key. Defaults to
            ``NVIDIA_NIM_API_KEY``, falling back to ``NVIDIA_API_KEY``.
            When both are empty, the fallback is disabled and
            :meth:`ask_agent_with_fallback` reports primary results
            (or raises when the primary is unusable).
        fallback_base_url: NIM base URL. Defaults to
            ``NVIDIA_NIM_BASE_URL`` or ``DEFAULT_FALLBACK_BASE_URL``.
        fallback_model: Fallback model id. Defaults to
            ``NVIDIA_NIM_MODEL`` or ``DEFAULT_FALLBACK_MODEL``.

    Raises:
        EnvironmentError: If no API key is available after checking the
            explicit argument and the environment.

    Example:
        ```python
        orchestrator = Orchestrator()  # reads .env automatically
        ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
        fallback_api_key: Optional[str] = None,
        fallback_base_url: Optional[str] = None,
        fallback_model: Optional[str] = None,
    ) -> None:
        load_dotenv()  # No-op if no .env file; never overrides real env vars.

        resolved_key = (api_key or os.getenv("OPENROUTER_API_KEY") or "").strip()
        if not resolved_key:
            raise EnvironmentError(
                "Missing OpenRouter API key. Set the OPENROUTER_API_KEY "
                "environment variable (or place it in a UTF-8 encoded .env "
                "file) or pass api_key='...' explicitly."
            )
        resolved_base_url = (
            base_url or os.getenv("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL
        ).strip().rstrip("/") or DEFAULT_BASE_URL
        resolved_model = (
            (model or os.getenv("OPENROUTER_MODEL") or DEFAULT_MODEL).strip()
            or DEFAULT_MODEL
        )

        self.api_key: str = resolved_key
        self.base_url: str = resolved_base_url
        self.model: str = resolved_model
        self.timeout: float = timeout
        self._total_api_calls: int = 0
        #: Per-provider call counts, e.g. {"opencode-zen": 12, "openrouter": 3}.
        self.provider_stats: Dict[str, int] = {
            PROVIDER_ZEN: 0,
            PROVIDER_OPENROUTER: 0,
            PROVIDER_NIM: 0,
        }
        #: Provider that served the most recent successful call.
        self.last_provider: Optional[str] = None

        # -- Primary #1: OpenCode Zen (optional) ------------------------------
        self.zen: ZenClient = ZenClient(timeout=timeout)
        if self.zen.enabled:
            _console.print(
                f"[green]Zen primary ready[/green] — model="
                f"[bold]{self.zen.model}[/bold] base_url={self.zen.base_url}"
            )
        else:
            _console.print(
                "[yellow]No OPENCODE_ZEN_API_KEY — Zen primary disabled. "
                "Cascade starts at OpenRouter.[/yellow]"
            )

        # Optional OpenRouter attribution headers (harmless if unset).
        default_headers: Dict[str, str] = {}
        site_url = os.getenv("OPENROUTER_SITE_URL", "").strip()
        app_name = os.getenv("OPENROUTER_APP_NAME", "").strip()
        if site_url:
            default_headers["HTTP-Referer"] = site_url
        if app_name:
            default_headers["X-Title"] = app_name

        # Path handling is Windows-safe via pathlib; nothing file-based here
        # except the .env load above, which python-dotenv reads as UTF-8.
        _ = Path(self.base_url)  # Validates the value is path-parseable.

        self.client: OpenAI = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            default_headers=default_headers or None,
        )
        _console.print(
            f"[green]Orchestrator ready[/green] — model="
            f"[bold]{self.model}[/bold] base_url={self.base_url}"
        )

        # -- Fallback model (NVIDIA NIM, optional) ----------------------------
        # Standard `openai` client, same library as the primary client.
        resolved_fallback_key = (
            fallback_api_key
            or os.getenv("NVIDIA_NIM_API_KEY")
            or os.getenv("NVIDIA_API_KEY")
            or ""
        ).strip()
        self.fallback_base_url: str = (
            fallback_base_url or os.getenv("NVIDIA_NIM_BASE_URL")
            or DEFAULT_FALLBACK_BASE_URL
        ).strip().rstrip("/") or DEFAULT_FALLBACK_BASE_URL
        self.fallback_model: str = (
            (fallback_model or os.getenv("NVIDIA_NIM_MODEL")
             or DEFAULT_FALLBACK_MODEL).strip()
            or DEFAULT_FALLBACK_MODEL
        )
        self.fallback_client: Optional[OpenAI] = None
        if resolved_fallback_key:
            self.fallback_client = OpenAI(
                api_key=resolved_fallback_key,
                base_url=self.fallback_base_url,
                timeout=self.timeout,
            )
            _console.print(
                "[green]Fallback ready[/green] — model="
                f"[bold]{self.fallback_model}[/bold] "
                f"base_url={self.fallback_base_url}"
            )
        else:
            _console.print(
                "[yellow]No NVIDIA_NIM_API_KEY (or NVIDIA_API_KEY) — fallback "
                "model disabled. Set one to enable the fallback.[/yellow]"
            )

    @property
    def fallback_available(self) -> bool:
        """True when the NVIDIA NIM fallback client is configured."""
        return self.fallback_client is not None

    def build_context_prompt(self, state: ProjectState) -> str:
        """Build a grounding context block from the current project state.

        Args:
            state: The live :class:`ProjectState` to summarise.

        Returns:
            A clearly formatted multi-line string containing the current
            phase, scout findings (if any), product name, and API call count,
            plus closely related fields (review status, sales description)
            that further reduce hallucination risk.
        """
        lines = [
            "=== LIVE PROJECT CONTEXT (single source of truth — do not invent values) ===",
            f"Current phase: {state.phase}",
            f"Product name: {state.product_name or '(not set yet)'}",
            f"Total API calls so far: {state.api_calls_count}",
        ]
        if state.scout_findings:
            try:
                findings = json.dumps(
                    state.scout_findings, indent=2, ensure_ascii=False
                )
            except (TypeError, ValueError):
                findings = str(state.scout_findings)
            # Truncate very large findings so the context window stays usable.
            if len(findings) > 4000:
                findings = findings[:4000] + "\n... [truncated]"
            lines.append(f"Scout findings:\n{findings}")
        else:
            lines.append("Scout findings: (none yet)")
        if state.review_status:
            lines.append(f"Review status: {state.review_status}")
        if state.review_feedback:
            lines.append(f"Review feedback: {state.review_feedback}")
        if state.sales_description:
            lines.append(f"Sales description: {state.sales_description}")
        if state.product_code:
            lines.append(
                f"Product code exists: yes ({len(state.product_code)} characters)"
            )
        else:
            lines.append("Product code exists: no")
        lines.append(
            "Use ONLY the context above as fact. "
            "If a value is marked as not set/none, say so instead of guessing."
        )
        return "\n".join(lines)

    def get_model_info(self) -> dict:
        """Return debugging/monitoring info about this orchestrator.

        Returns:
            Dict with ``model_name``, ``base_url``, and ``total_api_calls``
            (successful calls made through this instance in this process).
        """
        return {
            "model_name": self.model,
            "base_url": self.base_url,
            "total_api_calls": self._total_api_calls,
        }

    def ask_agent(
        self,
        system_prompt: str,
        user_prompt: str,
        state_manager: StateManager,
    ) -> str:
        """Ask Space Bunny Alpha with live state context injected.

        Before the API call, fresh state is loaded from ``state_manager`` and
        rendered via :meth:`build_context_prompt`, then appended to
        ``system_prompt`` so the model reasons from ground truth.

        Args:
            system_prompt: Role / behaviour instructions for the agent.
            user_prompt: The task to perform this turn.
            state_manager: The project's :class:`StateManager`, used both for
                context injection and for usage tracking.

        Returns:
            The model's response content as a string.

        Raises:
            TypeError: If ``state_manager`` is not a :class:`StateManager`.
            ValueError: If either prompt is empty or the model returns no content.
            RuntimeError: If the API call fails after one retry, with the
                underlying details included (the API key is never logged).
        """
        return self.ask_primary_agent(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            state_manager=state_manager,
        )

    def _stream_primary_once(
        self, messages: list, label: str
    ) -> tuple[str, Any, float]:
        """Run one streaming primary call.

        Returns:
            ``(content, usage, elapsed_seconds)`` with content stripped
            (possibly empty when the provider returns nothing).
        """
        start = time.perf_counter()
        with _console.status(
            f"[bold cyan]Streaming from {self.model} ({label})…[/bold cyan]",
            spinner="dots",
        ):
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                stream=True,
                temperature=DEFAULT_TEMPERATURE,
                timeout=self.timeout,
                extra_body={
                    "provider": {
                        "only": ["stealth"],
                        "allow_fallbacks": False,
                    }
                },
            )
            full_response = ""
            stream_usage = None
            for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                if choices:
                    delta = getattr(choices[0], "delta", None)
                    piece = getattr(delta, "content", None)
                    if piece:
                        full_response += piece
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    stream_usage = chunk_usage
        return full_response.strip(), stream_usage, time.perf_counter() - start

    def _retry_empty_primary(
        self, messages: list
    ) -> tuple[str, Any, float]:
        """Retry an empty OpenRouter body with escalating backoff.

        An empty body is a transient provider fault, not a verdict, so the
        same call is repeated up to twice (2s, then 5s) before the caller
        falls through to the fallback model.

        Returns:
            ``(content, usage, elapsed_seconds)`` for the last attempt.
            ``content`` is ``""`` when every attempt came back empty, and
            ``(None, 0.0)`` when every attempt raised.
        """
        content, stream_usage, elapsed = "", None, 0.0
        for retry_no, backoff in ((1, 2.0), (2, 5.0)):
            _console.print(
                "[yellow]Primary returned empty response, "
                f"retrying (attempt {retry_no}/2)…[/yellow]"
            )
            time.sleep(backoff)
            try:
                content, stream_usage, elapsed = self._stream_primary_once(
                    messages, f"empty-retry {retry_no}/2"
                )
            except Exception as exc:
                # An exception means no body at all; keep the empty content
                # so the caller's own error handling decides what happens.
                _console.print(
                    f"[yellow]Empty-retry {retry_no}/2 raised "
                    f"({type(exc).__name__}): {exc}[/yellow]"
                )
                continue
            if content:
                break
        return content, stream_usage, elapsed

    def call_with_retry(
        self,
        call_fn,
        failure_type: str,
        max_retries: int = 1,
        backoff_seconds: tuple = (2.0, 5.0),
    ):
        """Run ``call_fn`` under one central retry policy.

        The three failure classes get different treatment:

        * ``"transient"`` — network-level faults: retry with backoff.
        * ``"stall"`` — hung upstream: NEVER retry, raise immediately so the
          caller fails over without burning another full timeout.
        * ``"empty"`` — provider answered but said nothing: retry with
          backoff (an empty body is usually transient, not a verdict).

        Args:
            call_fn: Zero-argument callable returning the response (falsy
                means "empty response" for the ``"empty"`` policy).
            failure_type: One of ``"transient"``, ``"stall"``, ``"empty"``.
            max_retries: Retries after the initial attempt (0 = try once).
            backoff_seconds: Sleeps between attempts; reused cyclically.

        Returns:
            Whatever ``call_fn`` returned (possibly falsy when retries are
            exhausted on an ``"empty"`` policy — callers decide what empty
            means).

        Raises:
            ValueError: On an unknown ``failure_type``.
            Exception: Whatever ``call_fn`` raised on its final attempt
                (immediately for ``"stall"``).
        """
        if failure_type not in ("transient", "stall", "empty"):
            raise ValueError(
                f"Unknown failure_type {failure_type!r}; expected "
                "'transient', 'stall', or 'empty'."
            )
        backoffs = list(backoff_seconds) or [0.0]
        max_retries = max(0, int(max_retries))
        for attempt in range(max_retries + 1):
            try:
                result = call_fn()
                if result or failure_type != "empty" or attempt >= max_retries:
                    return result
                delay = backoffs[min(attempt, len(backoffs) - 1)]
                _console.print(
                    f"[yellow]Empty response, retrying in {delay:g}s "
                    f"(attempt {attempt + 1}/{max_retries})…[/yellow]"
                )
                time.sleep(delay)
            except Exception:
                if failure_type == "stall":
                    raise
                if attempt >= max_retries:
                    raise
                delay = backoffs[min(attempt, len(backoffs) - 1)]
                _console.print(
                    f"[yellow]Transient failure, retrying in {delay:g}s "
                    f"(attempt {attempt + 1}/{max_retries})…[/yellow]"
                )
                time.sleep(delay)
        return result

    def ask_primary_agent(
        self,
        system_prompt: str,
        user_prompt: str,
        state_manager: StateManager,
    ) -> str:
        """Ask the primary model through the provider cascade.

        Priority order:

        1. **OpenCode Zen** (``muse-spark-1.3``) — streaming, 60s stall guard,
           and the 2-retry empty-response backoff.
        2. **OpenRouter** (``stealth/space-bunny-alpha``) — the original path,
           including the pinned-stealth ``extra_body``.
        3. **NVIDIA NIM** — reached via
           :meth:`ask_agent_with_fallback` when both primaries fail.

        Every hop applies the same resilience features, and the provider that
        actually served the request is recorded in :attr:`provider_stats` and
        reported as ``Provider used: <name>``.

        Args:
            system_prompt: Role / behaviour instructions for the agent.
            user_prompt: The task to perform this turn.
            state_manager: The project's :class:`StateManager`, used both for
                context injection and for usage tracking.

        Returns:
            The full response content as a string.

        Raises:
            TypeError: If ``state_manager`` is not a :class:`StateManager`.
            ValueError: If either prompt is empty.
            RuntimeError: If every provider in the cascade fails.
        """
        if not isinstance(state_manager, StateManager):
            raise TypeError(
                "state_manager must be a core.state_manager.StateManager instance, "
                f"got {type(state_manager).__name__}."
            )
        if not system_prompt or not system_prompt.strip():
            raise ValueError("system_prompt must be a non-empty string.")
        if not user_prompt or not user_prompt.strip():
            raise ValueError("user_prompt must be a non-empty string.")

        # --- Inject ground-truth context (anti-hallucination) -----------------
        try:
            live_state: ProjectState = state_manager.get_state()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load project state for '{state_manager.project_id}' "
                f"before calling model '{self.model}': {exc}"
            ) from exc
        context_block = self.build_context_prompt(live_state)
        grounded_system_prompt = (
            f"{system_prompt.strip()}\n\n{context_block}"
        )

        messages = [
            {"role": "system", "content": grounded_system_prompt},
            {"role": "user", "content": user_prompt.strip()},
        ]

        # -- Cascade hop 1: OpenCode Zen -------------------------------------
        if self.zen.enabled:
            zen_error: Optional[str] = None
            zen_exc: Optional[BaseException] = None
            content = ""
            zen_attempted = False
            if self._zen_stall_count(state_manager) >= ZEN_CIRCUIT_BREAKER_TRIPS:
                # Deliberate skip, NOT a failure: do not log it as one, or
                # provider_stats shows Zen failing on every later call.
                zen_error = (
                    "circuit breaker open after "
                    f"{ZEN_CIRCUIT_BREAKER_TRIPS} consecutive stalls"
                )
                _console.print(
                    f"[yellow]Circuit breaker: Zen skipped after "
                    f"{ZEN_CIRCUIT_BREAKER_TRIPS} consecutive stalls"
                    " (rest of this run)[/yellow]"
                )
            else:
                zen_attempted = True
                try:
                    # Stall policy: NO retry — a hung upstream gets one
                    # 60s window, then the cascade fails over immediately.
                    content = self.call_with_retry(
                        lambda: self.zen.stream_once(messages, "initial"),
                        failure_type="stall",
                        max_retries=0,
                    )
                except Exception as exc:
                    zen_error = f"{type(exc).__name__}: {exc}"
                    zen_exc = exc
                    _console.print(
                        f"[yellow]Zen call failed: {zen_error}[/yellow]"
                    )
                    if self._is_stall_error(exc):
                        self._record_zen_stall(state_manager)
                if not content and zen_exc is None:
                    # Empty body (not a stall): retry once with backoff.
                    # Budget stays at 3 Zen attempts per call (initial +
                    # 2 retries), as before — the stall-phase call above
                    # already spent the first attempt.
                    try:
                        content = self.call_with_retry(
                            lambda: self.zen.stream_once(
                                messages, "empty-retry"
                            ),
                            failure_type="empty",
                            max_retries=1,
                            backoff_seconds=(2.0,),
                        )
                    except Exception as exc:
                        zen_error = f"{type(exc).__name__}: {exc}"
                        zen_exc = exc
                        _console.print(
                            f"[yellow]Zen call failed: {zen_error}[/yellow]"
                        )
                        if self._is_stall_error(exc):
                            self._record_zen_stall(state_manager)
            if content:
                self._reset_zen_stalls(state_manager)
                self._record_provider(PROVIDER_ZEN)
                try:
                    state_manager.log_api_call()
                except Exception as exc:
                    raise RuntimeError(
                        "Zen call succeeded but failed to log the API call for "
                        f"project '{live_state.project_id}': {exc}"
                    ) from exc
                self._total_api_calls += 1
                _console.print(f"[green]Provider used: {PROVIDER_ZEN}[/green]")
                return content
            if zen_attempted:
                self._record_provider_failure(
                    PROVIDER_ZEN, zen_exc, state_manager
                )
            _console.print(
                f"[yellow]Provider used: {PROVIDER_OPENROUTER} "
                f"(zen failed: {zen_error or 'empty response'})[/yellow]"
            )
        else:
            _console.print(
                f"[yellow]Provider used: {PROVIDER_OPENROUTER} "
                "(zen not configured)[/yellow]"
            )

        # -- Cascade hop 2: OpenRouter (original implementation) --------------
        try:
            content = self._ask_openrouter(
                messages, live_state, state_manager
            )
        except Exception as exc:
            self._record_provider_failure(
                PROVIDER_OPENROUTER, exc, state_manager
            )
            _console.print(
                f"[yellow]Provider used: {PROVIDER_NIM} "
                f"(openrouter failed: {type(exc).__name__})[/yellow]"
            )
            raise RuntimeError(
                f"All primary providers failed for project "
                f"'{live_state.project_id}'. Zen: "
                f"{'not configured' if not self.zen.enabled else 'failed'}; "
                f"OpenRouter: {type(exc).__name__}: {exc}"
            ) from exc
        self._record_provider(PROVIDER_OPENROUTER)
        _console.print(f"[green]Provider used: {PROVIDER_OPENROUTER}[/green]")
        return content

    def _record_provider(self, provider: str) -> None:
        """Increment the success counter for ``provider``."""
        self.provider_stats[provider] = self.provider_stats.get(provider, 0) + 1
        self.last_provider = provider

    def _record_provider_failure(
        self,
        provider: str,
        exc: Optional[BaseException] = None,
        state_manager: Optional[StateManager] = None,
    ) -> None:
        """Record a provider hop failure with full traceback for post-mortem.

        The Maiden Voyage 011 ``UnboundLocalError`` proved that a bare
        ``type(exc).__name__`` is not enough to debug a cascade failure, so
        every hop failure now captures the complete traceback twice: once to
        the terminal log, once into ``state.build_history``.

        Args:
            provider: The provider that failed (``opencode-zen``,
                ``openrouter``, ``nvidia-nim``).
            exc: The exception that caused the failure, or None.
            state_manager: When given, the failure record is appended to
                ``state.build_history`` as ``{"stage": "provider-failure",
                ...}``. Persistence failures are logged, never raised.
        """
        stats = self.provider_stats.setdefault("failures", {})  # type: ignore[arg-type]
        stats[provider] = stats.get(provider, 0) + 1  # type: ignore[union-attr]
        entry: Dict[str, Any] = {
            "stage": "provider-failure",
            "provider": provider,
            "error_type": type(exc).__name__ if exc is not None else "Unknown",
            "error": str(exc)[:500] if exc is not None else "",
            "traceback": "".join(traceback.format_exception(
                type(exc), exc, exc.__traceback__
            )) if exc is not None else "",
        }
        _console.print(
            f"[red]Provider '{provider}' failed "
            f"({entry['error_type']}):[/red] {entry['error'][:200]}"
        )
        if entry["traceback"]:
            _console.print(
                f"[dim]{entry['traceback'][:2000]}[/dim]"
            )
        if state_manager is None:
            return
        try:
            state = state_manager.get_state()
            history = list(state.build_history or [])
            history.append({
                "at": datetime.now(timezone.utc).isoformat(),
                **entry,
            })
            state.build_history = history
            state_manager.save(state)
        except Exception as persist_exc:
            _console.print(
                f"[yellow]Could not persist provider-failure record: "
                f"{persist_exc}[/yellow]"
            )

    @staticmethod
    def _is_stall_error(exc: BaseException) -> bool:
        """True when an exception is a provider stall, not a verdict."""
        return "stalled" in str(exc).lower()

    def _zen_stall_count(self, state_manager: StateManager) -> int:
        """Return consecutive Zen stalls recorded for this run (0 on doubt)."""
        try:
            return int(state_manager.get_state().zen_stall_count or 0)
        except Exception:
            return 0

    def _record_zen_stall(self, state_manager: StateManager) -> int:
        """Increment the consecutive-Zen-stall counter; return the new count.

        Best effort: a persistence failure warns and reports 0 (breaker
        degrades to inactive) rather than breaking the cascade.
        """
        try:
            state = state_manager.get_state()
            count = int(state.zen_stall_count or 0) + 1
            state.zen_stall_count = count
            state_manager.save(state)
            return count
        except Exception as exc:
            _console.print(
                f"[yellow]Could not record Zen stall count: {exc}[/yellow]"
            )
            return 0

    def _reset_zen_stalls(self, state_manager: StateManager) -> None:
        """Clear the consecutive-stall counter after a Zen success."""
        try:
            state = state_manager.get_state()
            if state.zen_stall_count:
                state.zen_stall_count = 0
                state_manager.save(state)
        except Exception as exc:
            _console.print(
                f"[yellow]Could not reset Zen stall count: {exc}[/yellow]"
            )

    def _ask_openrouter(
        self,
        messages: list,
        live_state: ProjectState,
        state_manager: StateManager,
    ) -> str:
        """Run the original streaming OpenRouter call (cascade hop 2).

        Split out of :meth:`ask_primary_agent` so the cascade can wrap it
        without duplicating the retry/usage-tracking logic.

        Args:
            messages: Grounded system+user messages.
            live_state: The already-loaded project state.
            state_manager: State manager for usage logging.

        Returns:
            The streamed response content.

        Raises:
            RuntimeError: If the call fails after all retries.
        """
        _console.print(
            f"[cyan]Calling model[/cyan] [bold]{self.model}[/bold] "
            f"for project '{live_state.project_id}' "
            f"(phase={live_state.phase}, api_calls={live_state.api_calls_count}, "
            "streaming, provider=stealth-only)…"
        )

        last_error: Optional[BaseException] = None
        # Bound before the loop: an exception on the first attempt must not
        # escape as NameError (the MV011-class bug), nor may a later attempt
        # return a previous attempt's stale content.
        content = ""
        stream_usage = None
        elapsed = 0.0
        for attempt in (1, 2):  # Initial try + exactly one retry.
            try:
                content, stream_usage, elapsed = self._stream_primary_once(
                    messages, f"attempt {attempt}/2"
                )
                if not content:
                    # Transient provider issue ("empty response"): retry the
                    # SAME call with escalating backoff before giving up to
                    # the fallback path.
                    content, stream_usage, elapsed = self._retry_empty_primary(
                        messages
                    )
                if not content:
                    raise ValueError(
                        "Model returned empty content (empty stream)."
                    )
                self._report_usage(
                    SimpleNamespace(usage=stream_usage), elapsed
                )
                # --- Track usage only after a confirmed success ---------------
                try:
                    state_manager.log_api_call()
                except Exception as exc:
                    raise RuntimeError(
                        "Model call succeeded but failed to log the API call "
                        f"for project '{live_state.project_id}': {exc}"
                    ) from exc
                self._total_api_calls += 1
                return content
            except (APITimeoutError, APIConnectionError, TimeoutError) as exc:
                last_error = exc
                _console.print(
                    f"[yellow]Attempt {attempt}/2 failed with transient "
                    f"network error ({type(exc).__name__}): {exc}[/yellow]"
                )
                if attempt == 2:
                    break
                time.sleep(2.0)  # Brief backoff before the single retry.
            except Exception as exc:
                if "empty response" in str(exc).lower():
                    # Same transient class as an empty stream: retry the
                    # SAME call with backoff before falling through to
                    # the fallback model.
                    content, stream_usage, elapsed = self._retry_empty_primary(
                        messages
                    )
                    if content:
                        self._report_usage(
                            SimpleNamespace(usage=stream_usage), elapsed
                        )
                        try:
                            state_manager.log_api_call()
                        except Exception as log_exc:
                            raise RuntimeError(
                                "Model call succeeded but failed to log the API call "
                                f"for project '{live_state.project_id}': {log_exc}"
                            ) from log_exc
                        self._total_api_calls += 1
                        return content
                # Non-retryable (auth, bad request, SDK misuse, …): fail fast
                # with context but without leaking the API key.
                raise RuntimeError(
                    f"OpenRouter API call failed (model='{self.model}', "
                    f"base_url='{self.base_url}', project="
                    f"'{live_state.project_id}'): "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

        raise RuntimeError(
            f"OpenRouter API call failed after 1 retry (model='{self.model}', "
            f"base_url='{self.base_url}', project='{live_state.project_id}'): "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    def ask_agent_with_fallback(
        self,
        system_prompt: str,
        user_prompt: str,
        state_manager: StateManager,
        expected_sections: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Ask the primary model, falling back when needed.

        Tries Space Bunny Alpha first via :meth:`ask_agent` (unchanged
        behaviour, including its retry and usage logging). The response is
        then completeness-checked (length over 100 characters, no
        mid-sentence/mid-table ending, all ``expected_sections`` present
        when given). If the primary fails or looks truncated, the same
        prompts go to the fallback model.

        Args:
            system_prompt: Role / behaviour instructions for the agent.
            user_prompt: The task to perform this turn.
            state_manager: The project's :class:`StateManager`.
            expected_sections: Optional substrings that must all appear in
                a complete response (e.g. template section titles).

        Returns:
            Dict with ``content`` (str), ``model_used``
            (``"primary"``/``"fallback"``), and ``was_truncated`` (True
            whenever the primary output was unusable; the content is then
            either the fallback answer or — if the fallback also failed —
            the truncated primary text so the pipeline can continue).

        Raises:
            TypeError / ValueError: Same input validation as :meth:`ask_agent`.
            RuntimeError: If the primary is unusable, no fallback is
                configured, and there is no primary text to degrade to.
        """
        if not isinstance(state_manager, StateManager):
            raise TypeError(
                "state_manager must be a core.state_manager.StateManager instance, "
                f"got {type(state_manager).__name__}."
            )
        if not system_prompt or not system_prompt.strip():
            raise ValueError("system_prompt must be a non-empty string.")
        if not user_prompt or not user_prompt.strip():
            raise ValueError("user_prompt must be a non-empty string.")

        primary_content: Optional[str] = None
        primary_error: Optional[str] = None
        try:
            primary_content = self.ask_agent(
                system_prompt, user_prompt, state_manager
            )
        except Exception as exc:
            primary_error = f"{type(exc).__name__}: {exc}"

        if primary_content is not None and self._looks_complete(
            primary_content, expected_sections
        ):
            return {
                "content": primary_content,
                "model_used": "primary",
                "was_truncated": False,
            }

        reason = (
            f"primary failed ({primary_error})"
            if primary_content is None
            else "primary response truncated/incomplete"
        )
        if not self.fallback_available:
            raise RuntimeError(
                f"Primary model {reason} and no fallback is configured "
                "(set NVIDIA_NIM_API_KEY or NVIDIA_API_KEY to enable "
                "the fallback)."
            )
        _console.print(
            "[yellow]Primary model truncated/failed, switching to fallback "
            f"({reason}).[/yellow]"
        )
        try:
            fallback_content = self.ask_fallback_agent(
                system_prompt, user_prompt, state_manager
            )
        except Exception as exc:
            # Never crash the pipeline on fallback trouble: hand back the
            # primary response (even truncated) so callers can continue.
            _console.print(
                "[yellow]Fallback timed out after 60s or failed, server "
                "likely overloaded. Degrading gracefully.[/yellow]"
            )
            if primary_content is None:
                raise RuntimeError(
                    "Primary model failed and fallback also failed "
                    f"({type(exc).__name__}: {exc}); nothing to return."
                ) from exc
            return {
                "content": primary_content,
                "model_used": "primary",
                "was_truncated": True,
            }
        return {
            "content": fallback_content,
            "model_used": "fallback",
            "was_truncated": True,
        }

    def ask_fallback_agent(
        self,
        system_prompt: str,
        user_prompt: str,
        state_manager: StateManager,
    ) -> str:
        """Call the fallback model directly with state context injected.

        Args:
            system_prompt: Role / behaviour instructions for the agent.
            user_prompt: The task to perform this turn.
            state_manager: The project's :class:`StateManager`, used both
                for context injection and for usage tracking.

        Returns:
            The fallback model's response content as a string.

        Raises:
            TypeError / ValueError: Same input validation as :meth:`ask_agent`.
            RuntimeError: If no fallback is configured or the call fails
                after one retry (key never logged).
        """
        if not isinstance(state_manager, StateManager):
            raise TypeError(
                "state_manager must be a core.state_manager.StateManager instance, "
                f"got {type(state_manager).__name__}."
            )
        if not system_prompt or not system_prompt.strip():
            raise ValueError("system_prompt must be a non-empty string.")
        if not user_prompt or not user_prompt.strip():
            raise ValueError("user_prompt must be a non-empty string.")
        if not self.fallback_available:
            raise RuntimeError(
                "Fallback model requested but neither NVIDIA_NIM_API_KEY "
                "nor NVIDIA_API_KEY is set."
            )
        try:
            live_state: ProjectState = state_manager.get_state()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load project state for '{state_manager.project_id}' "
                f"before calling fallback model '{self.fallback_model}': {exc}"
            ) from exc
        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]
        return self._call_fallback(messages, live_state.project_id, state_manager)

    def _call_fallback(
        self,
        messages: list,
        project_id: str,
        state_manager: StateManager,
    ) -> str:
        """Execute one fallback chat call with stall guard + usage logging.

        Every attempt runs under a hard ``FALLBACK_HARD_TIMEOUT`` wall-clock
        guard (daemon thread): an overloaded NIM server can never hang the
        swarm. A stall raises ``TimeoutError`` immediately with NO retry —
        retrying a saturated server only doubles the wait. Other transient
        network errors still get exactly one retry.
        """
        assert self.fallback_client is not None
        _console.print(
            f"[cyan]Calling fallback model[/cyan] [bold]{self.fallback_model}[/bold] "
            f"for project '{project_id}'…"
        )
        last_error: Optional[BaseException] = None
        for attempt in (1, 2):
            try:
                start = time.perf_counter()
                with _console.status(
                    f"[bold cyan]Waiting for {self.fallback_model} "
                    f"(attempt {attempt}/2, max {FALLBACK_HARD_TIMEOUT:g}s)…"
                    "[/bold cyan]",
                    spinner="dots",
                ):
                    response = self._create_guarded(messages)
                elapsed = time.perf_counter() - start
                content = self._extract_content(response)
                self._report_usage(response, elapsed)
                try:
                    state_manager.log_api_call()
                except Exception as exc:
                    raise RuntimeError(
                        "Fallback call succeeded but failed to log the API call "
                        f"for project '{project_id}': {exc}"
                    ) from exc
                self._total_api_calls += 1
                self._record_provider(PROVIDER_NIM)
                _console.print(
                    f"[green]Provider used: {PROVIDER_NIM}[/green]"
                )
                return content
            except TimeoutError as exc:
                # Server stall: fail fast, never retry an overloaded server.
                raise RuntimeError(
                    f"Fallback timed out after {FALLBACK_HARD_TIMEOUT:g}s "
                    f"(model='{self.fallback_model}', project='{project_id}'): "
                    "server likely overloaded."
                ) from exc
            except (APITimeoutError, APIConnectionError) as exc:
                last_error = exc
                _console.print(
                    f"[yellow]Fallback attempt {attempt}/2 failed with transient "
                    f"network error ({type(exc).__name__}): {exc}[/yellow]"
                )
                if attempt == 2:
                    break
                time.sleep(2.0)
            except Exception as exc:
                raise RuntimeError(
                    f"NVIDIA NIM fallback call failed (model='{self.fallback_model}', "
                    f"base_url='{self.fallback_base_url}', project='{project_id}'): "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
        raise RuntimeError(
            f"NVIDIA NIM fallback call failed after 1 retry "
            f"(model='{self.fallback_model}', project='{project_id}'): "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    def _create_guarded(self, messages: list) -> Any:
        """Run ``chat.completions.create`` under the hard stall guard.

        The OpenAI call executes on a daemon thread (so a hung socket can
        never block interpreter shutdown); if it outlives
        ``FALLBACK_HARD_TIMEOUT`` seconds, ``TimeoutError`` is raised.

        Raises:
            TimeoutError: If the call exceeds the hard timeout.
            Exception: Whatever the underlying API call raised.
        """
        assert self.fallback_client is not None
        box: Dict[str, Any] = {}

        def _target() -> None:
            try:
                box["response"] = self.fallback_client.chat.completions.create(
                    model=self.fallback_model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=DEFAULT_TEMPERATURE,
                    timeout=min(self.timeout, FALLBACK_HARD_TIMEOUT),
                )
            except Exception as exc:  # Captured and re-raised by the joiner.
                box["error"] = exc

        worker = threading.Thread(target=_target, daemon=True)
        worker.start()
        worker.join(timeout=FALLBACK_HARD_TIMEOUT)
        if worker.is_alive():
            raise TimeoutError(
                f"Fallback call exceeded {FALLBACK_HARD_TIMEOUT:g}s hard timeout "
                f"(model='{self.fallback_model}')."
            )
        if "error" in box:
            raise box["error"]
        return box.get("response")

    @staticmethod
    def _looks_complete(
        content: str, expected_sections: Optional[list] = None
    ) -> bool:
        """Heuristic completeness check for a model response."""
        text = (content or "").strip()
        if len(text) <= 100:
            return False
        if re.search(r"(\.\.\.|…|:|,|-|—)\s*$", text[-120:]):
            return False
        lowered = text.lower()
        for marker in ("to be continued", "[truncated", "(continued"):
            if marker in lowered:
                return False
        for section in expected_sections or []:
            if str(section).strip().lower() not in lowered:
                return False
        return True

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _extract_content(response: Any) -> str:
        """Pull the text content out of a chat-completion response.

        Raises:
            ValueError: If no choices / message content is present.
        """
        try:
            choices = getattr(response, "choices", None) or []
            if not choices:
                raise ValueError("response contained no choices.")
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(
                f"Could not parse model response: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                "Model returned empty content (no text in choices[0].message.content)."
            )
        return content.strip()

    @staticmethod
    def _report_usage(response: Any, elapsed_seconds: float) -> None:
        """Display token usage (when provided) and response time via rich."""
        usage: Any = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        if (
            isinstance(prompt_tokens, int)
            and isinstance(completion_tokens, int)
            and isinstance(total_tokens, int)
        ):
            _console.print(
                "[green]Response received[/green] "
                f"in {elapsed_seconds:.2f}s — tokens: "
                f"prompt={prompt_tokens}, completion={completion_tokens}, "
                f"total={total_tokens}"
            )
        else:
            _console.print(
                f"[green]Response received[/green] in {elapsed_seconds:.2f}s "
                "(token usage not reported by provider)"
            )
