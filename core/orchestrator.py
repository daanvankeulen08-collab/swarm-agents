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

All status output uses :mod:`rich` (coloured messages, spinner, token usage
and response-time reporting). This module is Windows-compatible: it uses
:class:`pathlib.Path` for paths and UTF-8 for every file operation.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, OpenAI
from rich.console import Console

from .state_manager import ProjectState, StateManager

__all__ = ["Orchestrator", "DEFAULT_BASE_URL", "DEFAULT_MODEL"]

#: Default OpenRouter base URL when ``OPENROUTER_BASE_URL`` is unset.
DEFAULT_BASE_URL: str = "https://openrouter.ai/api/v1"

#: Default model when ``OPENROUTER_MODEL`` is unset.
DEFAULT_MODEL: str = "stealth/space-bunny-alpha"

#: Sampling temperature: balances creativity against consistency.
DEFAULT_TEMPERATURE: float = 0.7

_console = Console()


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

        _console.print(
            f"[cyan]Calling model[/cyan] [bold]{self.model}[/bold] "
            f"for project '{live_state.project_id}' "
            f"(phase={live_state.phase}, api_calls={live_state.api_calls_count})…"
        )

        last_error: Optional[BaseException] = None
        for attempt in (1, 2):  # Initial try + exactly one retry.
            try:
                start = time.perf_counter()
                with _console.status(
                    f"[bold cyan]Waiting for {self.model} "
                    f"(attempt {attempt}/2)…[/bold cyan]",
                    spinner="dots",
                ):
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,  # type: ignore[arg-type]
                        temperature=DEFAULT_TEMPERATURE,
                        timeout=self.timeout,
                    )
                elapsed = time.perf_counter() - start
                content = self._extract_content(response)
                self._report_usage(response, elapsed)
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
