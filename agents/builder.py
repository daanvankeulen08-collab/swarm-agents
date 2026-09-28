"""Builder Agent: turns Scout research into complete digital products.

The builder is the second agent in the swarm pipeline (phase ``"building"``).
It takes a validated opportunity dict — either passed directly or read from
``state_manager.scout_findings`` (the Scout agent's output) — asks Space Bunny
Alpha (via the :class:`~core.orchestrator.Orchestrator`) for a complete,
working product, validates the result, and persists it both in state
(``product_code``) and on disk under ``products/``.

For production use, prefer :meth:`BuilderAgent.build_with_testing`: it builds
the product, validates it with :class:`~core.test_runner.TestRunner`,
rebuilds with the test errors as fix feedback (up to ``max_attempts`` times),
and packages successes with
:class:`~core.product_packager.ProductPackager` into ``packages/*.zip``.

Example usage (building from Scout findings):
    ```python
    from agents.builder import BuilderAgent
    from agents.scout import ScoutAgent
    from core.orchestrator import Orchestrator
    from core.state_manager import StateManager

    state_manager = StateManager(project_id="demo")
    orchestrator = Orchestrator()  # reads OPENROUTER_* from env / .env

    scout = ScoutAgent(orchestrator, state_manager)
    scout.research_opportunities("Notion templates for students")

    builder = BuilderAgent(orchestrator, state_manager)
    result = builder.build_with_testing(max_attempts=3)  # uses 1st finding
    print(result["success"], result["package_path"])

    # Or build a specific opportunity explicitly:
    # product_code = builder.build_product(opportunity={...})
    ```

Sampling temperature (0.7, balancing creativity and reliability) is enforced
centrally by :meth:`Orchestrator.ask_agent`. All terminal output uses
:mod:`rich`. This module is Windows-compatible: :class:`pathlib.Path` for
every file operation and UTF-8 for all reads/writes.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

from core.orchestrator import Orchestrator
from core.product_packager import ProductPackager
from core.state_manager import StateManager
from core.test_runner import TestRunner

__all__ = ["BuilderAgent", "BUILDER_SYSTEM_PROMPT", "PRODUCTS_DIR"]

#: Role instructions sent as the system prompt for every build call.
BUILDER_SYSTEM_PROMPT: str = (
    "You are an expert developer and product creator. You build complete, "
    "production-ready digital products (Python scripts, templates, tools) "
    "that deliver exactly what was promised."
)

#: Directory (relative to the working directory) holding built product files.
PRODUCTS_DIR: str = "products"

_console = Console()


class BuilderAgent:
    """Build complete digital products from Scout opportunities.

    Args:
        orchestrator: The swarm's :class:`Orchestrator` used for all model
            calls (usage is tracked automatically via the state manager).
        state_manager: The project's :class:`StateManager` where the built
            product (``product_code`` / ``product_name``) is persisted and
            the phase is set to ``"building"``.

    Example:
        ```python
        builder = BuilderAgent(orchestrator, state_manager)
        code = builder.build_product()
        ```
    """

    def __init__(
        self, orchestrator: Orchestrator, state_manager: StateManager
    ) -> None:
        # Duck-typed validation (rather than isinstance) so the agent accepts
        # the real Orchestrator/StateManager as well as test doubles/mocks
        # that expose the same interface.
        for name in ("ask_agent", "build_context_prompt", "get_model_info"):
            if not hasattr(orchestrator, name) or not callable(
                getattr(orchestrator, name, None)
            ):
                raise TypeError(
                    "orchestrator must expose the Orchestrator interface "
                    f"(missing callable {name!r}); "
                    f"got {type(orchestrator).__name__}."
                )
        for name in ("update_phase", "get_state", "save", "log_api_call"):
            if not hasattr(state_manager, name) or not callable(
                getattr(state_manager, name, None)
            ):
                raise TypeError(
                    "state_manager must expose the StateManager interface "
                    f"(missing callable {name!r}); "
                    f"got {type(state_manager).__name__}."
                )
        self.orchestrator: Orchestrator = orchestrator
        self.state_manager: StateManager = state_manager
        self.products_dir: Path = Path(PRODUCTS_DIR)
        # Chronological per-attempt records from build_with_testing().
        # Each entry: attempt, product_name, product_type, build_ok,
        # tests_passed, package_path, error, timestamp.
        self._build_history: List[Dict[str, Any]] = []

    def build_product(
        self,
        opportunity: Dict[str, Any] = None,
        feedback: Optional[str] = None,
    ) -> str:
        """Build a complete product for ``opportunity`` via Space Bunny Alpha.

        Args:
            opportunity: Validated opportunity dict as produced by
                :meth:`ScoutAgent.research_opportunities` (keys such as
                ``product_name``, ``target_audience``, ``pain_point``,
                ``estimated_price``, ``unique_value_proposition``,
                ``estimated_build_time``). If ``None``, the first opportunity
                in ``state_manager.scout_findings["opportunities"]`` is used.
            feedback: Optional fix feedback from a previous failed attempt
                (e.g. test errors). When given, it is appended to the user
                prompt so the model returns the complete corrected product.

        Returns:
            The generated product content (markdown fences stripped) as a
            string. It is also persisted via :meth:`save_product`.

        Raises:
            ValueError: If no opportunity is available, the opportunity is
                malformed, or the generated product fails validation (with
                the reason included).
            RuntimeError: If the underlying API call fails.

        Side effects:
            Validates the output, then saves it to ``product_code`` + a file
            under ``products/``, sets ``product_name`` and the phase to
            ``"building"``, and persists state.
        """
        if opportunity is None:
            opportunity = self._load_first_opportunity()
        else:
            opportunity = self._checked_opportunity(opportunity)

        product_name = str(opportunity["product_name"]).strip()
        _console.print(
            f"[cyan]Builder creating product[/cyan] "
            f"[bold]{product_name}[/bold]…"
        )

        user_prompt = self._build_user_prompt(opportunity, feedback=feedback)
        try:
            raw_response: str = self.orchestrator.ask_agent(
                system_prompt=BUILDER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                state_manager=self.state_manager,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"Builder API call failed for product {product_name!r}: {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Builder API call failed for product {product_name!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        _console.print("[cyan]Validating generated product…[/cyan]")
        product_content = self._extract_product_content(raw_response)
        reason = self._validation_error(product_content)
        if reason is not None:
            _console.print(f"[red]Product validation failed: {reason}[/red]")
            raise ValueError(
                f"Generated product for {product_name!r} failed validation: "
                f"{reason}"
            )

        _console.print("[green]Product validated. Saving…[/green]")
        self.save_product(product_content, product_name)
        _console.print(
            f"[green]Build complete:[/green] [bold]{product_name}[/bold] "
            f"({len(product_content)} characters)"
        )
        return product_content

    def build_with_testing(
        self, opportunity: Dict[str, Any] = None, max_attempts: int = 3
    ) -> Dict[str, Any]:
        """Build, test, fix, and package a product (up to ``max_attempts``).

        Each attempt builds the product with :meth:`build_product`, runs it
        through :class:`TestRunner`, and — on success — packages it with
        :class:`ProductPackager`. Failed attempts produce a fix prompt from
        the test errors that is fed back into the next build. Every attempt
        is recorded (see :meth:`get_build_history`) and mirrored to
        ``state.build_history`` / ``state.last_test_results``.

        Args:
            opportunity: Opportunity dict, or ``None`` to use the first
                Scout finding in state (same rule as :meth:`build_product`).
            max_attempts: Maximum build attempts (must be >= 1).

        Returns:
            Dict with exactly these keys:

            * ``success`` (bool): True if a tested package was produced.
            * ``attempts`` (int): Number of build attempts performed.
            * ``final_product_code`` (str | None): Last built product text.
            * ``test_results`` (dict): Last :class:`TestRunner` report
              (``{"success": ..., "errors": ..., "test_results": ...}``).
            * ``package_path`` (str | None): ZIP path when packaging
              succeeded, else None.
            * ``errors`` (list): Human-readable failure descriptions.

        Raises:
            ValueError: If inputs are invalid (bad opportunity, or
                ``max_attempts < 1``).
        """
        if opportunity is None:
            opportunity = self._load_first_opportunity()
        else:
            opportunity = self._checked_opportunity(opportunity)
        if (
            not isinstance(max_attempts, int)
            or isinstance(max_attempts, bool)
            or max_attempts < 1
        ):
            raise ValueError(
                f"max_attempts must be an integer >= 1, got {max_attempts!r}."
            )
        product_name = str(opportunity["product_name"]).strip()

        tester = TestRunner(self.state_manager)
        packager = ProductPackager(self.state_manager)

        errors: List[str] = []
        attempts = 0
        final_code: Optional[str] = None
        package_path: Optional[str] = None
        last_report: Dict[str, Any] = {
            "success": False,
            "output": "",
            "errors": ["No attempts performed yet."],
            "test_results": {},
        }
        feedback: Optional[str] = None

        for attempt in range(1, max_attempts + 1):
            attempts = attempt
            _console.print(
                f"[bold cyan]Build attempt {attempt}/{max_attempts}[/bold cyan] "
                f"for [bold]{product_name}[/bold]"
                + (" (retry with fix feedback)" if feedback else "")
            )
            entry: Dict[str, Any] = {
                "attempt": attempt,
                "product_name": product_name,
                "product_type": None,
                "build_ok": False,
                "tests_passed": False,
                "package_path": None,
                "error": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            try:
                code = self.build_product(
                    opportunity=opportunity, feedback=feedback
                )
            except (RuntimeError, ValueError) as exc:
                message = f"Attempt {attempt}: build failed: {exc}"
                _console.print(f"[red]{message}[/red]")
                errors.append(message)
                entry["error"] = str(exc)
                self._build_history.append(entry)
                feedback = (
                    "The previous build attempt failed before testing: "
                    f"{exc}\nReturn the COMPLETE corrected product."
                )
                continue

            entry["build_ok"] = True
            final_code = code
            product_type = self._detect_product_type(code)
            entry["product_type"] = product_type

            _console.print(
                f"[cyan]Attempt {attempt}: running tests "
                f"({product_type})…[/cyan]"
            )
            report = tester.run_tests(code, product_type=product_type)
            last_report = {
                "success": report["success"],
                "output": report["output"],
                "errors": list(report["errors"]),
                "test_results": report["test_results"],
            }
            if report["success"]:
                entry["tests_passed"] = True
                _console.print(
                    f"[green]Attempt {attempt}: tests passed. "
                    "Packaging…[/green]"
                )
                try:
                    zip_path = packager.create_package(
                        product_name, code, product_type=product_type
                    )
                except (ValueError, IOError, RuntimeError) as exc:
                    message = f"Attempt {attempt}: packaging failed: {exc}"
                    _console.print(f"[red]{message}[/red]")
                    errors.append(message)
                    entry["error"] = str(exc)
                    self._build_history.append(entry)
                    break
                package_path = str(zip_path)
                entry["package_path"] = package_path
                self._build_history.append(entry)
                _console.print(
                    f"[bold green]Build succeeded on attempt {attempt}: "
                    f"'{zip_path}'[/bold green]"
                )
                break

            message = (
                f"Attempt {attempt}: tests failed "
                f"({len(report['errors'])} error(s))."
            )
            _console.print(f"[yellow]{message}[/yellow]")
            errors.append(message + " " + " ".join(report["errors"][:3]))
            entry["error"] = "; ".join(report["errors"][:5])
            self._build_history.append(entry)
            feedback = self._build_fix_prompt(code, report)

        success = package_path is not None
        self._persist_history(last_report)
        if not success:
            _console.print(
                f"[red]Build failed after {attempts} attempt(s). "
                "See errors for details.[/red]"
            )
        return {
            "success": success,
            "attempts": attempts,
            "final_product_code": final_code,
            "test_results": last_report,
            "package_path": package_path,
            "errors": errors,
        }

    def get_build_history(self) -> List[Dict[str, Any]]:
        """Return all recorded build attempts and their outcomes.

        Returns:
            A list of per-attempt dicts (attempt number, product name/type,
            build/test/package outcomes, error text, timestamp), oldest
            first. Returns an empty list if :meth:`build_with_testing` has
            not run yet in this process. A copy is returned, so callers
            cannot mutate internal records.
        """
        return [dict(entry) for entry in self._build_history]

    def save_product(self, product_content: str, product_name: str) -> Path:
        """Persist a validated product to state and to ``products/``.

        Args:
            product_content: The complete product text (code or template).
            product_name: Human-readable name; sanitised for the filename
                (special chars removed, spaces become underscores).

        Returns:
            The :class:`pathlib.Path` of the written product file
            (``products/{sanitised}.py`` or ``.md`` for templates).

        Raises:
            ValueError: If either argument is empty.
            IOError: If the product file cannot be written, with details.
            RuntimeError: If persisting state fails.
        """
        if not isinstance(product_content, str) or not product_content.strip():
            raise ValueError("product_content must be a non-empty string.")
        cleaned_name = product_name.strip() if isinstance(product_name, str) else ""
        if not cleaned_name:
            raise ValueError("product_name must be a non-empty string.")

        filename = self._sanitise_filename(cleaned_name)
        extension = self._guess_extension(product_content)
        try:
            self.products_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise IOError(
                f"Could not create products directory '{self.products_dir}': {exc}"
            ) from exc
        target = self.products_dir / f"{filename}{extension}"
        try:
            target.write_text(product_content, encoding="utf-8")
        except OSError as exc:
            raise IOError(
                f"Could not write product file '{target}': {exc}"
            ) from exc

        try:
            self.state_manager.update_phase("building")
            state = self.state_manager.get_state()
            state.product_code = product_content
            state.product_name = cleaned_name
            self.state_manager.save(state)
        except Exception as exc:
            raise RuntimeError(
                f"Product file '{target}' written but persisting state failed "
                f"for project '{self.state_manager.project_id}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        _console.print(
            f"[green]Saved product '{cleaned_name}' to '{target}' "
            "and updated state (phase=building).[/green]"
        )
        return target

    def validate_product(self, product_content: str) -> bool:
        """Check whether generated content is a real, usable product.

        Rules:

        * Empty/whitespace-only content is invalid.
        * Content that compiles as Python (``compile()``) is valid, provided
          it has minimal substance (length and real code lines, not just
          comments).
        * Otherwise the content is treated as a template: it is valid if it
          shows template structure (headings, sections, placeholders, or
          example data) with sufficient length.

        Args:
            product_content: Candidate product text.

        Returns:
            True if valid, False otherwise. Never raises for malformed input.
        """
        return self._validation_error(product_content) is None

    # -- internals ----------------------------------------------------------

    def _load_first_opportunity(self) -> Dict[str, Any]:
        """Return the first Scout opportunity from state, or raise ValueError."""
        try:
            state = self.state_manager.get_state()
        except Exception as exc:
            raise ValueError(
                "No opportunity given and loading state failed: "
                f"{type(exc).__name__}: {exc}. "
                "Run the Scout agent first or pass opportunity={...}."
            ) from exc
        findings = state.scout_findings
        if not isinstance(findings, dict):
            raise ValueError(
                "No opportunity given and state holds no scout_findings. "
                "Run the Scout agent first or pass opportunity={...}."
            )
        opportunities = findings.get("opportunities")
        if not isinstance(opportunities, list) or not opportunities:
            raise ValueError(
                "No opportunity given and scout_findings contains no "
                "opportunities list. Run the Scout agent first or pass "
                "opportunity={...}."
            )
        first = opportunities[0]
        _console.print(
            f"[cyan]Using Scout opportunity 1/{len(opportunities)}:[/cyan] "
            f"[bold]{first.get('product_name', '(unnamed)') if isinstance(first, dict) else first!r}[/bold]"
        )
        return self._checked_opportunity(first)

    @staticmethod
    def _checked_opportunity(opportunity: Any) -> Dict[str, Any]:
        """Ensure the opportunity is a dict with at least a product_name."""
        if not isinstance(opportunity, dict):
            raise ValueError(
                "opportunity must be a dict like Scout's findings entries, "
                f"got {type(opportunity).__name__}."
            )
        name = opportunity.get("product_name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "opportunity must contain a non-empty 'product_name' string."
            )
        return opportunity

    @staticmethod
    def _build_user_prompt(
        opportunity: Dict[str, Any], feedback: Optional[str] = None
    ) -> str:
        """Compose the user prompt carrying the full opportunity details."""
        details = "\n".join(
            f"- {key}: {opportunity.get(key, '(not specified)')}"
            for key in (
                "product_name",
                "target_audience",
                "pain_point",
                "estimated_price",
                "unique_value_proposition",
                "estimated_build_time",
            )
        )
        prompt = (
            "Build the complete, production-ready digital product for this "
            "validated market opportunity:\n\n"
            f"{details}\n\n"
            "Requirements:\n"
            "- Create the ACTUAL product — complete, working, requiring NO "
            "modifications before use.\n"
            "- Include the main functionality, clear inline comments, concise "
            "usage instructions (as comments at the top), and a dependencies "
            "list (as comments; standard library preferred).\n"
            "- For a Python script: return ONE complete, runnable file with an "
            '`if __name__ == "__main__":` block demonstrating real usage.\n'
            "- For a template: return the COMPLETE structure with realistic "
            "example data filled in, not placeholders alone.\n"
            "- Output ONLY the product content (fenced in a single code block "
            "is fine). No marketing pitch, no explanations outside the product."
        )
        if feedback and feedback.strip():
            prompt += (
                "\n\nIMPORTANT — FIX REQUESTED:\n"
                f"{feedback.strip()}\n"
                "Return the ENTIRE corrected product, not a patch or diff."
            )
        return prompt

    @staticmethod
    def _detect_product_type(product_content: str) -> str:
        """Return ``"python"`` if the content compiles, else ``"template"``."""
        try:
            compile(product_content, "<product>", "exec")
            return "python"
        except (SyntaxError, ValueError):
            return "template"

    @staticmethod
    def _build_fix_prompt(failed_code: str, report: Dict[str, Any]) -> str:
        """Compose fix feedback from a failed TestRunner report."""
        failures = "\n".join(
            f"- {error}" for error in report.get("errors", [])[:10]
        ) or "- (no error details captured)"
        output_tail = (report.get("output") or "").strip()[-1500:]
        code_head = (failed_code or "")[:6000]
        return (
            "The previous attempt produced output that FAILED automated "
            "testing. Fix EVERY issue below and return the COMPLETE corrected "
            "product (never a patch or diff).\n\n"
            f"Test failures:\n{failures}\n\n"
            f"Captured output (tail):\n{output_tail or '(empty)'}\n\n"
            f"Previously generated code:\n{code_head}"
        )

    def _persist_history(self, last_report: Dict[str, Any]) -> None:
        """Mirror build history + latest test report onto the state."""
        try:
            state = self.state_manager.get_state()
            state.build_history = [dict(entry) for entry in self._build_history]
            state.last_test_results = {
                "success": last_report.get("success", False),
                "errors": list(last_report.get("errors", [])),
                "test_results": last_report.get("test_results", {}),
            }
            self.state_manager.save(state)
        except Exception as exc:  # Never mask the build outcome.
            _console.print(
                f"[yellow]Could not persist build history to state: {exc}[/yellow]"
            )

    @staticmethod
    def _extract_product_content(raw_response: str) -> str:
        """Strip a single surrounding markdown code fence, if present."""
        text = raw_response.strip() if isinstance(raw_response, str) else ""
        if text.startswith("```"):
            lines = text.splitlines()[1:]  # drop opening fence (``` or ```python)
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]  # drop closing fence
            text = "\n".join(lines).strip()
        return text

    @staticmethod
    def _sanitise_filename(product_name: str) -> str:
        """Sanitise a product name for use as a filename stem.

        Spaces become underscores; anything that is not a letter, digit,
        underscore, or hyphen is removed; repeats are collapsed. Falls back
        to ``"product"`` if nothing remains.
        """
        name = product_name.strip().replace(" ", "_")
        name = re.sub(r"[^A-Za-z0-9_-]", "", name)
        name = re.sub(r"_+", "_", name).strip("_-")
        return name or "product"

    @staticmethod
    def _guess_extension(product_content: str) -> str:
        """Pick ``.py`` for Python content, ``.md`` for templates."""
        try:
            compile(product_content, "<product>", "exec")
            return ".py"
        except (SyntaxError, ValueError):
            pass
        if re.search(r"^#{1,6}\s+\S", product_content, re.MULTILINE):
            return ".md"
        return ".py"

    @classmethod
    def _validation_error(cls, product_content: Any) -> Optional[str]:
        """Return None when valid, else a human-readable reason."""
        if not isinstance(product_content, str) or not product_content.strip():
            return "content is empty."
        text = product_content.strip()
        try:
            compile(text, "<product>", "exec")
        except (SyntaxError, ValueError) as exc:
            return cls._template_error(text, f"invalid Python syntax ({exc})")
        code_lines = [
            line
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if len(text) < 50:
            return "Python content is too short to be a complete product."
        if len(code_lines) < 3:
            return "Python content has no substantive code (only comments/blank lines)."
        return None

    @staticmethod
    def _template_error(text: str, python_reason: str) -> Optional[str]:
        """Validate non-Python content as a template; None when valid."""
        if len(text) < 100:
            return (
                f"content is not valid Python ({python_reason}) and is too "
                "short to be a usable template."
            )
        has_heading = re.search(r"^#{1,6}\s+\S", text, re.MULTILINE) is not None
        has_section = (
            re.search(r"^(.+)\n([=-])\1*\s*$", text, re.MULTILINE) is not None
            or "---" in text
        )
        has_placeholder_or_example = (
            re.search(r"[\{\}\[\]]", text) is not None
            or "example" in text.lower()
        )
        if has_heading or (has_section and has_placeholder_or_example):
            return None
        return (
            f"content is not valid Python ({python_reason}) and shows no "
            "template structure (headings, sections, placeholders, examples)."
        )
