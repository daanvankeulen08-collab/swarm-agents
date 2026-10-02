"""TestRunner: executes and validates built products before packaging.

The TestRunner is the quality gate of the swarm pipeline. It takes raw
product code (as produced by :class:`~agents.builder.BuilderAgent`), runs it
in an isolated temporary directory via :mod:`subprocess` with a hard timeout,
and returns a detailed report dict. Only products that pass reach the
:class:`~core.product_packager.ProductPackager`.

Example usage:
    ```python
    from core.state_manager import StateManager
    from core.test_runner import TestRunner

    state_manager = StateManager(project_id="demo")
    runner = TestRunner(state_manager)

    report = runner.run_tests(product_code=open("candidate.py").read())
    print(report["success"], report["errors"])

    template_report = runner.run_tests(template_text, product_type="template")
    ```

Security notes: tests run in a fresh temp directory that is always removed
afterwards, each subprocess call has a 30-second timeout, no network access
is required or used, and output is captured (never streamed to the terminal
raw). This module is Windows-compatible: :class:`pathlib.Path` everywhere,
UTF-8 for all file I/O, and :data:`sys.executable` (never ``"python"``) for
subprocess invocation so the correct interpreter is used.
"""

from __future__ import annotations

import inspect as _inspect  # noqa: F401  (re-exported for generated scripts)
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

from .placeholders import detect_abandoned_content, detect_ungrounded_claims
from .state_manager import StateManager

__all__ = [
    "TestRunner",
    "TEST_TIMEOUT_SECONDS",
    "PLACEHOLDER_MARKERS",
    "PYTHON_CHECK_SCRIPT",
]

#: Hard timeout (seconds) for every product subprocess execution.
TEST_TIMEOUT_SECONDS: int = 30

#: Case-insensitive markers that indicate unfilled template content.
#:
#: ``tbd`` was removed deliberately: in a planner template an empty field
#: *is* the product, and a blanket substring match rejected a valid
#: 28,702-char product during Maiden Voyage 3. Abandoned content is now
#: detected structurally by :func:`core.placeholders.detect_abandoned_content`.
PLACEHOLDER_MARKERS: tuple = (
    "todo",
    "fixme",
    "xxx",
    "[insert",
    "<insert",
)

#: Probe script written next to the product; imports it, exercises public
#: zero-argument functions, and prints a JSON-ish summary to stdout.
PYTHON_CHECK_SCRIPT: str = textwrap.dedent(
    """\
    import inspect
    import json
    import test_module

    summary = {"import_ok": True, "functions": [], "errors": []}
    for name, member in inspect.getmembers(test_module):
        if name.startswith("_") or not callable(member):
            continue
        if inspect.isclass(member):
            summary["functions"].append({"name": name, "kind": "class", "called": False})
            continue
        try:
            sig = inspect.signature(member)
        except (TypeError, ValueError):
            continue
        required = [
            p for p in sig.parameters.values()
            if p.default is p.empty
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        ]
        has_var = any(
            p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            for p in sig.parameters.values()
        )
        entry = {"name": name, "kind": "function", "called": False, "result": None}
        if not required and not has_var:
            try:
                result = member()
                entry["called"] = True
                entry["result"] = repr(result)[:500]
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
                summary["errors"].append(f"{name}: {entry['error']}")
        summary["functions"].append(entry)
    print("TEST_MODULE_SUMMARY:" + json.dumps(summary))
    """
)

_console = Console()


class TestRunner:
    """Execute and validate product code in an isolated temp directory.

    Args:
        state_manager: The project's :class:`StateManager`. The latest
            report is stored on ``state.last_test_results`` after each run
            so the whole swarm (and ``main.py``) can inspect it.

    Example:
        ```python
        runner = TestRunner(state_manager)
        report = runner.run_tests(product_code)
        assert report["success"]
        ```
    """

    def __init__(self, state_manager: Optional[StateManager] = None) -> None:
        # Duck-typed validation so test doubles/mocks work too. Optional so
        # content-level checks (e.g. _detect_abandoned_content) can be used
        # standalone without a project state file.
        if state_manager is not None:
            for name in ("update_phase", "get_state", "save", "log_api_call"):
                if not hasattr(state_manager, name) or not callable(
                    getattr(state_manager, name, None)
                ):
                    raise TypeError(
                        "state_manager must expose the StateManager interface "
                        f"(missing callable {name!r}); "
                        f"got {type(state_manager).__name__}."
                    )
        self.state_manager: Optional[StateManager] = state_manager

    def _detect_abandoned_content(self, content: str) -> List[str]:
        """Detect content the generator abandoned rather than left fillable.

        A planner template legitimately ships empty fields
        (``| Client Name | TBD |``). What must not ship is a section that
        simply stops at ``TBD``. This delegates to
        :func:`core.placeholders.detect_abandoned_content` so the test gate
        and the reviewer cannot disagree.

        Args:
            content: Generated template or section text.

        Returns:
            List of defect descriptions; empty means clean.
        """
        return detect_abandoned_content(content)

    def _detect_ungrounded_claims(self, content: str) -> List[str]:
        """Detect content that contradicts itself or the supplied input.

        Separate from abandonment: a section can be fully formatted and still
        be unusable because it announces it received nothing, denies its own
        existence, or presents totals computed from inputs it declares
        absent. Delegates to
        :func:`core.placeholders.detect_ungrounded_claims` so the test gate
        and the reviewer cannot disagree.

        Args:
            content: Generated template or section text.

        Returns:
            List of defect descriptions; empty means clean.
        """
        return detect_ungrounded_claims(content)

    def run_tests(
        self, product_code: str, product_type: str = "python"
    ) -> Dict[str, Any]:
        """Run the product and return a detailed report dict.

        Args:
            product_code: Raw product text from the Builder.
            product_type: ``"python"`` (executed via subprocess) or
                ``"template"`` (structurally validated, never executed).

        Returns:
            Dict with exactly these keys:

            * ``success`` (bool): True only if every check passed.
            * ``output`` (str): Captured stdout (or structural summary).
            * ``errors`` (list): Human-readable failure descriptions.
            * ``test_results`` (dict): Per-check details (what ran, return
              codes, functions exercised, timings, …).

        The report is also persisted to ``state.last_test_results`` (best
        effort — a state-persistence failure never masks the test outcome).
        """
        if not isinstance(product_code, str) or not product_code.strip():
            return self._finalise(_report(
                success=False,
                output="",
                errors=["product_code is empty; nothing to test."],
                test_results={"check": "non-empty input", "passed": False},
            ))
        normalised = product_type.strip().lower() if isinstance(product_type, str) else ""
        if normalised not in ("python", "template"):
            return self._finalise(_report(
                success=False,
                output="",
                errors=[
                    f"Unknown product_type {product_type!r}; "
                    "expected 'python' or 'template'."
                ],
                test_results={"check": "product_type", "passed": False},
            ))
        _console.print(
            f"[cyan]TestRunner testing {normalised} product "
            f"({len(product_code)} characters)…[/cyan]"
        )
        if normalised == "python":
            report = self._run_python_tests(product_code)
        else:
            report = self._run_template_tests(product_code)
        _console.print(
            "[green]Tests PASSED[/green]"
            if report["success"]
            else f"[red]Tests FAILED ({len(report['errors'])} error(s))[/red]"
        )
        return self._finalise(report)

    # -- python -------------------------------------------------------------

    def _run_python_tests(self, product_code: str) -> Dict[str, Any]:
        """Execute a Python product in a temp dir; return the report."""
        errors: List[str] = []
        test_results: Dict[str, Any] = {}
        combined_output = ""

        try:
            compile(product_code, "<product>", "exec")
        except (SyntaxError, ValueError) as exc:
            return _report(
                success=False,
                output="",
                errors=[f"Syntax check failed: {exc}"],
                test_results={"syntax_check": {"passed": False, "error": str(exc)}},
            )
        test_results["syntax_check"] = {"passed": True}

        with tempfile.TemporaryDirectory(prefix="swarm_test_") as tmp:
            tmpdir = Path(tmp)
            module_path = tmpdir / "test_module.py"
            probe_path = tmpdir / "run_checks.py"
            module_path.write_text(product_code, encoding="utf-8")
            probe_path.write_text(PYTHON_CHECK_SCRIPT, encoding="utf-8")

            # Step 1: run the module as a script (exercises __main__).
            _console.print("[cyan]  • running module as script…[/cyan]")
            main_result = self._execute(
                [sys.executable, "test_module.py"], cwd=tmpdir
            )
            test_results["main_run"] = main_result
            combined_output += main_result["stdout"]
            if main_result["timed_out"]:
                errors.append("Execution as script timed out after "
                              f"{TEST_TIMEOUT_SECONDS}s.")
            elif main_result["returncode"] != 0:
                errors.append(
                    "Execution as script failed "
                    f"(exit {main_result['returncode']}): "
                    f"{_last_line(main_result['stderr'])}"
                )

            # Step 2: import probe — exercises public callables.
            _console.print("[cyan]  • probing public functions…[/cyan]")
            probe_result = self._execute(
                [sys.executable, "run_checks.py"], cwd=tmpdir
            )
            test_results["probe_run"] = {
                k: v for k, v in probe_result.items() if k != "stdout"
            }
            combined_output += probe_result["stdout"]
            summary = _parse_probe_summary(probe_result["stdout"])
            test_results["function_probe"] = summary
            if probe_result["timed_out"]:
                errors.append("Function probe timed out after "
                              f"{TEST_TIMEOUT_SECONDS}s.")
            elif probe_result["returncode"] != 0:
                errors.append(
                    "Importing the module failed "
                    f"(exit {probe_result['returncode']}): "
                    f"{_last_line(probe_result['stderr'])}"
                )
            else:
                for entry in summary.get("functions", []):
                    if entry.get("error"):
                        errors.append(
                            f"Calling {entry['name']}() raised {entry['error']}."
                        )
                test_results["functions_called"] = sum(
                    1 for f in summary.get("functions", []) if f.get("called")
                )

            # Step 3: output sanity — no tracebacks, something observable.
            stderr_all = main_result["stderr"] + probe_result["stderr"]
            if "Traceback (most recent call last)" in stderr_all:
                errors.append("Error traceback detected in captured stderr.")
                test_results["no_traceback"] = {"passed": False}
            else:
                test_results["no_traceback"] = {"passed": True}
            observable = bool(combined_output.strip()) or bool(
                test_results.get("functions_called")
            )
            test_results["reasonable_output"] = {"passed": observable}
            if not observable:
                errors.append(
                    "No observable output: the script printed nothing and no "
                    "public function could be exercised."
                )

        success = not errors
        return _report(
            success=success,
            output=combined_output.strip(),
            errors=errors,
            test_results=test_results,
        )

    def _execute(self, argv: List[str], cwd: Path) -> Dict[str, Any]:
        """Run ``argv`` in ``cwd`` with timeout; capture everything.

        Never raises for product failures — they are encoded in the result.
        """
        try:
            completed = subprocess.run(
                argv,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=TEST_TIMEOUT_SECONDS,
                check=False,
            )
            return {
                "argv": list(argv),
                "returncode": completed.returncode,
                "stdout": completed.stdout or "",
                "stderr": completed.stderr or "",
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout
            stderr = exc.stderr
            return {
                "argv": list(argv),
                "returncode": None,
                "stdout": _to_str(stdout),
                "stderr": _to_str(stderr),
                "timed_out": True,
            }
        except OSError as exc:
            return {
                "argv": list(argv),
                "returncode": None,
                "stdout": "",
                "stderr": f"Could not launch subprocess: {exc}",
                "timed_out": False,
                "launch_error": str(exc),
            }

    # -- templates ----------------------------------------------------------

    def _run_template_tests(self, product_code: str) -> Dict[str, Any]:
        """Structurally validate a template product (never executed)."""
        errors: List[str] = []
        text = product_code.strip()
        has_heading = re.search(r"^#{1,6}\s+\S", text, re.MULTILINE) is not None
        has_sections = (
            len(re.findall(r"^#{1,6}\s+\S", text, re.MULTILINE)) >= 2
            or "---" in text
        )
        has_examples = (
            re.search(r"[\{\}\[\]]", text) is not None
            or "example" in text.lower()
        )
        leftover = sorted({
            marker for marker in PLACEHOLDER_MARKERS
            if marker in text.lower()
        })
        # Structural check replaces the old blanket 'tbd' scan: fillable
        # fields are the product, abandoned content is a defect.
        abandoned = self._detect_abandoned_content(text)
        ungrounded = self._detect_ungrounded_claims(text)
        long_enough = len(text) >= 100
        test_results = {
            "length": {"passed": long_enough, "characters": len(text)},
            "has_heading": {"passed": has_heading},
            "has_sections": {"passed": has_sections},
            "has_examples": {"passed": has_examples},
            "leftover_placeholders": leftover,
            "abandoned_content": abandoned,
            "ungrounded_claims": ungrounded,
        }
        _console.print("[cyan]  • checking template structure…[/cyan]")
        if not long_enough:
            errors.append("Template is too short to be a complete product.")
        if not has_heading:
            errors.append("Template has no headings/sections.")
        if not has_sections:
            errors.append("Template lacks section structure.")
        if not has_examples:
            errors.append("Template has no examples or placeholders.")
        if leftover:
            errors.append(
                "Unfilled placeholder text remains: " + ", ".join(leftover) + "."
            )
        if abandoned:
            errors.append(
                f"Abandoned content detected ({len(abandoned)} defect(s)): "
                + " | ".join(abandoned[:5])
            )
        if ungrounded:
            errors.append(
                f"Ungrounded content detected ({len(ungrounded)} defect(s)): "
                + " | ".join(ungrounded[:5])
            )
        summary = (
            f"Template check: {len(text)} chars, "
            f"headings={'yes' if has_heading else 'no'}, "
            f"examples={'yes' if has_examples else 'no'}, "
            f"leftover={leftover or 'none'}, "
            f"abandoned={len(abandoned)}, "
            f"ungrounded={len(ungrounded)}."
        )
        return _report(
            success=not errors,
            output=summary,
            errors=errors,
            test_results=test_results,
        )

    # -- state --------------------------------------------------------------

    def _finalise(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the report to state (best effort) and return it."""
        if self.state_manager is None:
            return report
        try:
            state = self.state_manager.get_state()
            state.last_test_results = {
                "success": report["success"],
                "errors": list(report["errors"]),
                "test_results": report["test_results"],
            }
            self.state_manager.save(state)
        except Exception as exc:  # Never mask the test outcome.
            _console.print(
                f"[yellow]Could not persist test results to state: {exc}[/yellow]"
            )
        return report


def _report(
    success: bool,
    output: str,
    errors: List[str],
    test_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the canonical report dict."""
    return {
        "success": bool(success),
        "output": output or "",
        "errors": list(errors),
        "test_results": dict(test_results),
    }


def _to_str(value: Any) -> str:
    """Best-effort decode of subprocess timeout partial output."""
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return ""


def _last_line(text: str) -> str:
    """Return the last non-empty line of ``text`` for compact errors."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[-1][:300] if lines else "(no stderr)"


def _parse_probe_summary(stdout: str) -> Dict[str, Any]:
    """Extract the TEST_MODULE_SUMMARY payload from probe stdout."""
    import json as _json

    fallback: Dict[str, Any] = {"import_ok": False, "functions": [], "errors": []}
    for line in (stdout or "").splitlines():
        if line.startswith("TEST_MODULE_SUMMARY:"):
            try:
                payload = _json.loads(line.split(":", 1)[1])
                if isinstance(payload, dict):
                    return payload
            except ValueError:
                continue
    return fallback
