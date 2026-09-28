"""ProductPackager: turns raw Builder code into a sellable ZIP package.



The packager is the final stage of the swarm pipeline. It takes validated

product code, assembles a buyer-ready folder (code, README, requirements,

examples, tests), ZIPs it into ``packages/{product_name}.zip``, and records

the package path + timestamp on the :class:`~core.state_manager.StateManager`.



Example usage:

    ```python

    from core.product_packager import ProductPackager

    from core.state_manager import StateManager



    state_manager = StateManager(project_id="demo")

    packager = ProductPackager(state_manager)



    zip_path = packager.create_package(

        product_name="Study Sprint Planner",

        product_code=open("products/Study_Sprint_Planner.py").read(),

    )

    print(zip_path)  # packages/Study_Sprint_Planner.zip

    ```



Python packages contain ``main.py``, ``README.md``, ``requirements.txt``,

``examples/example_usage.py`` and ``tests/test_basic.py``. Template packages

contain the template file, ``README.md`` and an ``examples/`` folder.

Dependencies are auto-detected by scanning import statements with :mod:`ast`

(standard-library modules excluded). All file operations use

:class:`pathlib.Path` with UTF-8 encoding, so the packager works on Windows

and POSIX alike. Temporary staging directories are always cleaned up.

"""



from __future__ import annotations



import ast

import sys

import tempfile

import zipfile

from datetime import datetime, timezone

from pathlib import Path

from typing import List



from rich.console import Console



from .state_manager import StateManager



__all__ = ["ProductPackager", "PACKAGES_DIR", "sanitise_product_name"]



#: Directory (relative to the working directory) holding finished ZIPs.

PACKAGES_DIR: str = "packages"



_console = Console()





def sanitise_product_name(product_name: str) -> str:

    """Sanitise a product name for use as a filename stem.



    Spaces become underscores; anything that is not a letter, digit,

    underscore, or hyphen is removed; repeats are collapsed. Falls back to

    ``"product"`` if nothing remains.



    Args:

        product_name: Human-readable product name.



    Returns:

        Filesystem-safe stem (no extension, no directory components).

    """

    import re as _re



    stem = product_name.strip().replace(" ", "_")

    stem = _re.sub(r"[^A-Za-z0-9_-]", "", stem)

    stem = _re.sub(r"_+", "_", stem).strip("_-")

    return stem or "product"





class ProductPackager:

    """Assemble, ZIP, and record sellable product packages.



    Args:

        state_manager: The project's :class:`StateManager`. After a

            successful build, ``state.package_path`` and

            ``state.packaged_at`` are updated so downstream agents and

            ``main.py`` can find the deliverable.



    Example:

        ```python

        packager = ProductPackager(state_manager)

        path = packager.create_package(name, code)

        ```

    """



    def __init__(self, state_manager: StateManager) -> None:

        # Duck-typed validation so test doubles/mocks work too.

        for name in ("update_phase", "get_state", "save", "log_api_call"):

            if not hasattr(state_manager, name) or not callable(

                getattr(state_manager, name, None)

            ):

                raise TypeError(

                    "state_manager must expose the StateManager interface "

                    f"(missing callable {name!r}); "

                    f"got {type(state_manager).__name__}."

                )

        self.state_manager: StateManager = state_manager

        self.packages_dir: Path = Path(PACKAGES_DIR)



    def create_package(

        self,

        product_name: str,

        product_code: str,

        product_type: str = "python",

    ) -> Path:

        """Create a sellable ZIP package for a validated product.



        Args:

            product_name: Human-readable name (sanitised for the ZIP name).

            product_code: Complete, tested product text from the Builder.

            product_type: ``"python"`` for code products, ``"template"``

                for template/markdown products.



        Returns:

            :class:`pathlib.Path` to the created

            ``packages/{sanitised_name}.zip`` file.



        Raises:

            ValueError: If the name/code is empty, the product type is

                unknown, or Python code has invalid syntax.

            IOError: If the ZIP file cannot be created, with details.



        Side effects:

            Creates ``packages/`` if needed, removes the staging directory

            afterwards, and records ``package_path`` + ``packaged_at`` on

            the state.

        """

        cleaned_name = product_name.strip() if isinstance(product_name, str) else ""

        if not cleaned_name:

            raise ValueError("product_name must be a non-empty string.")

        if not isinstance(product_code, str) or not product_code.strip():

            raise ValueError("product_code must be a non-empty string.")

        normalised = product_type.strip().lower() if isinstance(product_type, str) else ""

        if normalised not in ("python", "template"):

            raise ValueError(

                f"Unknown product_type {product_type!r}; "

                "expected 'python' or 'template'."

            )

        if normalised == "python":

            try:

                compile(product_code, "<product>", "exec")

            except (SyntaxError, ValueError) as exc:

                raise ValueError(

                    f"Cannot package product {cleaned_name!r}: invalid Python "

                    f"syntax ({exc}). Test the product before packaging."

                ) from exc



        stem = sanitise_product_name(cleaned_name)

        try:

            self.packages_dir.mkdir(parents=True, exist_ok=True)

        except OSError as exc:

            raise IOError(

                f"Could not create packages directory '{self.packages_dir}': {exc}"

            ) from exc

        zip_path = self.packages_dir / f"{stem}.zip"



        _console.print(

            f"[cyan]Packaging {normalised} product[/cyan] "

            f"[bold]{cleaned_name}[/bold] -> '{zip_path}'…"

        )

        with tempfile.TemporaryDirectory(prefix="swarm_pkg_") as staging:

            root = Path(staging) / stem

            root.mkdir(parents=True, exist_ok=True)

            if normalised == "python":

                self._stage_python_package(root, cleaned_name, product_code)

            else:

                self._stage_template_package(root, cleaned_name, product_code)

            _console.print("[cyan]  • compressing ZIP archive…[/cyan]")

            try:

                with zipfile.ZipFile(

                    zip_path, "w", compression=zipfile.ZIP_DEFLATED

                ) as archive:

                    for file_path in sorted(root.rglob("*")):

                        if file_path.is_file():

                            archive.write(

                                file_path,

                                arcname=file_path.relative_to(root.parent),

                            )

            except OSError as exc:

                raise IOError(

                    f"Could not write package ZIP '{zip_path}': {exc}"

                ) from exc

        # TemporaryDirectory removes the staging tree here, even on failure.



        try:

            state = self.state_manager.get_state()

            state.package_path = str(zip_path)

            state.packaged_at = datetime.now(timezone.utc).isoformat()

            self.state_manager.save(state)

        except Exception as exc:

            raise RuntimeError(

                f"Package '{zip_path}' created but recording it on state "

                f"failed for project '{self.state_manager.project_id}': "

                f"{type(exc).__name__}: {exc}"

            ) from exc



        _console.print(f"[green]Package ready: '{zip_path}'[/green]")

        return zip_path



    # -- staging --------------------------------------------------------



    def _stage_python_package(self, root: Path, name: str, code: str) -> None:

        """Write main.py, README, requirements, examples, and tests."""

        description = self._describe_product(name)

        dependencies = self._detect_dependencies(code)



        _console.print("[cyan]  • writing main.py…[/cyan]")

        (root / "main.py").write_text(code, encoding="utf-8")



        _console.print("[cyan]  • writing README.md…[/cyan]")

        (root / "README.md").write_text(

            self._render_readme(name, description, dependencies, python=True),

            encoding="utf-8",

        )



        _console.print("[cyan]  • writing requirements.txt…[/cyan]")

        (root / "requirements.txt").write_text(

            "".join(f"{dep}\n" for dep in dependencies), encoding="utf-8"

        )



        _console.print("[cyan]  • writing examples/example_usage.py…[/cyan]")

        examples_dir = root / "examples"

        examples_dir.mkdir(parents=True, exist_ok=True)

        (examples_dir / "example_usage.py").write_text(

            self._render_example(name), encoding="utf-8"

        )



        _console.print("[cyan]  • writing tests/test_basic.py…[/cyan]")

        tests_dir = root / "tests"

        tests_dir.mkdir(parents=True, exist_ok=True)

        (tests_dir / "test_basic.py").write_text(

            self._render_test_module(), encoding="utf-8"

        )



    def _stage_template_package(self, root: Path, name: str, code: str) -> None:

        """Write the template file, README, and an examples folder."""

        description = self._describe_product(name)

        _console.print("[cyan]  • writing template file…[/cyan]")

        (root / f"{sanitise_product_name(name)}.md").write_text(

            code, encoding="utf-8"

        )

        _console.print("[cyan]  • writing README.md…[/cyan]")

        (root / "README.md").write_text(

            self._render_readme(name, description, [], python=False),

            encoding="utf-8",

        )

        _console.print("[cyan]  • writing examples/…[/cyan]")

        examples_dir = root / "examples"

        examples_dir.mkdir(parents=True, exist_ok=True)

        (examples_dir / "example_filled.md").write_text(

            "# Filled example\n\n"

            "Duplicate the template file and fill in your own content. "

            "Below is the template as shipped:\n\n---\n\n" + code,

            encoding="utf-8",

        )



    # -- content rendering ------------------------------------------------



    def _describe_product(self, product_name: str) -> str:

        """Best-effort product description from live state, with fallback."""

        try:

            state = self.state_manager.get_state()

        except Exception:

            state = None

        if state is not None:

            for candidate in (state.sales_description, state.product_name):

                if isinstance(candidate, str) and candidate.strip():

                    if candidate.strip() != product_name:

                        return candidate.strip()

                    break

            findings = state.scout_findings

            if isinstance(findings, dict):

                for opp in findings.get("opportunities", []):

                    if (

                        isinstance(opp, dict)

                        and opp.get("product_name") == product_name

                        and isinstance(opp.get("pain_point"), str)

                    ):

                        return (

                            f"Solves: {opp['pain_point'].strip()} "

                            f"(for {opp.get('target_audience', 'you')})."

                        )

        return f"{product_name} — a ready-to-use digital product."



    @staticmethod

    def _detect_dependencies(product_code: str) -> List[str]:

        """Scan import statements via :mod:`ast`; drop stdlib modules.



        Returns:

            Sorted list of top-level third-party module names (may be empty).

        """

        try:

            tree = ast.parse(product_code)

        except (SyntaxError, ValueError):

            return []

        stdlib = set(sys.stdlib_module_names)

        found = set()

        for node in ast.walk(tree):

            if isinstance(node, ast.Import):

                for alias in node.names:

                    found.add(alias.name.split(".")[0])

            elif isinstance(node, ast.ImportFrom):

                if node.module and node.level == 0:

                    found.add(node.module.split(".")[0])

        return sorted(

            name for name in found if name and name not in stdlib

        )



    @staticmethod

    def _render_readme(

        name: str, description: str, dependencies: List[str], python: bool

    ) -> str:

        """Render buyer-facing README markdown."""

        if python:

            install = (

                "## Installation\n\n"

                "No build step needed — plain Python 3.10+.\n\n"

                "```bash\n"

                "python -m venv .venv\n"

                "source .venv/bin/activate  # Windows: .venv\\Scripts\\activate\n"

            )

            if dependencies:

                install += "pip install -r requirements.txt\n"

            install += "```\n\n## Usage\n\n```bash\npython main.py\n```\n"

            install += (

                "\nSee `examples/example_usage.py` for a worked example and "

                "`tests/test_basic.py` to verify your setup (`python -m "

                "pytest tests/`).\n"

            )

            requirements = (

                "## Requirements\n\n"

                + (

                    "".join(f"- `{dep}`\n" for dep in dependencies)

                    if dependencies

                    else "- Python 3.10+ (standard library only — nothing to install)\n"

                )

            )

        else:

            install = (

                "## Installation\n\n"

                "No installation needed. Duplicate the `.md` template file "

                "and fill in your own content.\n\n"

                "## Usage\n\n"

                "1. Open the template file in any Markdown editor (Notion, "

                "Obsidian, VS Code).\n"

                "2. See `examples/example_filled.md` for a filled-in example.\n"

            )

            requirements = "## Requirements\n\n- Any Markdown editor\n"

        return (

            f"# {name}\n\n{description}\n\n"

            "## Features\n\n"

            "- Complete and ready to use — no modifications required\n"

            "- Tested before packaging\n"

            "- Includes usage examples\n\n" + install + "\n" + requirements

        )



    @staticmethod

    def _render_example(product_name: str) -> str:

        """Render a minimal runnable usage example for the buyer."""

        return (

            f'"""Simple usage example for {product_name}."""\n'

            "import sys\n"

            "from pathlib import Path\n\n"

            "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n\n"

            "import main\n\n"

            "print(f\"Loaded product module: {main.__name__}\")\n"

            "public = [n for n in dir(main) if not n.startswith(\"_\")]\n"

            "print(f\"Public names: {public}\")\n"

        )



    @staticmethod

    def _render_test_module() -> str:

        """Render a smoke test the buyer can run with pytest or plain Python."""

        return (

            '"""Basic smoke test for the packaged product."""\n'

            "import inspect\n"

            "import sys\n"

            "from pathlib import Path\n\n"

            "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n\n"

            "import main\n\n\n"

            "def test_module_imports():\n"

            '    """The product module must import cleanly."""\n'

            "    assert main.__name__ == \"main\"\n\n\n"

            "def test_public_callables_do_not_crash():\n"

            '    """Zero-argument public functions must run without errors."""\n'

            "    for name, member in inspect.getmembers(main):\n"

            "        if name.startswith(\"_\") or not callable(member):\n"

            "            continue\n"

            "        if inspect.isclass(member):\n"

            "            continue\n"

            "        try:\n"

            "            sig = inspect.signature(member)\n"

            "        except (TypeError, ValueError):\n"

            "            continue\n"

            "        if all(\n"

            "            p.default is not p.empty\n"

            "            for p in sig.parameters.values()\n"

            "        ):\n"

            "            member()  # must not raise\n\n\n"

            'if __name__ == "__main__":\n'

            "    test_module_imports()\n"

            "    test_public_callables_do_not_crash()\n"

            '    print(\"All basic tests passed.\")\n'

        )

