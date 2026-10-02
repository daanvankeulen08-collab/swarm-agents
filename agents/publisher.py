"""PublisherAgent: prepares (and optionally submits) Payhip product listings.

The publisher takes the approved product, formats store-ready product data
with the **dynamic customer-driven price**, and writes a
``payhip_product.json`` manifest next to the ZIP in ``publish_ready/`` so a
listing can be created without retyping anything.

Live submission policy: the agent only attempts an HTTP upload when the
operator explicitly configures ``PAYHIP_API_URL``. No endpoint is guessed —
without that variable, :meth:`publish_product` returns a
``published: False`` result directing a manual upload of the prepared
manifest. ``PAYHIP_API_KEY`` (when set) is sent as a Bearer token.

Example:
    ```python
    from agents.publisher import PublisherAgent
    from core.state_manager import StateManager

    publisher = PublisherAgent(StateManager(project_id="demo"))
    data = publisher.format_product_data(
        product_name="My Planner",
        description="Weekly planning template.",
        price=12.5,
        file_path="publish_ready/My_Planner.zip",
    )
    publisher.write_manifest("publish_ready", data)
    ```
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from rich.console import Console

from core.state_manager import StateManager

__all__ = ["PublisherAgent"]

_console = Console()


class PublisherAgent:
    """Format Payhip product data and stage publish manifests.

    Args:
        state_manager: The project's :class:`StateManager` (used for
            state reads; no model calls are made here).
    """

    def __init__(self, state_manager: StateManager) -> None:
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

    def format_product_data(
        self,
        product_name: str,
        description: str,
        price: float,
        file_path: str,
        currency: str = "EUR",
    ) -> Dict[str, Any]:
        """Build a Payhip-ready product payload with a dynamic price.

        Args:
            product_name: Store listing title (non-empty).
            description: Listing description (non-empty).
            price: Customer-driven price in euros. Must be a positive
                number — the pipeline passes the clamped
                ``willingness_to_pay`` (5.0–25.0) here.
            file_path: Path to the deliverable ZIP.
            currency: ISO currency code (default ``"EUR"``).

        Returns:
            Dict with ``name``, ``description``, ``price`` (rounded 2dp),
            ``currency``, ``file_path``, and ``created_at`` (ISO).

        Raises:
            ValueError: On empty name/description/file or non-positive price.
        """
        cleaned_name = product_name.strip() if isinstance(product_name, str) else ""
        if not cleaned_name:
            raise ValueError("product_name must be a non-empty string.")
        cleaned_desc = description.strip() if isinstance(description, str) else ""
        if not cleaned_desc:
            raise ValueError("description must be a non-empty string.")
        if isinstance(price, bool) or not isinstance(price, (int, float)):
            raise ValueError(f"price must be a number, got {price!r}.")
        if not float(price) > 0:
            raise ValueError(f"price must be positive, got {price!r}.")
        cleaned_file = str(file_path or "").strip()
        if not cleaned_file:
            raise ValueError("file_path must be a non-empty string.")
        data = {
            "name": cleaned_name,
            "description": cleaned_desc,
            "price": round(float(price), 2),
            "currency": (currency or "EUR").strip().upper() or "EUR",
            "file_path": cleaned_file,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _console.print(
            f"[green]Payhip product data ready: '{cleaned_name}' at "
            f"{data['price']:g} {data['currency']}.[/green]"
        )
        return data

    def write_manifest(
        self, publish_dir: str | Path, product_data: Dict[str, Any]
    ) -> Path:
        """Write ``payhip_product.json`` manifest next to the ZIP (UTF-8).

        Returns:
            Path of the written manifest. Raises IOError on write failure.
        """
        directory = Path(publish_dir)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / "payhip_product.json"
            target.write_text(
                json.dumps(product_data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise IOError(f"Could not write manifest in '{directory}': {exc}") from exc
        _console.print(f"[green]Manifest written to '{target}'.[/green]")
        return target

    def publish_product(
        self, product_data: Dict[str, Any], timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Submit the listing, or report manual-upload instructions.

        Only attempts HTTP when ``PAYHIP_API_URL`` is explicitly configured
        (no endpoint is ever guessed). Returns ``{"published": bool, ...}``
        and never raises for transport problems.
        """
        load_dotenv()
        api_url = (os.getenv("PAYHIP_API_URL") or "").strip().rstrip("/")
        if not api_url:
            return {
                "published": False,
                "reason": (
                    "PAYHIP_API_URL is not configured — manifest is ready "
                    "for manual upload in publish_ready/payhip_product.json."
                ),
                "product_data": product_data,
            }
        api_key = (os.getenv("PAYHIP_API_KEY") or "").strip()
        try:
            import requests as _requests

            response = _requests.post(
                f"{api_url}/products",
                json=product_data,
                headers=(
                    {"Authorization": f"Bearer {api_key}"} if api_key else {}
                ),
                timeout=timeout,
            )
            if 200 <= response.status_code < 300:
                return {"published": True, "status": response.status_code,
                        "product_data": product_data}
            return {"published": False,
                    "reason": f"Payhip API returned HTTP {response.status_code}: "
                              f"{response.text[:300]}",
                    "product_data": product_data}
        except Exception as exc:
            return {"published": False,
                    "reason": f"Payhip upload failed: {type(exc).__name__}: {exc}",
                    "product_data": product_data}
