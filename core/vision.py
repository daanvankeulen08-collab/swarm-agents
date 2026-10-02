"""VisionClient: image understanding via the NVIDIA NIM vision API.

Wraps an OpenAI-compatible client pointed at
``https://integrate.api.nvidia.com/v1`` and exposes two helpers:
:meth:`VisionClient.analyze_image` (local file, base64 data URI) and
:meth:`VisionClient.analyze_image_url` (remote URL, passed through).

The default model is ``meta/llama-3.2-11b-vision-instruct``; override with
the ``model`` argument or the ``NVIDIA_NIM_VISION_MODEL`` environment
variable. (``deepseek-ai/deepseek-v4.1-flash`` is text-only on NIM and
returns 404 for image requests; the other catalog vision entries —
``microsoft/phi-3-vision-128k-instruct`` and ``nvidia/cosmos-reason2-8b`` —
are not enabled for this account.)
If no API key is configured the client reports ``enabled=False`` and every
call returns an error dict instead of raising — vision is always optional.

Example:
    ```python
    from core.vision import VisionClient

    vision = VisionClient()
    if vision.enabled:
        result = vision.analyze_image("shot.png", "Describe this image.")
        print(result.get("text"))
    ```
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console

__all__ = [
    "VisionClient",
    "DEFAULT_VISION_MODEL",
    "DEFAULT_VISION_BASE_URL",
    "VISION_TIMEOUT",
]

#: Default vision-capable model on NVIDIA NIM. ``deepseek-ai/deepseek-v4.1-flash``
#: is text-only (404 on image requests), and the other catalog vision models
#: are not entitled for this account, so Llama 3.2 Vision is used instead.
DEFAULT_VISION_MODEL: str = "meta/llama-3.2-11b-vision-instruct"

#: NVIDIA NIM OpenAI-compatible endpoint.
DEFAULT_VISION_BASE_URL: str = "https://integrate.api.nvidia.com/v1"

#: Per-request timeout for vision calls. Deliberately generous (10 minutes):
#: NIM judges are free and stable, and timing out a visual review mid-run
#: kills the pipeline for no benefit. The 60s stall guard stays ONLY on the
#: Zen/OpenRouter cascade hops — never on dedicated NIM judges.
VISION_TIMEOUT: float = 600.0

_console = Console()


class VisionClient:
    """Image analysis client over the NVIDIA NIM vision API.

    Args:
        model: Model id. Defaults to ``NVIDIA_NIM_VISION_MODEL`` or
            ``DEFAULT_VISION_MODEL``.
        api_key: API key. Defaults to ``NVIDIA_NIM_API_KEY`` (then
            ``NVIDIA_API_KEY``). Empty means disabled, never an error.
        base_url: Endpoint. Defaults to ``NVIDIA_NIM_BASE_URL`` or
            ``DEFAULT_VISION_BASE_URL``.
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = VISION_TIMEOUT,
    ) -> None:
        load_dotenv()
        self.model: str = (
            (model or os.getenv("NVIDIA_NIM_VISION_MODEL")
             or DEFAULT_VISION_MODEL).strip()
            or DEFAULT_VISION_MODEL
        )
        self.base_url: str = (
            (base_url or os.getenv("NVIDIA_NIM_BASE_URL")
             or DEFAULT_VISION_BASE_URL).strip().rstrip("/")
            or DEFAULT_VISION_BASE_URL
        )
        resolved_key = (
            api_key or os.getenv("NVIDIA_NIM_API_KEY")
            or os.getenv("NVIDIA_API_KEY") or ""
        ).strip()
        self.timeout: float = timeout
        self.enabled: bool = bool(resolved_key)
        self.client: Optional[OpenAI] = None
        if self.enabled:
            self.client = OpenAI(
                api_key=resolved_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
            _console.print(
                "[green]VisionClient ready[/green] — model="
                f"[bold]{self.model}[/bold]"
            )
        else:
            _console.print(
                "[yellow]VisionClient disabled (no NVIDIA NIM API key). "
                "Visual review will be skipped.[/yellow]"
            )

    def analyze_image(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """Analyze a local image file with a text prompt.

        Args:
            image_path: Path to a PNG/JPEG/WEBP file.
            prompt: What to ask about the image.

        Returns:
            ``{"ok": True, "text": ..., "model": ...}`` on success, or
            ``{"ok": False, "error": ...}`` on any failure (missing file,
            disabled client, network/API error). Never raises.
        """
        path = Path(image_path)
        if not path.is_file():
            return {"ok": False, "error": f"Image not found: {image_path}"}
        if not self.enabled or self.client is None:
            return {"ok": False, "error": "VisionClient disabled (no API key)."}
        if not prompt or not prompt.strip():
            return {"ok": False, "error": "Prompt must be a non-empty string."}
        try:
            raw = path.read_bytes()
        except OSError as exc:
            return {"ok": False, "error": f"Cannot read image: {exc}"}
        mime = _guess_mime(path.suffix.lower())
        data_uri = (
            f"data:{mime};base64,"
            + base64.b64encode(raw).decode("ascii")
        )
        return self._complete(data_uri, prompt.strip())

    def analyze_image_url(self, image_url: str, prompt: str) -> Dict[str, Any]:
        """Analyze a remote image URL with a text prompt.

        Args:
            image_url: Public ``http(s)://`` image URL.
            prompt: What to ask about the image.

        Returns:
            Same ``ok``/``error`` contract as :meth:`analyze_image`.
        """
        url = (image_url or "").strip()
        if not url.lower().startswith(("http://", "https://")):
            return {"ok": False, "error": f"Invalid image URL: {image_url!r}"}
        if not self.enabled or self.client is None:
            return {"ok": False, "error": "VisionClient disabled (no API key)."}
        if not prompt or not prompt.strip():
            return {"ok": False, "error": "Prompt must be a non-empty string."}
        return self._complete(url, prompt.strip())

    # -- internals ----------------------------------------------------------

    def _complete(self, image_ref: str, prompt: str) -> Dict[str, Any]:
        """Send one vision chat call; map every outcome to a result dict."""
        assert self.client is not None
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": image_ref}},
                    ],
                }],
                temperature=0.2,
                timeout=self.timeout,
            )
            choices = getattr(response, "choices", None) or []
            message = getattr(choices[0], "message", None) if choices else None
            text = (getattr(message, "content", None) or "").strip()
            if not text:
                return {"ok": False, "error": "Vision model returned no text."}
            usage = getattr(response, "usage", None)
            return {
                "ok": True,
                "text": text,
                "model": self.model,
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                },
            }
        except Exception as exc:
            _console.print(f"[yellow]Vision call failed: {exc}[/yellow]")
            return {"ok": False,
                    "error": f"{type(exc).__name__}: {exc}"}


def _guess_mime(suffix: str) -> str:
    """Map a file suffix to an image MIME type (default PNG)."""
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix, "image/png")
