"""ImageGenerator: product-cover art via NVIDIA NIM image models.

Uses the existing ``NVIDIA_NIM_API_KEY`` against NVIDIA's genai endpoint
(``https://ai.api.nvidia.com/v1/genai/<model>``) with FLUX.1-dev as the
default model. No new API key needed.

Why FLUX and not Qwen-Image
----------------------------
Qwen-Image (Alibaba) is not reachable with any key this swarm holds: it is
absent from the NIM catalog (81 models checked), absent from the Zen catalog
(40 models checked), and OpenRouter lists no Qwen image model — while the
OpenRouter key cannot afford image calls anyway (HTTP 402, ~$0 usable).
FLUX.1-dev answered HTTP 200 with valid JPEG bytes on the first probe using
the key already in ``.env``, so it is the proven default. The model id is
configurable (``NVIDIA_IMAGE_MODEL``), so a DashScope-backed Qwen-Image can
be slotted in later without touching callers.

Response contract mirrors :class:`~core.vision.VisionClient`: every method
returns ``{"ok": True, ...}`` or ``{"ok": False, "error": ...}`` and never
raises for provider problems, so image generation stays optional.

Example:
    ```python
    from core.image_generator import ImageGenerator

    gen = ImageGenerator()
    if gen.enabled:
        result = gen.generate_cover(
            product_name="Weekly Planner for Freelance Medical Interpreters",
            niche="medical interpreter",
            project_id="demo",
        )
        print(result.get("path"))
    ```
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv
from rich.console import Console

__all__ = [
    "ImageGenerator",
    "DEFAULT_IMAGE_MODEL",
    "DEFAULT_IMAGE_BASE_URL",
    "DEFAULT_IMAGE_TIMEOUT",
    "COVER_SIZE",
]

#: Proven working default (HTTP 200 + valid JPEG on live probe).
DEFAULT_IMAGE_MODEL: str = "black-forest-labs/flux.1-dev"

#: NVIDIA genai root; the model id is appended as the function path.
DEFAULT_IMAGE_BASE_URL: str = "https://ai.api.nvidia.com/v1/genai"

#: FLUX renders take a while; the HTTP timeout must cover it.
DEFAULT_IMAGE_TIMEOUT: float = 300.0

#: Default cover size in pixels (3:4 portrait, Payhip-friendly).
COVER_SIZE: tuple = (768, 1024)

_console = Console()


class ImageGenerator:
    """Product-cover image client over NVIDIA NIM image models.

    Args:
        model: NIM function id, e.g. ``black-forest-labs/flux.1-dev``.
            Defaults to ``NVIDIA_IMAGE_MODEL`` or :data:`DEFAULT_IMAGE_MODEL`.
        api_key: NIM/NGC key. Defaults to ``NVIDIA_NIM_API_KEY`` (then
            ``NVIDIA_API_KEY``). Empty means disabled, never an error.
        base_url: Genai root. Defaults to ``NVIDIA_IMAGE_BASE_URL`` or
            :data:`DEFAULT_IMAGE_BASE_URL`.
        timeout: Per-request HTTP timeout in seconds.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_IMAGE_TIMEOUT,
    ) -> None:
        load_dotenv()
        self.model: str = (
            (model or os.getenv("NVIDIA_IMAGE_MODEL")
             or DEFAULT_IMAGE_MODEL).strip()
            or DEFAULT_IMAGE_MODEL
        )
        self.base_url: str = (
            (base_url or os.getenv("NVIDIA_IMAGE_BASE_URL")
             or DEFAULT_IMAGE_BASE_URL).strip().rstrip("/")
            or DEFAULT_IMAGE_BASE_URL
        )
        self.api_key: str = (
            api_key or os.getenv("NVIDIA_NIM_API_KEY")
            or os.getenv("NVIDIA_API_KEY") or ""
        ).strip()
        self.timeout: float = timeout
        self.enabled: bool = bool(self.api_key)
        if self.enabled:
            _console.print(
                "[green]ImageGenerator ready[/green] — model="
                f"[bold]{self.model}[/bold]"
            )
        else:
            _console.print(
                "[yellow]ImageGenerator disabled (no NVIDIA NIM API key). "
                "Cover generation will be skipped.[/yellow]"
            )

    @property
    def endpoint(self) -> str:
        """Full function URL for the configured model."""
        return f"{self.base_url}/{self.model}"

    def generate(
        self,
        prompt: str,
        output_path: Optional[str] = None,
        seed: int = 0,
        steps: int = 30,
        cfg_scale: float = 3.5,
        size: tuple = COVER_SIZE,
    ) -> Dict[str, Any]:
        """Generate one image from a text prompt.

        Args:
            prompt: Image description (English works best for FLUX).
            output_path: Where to save the PNG/JPEG. Parent directories are
                created. When None, the raw bytes are returned without
                touching disk.
            seed: Diffusion seed (0 = random).
            steps: Denoising steps; 25-40 is the useful range.
            cfg_scale: Prompt adherence; 3-5 works well for FLUX.
            size: (width, height) in pixels; defaults to portrait
                :data:`COVER_SIZE`.

        Returns:
            ``{"ok": True, "path": ..., "bytes": n, "model", "seed",
            "seconds"}`` on success, or ``{"ok": False, "error": ...}``
            on any failure. Never raises for provider problems.
        """
        if not prompt or not prompt.strip():
            return {"ok": False, "error": "Prompt must be a non-empty string."}
        if not self.enabled:
            return {"ok": False, "error": "ImageGenerator disabled (no API key)."}
        width, height = size
        started = time.monotonic()
        try:
            response = httpx.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json={
                    "text_prompts": [
                        {"text": prompt.strip(), "weight": 1.0}
                    ],
                    "cfg_scale": cfg_scale,
                    "steps": steps,
                    "seed": seed,
                    "width": width,
                    "height": height,
                },
                timeout=self.timeout,
            )
        except Exception as exc:
            return {"ok": False,
                    "error": f"{type(exc).__name__}: {exc}"}
        if response.status_code != 200:
            return {"ok": False, "error": (
                f"HTTP {response.status_code}: {response.text[:300]}")}
        try:
            payload = response.json()
            artifacts = payload.get("artifacts") or []
            b64 = artifacts[0].get("base64", "") if artifacts else ""
            used_seed = (artifacts[0].get("seed", seed)
                         if artifacts else seed)
            raw = base64.b64decode(b64)
        except Exception as exc:
            return {"ok": False,
                    "error": f"Could not decode image response: {exc}"}
        if not raw:
            return {"ok": False, "error": "Empty image in provider response."}
        result: Dict[str, Any] = {
            "ok": True,
            "bytes": len(raw),
            "model": self.model,
            "seed": used_seed,
            "seconds": round(time.monotonic() - started, 1),
        }
        if output_path:
            try:
                target = Path(output_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
            except OSError as exc:
                return {"ok": False,
                        "error": f"Could not write image file: {exc}"}
            result["path"] = str(target)
            _console.print(f"[green]Cover saved to '{target}'.[/green]")
        else:
            result["raw"] = raw
        return result

    def generate_cover(
        self,
        product_name: str,
        niche: str,
        project_id: str = "cover",
        subtitle: str = "",
        style: str = "",
    ) -> Dict[str, Any]:
        """Generate a product cover for a niche (FASE 1: single variant).

        Builds a cover prompt from the product name + niche visual language
        and saves ``state/output/<project_id>/cover.png``.

        Args:
            product_name: Title shown in the prompt (kept short on purpose:
                diffusion models mangle long text).
            niche: Niche key, e.g. ``"medical interpreter"``. Selects the
                palette/motifs from :func:`cover_prompt`.
            project_id: Used for the output path only.
            subtitle: Optional one-line subtitle hint for the composition.
            style: Optional style override (default: clean flat vector).

        Returns:
            Same contract as :meth:`generate`, plus ``"prompt"`` (the exact
            prompt sent, so FASE 3 can learn from it).
        """
        if not product_name or not product_name.strip():
            return {"ok": False, "error": "product_name must be non-empty."}
        prompt = cover_prompt(
            product_name.strip(), niche.strip() if niche else "", subtitle,
            style)
        out = Path("state") / "output" / project_id / "cover.png"
        result = self.generate(prompt, str(out))
        result["prompt"] = prompt
        return result


#: Niche visual languages for covers. Unknown niches fall back to "default".
NICHE_STYLES: Dict[str, str] = {
    "medical interpreter": (
        "palette of clinical teal, navy blue and white with one warm amber "
        "accent; motifs: calendar grid, speech bubbles between two figures, "
        "stethoscope, hospital building silhouette, clock showing time zones"
    ),
    "default": (
        "palette of deep indigo, clean white and one coral accent; motifs: "
        "minimal geometric shapes, a tidy planner grid, subtle depth"
    ),
}


def cover_prompt(product_name: str, niche: str = "",
                 subtitle: str = "", style: str = "") -> str:
    """Build a FLUX cover prompt from product + niche.

    Keeps rendered text minimal (short title only): diffusion models garble
    long text, so subtitles and feature lists stay OUT of the image and go
    on the Payhip description instead (FASE 2 will overlay them crisply).

    Args:
        product_name: Short product title.
        niche: Niche key for palette/motifs (see :data:`NICHE_STYLES`).
        subtitle: Compositional hint only, never rendered as text.
        style: Style override; defaults to clean flat vector illustration.

    Returns:
        The exact English prompt string.
    """
    visual = NICHE_STYLES.get(niche.lower(), NICHE_STYLES["default"])
    style_text = style.strip() or (
        "clean flat vector illustration, professional digital-product cover, "
        "balanced composition with clear focal point, no photo, no watermark"
    )
    short_title = product_name.strip()
    if len(short_title) > 60:
        short_title = short_title[:57].rstrip() + "..."
    parts = [
        f"Digital product cover art for a planner titled "
        f"\"{short_title}\".",
        style_text + ".",
        f"Visual language: {visual}.",
        "Leave clear negative space at the top for the title text. "
        "No text, no letters, no words anywhere in the image.",
    ]
    if subtitle.strip():
        parts.append(f"Mood hint (do not render as text): "
                     f"{subtitle.strip()}.")
    return " ".join(parts)
