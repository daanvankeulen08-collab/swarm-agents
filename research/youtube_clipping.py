"""YouTubeClippingResearch: separates safe clipping tactics from dangerous ones.

Clipper channels sit on a legal fault line: transformative, licensed, or
original-narration formats can be legitimate, while re-uploading other
creators' footage, podcasts, or sports highlights invites DMCA strikes and
channel termination. This module researches clipping strategies for a niche
(public video metadata via :class:`YouTubeResearcher`), screens every
strategy with :class:`LegalAnalyzer`, and returns explicit safe vs dangerous
lists with recommendations.

Example:
    ```python
    from core.orchestrator import Orchestrator
    from core.state_manager import StateManager
    from research.youtube_clipping import YouTubeClippingResearch

    sm = StateManager(project_id="demo")
    clipping = YouTubeClippingResearch(
        orchestrator=Orchestrator(), state_manager=sm
    )
    result = clipping.research("personal finance")
    print(result["safe_strategies"])
    print(result["dangerous_strategies"])
    ```
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rich.console import Console

from core.orchestrator import Orchestrator
from core.state_manager import StateManager
from research.data_sources import YouTubeResearcher
from research.legal_analyzer import LegalAnalyzer

__all__ = ["YouTubeClippingResearch", "CLIPPING_STRATEGIES"]

#: Candidate clipping strategies screened per niche (description templates).
CLIPPING_STRATEGIES: List[Dict[str, str]] = [
    {
        "name": "Original narration over licensed stock",
        "description": "Faceless videos with self-written scripts, own voiceover "
                       "or licensed TTS, and commercially licensed stock footage.",
    },
    {
        "name": "Transformative commentary and critique",
        "description": "Short third-party clips embedded inside longer original "
                       "commentary/critique that adds new meaning (classic fair-use framing).",
    },
    {
        "name": "Public-domain and Creative-Commons compilations",
        "description": "Compilations built only from public-domain or CC-licensed "
                       "material with attribution and license checks per asset.",
    },
    {
        "name": "Podcast clip re-uploads",
        "description": "Re-uploading podcast video clips from other creators with "
                       "minimal or no added commentary.",
    },
    {
        "name": "Sports and TV highlight reposts",
        "description": "Reposting sports highlights, TV scenes, or movie clips "
                       "owned by leagues and studios.",
    },
    {
        "name": "Viral TikTok compilation reposts",
        "description": "Compilations of other creators' viral short videos "
                       "re-uploaded without permission or transformation.",
    },
]

_console = Console()


class YouTubeClippingResearch:
    """Research clipping strategies for a niche and classify their risk.

    Args:
        orchestrator: The swarm's :class:`Orchestrator` (needed to build a
            default :class:`LegalAnalyzer` when none is supplied).
        state_manager: The project's :class:`StateManager`.
        youtube_researcher: Optional pre-built researcher (DI for tests).
        legal_analyzer: Optional pre-built analyzer (DI for tests).
    """

    def __init__(
        self,
        orchestrator: Optional[Orchestrator] = None,
        state_manager: Optional[StateManager] = None,
        youtube_researcher: Optional[YouTubeResearcher] = None,
        legal_analyzer: Optional[LegalAnalyzer] = None,
    ) -> None:
        if legal_analyzer is None and (orchestrator is None or state_manager is None):
            raise ValueError(
                "Provide a legal_analyzer, or both orchestrator and "
                "state_manager so one can be built."
            )
        self.orchestrator = orchestrator
        self.state_manager = state_manager
        self.youtube = youtube_researcher or YouTubeResearcher()
        self.legal = legal_analyzer or LegalAnalyzer(orchestrator, state_manager)

    def research(
        self, niche: str, include_videos: bool = True
    ) -> Dict[str, Any]:
        """Screen clipping strategies for ``niche``.

        Args:
            niche: Topic area, e.g. ``"personal finance"``.
            include_videos: When True (default), pull public video metadata
                for context; set False for AI/legal-only analysis
                (used by ``--skip-scraping``).

        Returns:
            Dict with ``niche``, ``videos`` (metadata, may be empty),
            ``safe_strategies`` (low-risk names + why),``dangerous_strategies``
            (medium/high names + why), ``assessments`` (per-strategy legal
            reports), and ``recommendations``.

        Raises:
            ValueError: On empty niche.
        """
        if not isinstance(niche, str) or not niche.strip():
            raise ValueError("niche must be a non-empty string.")
        niche = niche.strip()
        _console.print(f"[cyan]Researching clipping strategies for {niche!r}…[/cyan]")

        videos: List[Dict[str, Any]] = []
        if include_videos:
            try:
                videos = self.youtube.research(niche).get("videos", [])
            except Exception as exc:
                _console.print(
                    f"[yellow]Video scan failed ({exc}); continuing "
                    "with legal screening only.[/yellow]"
                )

        safe: List[Dict[str, str]] = []
        dangerous: List[Dict[str, str]] = []
        assessments: List[Dict[str, Any]] = []
        for strategy in CLIPPING_STRATEGIES:
            label = f"{strategy['name']} in the {niche} niche: {strategy['description']}"
            try:
                assessment = self.legal.analyze(label)
            except Exception as exc:
                _console.print(
                    f"[yellow]Legal screening failed for "
                    f"'{strategy['name']}': {exc}[/yellow]"
                )
                assessment = {
                    "revenue_stream": label,
                    "risk_level": "high",
                    "risk_factors": [f"Could not complete screening: {exc}"],
                    "laws_cited": [],
                    "platform_terms": [],
                    "mitigations": ["Require human legal review before proceeding."],
                    "explanation": "Screening failed; treat as high risk.",
                    "pattern_hits": [],
                    "disclaimer": "Treat as high risk pending review.",
                }
            assessments.append({"strategy": strategy["name"], **assessment})
            entry = {
                "strategy": strategy["name"],
                "why": assessment.get("explanation", "")[:300],
            }
            if assessment.get("risk_level") == "low":
                safe.append(entry)
            else:
                dangerous.append({
                    **entry,
                    "risk_level": assessment.get("risk_level", "high"),
                })

        recommendations = self._recommendations(safe, dangerous)
        _console.print(
            f"[green]Clipping research done: {len(safe)} safe, "
            f"{len(dangerous)} dangerous strategies.[/green]"
        )
        return {
            "niche": niche,
            "videos": videos,
            "safe_strategies": safe,
            "dangerous_strategies": dangerous,
            "assessments": assessments,
            "recommendations": recommendations,
        }

    @staticmethod
    def _recommendations(
        safe: List[Dict[str, str]], dangerous: List[Dict[str, str]]
    ) -> List[str]:
        """Build actionable clipping guidance from the classification."""
        recs = [
            "Only publish formats from the safe list; keep licenses and "
            "attribution records for every third-party asset.",
            "Add genuine transformation to anything borrowed: original "
            "script, critique, or educational framing — never bare reposts.",
        ]
        if safe:
            recs.insert(0, f"Start with: {safe[0]['strategy']}.")
        if dangerous:
            names = ", ".join(d["strategy"] for d in dangerous[:3])
            recs.append(f"Absolutely avoid: {names}.")
        return recs
