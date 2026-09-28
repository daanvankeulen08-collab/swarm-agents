"""LegalAnalyzer: structured legal-risk screening for revenue streams.

The analyzer combines deterministic keyword screening against
:data:`KNOWN_RISK_PATTERNS` with an AI review via the swarm
:class:`~core.orchestrator.Orchestrator`, and returns a structured risk
assessment covering copyright (DMCA, fair use), platform terms of service,
consumer-protection, and data-protection (GDPR) basics.

Focus areas: copyright law, platform terms (YouTube, Gumroad, Etsy, Creative
Market), consumer-protection law, GDPR basics.

Example:
    ```python
    from core.orchestrator import Orchestrator
    from core.state_manager import StateManager
    from research.legal_analyzer import LegalAnalyzer

    analyzer = LegalAnalyzer(Orchestrator(), StateManager(project_id="demo"))
    report = analyzer.analyze("Selling original Notion templates on Gumroad")
    print(report["risk_level"], report["laws_cited"])
    ```

Important: this tool provides general legal *information*, not legal advice.
Every assessment carries :data:`DISCLAIMER` and recommends consulting a
qualified attorney before acting. All analysis is ethical: no private data,
no circumvention, public information only.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from rich.console import Console

from core.orchestrator import Orchestrator
from core.state_manager import StateManager

__all__ = ["LegalAnalyzer", "KNOWN_RISK_PATTERNS", "DISCLAIMER", "RISK_LEVELS"]

#: Valid risk levels, lowest to highest.
RISK_LEVELS: tuple = ("low", "medium", "high")

#: Appended to every assessment — this is information, not counsel.
DISCLAIMER: str = (
    "Disclaimer: this assessment provides general legal information only, "
    "not legal advice, and may be incomplete or out of date. Consult a "
    "qualified attorney before pursuing any revenue stream."
)

#: Keyword-driven screens applied before (and merged with) AI analysis.
#: Each entry maps trigger keywords to a baseline level and explanation.
KNOWN_RISK_PATTERNS: List[Dict[str, Any]] = [
    {
        "keywords": ("re-upload", "reupload", "repost", "rip", "download",
                     "clipper", "compilation", "someone else's", "others'"),
        "context_keywords": ("youtube", "video", "movie", "music", "song",
                             "clip", "podcast", "tiktok"),
        "risk_level": "high",
        "reason": "Re-uploading third-party audio/video risks DMCA takedowns, "
                  "Content ID claims, and channel termination; 'fair use' "
                  "rarely protects wholesale reposts.",
        "laws": ["17 U.S.C. § 512 (DMCA takedown/counter-notice)",
                 "17 U.S.C. § 107 (fair use, four-factor test)",
                 "YouTube Terms of Service (copyright policy)"],
    },
    {
        "keywords": ("plr", "private label rights", "resell rights", "mrr",
                     "master resell"),
        "context_keywords": (),
        "risk_level": "medium",
        "reason": "PLR/MRR content is only sellable within the exact license "
                  "terms; saturation and license violations are common, and "
                  "buyers can demand refunds for misrepresented rights.",
        "laws": ["License/contract terms of the PLR provider",
                 "EU Consumer Rights Directive 2011/83 (digital content)"],
    },
    {
        "keywords": ("scrape", "scraping", "bot", "automated collection",
                     "crawl"),
        "context_keywords": ("personal data", "email", "user data", "profile"),
        "risk_level": "high",
        "reason": "Collecting personal data at scale can breach GDPR and "
                  "platform anti-scraping terms, and may trigger CFAA-style "
                  "unauthorized-access claims.",
        "laws": ["GDPR Arts. 5–6 (lawfulness, data minimisation)",
                 "Platform Terms of Service (anti-scraping clauses)"],
    },
    {
        "keywords": ("original", "own template", "my own", "created by me",
                     "handmade", "from scratch"),
        "context_keywords": ("template", "script", "ebook", "guide", "tool",
                             "notion", "etsy", "gumroad"),
        "risk_level": "low",
        "reason": "Wholly original works carry the lowest IP risk, provided "
                  "no third-party assets (fonts, images, music) are bundled "
                  "without a license.",
        "laws": ["17 U.S.C. § 102 (copyright subsists in original works)",
                 "Gumroad/Etsy seller terms (own-content warranties)"],
    },
    {
        "keywords": ("affiliate", "referral link", "commission"),
        "context_keywords": (),
        "risk_level": "medium",
        "reason": "Affiliate income is legitimate but requires clear "
                  "disclosure of the commercial relationship in most "
                  "jurisdictions; undisclosed endorsements draw FTC/EU "
                  "enforcement.",
        "laws": ["FTC Endorsement Guides (disclosure duties)",
                 "EU Unfair Commercial Practices Directive"],
    },
    {
        "keywords": ("ai-generated", "ai generated", "generated content",
                     "midjourney", "stock photo", "font"),
        "context_keywords": (),
        "risk_level": "medium",
        "reason": "AI outputs and bundled third-party assets (fonts, stock, "
                  "music) need license checks: generator terms, commercial-use "
                  "tiers, and unsettled authorship questions.",
        "laws": ["Generator platform license terms",
                 "17 U.S.C. § 107 (fair use, training-data disputes)"],
    },
]

SYSTEM_PROMPT: str = (
    "You are a meticulous legal-risk researcher (not a lawyer). Analyse the "
    "described online revenue stream for legal risk across: copyright law "
    "(DMCA, fair use), platform terms of service (YouTube, Gumroad, Etsy), "
    "consumer-protection law, and GDPR basics. Respond with VALID JSON ONLY, "
    "no commentary, using exactly this shape: "
    '{"risk_level": "low|medium|high", '
    '"risk_factors": ["..."], '
    '"laws_cited": ["..."], '
    '"platform_terms": ["..."], '
    '"mitigations": ["..."], '
    '"explanation": "..."}'
)

_console = Console()


class LegalAnalyzer:
    """Screen a revenue-stream description for legal risk.

    Args:
        orchestrator: The swarm's :class:`Orchestrator` for AI analysis.
        state_manager: The project's :class:`StateManager` (API usage is
            logged through it on every AI call).

    Deterministic :data:`KNOWN_RISK_PATTERNS` screening always runs; the AI
    review refines it. The higher of the two risk levels wins, so the tool
    errs toward caution.
    """

    def __init__(
        self, orchestrator: Orchestrator, state_manager: StateManager
    ) -> None:
        # Duck-typed checks so test doubles work (swarm-wide convention).
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

    def analyze(self, revenue_stream: str) -> Dict[str, Any]:
        """Assess legal risk for one revenue-stream description.

        Args:
            revenue_stream: Plain-language description, e.g.
                ``"Selling original Notion templates on Gumroad"``.

        Returns:
            Dict with ``revenue_stream``, ``risk_level`` (low/medium/high),
            ``risk_factors``, ``laws_cited``, ``platform_terms``,
            ``mitigations``, ``explanation``, ``pattern_hits`` (which known
            patterns fired), and ``disclaimer``.

        Raises:
            ValueError: If the description is empty.
            RuntimeError: If the AI review fails (pattern screening results
                are still included in the error context).
        """
        cleaned = revenue_stream.strip() if isinstance(revenue_stream, str) else ""
        if not cleaned:
            raise ValueError("revenue_stream must be a non-empty string.")
        _console.print(
            f"[cyan]Legal screening:[/cyan] {cleaned[:100]}"
            f"{'…' if len(cleaned) > 100 else ''}"
        )

        pattern_report = self.screen_patterns(cleaned)
        ai_report: Dict[str, Any] = {}
        try:
            raw = self.orchestrator.ask_agent(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=(
                    "Analyse the legal risk of this revenue stream:\n\n"
                    f"{cleaned}\n\nRespond with VALID JSON ONLY."
                ),
                state_manager=self.state_manager,
            )
            parsed = _parse_json(raw)
            if isinstance(parsed, dict):
                ai_report = parsed
        except Exception as exc:
            raise RuntimeError(
                f"AI legal review failed for {cleaned!r} "
                f"(pattern screen found: {pattern_report['risk_level']}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        merged = self._merge(pattern_report, ai_report)
        merged["revenue_stream"] = cleaned
        merged["disclaimer"] = DISCLAIMER
        _console.print(
            f"[green]Legal screening done: risk = "
            f"[bold]{merged['risk_level']}[/bold][/green]"
        )
        return merged

    def screen_patterns(self, revenue_stream: str) -> Dict[str, Any]:
        """Run deterministic keyword screening (no AI call).

        Args:
            revenue_stream: Description to screen (case-insensitive).

        Returns:
            Dict with ``risk_level`` (highest hit, default ``"low"``),
            ``risk_factors``, ``laws_cited``, and ``pattern_hits`` (indices
            into :data:`KNOWN_RISK_PATTERNS`). Never raises for string input.
        """
        text = revenue_stream.lower()
        hits: List[int] = []
        factors: List[str] = []
        laws: List[str] = []
        for i, pattern in enumerate(KNOWN_RISK_PATTERNS):
            if not any(k in text for k in pattern["keywords"]):
                continue
            context = pattern.get("context_keywords", ())
            if context and not any(c in text for c in context):
                continue
            hits.append(i)
            factors.append(pattern["reason"])
            laws.extend(pattern.get("laws", []))
        level = "low"
        for i in hits:
            level = _higher(level, KNOWN_RISK_PATTERNS[i]["risk_level"])
        return {
            "risk_level": level,
            "risk_factors": factors,
            "laws_cited": _dedupe(laws),
            "pattern_hits": hits,
        }

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _merge(pattern: Dict[str, Any], ai: Dict[str, Any]) -> Dict[str, Any]:
        """Combine pattern + AI reports; the higher risk level wins."""
        ai_level = str(ai.get("risk_level", "low")).strip().lower()
        if ai_level not in RISK_LEVELS:
            ai_level = "low"
        return {
            "risk_level": _higher(pattern["risk_level"], ai_level),
            "risk_factors": _dedupe(
                [str(f) for f in pattern.get("risk_factors", [])]
                + [str(f) for f in ai.get("risk_factors", [])
                   if isinstance(f, str)]
            ),
            "laws_cited": _dedupe(
                [str(v) for v in pattern.get("laws_cited", [])]
                + [str(v) for v in ai.get("laws_cited", [])
                   if isinstance(v, str)]
            ),
            "platform_terms": _dedupe(
                [str(v) for v in ai.get("platform_terms", [])
                 if isinstance(v, str)]
            ),
            "mitigations": _dedupe(
                [str(v) for v in ai.get("mitigations", []) if isinstance(v, str)]
            ),
            "explanation": str(ai.get("explanation", "")).strip()
            or pattern.get("risk_factors", ["No specific risks identified."])[0],
            "pattern_hits": list(pattern.get("pattern_hits", [])),
        }


def _higher(first: str, second: str) -> str:
    """Return the higher of two risk levels."""
    order = {level: rank for rank, level in enumerate(RISK_LEVELS)}
    return first if order.get(first, 0) >= order.get(second, 0) else second


def _dedupe(values: List[str]) -> List[str]:
    """Order-preserving de-duplication of non-empty strings."""
    seen = set()
    out: List[str] = []
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            out.append(cleaned)
    return out


def _parse_json(raw: str) -> Any:
    """Parse model JSON tolerantly (strips code fences first)."""
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            return json.loads(text[start: end + 1])
        raise
