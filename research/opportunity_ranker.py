"""OpportunityRanker: transparent, deterministic ranking of revenue streams.

The ranker takes opportunity dicts (as produced by
:meth:`ResearchAgent.generate_opportunity_report` or AI-only brainstorming),
optionally folds in :class:`LegalAnalyzer` assessments matched by name, and
scores every candidate with a documented composite formula:

    score = feasibility (0–10, weight 2.0)
            − risk penalty (low 0 / medium 3 / high 8)
            + automation bonus (full 2 / partial 1 / manual 0)
            + revenue signal (0–2 from revenue_potential text cues)

Scores are clamped to 0–20 and every result carries its ``score_breakdown``
so rankings stay auditable. No AI calls, no network — pure and testable.

Example:
    ```python
    from research.opportunity_ranker import OpportunityRanker

    ranker = OpportunityRanker()
    ranked = ranker.rank(opportunities, legal_assessments)
    print(ranked[0]["name"], ranked[0]["score"])
    print(ranker.safe_bets(ranked))
    print(ranker.first_projects(ranked, n=3))
    ```
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from rich.console import Console

__all__ = [
    "OpportunityRanker",
    "RISK_PENALTY",
    "AUTOMATION_BONUS",
    "MAX_SCORE",
]

#: Score penalty per risk level (subtracted from the weighted feasibility).
RISK_PENALTY: Dict[str, float] = {"low": 0.0, "medium": 3.0, "high": 8.0}

#: Score bonus per automation level.
AUTOMATION_BONUS: Dict[str, float] = {"full": 2.0, "partial": 1.0, "manual": 0.0}

#: Upper clamp for composite scores (feasibility weight 2.0 × 10 = 20).
MAX_SCORE: float = 20.0

#: Revenue-text cues mapping to a 0–2 signal (first match wins, top-down).
REVENUE_CUES: tuple = (
    (re.compile(r"\$\s?[\d,]{4,}"), 2.0),          # $1,000+ figures
    (re.compile(r"\b\d{3,}\s?/mo\b", re.I), 2.0),  # 500/mo style
    (re.compile(r"\$\s?[\d,]{2,}"), 1.0),          # any $ amount
    (re.compile(r"\b(high|strong|proven)\b", re.I), 1.0),
    (re.compile(r"\b(moderate|medium|steady)\b", re.I), 0.5),
)

_console = Console()


class OpportunityRanker:
    """Rank revenue opportunities with a transparent composite score.

    The ranker is stateless and defensive: missing fields get safe defaults
    (feasibility 5, risk ``"medium"``, automation ``"partial"``) and inputs
    are never mutated — ranked copies are returned.
    """

    def rank(
        self,
        opportunities: List[Dict[str, Any]],
        legal_assessments: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Score and sort opportunities, best first.

        Args:
            opportunities: Candidate dicts with (optionally) ``name``,
                ``feasibility_score``, ``risk_level``, ``automation_level``,
                ``revenue_potential``.
            legal_assessments: Optional LegalAnalyzer outputs. When an
                assessment's ``revenue_stream`` text mentions an
                opportunity's name (or vice versa), the opportunity's
                ``risk_level`` is upgraded to the harsher of the two and
                the assessment is attached as ``legal_assessment``.

        Returns:
            New list of enriched dicts (original keys plus ``score``,
            ``score_breakdown``, ``rank``, and possibly
            ``legal_assessment`` / ``needs_human_review``), sorted by
            ``score`` descending, then feasibility, then name.

        Raises:
            ValueError: If ``opportunities`` is empty or not a list.
        """
        if not isinstance(opportunities, list) or not opportunities:
            raise ValueError("opportunities must be a non-empty list.")
        legal = legal_assessments or []
        enriched = [self._score(dict(opp), legal) for opp in opportunities]
        enriched.sort(
            key=lambda o: (-o["score"], -o["feasibility_score"], o["name"])
        )
        for position, item in enumerate(enriched, start=1):
            item["rank"] = position
        _console.print(
            f"[green]Ranked {len(enriched)} opportunities "
            f"(best: '{enriched[0]['name']}' at {enriched[0]['score']}).[/green]"
        )
        return enriched

    def top(self, ranked: List[Dict[str, Any]], n: int = 5) -> List[Dict[str, Any]]:
        """Return the first ``n`` entries of a ranked list."""
        return list(ranked[: max(0, n)])

    def safe_bets(
        self, ranked: List[Dict[str, Any]], n: int = 5
    ) -> List[Dict[str, Any]]:
        """Return up to ``n`` low-risk entries, in rank order."""
        return [o for o in ranked if o.get("risk_level") == "low"][: max(0, n)]

    def avoid(self, ranked: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return high-risk entries with reasons, worst first."""
        return [o for o in reversed(ranked) if o.get("risk_level") == "high"]

    def first_projects(
        self, ranked: List[Dict[str, Any]], n: int = 3
    ) -> List[Dict[str, Any]]:
        """Recommend the first ``n`` projects to build with the BuilderAgent.

        Preference order: low-risk first (by rank), then the best remaining
        non-high-risk entries. Each recommendation gains a ``build_reason``
        string. Never recommends high-risk items unless nothing else exists
        (flagged via ``needs_human_review=True`` in that case).
        """
        picks: List[Dict[str, Any]] = []
        for item in ranked:
            if len(picks) >= n:
                break
            if item.get("risk_level") == "low":
                picks.append(item)
        for item in ranked:
            if len(picks) >= n:
                break
            if item in picks or item.get("risk_level") == "high":
                continue
            picks.append(item)
        if len(picks) < n:
            for item in ranked:
                if len(picks) >= n:
                    break
                if item in picks:
                    continue
                flagged = dict(item)
                flagged["needs_human_review"] = True
                picks.append(flagged)
        for item in picks:
            item.setdefault(
                "build_reason",
                f"Rank #{item.get('rank', '?')}: feasibility "
                f"{item.get('feasibility_score', '?')}/10, "
                f"{item.get('risk_level', '?')} risk, "
                f"{item.get('automation_level', '?')} automation.",
            )
        return picks[:n]

    # -- internals ----------------------------------------------------------

    def _score(
        self, opp: Dict[str, Any], legal: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Score one opportunity copy, folding in legal assessments."""
        name = str(opp.get("name", "Unnamed opportunity")).strip() or "Unnamed opportunity"
        opp["name"] = name
        feasibility = _clamp_number(opp.get("feasibility_score", 5), 1, 10)
        opp["feasibility_score"] = feasibility
        risk = str(opp.get("risk_level", "medium")).strip().lower()
        if risk not in RISK_PENALTY:
            risk = "medium"
        automation = str(opp.get("automation_level", "partial")).strip().lower()
        if automation not in AUTOMATION_BONUS:
            automation = "partial"

        attached = self._match_legal(name, legal)
        if attached is not None:
            legal_risk = str(attached.get("risk_level", "low")).strip().lower()
            if legal_risk in RISK_PENALTY and RISK_PENALTY[legal_risk] > RISK_PENALTY[risk]:
                risk = legal_risk
            opp["legal_assessment"] = attached
        opp["risk_level"] = risk
        opp["automation_level"] = automation

        revenue_text = str(opp.get("revenue_potential", ""))
        revenue_signal = 0.0
        for pattern, value in REVENUE_CUES:
            if pattern.search(revenue_text):
                revenue_signal = value
                break

        feasibility_part = feasibility * 2.0
        penalty = RISK_PENALTY[risk]
        bonus = AUTOMATION_BONUS[automation]
        score = round(
            max(0.0, min(MAX_SCORE, feasibility_part - penalty + bonus + revenue_signal)),
            2,
        )
        opp["score"] = score
        opp["score_breakdown"] = {
            "feasibility_part": feasibility_part,
            "risk_penalty": penalty,
            "automation_bonus": bonus,
            "revenue_signal": revenue_signal,
        }
        if risk == "high":
            opp["needs_human_review"] = True
        return opp

    @staticmethod
    def _match_legal(
        name: str, legal: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Attach the first assessment whose stream text matches ``name``."""
        needle = name.lower()
        for assessment in legal:
            if not isinstance(assessment, dict):
                continue
            stream = str(assessment.get("revenue_stream", "")).lower()
            if needle and (needle in stream or (stream and stream in needle)):
                return assessment
        return None


def _clamp_number(value: Any, low: float, high: float) -> int:
    """Coerce to int and clamp into ``[low, high]`` (default: midpoint)."""
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        number = int((low + high) // 2)
    return max(int(low), min(int(high), number))
