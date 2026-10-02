"""FactCheckerAgent: deterministic citation-coverage checking.

This is intentionally not an external truth verifier. It cannot browse the
web, inspect USDA records, or prove that a cited authority is correct. What
it can prove deterministically is whether a factual-looking assertion is
accompanied by a textual source: a named authority, citation, URL, DOI, FDC
identifier, or matching entry in a Sources section.

That distinction matters. A bare ``"USDA guidelines"`` attribution means
``has_source=True`` but ``verifiable=False``. Only a resolvable artifact—a
URL, DOI, FDC ID, or other retrievable reference—is marked verifiable. No
source metadata is ever invented: :meth:`generate_sources_appendix` formats
only caller-supplied verified sources.

Worked examples, fillable fields, imperatives, and explicitly illustrative
numbers are not treated as factual assertions. A grocery quantity in a table
introduced as an illustrative example is product scaffolding, not a claim
about the external world.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from rich.console import Console

from core.orchestrator import Orchestrator
from core.state_manager import StateManager

__all__ = [
    "FactCheckerAgent",
    "SOURCE_SECTION_HEADINGS",
    "MAX_CLAIM_CHARS",
]

logger = logging.getLogger(__name__)
_console = Console()

#: Headings whose sections contain citations rather than new factual claims.
SOURCE_SECTION_HEADINGS: frozenset = frozenset(
    {
        "sources",
        "nutrition sources",
        "references",
        "bibliography",
        "works cited",
    }
)

#: Maximum retained claim text. The source document remains authoritative.
MAX_CLAIM_CHARS: int = 500

#: Sentences containing these patterns assert external/research knowledge.
_CLAIM_HINT_RE = re.compile(
    r"\b(?:studies?|research|researchers?|scientists?|doctors?|dietitians?|"
    r"nutritionists?|experts?|survey|evidence|data|guidelines?|"
    r"recommendations?|recommended|requirements?|required|standards?|"
    r"according to|as recommended by|published by|reported by)\b"
    r"|recommended daily|daily (?:intake|value)|deficiency|toxicity",
    re.IGNORECASE,
)

#: Numeric values that can carry an external factual assertion.
_NUMERIC_CLAIM_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:%|percent|kcal|kilocalories?|calories?|cal\b|"
    r"mg|g\b|kg|°F|°C|IU|mcg|ml|l\b)",
    re.IGNORECASE,
)

#: Words marking illustrative/template content rather than factual assertions.
_EXAMPLE_MARKER_RE = re.compile(
    r"\b(?:illustrative|illustration|example|examples|sample|placeholder|"
    r"fill(?:-|\s+)?in|adjust|replace them|your own|hypothetical|fictional)\b",
    re.IGNORECASE,
)

#: Imperative instructions are product directions, not external claims.
_IMPERATIVE_START_RE = re.compile(
    r"^\s*(?:check|compare|move|delete|shop|use|wash|cook|season|pack|"
    r"refrigerate|freeze|label|store|record|enter|confirm|review|add|"
    r"transfer|preheat|assemble|clean|divide|combine)\b",
    re.IGNORECASE,
)

#: Explicit attribution to a named authority.
_ATTRIBUTION_RES = (
    re.compile(
        r"\baccording to\s+([^.;()]{3,120})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bas\s+(?:recommended|stated|reported|published|required)\s+by\s+"
        r"([^.;()]{3,120})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b([A-Z][\w&'’.-]*(?:\s+[A-Z][\w&'’.-]*){0,5})\s+"
        r"(guidelines?|recommendations?|research|researchers?|study|studies|"
        r"survey|data|report|experts?|scientists?|doctors?)\b"
    ),
)

#: Citation artifacts that can identify a resolvable source.
_URL_RE = re.compile(r"https?://[^\s)>\]]+")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]{1,200})\]\((https?://[^)\s]+)\)")
_DOI_RE = re.compile(r"\b10\.\d{4,}/[^\s)>\]]+", re.IGNORECASE)
_FDC_ID_RE = re.compile(r"\bFDC ID\s*[:#]?\s*(\d{4,})\b", re.IGNORECASE)
_FDC_URL_RE = re.compile(r"food-details/(\d{4,})/nutrients")
_AUTHOR_YEAR_RE = re.compile(r"\([A-Z][^()]{1,80}?,\s*\d{4}\)")
_FOOTNOTE_RE = re.compile(r"\[\^?[1-9][0-9]?\]")


class FactCheckerAgent:
    """Check whether factual-looking claims carry textual sources.

    Args:
        orchestrator: The swarm's :class:`Orchestrator`. Retained for future
            live source retrieval; the current check is deterministic and
            makes no model calls.
        state_manager: The project's :class:`StateManager`.
    """

    def __init__(
        self, orchestrator: Orchestrator, state_manager: StateManager
    ) -> None:
        for name in (
            "ask_primary_agent",
            "build_context_prompt",
            "get_model_info",
        ):
            if not hasattr(orchestrator, name) or not callable(
                getattr(orchestrator, name, None)
            ):
                raise TypeError(
                    "orchestrator must expose the Orchestrator interface "
                    f"(missing callable {name!r}); "
                    f"got {type(orchestrator).__name__}."
                )
        for name in (
            "get_state",
            "log_api_call",
            "log_fact_check",
            "save",
            "update_phase",
        ):
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

    def check_facts(self, content: str) -> Dict[str, Any]:
        """Extract factual claims and check them for textual sources.

        Args:
            content: Product Markdown/text under review.

        Returns:
            Dict with ``claims``, ``sources``, and ``summary``. ``sources``
            contains only citation artifacts actually found in the content;
            nothing is invented.
        """
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string.")
        lines = content.splitlines()
        source_entries = _parse_source_entries(lines)
        claims = self._extract_factual_claims(content)
        checked = [
            self._verify_source(claim, lines, source_entries) for claim in claims
        ]
        sourced = sum(1 for item in checked if item["has_source"])
        total = len(checked)
        summary = {
            "total_claims": total,
            "sourced_claims": sourced,
            "unsourced_claims": total - sourced,
            "sourcing_rate": round(sourced / total, 3) if total else 1.0,
        }
        return {
            "claims": checked,
            "sources": _normalize_sources(source_entries),
            "summary": summary,
        }

    def _extract_factual_claims(self, content: str) -> List[Dict[str, Any]]:
        """Extract sentences that assert external/research knowledge."""
        lines = content.splitlines()
        claims: List[Dict[str, Any]] = []
        in_source_section = False
        for number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if _is_source_heading(stripped):
                in_source_section = True
                continue
            if stripped.startswith("#"):
                in_source_section = False
            if not stripped or in_source_section:
                continue
            context = _example_context(lines, number)
            for sentence in _split_sentences(stripped):
                cleaned = sentence.strip()
                if len(cleaned) < 20 or len(cleaned) > MAX_CLAIM_CHARS + 100:
                    continue
                if _EXAMPLE_MARKER_RE.search(cleaned) or _EXAMPLE_MARKER_RE.search(
                    context
                ):
                    continue
                if _IMPERATIVE_START_RE.search(cleaned):
                    continue
                if not _CLAIM_HINT_RE.search(
                    cleaned
                ) and not _NUMERIC_CLAIM_RE.search(cleaned):
                    continue
                # A bare number in instructional/template prose is not enough:
                # require an external-knowledge cue as well.
                if _NUMERIC_CLAIM_RE.search(
                    cleaned
                ) and not _CLAIM_HINT_RE.search(cleaned):
                    continue
                claims.append(
                    {
                        "claim": cleaned[:MAX_CLAIM_CHARS],
                        "line": number,
                        "context": context[:200],
                    }
                )
        return claims

    def _verify_source(
        self,
        claim: Dict[str, Any],
        lines: List[str],
        source_entries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Check one claim for a textual source in nearby content."""
        text = claim["claim"]
        number = claim["line"]
        nearby = "\n".join(
            lines[max(0, number - 3):min(len(lines), number + 2)]
        )
        attributions = _attributions_in_text(text) or _attributions_in_text(
            nearby
        )
        citations = _citations_in_text(f"{text}\n{nearby}")
        matched_entry = _match_source_entry(text, source_entries)
        if matched_entry is not None:
            citations.extend(_citations_in_text(matched_entry["text"]))

        source_text = "; ".join(
            dict.fromkeys(
                attributions
                + [item["text"] for item in citations]
                + ([matched_entry["text"]] if matched_entry else [])
            )
        )[:300]
        has_source = bool(attributions or citations or matched_entry)
        verifiable = any(
            item.get("url")
            or item.get("doi")
            or item.get("fdc_id")
            for item in citations
        ) or (
            matched_entry is not None
            and bool(
                matched_entry.get("url")
                or matched_entry.get("doi")
                or matched_entry.get("fdc_id")
            )
        )
        return {
            "claim": text,
            "line": number,
            "has_source": has_source,
            "source": source_text or None,
            "verifiable": verifiable,
        }

    def generate_sources_appendix(self, sources: List[Dict[str, Any]]) -> str:
        """Format caller-supplied verified sources as a Sources appendix.

        Args:
            sources: Source dicts with a title/text plus a URL, DOI, FDC ID,
                or other retrievable identifier.

        Returns:
            Markdown appendix text, or an empty string when no sources are
            supplied. Unverifiable entries raise rather than being formatted.
        """
        if not isinstance(sources, list):
            raise ValueError("sources must be a list.")
        if not sources:
            return ""
        formatted = []
        for position, source in enumerate(sources, start=1):
            if not isinstance(source, dict):
                raise ValueError(
                    f"Source {position} must be a mapping, got "
                    f"{type(source).__name__}."
                )
            title = str(
                source.get("title") or source.get("text") or ""
            ).strip()
            url = str(source.get("url") or "").strip()
            identifier = str(
                source.get("doi")
                or source.get("fdc_id")
                or source.get("identifier")
                or ""
            ).strip()
            if not title:
                raise ValueError(f"Source {position} has no title or text.")
            if not url and not identifier:
                raise ValueError(
                    f"Source {position} has no URL, DOI, FDC ID, or other "
                    "retrievable identifier and cannot be called verified."
                )
            detail = url or identifier
            formatted.append(f"{position}. {title} {detail}".rstrip())
        return "\n---\n\n## Sources\n\n" + "\n\n".join(formatted) + "\n"


def _split_sentences(line: str) -> List[str]:
    """Split one Markdown line into sentence-like units."""
    text = re.sub(r"[*_`]+", "", line).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def _is_source_heading(line: str) -> bool:
    """True when a Markdown heading starts a sources section."""
    match = re.match(r"^#{1,6}\s*(.+?)\s*$", line)
    return bool(
        match and match.group(1).strip().lower() in SOURCE_SECTION_HEADINGS
    )


def _example_context(lines: List[str], number: int) -> str:
    """Return nearby nonblank lines used for example-context detection."""
    context = [
        lines[index].strip()
        for index in range(max(0, number - 9), min(len(lines), number + 1))
        if lines[index].strip()
    ]
    return " ".join(context[-4:])


def _attributions_in_text(text: str) -> List[str]:
    """Extract named-authority attributions from text."""
    found = []
    for pattern in _ATTRIBUTION_RES:
        for match in pattern.finditer(text):
            candidate = re.sub(r"\s+", " ", match.group(1)).strip(" ,;:")
            if len(candidate) >= 3:
                found.append(candidate[:160])
    return found


def _citations_in_text(text: str) -> List[Dict[str, Any]]:
    """Extract resolvable citation artifacts from text."""
    citations = []
    for label, url in _MARKDOWN_LINK_RE.findall(text):
        citations.append(
            {"text": f"{label.strip()} ({url.strip()})", "url": url.strip()}
        )
    for url in _URL_RE.findall(text):
        citations.append({"text": url.strip(), "url": url.strip()})
    for doi in _DOI_RE.findall(text):
        citations.append({"text": doi.strip(), "doi": doi.strip()})
    for fdc_id in _FDC_ID_RE.findall(text):
        citations.append(
            {
                "text": f"FDC ID {fdc_id}",
                "fdc_id": int(fdc_id),
                "url": (
                    "https://fdc.nal.usda.gov/fdc-app.html#/food-details/"
                    f"{fdc_id}/nutrients"
                ),
            }
        )
    for fdc_id in _FDC_URL_RE.findall(text):
        citations.append(
            {
                "text": f"FDC ID {fdc_id}",
                "fdc_id": int(fdc_id),
                "url": (
                    "https://fdc.nal.usda.gov/fdc-app.html#/food-details/"
                    f"{fdc_id}/nutrients"
                ),
            }
        )
    for marker in _AUTHOR_YEAR_RE.findall(text):
        citations.append({"text": marker.strip()})
    for marker in _FOOTNOTE_RE.findall(text):
        citations.append({"text": marker.strip()})
    return citations


def _parse_source_entries(lines: List[str]) -> List[Dict[str, Any]]:
    """Parse Sources/References sections into citation entries."""
    entries = []
    in_sources = False
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if _is_source_heading(stripped):
            in_sources = True
            continue
        if stripped.startswith("#"):
            in_sources = False
        if in_sources and stripped:
            entries.append({"line": number, "text": stripped[:300]})
    return entries


def _normalize_sources(
    entries: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Normalize parsed source entries and their citation artifacts."""
    normalized = []
    for entry in entries:
        citations = _citations_in_text(entry["text"])
        normalized.append(
            {
                "text": entry["text"],
                "line": entry["line"],
                "url": next(
                    (
                        item["url"]
                        for item in citations
                        if item.get("url")
                    ),
                    None,
                ),
                "doi": next(
                    (
                        item["doi"]
                        for item in citations
                        if item.get("doi")
                    ),
                    None,
                ),
                "fdc_id": next(
                    (
                        item["fdc_id"]
                        for item in citations
                        if item.get("fdc_id")
                    ),
                    None,
                ),
            }
        )
    return normalized


def _significant_words(text: str) -> List[str]:
    """Return distinctive lowercase words for source-entry matching."""
    stopwords = {
        "the", "and", "for", "with", "from", "that", "this", "these",
        "those", "according", "guidelines", "guideline", "recommended",
        "daily", "intake", "values", "value", "food", "foods", "product",
    }
    words = re.findall(r"[a-z0-9]{4,}", text.lower())
    return [word for word in words if word not in stopwords]


def _match_source_entry(
    claim: str, entries: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Match a claim to a source entry by distinctive shared wording."""
    claim_words = set(_significant_words(claim))
    if not claim_words:
        return None
    best: Optional[Dict[str, Any]] = None
    best_overlap = 0
    for entry in entries:
        overlap = len(claim_words & set(_significant_words(entry["text"])))
        if overlap > best_overlap:
            best_overlap = overlap
            best = entry
    # Three distinctive shared words avoids matching every nutrition sentence
    # to a generic "USDA FoodData Central" line by the word "USDA" alone.
    return best if best_overlap >= 3 else None
