"""KcalChecker: validate product nutrition claims against USDA reference data.

Models are fluent enough to invent plausible-looking kcal values. This agent
therefore treats every energy number as guilty until USDA FoodData Central
confirms it: it extracts food/energy/serving triples deterministically, asks
:class:`~core.usda_client.USDAClient` to resolve the same serving in grams,
and accepts only values within tolerance. Anything unparseable or
unresolvable is reported as unresolved rather than guessed.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from core.usda_client import USDAClient, food_url

__all__ = ["KcalChecker", "DEFAULT_TOLERANCE_PERCENT", "MAX_CLAIMS"]

logger = logging.getLogger(__name__)

#: Rounding/model-source tolerance used when USDA comparison is performed.
DEFAULT_TOLERANCE_PERCENT: float = 5.0

#: Upper bound on USDA lookups for one product. Nutrition-dense products
#: should be validated in sections rather than exhausting a shared API quota.
MAX_CLAIMS: int = 200

#: Numeric energy mention, for example ``95 kcal`` or ``200 calories``.
_ENERGY_RE = re.compile(
    r"(?P<kcal>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>kcal|kilocalories?|calories?|cal)\b",
    re.IGNORECASE,
)

#: Markdown table divider row.
_TABLE_DIVIDER_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")

#: Header synonyms used to identify food, serving, and energy columns.
_FOOD_HEADERS = {
    "food", "item", "ingredient", "meal", "snack", "fruit", "product", "dish"
}
_SERVING_HEADERS = {
    "serving", "servings", "portion", "portions", "size", "amount", "quantity"
}
_ENERGY_HEADERS = {
    "energy", "kcal", "calories", "calorie", "cal", "kilocalories"
}

#: Claim bases understood by USDA validation. Per-serving/countable claims
#: are intentionally not accepted as-is; they fail closed and ask the author
#: to restate the value per 100 g or with an explicit metric mass.
UNIT_PER_100G = "per_100g"
UNIT_METRIC_SERVING = "metric_serving"
UNIT_PER_VOLUME = "per_volume"
UNIT_PER_SERVING = "per_serving"

#: Multi-food dish totals (a whole recipe's energy, not one food's). These
#: can never be confirmed by single-food USDA lookup, so they fail with
#: guidance toward the inputs arithmetic verification will need — never
#: with "restate per 100 g," which is wrong advice for a dish total.
UNIT_RECIPE_TOTAL = "recipe_total"

#: Recipe codes (``R01``) and dish-ending words signalling a whole dish.
_RECIPE_CODE_RE = re.compile(r"\bR\d{2,}\b|\brecipe\b", re.IGNORECASE)
_DISH_WORDS = frozenset({
    "bowl", "bowls", "jar", "jars", "chili", "pasta", "salad", "salads",
    "soup", "soups", "stew", "stews", "curry", "curries", "casserole",
    "tacos", "stir-fry", "bake", "skillet", "platter", "platters",
})

#: Guidance for recipe totals: the exact inputs a future arithmetic check
#: needs, instead of advice that does not apply to dish totals.
RECIPE_TOTAL_GUIDANCE = (
    "Recipe-total energy cannot be verified by single-food USDA lookup. "
    "To make this verifiable, supply each ingredient's metric mass with "
    "its per-100g value so the dish total can be checked arithmetically."
)

#: Explicit metric masses that can be converted algebraically to per-100-gram
#: values without a food-specific density or portion assumption.
_METRIC_MASS_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mg|g|kg)\s*$",
    re.IGNORECASE,
)

#: A 100-gram reference amount expressed as serving text.
_PER_100G_SERVING_RE = re.compile(
    r"^\s*(?:per\s+)?100\s*g(?:rams?)?(?:\s+servings?)?\s*$",
    re.IGNORECASE,
)

#: Verbs/prepositions separating a food name from its nutritional statement.
_FOOD_CLAUSE_RE = re.compile(
    r"\s+(?:contains?|provides?|has|have|includes?|with|per)\s+",
    re.IGNORECASE,
)

#: Leading bullets, checkboxes, or enumeration markers.
_LEADING_MARKER_RE = re.compile(
    r"^\s*(?:[-*•‣▪>]|\d+[.)]|\[[ xX]\])\s*"
)

#: Parenthetical serving at the end of a food phrase, as in ``Apple (1 cup)``.
_PARENTHETICAL_SERVING_RE = re.compile(
    r"^(?P<food>.+?)\s*\((?P<serving>[^()]*)\)\s*$"
)


class KcalChecker:
    """Extract and USDA-validate every nutritional energy claim in a product.

    Args:
        usda_client: Configured :class:`~core.usda_client.USDAClient`.
        tolerance_percent: Allowed USDA deviation, in percent.
    """

    def __init__(
        self,
        usda_client: USDAClient,
        tolerance_percent: float = DEFAULT_TOLERANCE_PERCENT,
    ) -> None:
        for name in ("search_food", "get_nutrients", "validate_kcal"):
            if not hasattr(usda_client, name) or not callable(
                getattr(usda_client, name, None)
            ):
                raise TypeError(
                    "usda_client must expose the USDAClient interface "
                    f"(missing callable {name!r}); "
                    f"got {type(usda_client).__name__}."
                )
        if isinstance(tolerance_percent, bool):
            raise ValueError("tolerance_percent must be a number.")
        try:
            tolerance = float(tolerance_percent)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"tolerance_percent must be a number, got {tolerance_percent!r}."
            ) from exc
        if tolerance != tolerance or tolerance == float("inf") or tolerance < 0:
            raise ValueError("tolerance_percent must be finite and non-negative.")
        self.usda_client: USDAClient = usda_client
        self.tolerance_percent: float = tolerance

    def scan_content(self, content: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract claims and unparseable energy mentions without USDA calls."""
        claims = self._extract_nutritional_claims(content)
        skipped = self._find_unparsed_energy(content, claims)
        return {"claims": claims, "skipped": skipped}

    def check_product_nutrition(self, content: str) -> Dict[str, Any]:
        """Extract all nutritional claims and validate them against USDA.

        Args:
            content: Product Markdown/text under validation.

        Returns:
            Claims, skipped energy mentions, summary counts, and USDA
            sources. ``valid`` is True only for a confirmed USDA comparison;
            API failures and ambiguous servings are ``unresolved``, never
            silently treated as valid.
        """
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string.")
        scan = self.scan_content(content)
        claims = scan["claims"][:MAX_CLAIMS]
        skipped = scan["skipped"]
        validated: List[Dict[str, Any]] = []
        for claim in claims:
            try:
                validated.append(self._validate_claim(claim))
            except Exception as exc:
                logger.warning(
                    "USDA validation failed for %r: %s", claim.get("food"), exc
                )
                validated.append(
                    {
                        **claim,
                        "status": "unresolved",
                        "valid": False,
                        "usda_kcal": None,
                        "deviation_percent": None,
                        "fdc_id": None,
                        "food_name": None,
                        "food_url": None,
                        "error": (
                            f"USDA validation unavailable: {type(exc).__name__}."
                        ),
                    }
                )

        valid_claims = sum(1 for item in validated if item.get("status") == "valid")
        invalid_claims = sum(
            1 for item in validated if item.get("status") == "invalid"
        )
        unresolved_claims = sum(
            1 for item in validated if item.get("status") == "unresolved"
        )
        total = len(validated)
        summary = {
            "total_claims": total,
            "valid_claims": valid_claims,
            "invalid_claims": invalid_claims,
            "unresolved_claims": unresolved_claims,
            "unparsed_energy_mentions": len(skipped),
            "validation_rate": round(valid_claims / total, 3) if total else 1.0,
            "tolerance_percent": round(self.tolerance_percent, 2),
        }
        return {
            "claims": validated,
            "skipped": skipped,
            "summary": summary,
            "sources": self._sources(validated),
        }

    def _validate_claim(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """Validate one extracted claim against USDA reference data."""
        if claim.get("unit") == UNIT_RECIPE_TOTAL:
            # No USDA call: no single-food lookup can confirm a dish total,
            # so any comparison would be invented. Fail closed with the
            # guidance arithmetic verification will need.
            return {
                "food": claim["food"],
                "serving": claim.get("serving"),
                "unit": UNIT_RECIPE_TOTAL,
                "claimed_kcal": claim["claimed_kcal"],
                "line": claim.get("line"),
                "status": "unresolved",
                "valid": False,
                "usda_kcal": None,
                "deviation_percent": None,
                "fdc_id": None,
                "food_name": None,
                "food_url": None,
                "error": RECIPE_TOTAL_GUIDANCE,
            }
        validation = self.usda_client.validate_kcal(
            claim["food"],
            claim["claimed_kcal"],
            claim.get("serving"),
            unit=claim.get("unit", UNIT_PER_SERVING),
            tolerance_percent=self.tolerance_percent,
        )
        if validation.get("valid"):
            status = "valid"
        elif validation.get("usda_kcal") is None:
            status = "unresolved"
        else:
            status = "invalid"
        return {
            "food": claim["food"],
            "serving": claim.get("serving"),
            "unit": claim.get("unit", UNIT_PER_SERVING),
            "claimed_kcal": claim["claimed_kcal"],
            "line": claim.get("line"),
            "status": status,
            **validation,
            "valid": status == "valid",
        }

    def _extract_nutritional_claims(self, content: str) -> List[Dict[str, Any]]:
        """Extract food/energy/serving triples from Markdown and prose."""
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string.")
        lines = content.splitlines()
        claims: List[Dict[str, Any]] = []
        seen = set()
        header: Optional[List[str]] = None

        for number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                header = None
                continue
            if _is_table_row(stripped):
                if _TABLE_DIVIDER_RE.match(stripped):
                    continue
                cells = _split_table_cells(stripped)
                if header is None:
                    if _looks_like_header(cells):
                        header = [_normalize_header(cell) for cell in cells]
                    continue
                table_claim = _claim_from_table_row(cells, header, number, line)
                if table_claim is not None:
                    _add_claim(claims, seen, table_claim)
                    continue
                # A data row beneath a nutrition header that cannot be mapped
                # is still handled by the prose fallback below.
            else:
                header = None
            for match in _ENERGY_RE.finditer(line):
                sentence_claim = _claim_from_sentence(
                    line, match, number
                )
                if sentence_claim is not None:
                    _add_claim(claims, seen, sentence_claim)
        return claims

    def _find_unparsed_energy(
        self, content: str, claims: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find numeric energy values that extraction could not validate.

        A bare unit word such as a table-header ``Calories`` label is not a
        claim and is ignored. Only a number that cannot be attached to an
        identifiable food or serving blocks validation.
        """
        claimed_spans = {
            (claim.get("line"), round(float(claim.get("claimed_kcal", 0.0)), 2))
            for claim in claims
        }
        skipped: List[Dict[str, Any]] = []
        for number, line in enumerate(content.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or _TABLE_DIVIDER_RE.match(stripped):
                continue
            if _is_table_row(stripped) and _looks_like_header(
                _split_table_cells(stripped)
            ):
                continue
            numeric_spans = [
                (match.start(), match.end(), round(float(match.group("kcal")), 2))
                for match in _ENERGY_RE.finditer(line)
            ]
            claimed_values = {
                value for line_no, value in claimed_spans if line_no == number
            }
            for _start, _end, value in numeric_spans:
                if value in claimed_values:
                    continue
                skipped.append(
                    {
                        "line": number,
                        "text": line.strip()[:200],
                        "reason": (
                            "Energy value has no identifiable food or serving."
                        ),
                    }
                )
        return skipped

    def _sources(
        self, claims: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Deduplicate successful USDA sources for citation."""
        sources: List[Dict[str, Any]] = []
        seen = set()
        for claim in claims:
            if claim.get("status") != "valid":
                continue
            fdc_id = claim.get("fdc_id")
            if not isinstance(fdc_id, int) or fdc_id in seen:
                continue
            seen.add(fdc_id)
            sources.append(
                {
                    "food": claim.get("food_name") or claim.get("food"),
                    "fdc_id": fdc_id,
                    "url": claim.get("food_url") or food_url(fdc_id),
                }
            )
        return sources


def _is_table_row(line: str) -> bool:
    """True for a Markdown table row with at least two pipe boundaries."""
    return line.count("|") >= 2


def _split_table_cells(line: str) -> List[str]:
    """Split a Markdown table row into cell text."""
    text = line.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|"):
        text = text[:-1]
    return [cell.strip() for cell in text.split("|")]


def _normalize_header(cell: str) -> str:
    """Normalize table-header text for column identification."""
    return re.sub(r"\s+", " ", cell.strip().lower())


def _looks_like_header(cells: List[str]) -> bool:
    """True when a row names food/serving/energy columns."""
    normalized = {_normalize_header(cell) for cell in cells}
    return bool(
        (normalized & _FOOD_HEADERS)
        and (normalized & _ENERGY_HEADERS)
    )


def _column_index(normalized: List[str], vocabulary: set) -> Optional[int]:
    """Find the first header column belonging to a vocabulary."""
    for index, cell in enumerate(normalized):
        words = set(re.findall(r"[a-z0-9]+", cell))
        if words & vocabulary:
            return index
    return None


def _serving_is_plural(serving: Optional[str]) -> bool:
    """True when a serving describes more than one serving."""
    if not serving:
        return False
    text = serving.strip()
    count = re.match(r"(\d+)", text)
    if count and int(count.group(1)) > 1:
        return True
    return "serving" in text.lower()


def _is_recipe_total(food: str, serving: Optional[str]) -> bool:
    """True when an energy value is a whole-dish total, not one food's.

    Signals, in order: an explicit recipe code (``R01``), a multi-item food
    description (commas, "and", "with"), or a dish-ending word combined
    with a plural serving. Single foods in any serving form are never
    recipe totals.
    """
    lowered = (food or "").lower()
    if _RECIPE_CODE_RE.search(food or ""):
        return True
    if "," in (food or "") or " and " in lowered or " with " in lowered:
        return True
    tokens = re.findall(r"[a-z0-9-]+", lowered)
    if tokens and tokens[-1] in _DISH_WORDS and _serving_is_plural(serving):
        return True
    return False


def _classify_unit(
    serving: Optional[str],
    line: str,
    energy: re.Match,
    header_text: str = "",
) -> str:
    """Label a claim as per-100g, metric-serving, or countable-serving.

    Countable servings are deliberately labelled ``per_serving`` even when a
    numeric energy value is present. The validator must fail those claims
    closed rather than accept the household serving as-is.
    """
    context = f"{header_text} {line[energy.end():]}"
    if re.search(
        r"(?:/|per)\s*100\s*ml\b", context, re.IGNORECASE
    ) or (
        serving is not None
        and re.search(
            r"\d+(?:\.\d+)?\s*(?:ml|l|fl\s*oz|cups?|tbsp|tsp)\b",
            serving,
            re.IGNORECASE,
        )
    ):
        return UNIT_PER_VOLUME
    if re.search(
        r"(?:/|per)\s*100\s*g(?:rams?)?\b", context, re.IGNORECASE
    ) or (
        serving is not None and _PER_100G_SERVING_RE.match(serving)
    ):
        return UNIT_PER_100G
    if serving is not None and _METRIC_MASS_RE.search(serving):
        return UNIT_METRIC_SERVING
    return UNIT_PER_SERVING


def _claim_from_table_row(
    cells: List[str], header: List[str], number: int, line: str
) -> Optional[Dict[str, Any]]:
    """Map one Markdown table row through its identified header columns."""
    food_index = _column_index(header, _FOOD_HEADERS)
    energy_index = _column_index(header, _ENERGY_HEADERS)
    serving_index = _column_index(header, _SERVING_HEADERS)
    if food_index is None or energy_index is None:
        return None
    if max(food_index, energy_index) >= len(cells):
        return None
    food = _clean_food_candidate(cells[food_index])
    serving = cells[serving_index].strip() if (
        serving_index is not None and serving_index < len(cells)
    ) else None
    energy = _ENERGY_RE.search(cells[energy_index])
    if not food or energy is None:
        return None
    if _is_recipe_total(food, serving):
        unit = UNIT_RECIPE_TOTAL
    else:
        unit = _classify_unit(
            serving, line, energy, header_text=" ".join(header)
        )
    return {
        "food": food,
        "serving": serving or None,
        "unit": unit,
        "claimed_kcal": round(float(energy.group("kcal")), 2),
        "line": number,
        "source": line.strip()[:200],
    }


def _claim_from_sentence(
    line: str, match: re.Match, number: int
) -> Optional[Dict[str, Any]]:
    """Parse one prose energy mention into a food/serving claim."""
    prefix = line[: match.start()].rstrip()
    food_text, serving_text = _split_food_serving(prefix)
    food = _clean_food_candidate(food_text)
    if not food:
        return None
    serving = serving_text.strip() if serving_text and serving_text.strip() else None
    if _is_recipe_total(food, serving):
        unit = UNIT_RECIPE_TOTAL
    else:
        unit = _classify_unit(serving, line, match)
    return {
        "food": food,
        "serving": serving,
        "unit": unit,
        "claimed_kcal": round(float(match.group("kcal")), 2),
        "line": number,
        "source": line.strip()[:200],
    }


def _split_food_serving(prefix: str) -> Tuple[str, Optional[str]]:
    """Split text before an energy value into food and serving parts."""
    text = _LEADING_MARKER_RE.sub("", prefix).strip()
    if not text:
        return "", None
    # A label separator may follow the serving parenthetical, as in
    # "Apple (1 medium): 95 kcal".
    text = re.sub(r"\s*[:\-–—]\s*$", "", text).strip()
    if not text:
        return "", None
    parenthetical = _PARENTHETICAL_SERVING_RE.match(text)
    if parenthetical:
        return parenthetical.group("food"), parenthetical.group("serving")
    clause = _FOOD_CLAUSE_RE.split(text, maxsplit=1)
    if len(clause) == 2:
        food, remainder = clause
        serving_match = re.search(
            r"\bper\s+(?P<serving>.+?)\s*$", remainder.strip(), re.IGNORECASE
        )
        if serving_match:
            return food, serving_match.group("serving")
        return food, None
    # A final colon or dash normally separates a label from its value:
    # "Apple: 95", "Apple - 95".
    for separator in (":", "-", "–", "—"):
        if separator in text:
            food, _, _ = text.rpartition(separator)
            if food.strip():
                return food, None
    return text, None


def _clean_food_candidate(raw: str) -> str:
    """Remove Markdown, bullets, quantities, and clause verbs from food text."""
    text = _LEADING_MARKER_RE.sub("", raw or "").strip()
    text = re.sub(r"[*_`]+", "", text).strip()
    # Remove an enumeration or checkbox remnant missed by the marker pattern.
    text = re.sub(r"^\s*(?:\d+[.)]|[\[\(][ xX][\]\)])\s*", "", text).strip()
    # Remove trailing descriptors after verbs, quantities, or percentages.
    text = re.split(
        r"\s+(?:contains?|provides?|has|have|includes?|with|per)\s+",
        text,
        maxsplit=1,
    )[0].strip()
    text = re.split(r"\s*[:\-–—=]\s*", text)[0].strip()
    text = text.strip(" ,;")
    if len(text.split()) > 12:
        return ""
    if re.fullmatch(
        r"(?i)(?:total|subtotal|sum|summary|average|daily\s+value)s?",
        text,
    ):
        return ""
    return text


def _add_claim(
    claims: List[Dict[str, Any]],
    seen: set,
    claim: Dict[str, Any],
) -> None:
    """Append a claim unless the same line/value was already extracted."""
    key = (
        claim.get("line"),
        claim.get("food", "").lower(),
        (claim.get("serving") or "").lower(),
        claim.get("unit"),
        claim.get("claimed_kcal"),
    )
    if key in seen:
        return
    seen.add(key)
    claims.append(claim)
