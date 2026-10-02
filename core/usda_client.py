"""USDAClient: authoritative nutrition data from FoodData Central.

Models must not invent energy values. Every kcal validation in this swarm
therefore goes through USDA FoodData Central (FDC), whose public REST API
returns the food description, energy per 100 grams, portion gram weights,
and serving metadata used below.

Only the standard ``requests`` package is used. The API key is never logged
or included in exception text.
"""

from __future__ import annotations

import logging
import os
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from rich.console import Console

__all__ = [
    "USDAClient",
    "DEFAULT_USDA_BASE_URL",
    "DEFAULT_USDA_TIMEOUT",
    "DEFAULT_KCAL_TOLERANCE_PERCENT",
    "USDA_DATA_TYPES",
    "NON_BRANDED_DATA_TYPES",
    "DUTCH_FOOD_ALIASES",
]

logger = logging.getLogger(__name__)
_console = Console()

#: USDA FoodData Central REST root, verified against the official API guide.
DEFAULT_USDA_BASE_URL: str = "https://api.nal.usda.gov/fdc/v1"

#: Network ceiling for one USDA request. Food validation must be bounded.
DEFAULT_USDA_TIMEOUT: float = 30.0

#: Rounding/model-source tolerance used when comparing a claimed value with
#: USDA's reference value.
DEFAULT_KCAL_TOLERANCE_PERCENT: float = 5.0

#: FDC data types searched by default. All four are official USDA sources.
USDA_DATA_TYPES: tuple = (
    "Foundation",
    "SR Legacy",
    "Survey (FNDDS)",
    "Branded",
)

#: Default search scope for unbranded product wording. Branded records can
#: use sparse one-word descriptions (for example, ``APPLE``) without useful
#: portion data, so they are consulted only when explicitly requested.
NON_BRANDED_DATA_TYPES: tuple = (
    "Foundation",
    "SR Legacy",
    "Survey (FNDDS)",
)

#: Generic foods are preferred over branded reformulations when descriptions
#: are otherwise equally close. Standardized composition records are preferred
#: over survey-recall records for generic reference values.
_DATA_TYPE_PRIORITY: Dict[str, float] = {
    "SR Legacy": 1.0,
    "Foundation": 0.97,
    "Survey (FNDDS)": 0.94,
    "Branded": 0.90,
    "Experimental": 0.80,
}

#: Minimal Dutch-to-English food-name aliases for USDA lookup. These map food
#: wording only; they never supply nutritional values. Extend this table when
#: additional localized products are validated.
DUTCH_FOOD_ALIASES: Dict[str, str] = {
    "havermout": "oats",
    "melk halfvol": "milk, reduced fat",
    "banaan": "banana",
    "blauwe bessen": "blueberries",
    "honing": "honey",
    "amandelen": "almonds",
}

#: Mass units convertible to grams without food-specific density data.
_MASS_TO_GRAMS: Dict[str, float] = {
    "mg": 0.001,
    "milligram": 0.001,
    "milligrams": 0.001,
    "g": 1.0,
    "gram": 1.0,
    "grams": 1.0,
    "kg": 1000.0,
    "kilogram": 1000.0,
    "kilograms": 1000.0,
    "oz": 28.349523125,
    "ounce": 28.349523125,
    "ounces": 28.349523125,
    "lb": 453.59237,
    "lbs": 453.59237,
    "pound": 453.59237,
    "pounds": 453.59237,
}

#: Common quantity words accepted at the start of a serving description.
_NUMBER_WORDS: Dict[str, float] = {
    "a": 1.0,
    "an": 1.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "half": 0.5,
    "quarter": 0.25,
}

#: Unicode vulgar fractions accepted in serving text.
_FRACTION_CHARS: Dict[str, float] = {
    "¼": 0.25,
    "½": 0.5,
    "¾": 0.75,
    "⅓": 1.0 / 3.0,
    "⅔": 2.0 / 3.0,
    "⅛": 0.125,
}

#: Words that describe an unprepared generic food rather than a different dish.
_GENERIC_STATE_WORDS: frozenset = frozenset(
    {"fresh", "plain", "raw", "ripe", "uncooked", "unripe", "whole"}
)


class USDAClient:
    """Query USDA FoodData Central and validate energy claims.

    Args:
        api_key: USDA/data.gov API key. Defaults to ``USDA_API_KEY``. The
            client can be constructed without one so callers can report key
            status, but every network method requires a key.
        base_url: FDC API root. Defaults to :data:`DEFAULT_USDA_BASE_URL`.
        timeout: Per-request timeout in seconds.
        tolerance_percent: Default allowed deviation when comparing claimed
            and USDA kcal values.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_USDA_TIMEOUT,
        tolerance_percent: float = DEFAULT_KCAL_TOLERANCE_PERCENT,
    ) -> None:
        load_dotenv()
        self.base_url: str = (
            (base_url or DEFAULT_USDA_BASE_URL).strip().rstrip("/")
            or DEFAULT_USDA_BASE_URL
        )
        self.api_key: str = (api_key or os.getenv("USDA_API_KEY") or "").strip()
        self.timeout: float = float(timeout)
        self.tolerance_percent: float = float(tolerance_percent)
        if not self.timeout > 0:
            raise ValueError("timeout must be a positive number.")
        if not self.tolerance_percent >= 0:
            raise ValueError("tolerance_percent cannot be negative.")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "swarm-kcal-checker/1.0",
            }
        )

    @property
    def has_api_key(self) -> bool:
        """True when an API key is configured."""
        return bool(self.api_key)

    def search_food(
        self,
        food_name: str,
        page_size: int = 10,
        data_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search USDA for a food and return normalized candidate matches.

        Args:
            food_name: Food text from the product, for example ``"Apple"``.
            page_size: USDA candidates requested, from 1 to 25.
            data_types: Optional FDC data-type filter.

        Returns:
            Normalized candidates with ``fdc_id``, ``description``,
            ``data_type``, ``score`` when supplied, and alternate names.

        Raises:
            ValueError: For invalid input.
            EnvironmentError: When no API key is configured.
            RuntimeError: For transport, HTTP, or malformed-response failures.
        """
        cleaned = food_name.strip() if isinstance(food_name, str) else ""
        if not cleaned:
            raise ValueError("food_name must be a non-empty string.")
        if not isinstance(page_size, int) or isinstance(page_size, bool):
            raise ValueError("page_size must be an integer.")
        if not 1 <= page_size <= 25:
            raise ValueError("page_size must be between 1 and 25.")
        self._require_api_key()
        query = _translate_food_name(cleaned)
        payload: Dict[str, Any] = {
            "query": query,
            "pageSize": page_size,
            "pageNumber": 1,
        }
        selected_types = (
            data_types if data_types is not None else NON_BRANDED_DATA_TYPES
        )
        if selected_types:
            payload["dataType"] = list(selected_types)
        response = self._request(
            "POST", f"{self.base_url}/foods/search", json_body=payload
        )
        foods = response.get("foods", [])
        if not isinstance(foods, list) or not foods:
            raise LookupError(f"USDA found no match for {cleaned!r}.")
        candidates = []
        for food in foods:
            if not isinstance(food, dict):
                continue
            try:
                fdc_id = int(food.get("fdcId"))
            except (TypeError, ValueError):
                continue
            description = str(food.get("description", "")).strip()
            if not description:
                continue
            candidates.append(
                {
                    "fdc_id": fdc_id,
                    "description": description,
                    "data_type": str(food.get("dataType", "") or ""),
                    "score": food.get("score"),
                    "common_names": _string_list(food.get("commonNames")),
                    "additional_descriptions": _string_list(
                        food.get("additionalDescriptions")
                    ),
                }
            )
        if not candidates:
            raise LookupError(f"USDA returned no usable match for {cleaned!r}.")
        return candidates

    def lookup_food(
        self,
        food_name: str,
        page_size: int = 10,
        data_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Return the best USDA candidate and its full nutrient details.

        Args:
            food_name: Food text from the product.
            page_size: USDA candidates requested, from 1 to 25.
            data_types: Optional FDC data-type filter.

        Returns:
            Dict with the translated ``query``, selected ``candidate``,
            ``match_score``, and full ``details`` from
            :meth:`get_nutrients`.
        """
        candidates = self.search_food(
            food_name, page_size=page_size, data_types=data_types
        )
        best, match_score = _select_best_match(
            _translate_food_name(food_name), candidates
        )
        return {
            "query": _translate_food_name(food_name),
            "candidate": best,
            "match_score": round(match_score, 3),
            "details": self.get_nutrients(best["fdc_id"]),
        }

    def get_nutrients(self, fdc_id: int) -> Dict[str, Any]:
        """Get energy, macros, portions, and serving metadata for one food.

        Args:
            fdc_id: Positive USDA FoodData Central identifier.

        Returns:
            Normalized nutrient details. FDC energy is conventionally stated
            per 100 grams; branded label energy is retained separately when
            USDA supplies it.

        Raises:
            ValueError: For an invalid identifier.
            EnvironmentError: When no API key is configured.
            RuntimeError: For transport, HTTP, or malformed-response failures.
            LookupError: When USDA has no energy value for the food.
        """
        if isinstance(fdc_id, bool) or not isinstance(fdc_id, int) or fdc_id <= 0:
            raise ValueError(f"fdc_id must be a positive integer, got {fdc_id!r}.")
        self._require_api_key()
        details = self._request("GET", f"{self.base_url}/food/{fdc_id}")
        nutrients = details.get("foodNutrients", [])
        if not isinstance(nutrients, list):
            raise RuntimeError(f"USDA food {fdc_id} has malformed nutrients.")
        energy_kcal, energy_source = _energy_kcal_per_100_grams(nutrients)
        label_energy, label_serving = _label_energy(nutrients, details)
        return {
            "fdc_id": fdc_id,
            "description": str(details.get("description", "") or ""),
            "data_type": str(details.get("dataType", "") or ""),
            "energy_kcal_per_100g": energy_kcal,
            "energy_source": energy_source,
            "protein_g_per_100g": _macro_amount(nutrients, {"protein"}),
            "fat_g_per_100g": _macro_amount(nutrients, {"fat", "lipid"}),
            "carbohydrate_g_per_100g": _macro_amount(
                nutrients, {"carbohydrate", "carbs", "carb"}
            ),
            "label_energy_kcal": label_energy,
            "label_serving": label_serving,
            "portions": _normalize_portions(details.get("foodPortions", [])),
            "food_url": food_url(fdc_id),
        }

    def validate_kcal(
        self,
        food_name: str,
        claimed_kcal: float,
        serving_size: Optional[str] = None,
        unit: str = "100g",
        tolerance_percent: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Validate a kcal claim on a metric/per-100-gram basis.

        Serving-count wording such as ``"1 medium"`` is deliberately not
        resolved through USDA portion descriptions. Those descriptions vary
        by record and reintroduce the ambiguity this standard is meant to
        eliminate. An explicit metric mass is converted algebraically to
        kcal per 100 grams; any other non-per-100-gram claim is returned as
        unresolved rather than accepted.

        Args:
            food_name: Food named by the product.
            claimed_kcal: Claimed positive energy value, in kcal.
            serving_size: Serving text from the product. Only an explicit
                mass (for example ``"200 g"``) makes a non-per-100-gram
                value comparable.
            unit: ``"100g"`` for a value already stated per 100 grams,
                ``"metric_serving"`` for a value accompanied by an explicit
                metric mass, ``"per_volume"`` for a value stated per volume,
                or ``"per_serving"`` for a countable/household serving.
            tolerance_percent: Optional override for the client's default
                rounding tolerance.

        Returns:
            Validation record. ``valid`` is False both for a measured
            deviation above tolerance and for an unresolvable serving; the
            latter carries ``usda_kcal=None`` and an explanatory ``error``
            instead of inventing a comparison.
        """
        claimed = _positive_float(claimed_kcal, "claimed_kcal")
        tolerance = (
            self.tolerance_percent
            if tolerance_percent is None
            else _non_negative_float(tolerance_percent, "tolerance_percent")
        )
        normalized_unit = unit.strip().lower().replace("_", " ") if isinstance(
            unit, str
        ) else ""
        if normalized_unit in {"100g", "100 g", "per 100g", "per 100 g"}:
            normalized_unit = "100g"
        elif normalized_unit in {"per serving", "per serving.", "serving"}:
            normalized_unit = "per_serving"
        elif normalized_unit in {"metric serving", "metric", "explicit mass"}:
            normalized_unit = "metric_serving"
        elif normalized_unit in {
            "per volume",
            "per_volume",
            "per 100ml",
            "per 100 ml",
        }:
            normalized_unit = "per_volume"
        else:
            raise ValueError(
                "unit must be '100g', 'metric_serving', 'per_volume', or "
                f"'per_serving', got {unit!r}."
            )
        serving_text = (
            serving_size.strip()
            if isinstance(serving_size, str) and serving_size.strip()
            else None
        )

        candidates = self.search_food(_translate_food_name(food_name))
        best, match_score = _select_best_match(
            _translate_food_name(food_name), candidates
        )
        details = self.get_nutrients(best["fdc_id"])
        serving_grams = _explicit_mass_grams(serving_text)
        base = {
            "food": food_name.strip(),
            "usda_query": _translate_food_name(food_name),
            "serving": serving_text,
            "unit": "per_100g" if normalized_unit == "100g" else normalized_unit,
            "claimed_kcal": round(claimed, 2),
            "tolerance_percent": round(tolerance, 2),
            "fdc_id": best["fdc_id"],
            "food_name": best["description"],
            "data_type": best["data_type"],
            "match_score": round(match_score, 3),
            "food_url": details["food_url"],
            "energy_kcal_per_100g": details["energy_kcal_per_100g"],
            "serving_grams": serving_grams,
            "serving_basis": (
                "explicit serving mass" if serving_grams is not None else None
            ),
        }
        if normalized_unit == "100g" and serving_grams is None and (
            serving_text is None or _is_per_100g_serving(serving_text)
        ):
            claimed_per_100g = claimed
        elif serving_grams:
            claimed_per_100g = claimed * 100.0 / serving_grams
        elif normalized_unit == "per_volume" or _is_per_volume_serving(
            serving_text
        ):
            return {
                **base,
                "claimed_per_100g_kcal": None,
                "valid": False,
                "usda_kcal": None,
                "deviation_percent": None,
                "error": (
                    "Per-volume claims cannot be compared with USDA "
                    "per-100-gram values without a food-specific density. "
                    "Restate the claim per 100 g or give the serving's "
                    "metric mass."
                ),
            }
        else:
            return {
                **base,
                "claimed_per_100g_kcal": None,
                "valid": False,
                "usda_kcal": None,
                "deviation_percent": None,
                "error": (
                    "Non-metric serving cannot be compared without assuming "
                    "its gram weight. Restate the claim per 100 g or with "
                    "an explicit metric mass."
                ),
            }

        usda_kcal = details["energy_kcal_per_100g"]
        if usda_kcal > 0:
            deviation = abs(claimed_per_100g - usda_kcal) / usda_kcal * 100.0
        elif claimed_per_100g == 0:
            deviation = 0.0
        else:
            deviation = 100.0
        return {
            **base,
            "claimed_per_100g_kcal": round(claimed_per_100g, 2),
            "valid": bool(deviation <= tolerance),
            "usda_kcal": round(usda_kcal, 2),
            "deviation_percent": round(deviation, 2),
            "error": None,
        }

    # -- transport ---------------------------------------------------------

    def _require_api_key(self) -> None:
        """Require a configured key without ever exposing its value."""
        if not self.has_api_key:
            raise EnvironmentError(
                "USDA_API_KEY is not configured. Set USDA_API_KEY before "
                "calling USDA network methods."
            )

    def _request(
        self,
        method: str,
        url: str,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send one USDA request and return its decoded JSON object."""
        try:
            if method == "GET":
                response = self.session.get(
                    url,
                    params={"api_key": self.api_key},
                    timeout=self.timeout,
                )
            elif method == "POST":
                response = self.session.post(
                    url,
                    params={"api_key": self.api_key},
                    json=json_body or {},
                    timeout=self.timeout,
                )
            else:
                raise ValueError(f"Unsupported USDA method {method!r}.")
        except requests.RequestException as exc:
            raise RuntimeError(f"USDA request failed: {type(exc).__name__}.") from exc
        if response.status_code == 429:
            retry_after = _retry_after_seconds(response.headers)
            wait_text = (
                f" Retry after {retry_after:,} seconds."
                if retry_after is not None
                else ""
            )
            raise RuntimeError(
                "USDA rate limit exceeded (HTTP 429); retry later or use a "
                f"non-demo API key.{wait_text}"
            )
        if response.status_code == 403:
            raise RuntimeError(
                "USDA rejected the API key (HTTP 403)."
            )
        if response.status_code == 404:
            raise LookupError(f"USDA resource not found: {url}.")
        if not response.ok:
            body = (response.text or "")[:200].replace(self.api_key, "[redacted]")
            raise RuntimeError(
                f"USDA request failed with HTTP {response.status_code}: {body}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("USDA returned malformed JSON.") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("USDA returned an unexpected JSON shape.")
        return payload


def food_url(fdc_id: int) -> str:
    """Return the public FoodData Central detail URL for an FDC ID."""
    return (
        "https://fdc.nal.usda.gov/fdc-app.html#/food-details/"
        f"{int(fdc_id)}/nutrients"
    )


def _retry_after_seconds(headers: Any) -> Optional[int]:
    """Parse an HTTP Retry-After delay, if the provider supplies one."""
    if not isinstance(headers, dict):
        try:
            value = headers.get("Retry-After")
        except AttributeError:
            return None
    else:
        value = headers.get("Retry-After")
    try:
        delay = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return delay if delay >= 0 else None


def _string_list(value: Any) -> List[str]:
    """Coerce USDA name/description fields to a compact string list."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()][:10]
    return []


def _singularize_token(word: str) -> str:
    """Reduce a simple English plural to its singular for food matching."""
    if len(word) <= 3:
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("es"):
        stem = word[:-2]
        if stem.endswith(("s", "x", "z", "ch", "sh", "o")):
            return stem
        return word[:-1]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _translate_food_name(food_name: str) -> str:
    """Translate a supported non-English food name for USDA lookup.

    This changes query wording only. USDA remains the sole source of every
    nutritional value; no alias supplies energy or serving data.
    """
    normalized = _normalize_food_name(food_name)
    return DUTCH_FOOD_ALIASES.get(normalized, food_name.strip())


def _normalize_food_name(value: str) -> str:
    """Normalize a food name for exact and similarity matching."""
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [_singularize_token(token) for token in text.split()]
    return re.sub(r"\s+", " ", " ".join(tokens)).strip()


def _candidate_score(query: str, candidate: Dict[str, Any]) -> float:
    """Score one candidate without using any claimed nutritional value."""
    candidate_text = _normalize_food_name(candidate["description"])
    similarity = SequenceMatcher(None, query, candidate_text).ratio()
    weight = _DATA_TYPE_PRIORITY.get(candidate["data_type"], 0.75)
    extraneous = [
        token
        for token in candidate_text.split()
        if token and token not in query.split()
    ]
    penalty = sum(
        0.0 if token in _GENERIC_STATE_WORDS else 0.12 for token in extraneous
    )
    return max(0.0, similarity * weight - penalty)


def _select_best_match(
    food_name: str, candidates: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], float]:
    """Choose the candidate closest to the product's food wording.

    Exact normalized-description matches win; otherwise description
    similarity is weighted toward generic USDA sources over branded recipes.
    A low score raises instead of silently accepting a poor match.
    """
    query = _normalize_food_name(food_name)
    exact = [
        candidate
        for candidate in candidates
        if _normalize_food_name(candidate["description"]) == query
    ]
    if exact:
        exact.sort(
            key=lambda candidate: (
                -_DATA_TYPE_PRIORITY.get(candidate["data_type"], 0.75),
                candidate["fdc_id"],
            )
        )
        return exact[0], 1.0
    scored = [
        (_candidate_score(query, candidate), candidate)
        for candidate in candidates
    ]
    scored.sort(key=lambda item: (-item[0], item[1]["fdc_id"]))
    score, best = scored[0]
    if score < 0.45:
        raise LookupError(
            f"USDA has no reliable match for {food_name!r}; best candidate "
            f"was {best['description']!r}."
        )
    return best, score


def _nutrient_values(nutrients: List[Any]) -> List[Tuple[str, str, float]]:
    """Normalize FDC's two nutrient payload shapes to name/unit/amount."""
    values = []
    for nutrient in nutrients:
        if not isinstance(nutrient, dict):
            continue
        node = nutrient.get("nutrient")
        node = node if isinstance(node, dict) else {}
        nutrient_id = node.get("id", nutrient.get("nutrientId"))
        number = str(
            node.get("number", nutrient.get("nutrientNumber", "") or "")
        ).strip()
        name = str(
            node.get("name", nutrient.get("nutrientName", "") or "")
        ).strip()
        unit = str(
            node.get("unitName", nutrient.get("unitName", "") or "")
        ).strip()
        amount = nutrient.get("amount", nutrient.get("value"))
        try:
            amount_number = float(amount)
        except (TypeError, ValueError):
            continue
        # Keep identifiers alongside the human-readable fields: energy and
        # macros are matched by either their names or USDA numbers below.
        values.append(
            (
                f"{nutrient_id}|{number}|{name}".lower(),
                unit.lower(),
                amount_number,
            )
        )
    return values


def _energy_kcal_per_100_grams(
    nutrients: List[Any],
) -> Tuple[float, str]:
    """Extract reference energy in kcal per 100 grams.

    FDC detail/search nutrient amounts use the conventional per-100-gram
    basis. Kilojoules are converted rather than compared as if they were
    kcal—the classic unit error this validator exists to prevent.
    """
    kilojoules: Optional[float] = None
    for identity, unit, amount in _nutrient_values(nutrients):
        if "energy" not in identity and "|208|" not in identity:
            continue
        if unit in {"kcal", "cal"}:
            return float(amount), "energy_kcal_per_100g"
        if unit == "kj":
            kilojoules = float(amount)
    if kilojoules is not None:
        return round(kilojoules / 4.184, 2), "energy_kj_per_100g_converted"
    raise LookupError("USDA food has no energy value.")


def _macro_amount(nutrients: List[Any], names: set) -> Optional[float]:
    """Return the first matching macro amount, or None when absent."""
    for identity, _unit, amount in _nutrient_values(nutrients):
        if any(name in identity for name in names):
            return float(amount)
    return None


def _label_energy(
    nutrients: List[Any], details: Dict[str, Any]
) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
    """Extract branded label energy and its serving, when USDA supplies both."""
    label = details.get("labelNutrients")
    if not isinstance(label, dict):
        return None, None
    energy = label.get("energy")
    if not isinstance(energy, dict):
        return None, None
    try:
        value = float(energy.get("value"))
    except (TypeError, ValueError):
        return None, None
    serving_size = details.get("servingSize")
    serving_unit = str(details.get("servingSizeUnit", "") or "").strip()
    try:
        serving_amount = float(serving_size)
    except (TypeError, ValueError):
        return None, None
    grams = _mass_to_grams(serving_amount, serving_unit)
    if grams is None:
        return None, None
    return round(value, 2), {
        "grams": grams,
        "text": f"{serving_amount:g} {serving_unit}".strip(),
    }


def _normalize_portions(value: Any) -> List[Dict[str, Any]]:
    """Normalize USDA portion gram weights and descriptions.

    Newer FDC records use ``portionDescription``, while SR Legacy records
    commonly use ``modifier`` instead. Foundation records may supply only a
    measure unit. All three shapes are normalized so serving matching does
    not depend on the record's age.
    """
    portions = []
    if not isinstance(value, list):
        return portions
    for portion in value:
        if not isinstance(portion, dict):
            continue
        try:
            grams = float(portion.get("gramWeight"))
        except (TypeError, ValueError):
            continue
        if grams <= 0:
            continue
        description = str(
            portion.get("portionDescription")
            or portion.get("modifier")
            or ""
        ).strip()
        if not description:
            unit = portion.get("measureUnit")
            unit_name = ""
            if isinstance(unit, dict):
                unit_name = str(
                    unit.get("name") or unit.get("abbreviation") or ""
                ).strip()
            amount = portion.get("amount")
            try:
                amount_number = float(amount)
            except (TypeError, ValueError):
                amount_number = None
            if unit_name and amount_number is not None:
                description = f"{amount_number:g} {unit_name}"
            elif unit_name:
                description = unit_name
        if not description:
            continue
        portions.append({"description": description, "grams": grams})
    return portions


def _mass_to_grams(amount: float, unit: str) -> Optional[float]:
    """Convert an explicit mass quantity to grams."""
    factor = _MASS_TO_GRAMS.get((unit or "").strip().lower())
    if factor is None:
        return None
    return float(amount) * factor


def _parse_number_token(token: str) -> Optional[float]:
    """Parse decimals, fractions, and quantity words as a number."""
    cleaned = token.strip().lower()
    if cleaned in _NUMBER_WORDS:
        return _NUMBER_WORDS[cleaned]
    if cleaned in _FRACTION_CHARS:
        return _FRACTION_CHARS[cleaned]
    if re.fullmatch(r"\d+(?:\.\d+)?", cleaned):
        return float(cleaned)
    fraction = re.fullmatch(r"(\d+)\s*/\s*(\d+)", cleaned)
    if fraction and float(fraction.group(2)) != 0:
        return float(fraction.group(1)) / float(fraction.group(2))
    return None


def _parse_quantity(text: str) -> Tuple[Optional[float], str]:
    """Split serving text into an optional leading amount and descriptor."""
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return None, ""
    mixed = re.match(
        r"^\s*(\d+)\s+(\d+\s*/\s*\d+|[¼½¾⅓⅔⅛])\s*(.*)$", cleaned
    )
    if mixed:
        whole = float(mixed.group(1))
        fraction = _parse_number_token(mixed.group(2)) or 0.0
        return whole + fraction, mixed.group(3).strip()
    first, _, rest = cleaned.partition(" ")
    amount = _parse_number_token(first)
    if amount is not None:
        return amount, rest.strip()
    return None, cleaned


def _explicit_mass_grams(serving_text: Optional[str]) -> Optional[float]:
    """Convert only an explicit mass serving to grams.

    Countable servings (``"1 medium"``), volumes (``"100 ml"``), and package
    descriptions without a mass are intentionally rejected: converting them
    would require a density or portion assumption.
    """
    if not serving_text or not serving_text.strip():
        return None
    text = re.sub(r"\s+", " ", serving_text.strip())
    parenthetical = re.search(
        r"\(\s*(\d+(?:\.\d+)?)\s*"
        r"(mg|g|kg)s?\s*\)",
        text,
        re.IGNORECASE,
    )
    if parenthetical:
        grams = _mass_to_grams(
            float(parenthetical.group(1)), parenthetical.group(2)
        )
        return grams if grams is not None and grams > 0 else None

    amount, remainder = _parse_quantity(text)
    if amount is not None:
        multiplier_match = re.match(
            r"^\s*[x×]\s*(\d+(?:\.\d+)?)\s*"
            r"(mg|g|kg)s?\s*$",
            remainder,
            re.IGNORECASE,
        )
        if multiplier_match:
            grams = _mass_to_grams(
                amount * float(multiplier_match.group(1)),
                multiplier_match.group(2),
            )
            return grams if grams is not None and grams > 0 else None
        unit_match = re.fullmatch(
            r"(mg|g|kg)s?",
            remainder,
            re.IGNORECASE,
        )
        if unit_match:
            grams = _mass_to_grams(amount, unit_match.group(1))
            return grams if grams is not None and grams > 0 else None
        return None

    explicit = re.match(
        r"^\s*(\d+(?:\.\d+)?)\s*"
        r"(mg|g|kg)s?\s*$",
        text,
        re.IGNORECASE,
    )
    if explicit:
        grams = _mass_to_grams(float(explicit.group(1)), explicit.group(2))
        return grams if grams is not None and grams > 0 else None
    return None


def _is_per_volume_serving(serving_text: Optional[str]) -> bool:
    """True when serving text is a volume rather than a mass."""
    return bool(
        serving_text
        and re.search(
            r"\b\d+(?:\.\d+)?\s*(?:ml|l|fl\s*oz|cups?|tbsp|tsp)\b",
            serving_text,
            re.IGNORECASE,
        )
    )


def _is_per_100g_serving(serving_text: str) -> bool:
    """True when serving text already means a 100-gram reference amount."""
    return bool(
        re.match(
            r"^\s*(?:per\s+)?100\s*g(?:rams?)?(?:\s+servings?)?\s*$",
            serving_text,
            re.IGNORECASE,
        )
    )


def _positive_float(value: Any, name: str) -> float:
    """Coerce a claimed numeric value to a positive float."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got {value!r}.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}.") from exc
    if number != number or number == float("inf") or number <= 0:
        raise ValueError(f"{name} must be a positive finite number.")
    return number


def _non_negative_float(value: Any, name: str) -> float:
    """Coerce a tolerance to a non-negative finite float."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got {value!r}.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}.") from exc
    if number != number or number == float("inf") or number < 0:
        raise ValueError(f"{name} must be a non-negative finite number.")
    return number
