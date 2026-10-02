"""Section definitions: the single source of truth for the whole swarm.

Both :mod:`agents.builder` and :mod:`agents.reviewer` (and
:mod:`build_pipeline`) import from here, so a section can never be defined
twice with drifting names — the exact configuration drift that sank Maiden
Voyage 2.0, where the reviewer searched for ``protocol`` while the builder
wrote the heading ``Assignment Notes``.

Each definition carries three distinct concepts:

* ``id`` — the stable machine key used by state, logs and prompts.
* ``heading`` — the Markdown ``##`` heading the Builder writes.
* ``keywords`` — every accepted spelling the Reviewer will look for. The
  first entry is the *primary* keyword and should appear in reviewer
  prompts; the rest are tolerated variants.

The reviewer's completeness check passes when **any** keyword (including the
heading) is found, so a template that says "Protocol", "Assignment Notes",
or "Assignment protocol notes" all satisfy the same requirement.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

__all__ = [
    "SECTION_DEFINITIONS",
    "SECTION_IDS",
    "SECTION_HEADINGS",
    "section_by_id",
    "section_keywords",
    "reviewer_section_specs",
    "required_section_keywords",
    "section_requirement_lines",
    "section_guidance_by_heading",
    "normalize_required_sections",
]

#: Canonical section set. Order defines assembly order on disk.
SECTION_DEFINITIONS: Tuple[Dict[str, object], ...] = (
    {
        "id": "appointment_grid",
        "heading": "Weekly Appointment Grid",
        "keywords": ("appointment grid", "appointment", "weekly schedule"),
        "requirement": (
            "Weekly appointment grid with time zone support — day/time "
            "slots, facility, department, room and a time zone column"
        ),
    },
    {
        "id": "client_database",
        "heading": "Client Database",
        "keywords": ("client database", "facility database", "facility",
                     "client"),
        "requirement": (
            "Client/facility database — facility name, address, contact, "
            "specialty, billing reference and a standing-notes field"
        ),
    },
    {
        "id": "medical_terminology",
        "heading": "Medical Terminology",
        "keywords": ("medical terminology", "terminology", "glossary"),
        "requirement": (
            "Medical terminology quick-reference by specialty — bilingual "
            "term pairs grouped by specialty with a plain-language meaning"
        ),
        "guidance": (
            "Write as a definitive desk reference, not a cautious essay: "
            "each term gets a confident plain-language definition plus its "
            "standard meaning, normal range, or canonical use where one "
            "exists (e.g. 'Blood pressure: the force of blood against "
            "artery walls. Normal range: 90/60 to 120/80 mmHg.'). "
            "One short section-level note that examples are illustrative "
            "is enough — do not hedge, qualify, or disclaim individual "
            "rows."
        ),
    },
    {
        "id": "invoice_tracking",
        "heading": "Invoice Tracking",
        "keywords": ("invoice tracking", "invoice", "hours tracking",
                     "billing"),
        "requirement": (
            "Invoice/hours tracking per assignment — date, facility, hours, "
            "rate, amount, payment status and a running total"
        ),
    },
    {
        "id": "certification_deadlines",
        "heading": "Certification Deadlines",
        "keywords": ("certification deadline", "certification", "deadline",
                     "ceu"),
        "requirement": (
            "Certification deadline tracker — credential, CEU requirement, "
            "progress, expiry date and next renewal action"
        ),
    },
    {
        "id": "protocol",
        "heading": "Assignment Notes",
        "keywords": ("protocol", "assignment notes", "assignment", "protocols"),
        "requirement": (
            "Assignment protocol notes — pre-assignment confirmation, site "
            "arrival or remote-login checklist, privacy rules, encounter "
            "boundaries and escalation procedure"
        ),
    },
)

#: Stable ids, in order.
SECTION_IDS: Tuple[str, ...] = tuple(
    str(d["id"]) for d in SECTION_DEFINITIONS
)

#: Markdown headings, in order.
SECTION_HEADINGS: Tuple[str, ...] = tuple(
    str(d["heading"]) for d in SECTION_DEFINITIONS
)


def section_by_id(section_id: str) -> Dict[str, object]:
    """Return the definition for ``section_id``.

    Args:
        section_id: A key from :data:`SECTION_IDS`.

    Returns:
        The matching definition dict.

    Raises:
        KeyError: If the id is unknown.
    """
    for definition in SECTION_DEFINITIONS:
        if definition["id"] == section_id:
            return definition
    raise KeyError(
        f"Unknown section id {section_id!r}; expected one of {list(SECTION_IDS)}."
    )


def section_keywords(section_id: str) -> Tuple[str, ...]:
    """Return the accepted keyword spellings for ``section_id``."""
    definition = section_by_id(section_id)
    return tuple(definition["keywords"])  # type: ignore[arg-type]


def _accepted_spellings(definition: Dict[str, object]) -> Tuple[str, ...]:
    """Return the heading plus every keyword, de-duplicated, order-stable.

    The heading is itself an accepted spelling (the Builder writes it), which
    is what closes the ``protocol`` vs ``Assignment Notes`` gap.
    """
    ordered: List[str] = []
    for candidate in [str(definition["heading"]).lower(),
                      *definition["keywords"]]:  # type: ignore[misc]
        if candidate and candidate not in ordered:
            ordered.append(candidate)
    return tuple(ordered)


def reviewer_section_specs() -> List[Tuple[str, Tuple[str, ...]]]:
    """Return ``[(label, accepted_keywords), ...]`` for the reviewer's checks.

    The label is the human-facing requirement text, so reviewer feedback
    quotes what the section is *for* rather than a bare keyword.
    """
    return [
        (str(d["requirement"]), _accepted_spellings(d))
        for d in SECTION_DEFINITIONS
    ]


def required_section_keywords() -> Dict[str, Tuple[str, ...]]:
    """Return ``{section_id: accepted_keywords}`` for completeness checks."""
    return {
        str(d["id"]): _accepted_spellings(d)
        for d in SECTION_DEFINITIONS
    }


def normalize_required_sections(
    required: object = None,
) -> Dict[str, Tuple[str, ...]]:
    """Coerce a reviewer's ``required_sections`` argument into id->keywords.

    The reviewer accepts three shapes so no caller can reintroduce drift:

    * ``None`` / empty -> the canonical :data:`SECTION_DEFINITIONS`.
    * A mapping ``{id: keyword, ...}`` (e.g. ``{"protocol": "protocol"}``) —
      values may be a single string or a sequence of alternatives. Ids that
      are not in :data:`SECTION_IDS` are rejected so a typo cannot silently
      weaken a check.
    * A sequence of bare keywords/headings -> each is looked up against the
      canonical set; unknown entries are kept as their own single-keyword
      requirement, and a known heading or id is expanded to all its
      accepted spellings.

    Args:
        required: The caller's ``required_sections`` value.

    Returns:
        Mapping of requirement id to the accepted keyword tuple.

    Raises:
        ValueError: If a mapping contains an unknown section id.
    """
    if not required:
        return required_section_keywords()

    result: Dict[str, Tuple[str, ...]] = {}
    if isinstance(required, dict):
        for key, value in required.items():
            if key not in SECTION_IDS:
                raise ValueError(
                    f"Unknown section id {key!r} in required_sections; "
                    f"expected one of {list(SECTION_IDS)}."
                )
            if isinstance(value, str):
                result[key] = (value.lower(),)
            else:
                result[key] = tuple(str(v).lower() for v in value)
        return result

    for entry in required:
        needle = str(entry).strip().lower()
        if not needle:
            continue
        if needle in SECTION_IDS:
            # An id must expand to the heading AND every keyword, otherwise
            # `["protocol"]` would silently drop the "assignment notes"
            # spelling and reintroduce the drift this module prevents.
            result[needle] = _accepted_spellings(section_by_id(needle))
            continue
        for definition in SECTION_DEFINITIONS:
            spellings = _accepted_spellings(definition)
            if needle in spellings:
                result[str(definition["id"])] = spellings
                break
        else:
            # Not part of the canonical set: honour it as its own check.
            result[needle] = (needle,)
    return result


def section_requirement_lines() -> str:
    """Render the numbered requirement list injected into build prompts."""
    lines = []
    for index, definition in enumerate(SECTION_DEFINITIONS, start=1):
        lines.append(f"{index}) {definition['requirement']} "
                     f"(## {definition['heading']})")
    return "\n".join(lines)


def section_guidance_by_heading(heading: str) -> str:
    """Return extra generation guidance for a section heading, if any.

    Most sections need none (the generic requirements suffice); sections
    with a history of systematic failure carry targeted instructions.
    Currently only ``Medical Terminology`` — whose hedging habit reads as
    incompleteness to every judge — has guidance.

    Args:
        heading: Section display heading (case-insensitive).

    Returns:
        The guidance string, or ``""`` when the section has none.
    """
    needle = (heading or "").strip().lower()
    for definition in SECTION_DEFINITIONS:
        if str(definition["heading"]).lower() == needle:
            return str(definition.get("guidance", ""))
    return ""
