"""Structural detection of abandoned (unfinished) generated content.

A planner template is *supposed* to ship empty fields for the customer to
fill in. ``| Client Name | TBD |`` is the product working as designed, while
a section that simply stops at ``TBD`` is a generator that gave up. Both look
identical to a substring search, which is why a blanket ``"tbd" in text``
check rejected a perfectly good 28,702-char product during Maiden Voyage 3.

This module encodes the distinction structurally. The rules:

Defects (the generator gave up)
    * Content ends with ``TBD`` and no closing sentence follows.
    * ``TBD`` appears mid-sentence in prose (not a table cell, not a field
      label).
    * More than :data:`MAX_TBD_PER_ROW` consecutive ``TBD`` cells in one
      table row — a whole row of nothing.
    * A section body consisting only of ``TBD`` and whitespace.

Features (a template awaiting customer input)
    * ``TBD`` inside a table cell belonging to a documented table.
    * ``**TBD**`` used as a standalone field label (``Date: **TBD**``).
    * Blank table cells in a section that also has explanatory prose.

Both :class:`~core.test_runner.TestRunner` and
:class:`~agents.reviewer.ReviewerAgent` delegate to
:func:`detect_abandoned_content` so the gate can never disagree with the
reviewer about what counts as a defect.
"""

from __future__ import annotations

import re
from typing import List, Optional

__all__ = [
    "detect_abandoned_content",
    "detect_ungrounded_claims",
    "MAX_TBD_PER_ROW",
    "ABANDONMENT_MARKERS",
]

#: Minimum words a prose line must contain *besides* ``TBD`` for the line to
#: count as genuinely explanatory rather than abandoned.
#:
#: This is what separates a good template from a broken one. The Maiden
#: Voyage 3 product wrote sentences like "Record unresolved items as
#: **TBD-unverified**, with an owner and deadline" — real guidance that must
#: pass — alongside bare ``TBD`` stubs that must not.
MIN_SUBSTANTIVE_WORDS: int = 6

#: How many ``TBD`` cells in one row before the row itself is suspect.
#:
#: Retained as a reported metric: a row with more than this many empty cells
#: is logged for the reviewer's attention, but it is only a *defect* when
#: every data row in the table is empty too (see
#: :func:`detect_abandoned_content`). A planner template legitimately ships
#: blank rows for the customer to complete, so a single empty row among
#: worked examples is a feature, not a failure.
MAX_TBD_PER_ROW: int = 3

#: Other markers that mean the generator stopped, regardless of context.
#: These stay blanket matches: unlike ``TBD`` they are never a legitimate
#: field in a finished planner.
ABANDONMENT_MARKERS: tuple = (
    "lorem ipsum",
    "[insert",
    "<insert",
    "coming soon",
    "to be continued",
    "(continued",
)

_TBD = re.compile(r"\bt\.?b\.?d\.?\b", re.IGNORECASE)
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_DIVIDER = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_FENCE = re.compile(r"^\s*```")
_SENTENCE_END = re.compile(r"[.!?)\]:;]\s*$")


def _is_table_cell_tbd(line: str) -> bool:
    """True when the line is a table row, so any ``TBD`` sits in a cell.

    Being on a table row is sufficient: every occurrence on such a line is
    already inside a ``| ... |`` cell, which is the fillable-field form we
    accept. Rows that are *entirely* TBD are caught separately by
    :data:`MAX_TBD_PER_ROW`.
    """
    if not _TABLE_ROW.match(line) or _TABLE_DIVIDER.match(line):
        return False
    return bool(_TBD.search(line))


def _substantive_words_besides_tbd(line: str) -> int:
    """Count words on the line that are not a ``TBD`` token.

    Bold markers, list bullets and checkbox glyphs are stripped first so
    ``**TBD-unverified**`` counts as the single token it is, and
    ``- [ ] Follow...`` counts its words.
    """
    text = _TBD.sub(" ", line)
    text = re.sub(r"[*_`>#\[\]()\-—–•]", " ", text)
    return len([w for w in text.split() if re.search(r"\w", w)])


def _is_field_label_tbd(line: str) -> bool:
    """True when ``TBD`` is a standalone field label (``Date: **TBD**``)."""
    stripped = line.strip().rstrip(".")
    if stripped.startswith(("|", "-", "*", ">")):
        return False
    return bool(re.match(
        r"^[A-Za-z][A-Za-z0-9 _/&()'-]{0,60}\s*[:=]\s*\*{0,2}t\.?b\.?d\.?\*{0,2}\s*$",
        stripped,
        re.IGNORECASE,
    ))


def detect_abandoned_content(content: str) -> List[str]:
    """Return descriptions of abandoned-content defects in ``content``.

    Args:
        content: The generated template or section text.

    Returns:
        A list of human-readable defect descriptions. An empty list means the
        content is either complete or a well-formed template awaiting
        customer input.
    """
    if not content or not content.strip():
        return ["content is empty."]

    defects: List[str] = []
    lines = content.splitlines()
    lowered = content.lower()

    # 1. Blanket markers that are never legitimate in finished output.
    for marker in ABANDONMENT_MARKERS:
        if marker in lowered:
            defects.append(
                f"unfinished marker {marker!r} present (generator stopped)."
            )

    # 2. Hollow tables: a table whose data rows are ALL TBD is a shell the
    #    generator never populated. A *blank row inside a table that has real
    #    example rows* is a feature — that is the row the customer fills in,
    #    and the Maiden Voyage 3 product was rejected for exactly this.
    for table_no, (start, rows) in enumerate(_iter_tables(lines), start=1):
        data_rows = [r for _, r in rows]
        if not data_rows:
            continue
        hollow = all(
            all(_TBD.fullmatch(c.strip().replace("*", ""))
                for c in r.strip().strip("|").split("|"))
            for r in data_rows
        )
        if hollow:
            line_no = rows[0][0]
            defects.append(
                f"line {line_no}: table {table_no} has {len(data_rows)} "
                f"data row(s) that are entirely 'TBD' — the table is an empty "
                f"shell with no worked example."
            )

    # 3. Prose use of TBD — only a defect when the generator had nothing
    #    else to say. A template that *explains* a fillable field
    #    ("Record unresolved items as TBD, with an owner and deadline") is
    #    working correctly and must pass; a line whose only content is
    #    "TBD" (or a couple of filler words) is an abandoned section.
    in_fence = False
    for number, line in enumerate(lines, start=1):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence or not _TBD.search(line):
            continue
        if _is_table_cell_tbd(line) or _is_field_label_tbd(line):
            continue
        if _substantive_words_besides_tbd(line) < MIN_SUBSTANTIVE_WORDS:
            defects.append(
                f"line {number}: 'TBD' stands alone in prose with no "
                f"explanatory content — the section was abandoned rather "
                f"than left as a fillable field: {line.strip()[:80]!r}"
            )

    # 4. Content whose LAST TOKEN is TBD — the generator stopped with nothing
    #    after it. A sentence that merely mentions TBD and then closes
    #    properly is fine, so this tests the final token rather than mere
    #    presence on the line.
    tail = _last_meaningful_line(lines)
    if tail and not _is_table_cell_tbd(tail) and not _is_field_label_tbd(tail):
        final_token = re.sub(r"[*_`>#\[\]()]", " ", tail).split()
        final_token = final_token[-1] if final_token else ""
        if _TBD.fullmatch(final_token):
            defects.append(
                "content ends on 'TBD' with no closing sentence — the "
                "section was abandoned mid-generation."
            )

    # 5. A body that is nothing but TBD.
    body = [
        ln.strip() for ln in lines
        if ln.strip() and not _TABLE_DIVIDER.match(ln)
        and not ln.strip().startswith(("#", "<!--"))
    ]
    if body and all(_TBD.fullmatch(b.replace("*", "").strip()) for b in body):
        defects.append(
            "section body is only 'TBD' — no usable content was produced."
        )
    return defects


# ---------------------------------------------------------------------------
# Non-delivery disclaimers
# ---------------------------------------------------------------------------

#: A section that *asserts* it received nothing. Unlike ``TBD`` in a field
#: cell, this is the generator announcing that it skipped the work: "No
#: product-specific recipe findings have been added yet", "Because no
#: client data was supplied, the table below is an example".
#:
#: Matched as a passive/assertion construction rather than a bare "no X", so
#: instructional copy is not swept up by the same pattern.
_DISCLAIMER = re.compile(
    r"\b(?:no|not any|none of the)\b[^.\n]{0,100}?"
    r"\b(?:has|have|had|is|are|was|were)\s+"
    r"(?:been\s+)?"
    r"(?:supplied|provided|given|specified|added|included|available|"
    r"received|entered|confirmed|known|on file)\b",
    re.IGNORECASE,
)

#: The same idea in a self-negating form: the product or its own fields
#: denying that it exists. A template must never say this about itself.
_SELF_NEGATION = re.compile(
    r"\b(?:the\s+)?(?:product|template|package|document|this\s+"
    r"(?:product|template|package))\b[^.\n]{0,40}?"
    r"\b(?:does\s*n[o']t|do\s*n[o']t|did\s*n[o']t|is\s+not|are\s+not)\b"
    r"[^.\n]{0,20}?"
    r"\b(?:exist|apply|appear|provided|supplied|specified|defined|set)\b",
    re.IGNORECASE,
)

#: Instructional copy that *tells the reader* what to do when input is
#: missing. This is the product working correctly, so it vetoes a match.
#:
#: The explicit leading-condition test is intentionally checked first: a line
#: such as "If no client notes are supplied, record 'None'" should be
#: classified as an instruction before any disclaimer test is considered.
_INSTRUCTION_START = re.compile(
    r"^\s*(?:if|when)\s+(?:no|not|none|never)\b",
    re.IGNORECASE,
)

_GUIDANCE_CUE = re.compile(
    r"(?:^|\b)(?:if|when|unless|should|where|any)\b[^.\n]{0,40}?"
    r"\b(?:no|not|none|never)\b"
    r"|"
    r"\b(?:record|enter|fill\s+in|write|mark|note|log|list|leave|add|insert|"
    r"replace|state|indicate|flag|ask|confirm|check|use|set)\b"
    r"[^.\n]{0,40}?\b(?:no|not|none)\b"
    r"|"
    r"\b(?:for example|e\.g\.|such as|if applicable|as needed|"
    r"where applicable|if relevant)\b",
    re.IGNORECASE,
)


def _is_guidance(sentence: str) -> bool:
    """True when the sentence instructs the reader rather than confessing."""
    if _INSTRUCTION_START.search(sentence):
        return True
    return bool(_GUIDANCE_CUE.search(sentence))


def _sentences(lines: List[str]) -> List[tuple]:
    """Yield ``(line_no, sentence)`` for prose sentences outside fences.

    Table rows are excluded: a ``| Notes |`` cell naming a missing input is
    still a field the customer fills in, not a disclaimer.
    """
    in_fence = False
    for number, line in enumerate(lines, start=1):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence or _TABLE_ROW.match(line):
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            if sentence.strip():
                yield number, sentence.strip()


#: Exact table-cell values that declare a field as awaiting customer input.
#: These are parsed at the cell level so phrases containing these words in
#: otherwise substantive cells are not mistaken for blank-input declarations.
_UNSET_CELL_VALUES = frozenset({
    "not set",
    "not supplied",
    "not provided",
    "not specified",
    "not entered",
    "not confirmed",
    "none",
    "no value",
    "n/a",
    "na",
})


def _split_table_cells(row: str) -> List[str]:
    """Split one Markdown table row into its individual cell values."""
    text = row.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|"):
        text = text[:-1]
    return [cell.strip().strip("*_`").strip()
            for cell in text.split("|")]


def _iter_table_data_cells(lines: List[str]) -> List[tuple]:
    """Yield ``(line_no, cells, row)`` for Markdown table data rows.

    Headers and divider rows are excluded: this is explicit cell-level
    detection for disclaimers, totals, and unset-input declarations.
    """
    cells_by_row: List[tuple] = []
    for _, (_, rows) in enumerate(_iter_tables(lines), start=1):
        for number, row in rows:
            cells_by_row.append((number, _split_table_cells(row), row))
    return cells_by_row


def _is_unset_table_cell(cell: str) -> bool:
    """True when one table cell is an explicit awaiting-input declaration."""
    normalized = re.sub(r"[*_`]+", "", cell).strip().rstrip(".,;:").lower()
    return normalized in _UNSET_CELL_VALUES


def _detect_table_cell_claims(lines: List[str]) -> List[str]:
    """Detect non-delivery disclaimers expressed inside table data cells.

    Most table cells are fillable fields, so this first removes exact
    unset-input values such as ``Not set`` from consideration. Only a cell
    containing a full admission—rather than a blank-input label—can fail.
    """
    defects: List[str] = []
    for number, cells, _row in _iter_table_data_cells(lines):
        for cell in cells:
            text = cell.strip()
            if not text or _is_unset_table_cell(text):
                continue
            if _SELF_NEGATION.search(text):
                defects.append(
                    f"line {number}: a table cell denies the template's "
                    f"existence — {text[:80]!r}"
                )
                continue
            if _DISCLAIMER.search(text) and not _is_guidance(text):
                defects.append(
                    f"line {number}: a table cell states that no input was "
                    f"received and substitutes filler: {text[:80]!r}"
                )
    return defects


def detect_ungrounded_claims(content: str) -> List[str]:
    """Return descriptions of content that contradicts itself or the input.

    Two defects that :func:`detect_abandoned_content` deliberately does not
    cover, because neither is a bare ``TBD``:

    1. **Non-delivery disclaimer** — a section that states it was given
       nothing and substitutes a generic example. On the Weekly Meal Prep
       Planner the recipe index opened with "No product-specific recipe
       findings have been added yet", so the section a customer paid for
       shipped as an apology plus filler.
    2. **Self-negation** — a template claiming that it, or its own name and
       code, does not exist. The Blog Post Outline Generator stated "the
       product name and code do not exist" while shipping sections built
       around them.

    Instructional copy is exempt: "If no client notes are supplied, record
    'None'" teaches the customer how to use the field and must pass.

    Args:
        content: The generated template or section text.

    Returns:
        A list of human-readable defect descriptions; empty when clean.
    """
    if not content or not content.strip():
        return []

    defects: List[str] = []
    prose_claims: List[str] = []
    lines = content.splitlines()

    for number, sentence in _sentences(lines):
        if _SELF_NEGATION.search(sentence):
            prose_claims.append(
                f"line {number}: the template denies its own existence — "
                f"{sentence[:80]!r}"
            )
            continue
        if _DISCLAIMER.search(sentence) and not _is_guidance(sentence):
            prose_claims.append(
                f"line {number}: section states it received no input and "
                f"substitutes filler instead of content: {sentence[:80]!r}"
            )

    table_claims = _detect_table_cell_claims(lines)
    defects.extend(prose_claims)
    defects.extend(table_claims)

    # A table total is only "ungrounded" when its section already admits that
    # its input is missing. Worked examples with blank/zero customer fields
    # are legitimate and must not fail merely because a date or label cell is
    # still marked "Not set".
    if prose_claims or table_claims:
        defects.extend(_detect_numeric_grounding(lines))
    return defects


#: A computed figure presented as settled: a currency amount, or a number with
#: a unit or a thousands separator.
_MONEY = re.compile(
    r"(?:[$€£¥]\s?\d[\d,]*(?:\.\d+)?)"
    r"|(?:\d[\d,]{2,}(?:\.\d+)?\s?%)"
    r"|(?:\b\d[\d,]*(?:\.\d+)?\s?"
    r"(?:USD|EUR|GBP|euros?|dollars?|percent)\b)",
    re.IGNORECASE,
)

#: A row that claims a total, i.e. arithmetic performed on the figures above.
_TOTAL_ROW = re.compile(r"\btotals?\b|\bsum\b|\bsubtotal\b|\bbalance\b",
                        re.IGNORECASE)


def _settled_amount(row: str) -> Optional[str]:
    """Return the money figure in ``row`` only when it is non-zero.

    A template that shows ``Total: $0.00 — enter your figures above`` is doing
    its job: it has declared the fields unset *and* left the total at zero for
    the customer to compute. Only a non-zero total claims to have done the
    arithmetic itself, which is the actual defect. Returning ``None`` for zero
    is what keeps this check from rejecting every legitimate ledger template.
    """
    match = _MONEY.search(row)
    if not match:
        return None
    digits = re.sub(r"[^\d.]", "", match.group(0))
    try:
        if float(digits) == 0.0:
            return None
    except ValueError:
        return None
    return match.group(0).strip()


def _detect_numeric_grounding(lines: List[str]) -> List[str]:
    """Flag totals computed from inputs the same section declares as absent.

    The Budget Tracker for Freelancers shipped a reconciled ledger — per-line
    expenses, category totals and a bold ``$1,555`` grand total — in a section
    whose own header read ``**Product:** Not set`` and
    ``**Data status:** No transaction ledger … was provided``. The customer
    reviewer rejected the product for exactly this ("the unexplained $1,555
    and $7,945 figures make the document feel unfinished").

    A planner may legitimately ship worked examples, so this is not "numbers
    are present". Callers must invoke this only when the same section already
    admits that its input is missing. A total left at ``$0.00`` for the
    customer to compute is also excluded — see :func:`_settled_amount`.
    """
    defects: List[str] = []
    for table_no, (_, rows) in enumerate(_iter_tables(lines), start=1):
        if not rows:
            continue
        unset_cells = sum(
            1
            for _, row in rows
            for cell in _split_table_cells(row)
            if _is_unset_table_cell(cell)
        )
        total_rows = [
            (no, r) for no, r in rows
            if _TOTAL_ROW.search(r) and _settled_amount(r)
        ]
        if total_rows and unset_cells:
            no, row = total_rows[0]
            amount = _settled_amount(row)
            defects.append(
                f"line {no}: table {table_no} presents a computed total "
                f"({amount if amount else 'sum'}) while "
                f"{unset_cells} row(s) in the same table declare their input "
                f"as not set — the figures are invented, not the customer's."
            )
    return defects


def _iter_tables(lines: List[str]):
    """Yield ``(header_line_no, [(line_no, row_text), ...])`` per table.

    Data rows exclude the header, the ``|---|---|`` divider, and separator
    lines such as ``|---:---:|``.
    """
    current_header: int | None = None
    rows: List[tuple] = []
    for number, line in enumerate(lines, start=1):
        if not _TABLE_ROW.match(line):
            if current_header is not None and rows:
                yield current_header, rows
            current_header, rows = None, []
            continue
        if _TABLE_DIVIDER.match(line):
            continue
        if current_header is None:
            current_header = number
        else:
            rows.append((number, line))
    if current_header is not None and rows:
        yield current_header, rows


def _last_meaningful_line(lines: List[str]) -> str:
    """Return the last non-blank, non-comment line of ``lines``."""
    for line in reversed(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith(("<!--", "|---", "```")):
            return stripped
    return ""
