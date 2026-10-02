"""GrammarAgent: multi-model consensus language-quality checking.

The primary builder model can miss its own spelling, grammar, style, and
clarity mistakes. This agent therefore asks up to three independently
configured provider/model combinations to inspect the same text, clusters
near-duplicate findings, and only reports an issue when at least two
providers agree. Agreement is deliberately conservative: exact or
near-duplicate quoted text is required, because paraphrased findings from
different models are not evidence of the same defect.

The configured panel is:

* OpenCode Zen, using the model configured on the orchestrator;
* OpenRouter, using the orchestrator's primary model;
* NVIDIA NIM, using the dedicated language-judge model.

This intentionally does not use ``Orchestrator.ask_agent`` for the panel:
that method runs the Zen/OpenRouter/NIM cascade and therefore cannot force
the same check onto three different models. The training-feedback repair
does use ``ask_primary_agent`` because a single corrected document—not three
competing rewrites—is needed.

"Training" here means prompt- and context-based self-correction, not
gradient updates or persistent model-weight changes.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from rich.console import Console

from core.orchestrator import Orchestrator
from core.state_manager import StateManager

__all__ = [
    "GrammarAgent",
    "ERROR_TYPES",
    "MIN_MODELS_FOR_CONSENSUS",
    "SIMILARITY_THRESHOLD",
]

logger = logging.getLogger(__name__)
_console = Console()

#: Error categories accepted from every judging model.
ERROR_TYPES: tuple = ("spelling", "grammar", "style", "clarity")

#: Minimum number of distinct provider/model combinations that must report
#: substantially the same finding before it is treated as consensus.
MIN_MODELS_FOR_CONSENSUS: int = 2

#: Normalized-text similarity floor for grouping two findings. Exact matches
#: group first; this handles small quotation/punctuation differences without
#: treating two different sentences as one error.
SIMILARITY_THRESHOLD: float = 0.82

#: Long inputs are checked in independent chunks so one model call does not
#: have to return findings for an entire product at once. Calibrated to
#: ~2,000 characters by Maiden Voyage 015: 6,000-char chunks failed on every
#: provider (Zen stalled, OpenRouter returned empty, NIM returned truncated
#: non-JSON), while short inputs judged cleanly.
MAX_CHUNK_CHARS: int = 2000

#: Attempts per chunk per provider before that provider is marked failed.
#: Transient faults (stalls, empty responses, truncated JSON) often clear
#: on an immediate retry; a persistent failure still fails the provider
#: rather than the whole consensus run.
MAX_PROVIDER_ATTEMPTS: int = 2

#: Failure text retained per finding. Long prose is truncated because the
#: original document remains available to every later stage.
MAX_ERROR_TEXT_CHARS: int = 500

#: Cap on findings accepted from one provider response. A malformed response
#: containing hundreds of alleged errors is more likely to be invalid than
#: to describe a document that poorly.
MAX_ERRORS_PER_RESPONSE: int = 100

#: Output ceiling for one grammar judgment. Findings are compact JSON, not
#: rewritten prose.
MAX_RESPONSE_TOKENS: int = 4000

#: Deterministic judgments are wanted here; creativity would only add noise.
DIRECT_TEMPERATURE: float = 0.0

#: Dedicated NIM language judge. This deliberately does not reuse the
#: orchestrator's generic NIM fallback model: that slot can be configured for
#: another modality, while grammar checking always needs a text model.
NIM_GRAMMAR_MODEL: str = (
    os.getenv("NVIDIA_GRAMMAR_MODEL")
    or os.getenv("NVIDIA_REVIEWER_MODEL")
    or "nvidia/nemotron-3-ultra-550b-a55b"
).strip() or "nvidia/nemotron-3-ultra-550b-a55b"

#: NIM endpoint for the dedicated grammar-judge client.
NIM_GRAMMAR_BASE_URL: str = (
    os.getenv("NVIDIA_NIM_BASE_URL") or "https://integrate.api.nvidia.com/v1"
).strip().rstrip("/") or "https://integrate.api.nvidia.com/v1"

#: Generous direct-call timeout for the dedicated judge. This mirrors the
#: reviewer-agent policy: a language-judgment call should not be cut off by
#: the shorter cascade stall guards.
NIM_GRAMMAR_TIMEOUT: float = 600.0

#: Instructions shared by every independent judging call.
GRAMMAR_SYSTEM_PROMPT: str = (
    "You are a professional English grammar and style checker. "
    "Inspect only the supplied text; do not use outside knowledge, rewrite "
    "the whole document, or invent defects that are not quoted below."
)


class GrammarAgent:
    """Check language quality through independent multi-model agreement.

    Args:
        orchestrator: The swarm's :class:`Orchestrator`. Its already
            configured Zen, OpenRouter, and NIM credentials are reused; no
            new secrets are introduced.
        state_manager: The project's :class:`StateManager`. Successful model
            calls are usage-tracked, and grammar summaries are recorded.
        min_agreement: Distinct provider/model combinations required for
            consensus. Defaults to two.
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        state_manager: StateManager,
        min_agreement: int = MIN_MODELS_FOR_CONSENSUS,
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
        for name in ("get_state", "log_api_call", "log_grammar_check", "save"):
            if not hasattr(state_manager, name) or not callable(
                getattr(state_manager, name, None)
            ):
                raise TypeError(
                    "state_manager must expose the StateManager interface "
                    f"(missing callable {name!r}); "
                    f"got {type(state_manager).__name__}."
                )
        if (
            not isinstance(min_agreement, int)
            or isinstance(min_agreement, bool)
            or min_agreement < 1
        ):
            raise ValueError("min_agreement must be a positive integer.")

        self.orchestrator: Orchestrator = orchestrator
        self.state_manager: StateManager = state_manager
        self.min_agreement: int = min_agreement
        self.nim_model: str = NIM_GRAMMAR_MODEL
        self.nim_client: Optional[OpenAI] = None
        self._provider_targets: List[Dict[str, Any]] = []
        self._resolve_provider_targets()

    # -- public workflow ---------------------------------------------------

    def check_grammar(
        self, content: str, content_type: str = "product"
    ) -> Dict[str, Any]:
        """Run the independent-model consensus check and record the summary.

        Args:
            content: Text to inspect.
            content_type: Short label retained in the result and state
                record, for example ``"product"`` or ``"sample"``.

        Returns:
            Dict with ``errors`` and ``summary`` in the documented shape,
            plus provider/chunk metadata.

        Raises:
            ValueError: If ``content`` is empty.
            RuntimeError: If no provider returns a usable judgment.
        """
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string.")
        content_label = (
            content_type.strip()
            if isinstance(content_type, str) and content_type.strip()
            else "product"
        )
        _console.print(
            f"[cyan]Grammar consensus check ({content_label}, "
            f"{len(content):,} chars)…[/cyan]"
        )

        chunks = _chunk_content(content)
        provider_results, successful_providers = self._run_consensus_check(
            chunks, content_label
        )
        if successful_providers < 1:
            raise RuntimeError(
                "Grammar consensus failed: no provider returned a usable "
                "judgment."
            )
        consensus_errors = self._find_consensus(
            provider_results, successful_providers
        )
        summary = self._summarise(
            consensus_errors,
            provider_results,
            successful_providers,
            content_label,
            len(chunks),
        )
        record = {
            "content_type": content_label,
            "chunks": len(chunks),
            "providers": summary["providers"],
            "min_agreement": self.min_agreement,
            "summary": {
                "total_errors": summary["total_errors"],
                "spelling_errors": summary["spelling_errors"],
                "grammar_errors": summary["grammar_errors"],
                "style_errors": summary["style_errors"],
                "clarity_errors": summary.get("clarity_errors", 0),
                "consensus_rate": summary["consensus_rate"],
            },
            "errors": consensus_errors,
        }
        try:
            self.state_manager.log_grammar_check(record)
        except Exception as exc:
            logger.warning("Could not record grammar-check result: %s", exc)
            _console.print(
                f"[yellow]Could not record grammar-check result: {exc}[/yellow]"
            )
        return {"errors": consensus_errors, "summary": summary}

    def _training_feedback(
        self, original_content: str, consensus_errors: List[Dict[str, Any]]
    ) -> str:
        """Apply consensus findings through the primary repair model.

        This is refinement, not model training: the primary model receives
        the missed findings as explicit correction context and returns one
        corrected document.

        Args:
            original_content: Text the consensus panel inspected.
            consensus_errors: Consensus findings from :meth:`check_grammar`.

        Returns:
            The corrected document text.

        Raises:
            ValueError: If inputs are malformed.
            RuntimeError: If the repair model returns nothing.
        """
        if not isinstance(original_content, str) or not original_content.strip():
            raise ValueError("original_content must be a non-empty string.")
        if not isinstance(consensus_errors, list) or not all(
            isinstance(item, dict) for item in consensus_errors
        ):
            raise ValueError("consensus_errors must be a list of dicts.")
        if not consensus_errors:
            return original_content

        error_summary = "\n".join(
            f"- {error['type']}: '{error['original_text']}' → "
            f"'{error['suggestion']}' ({error['models_agreed']} models agreed)"
            for error in consensus_errors
        )
        try:
            corrected = self.orchestrator.ask_primary_agent(
                system_prompt=(
                    "You are correcting grammar errors in your own output. "
                    "Apply only the identified language fixes. Preserve the "
                    "meaning, document structure, code fences, tables, "
                    "placeholders, prices, and section order."
                ),
                user_prompt=(
                    "Your previous output had these grammar/style errors "
                    "caught by other models in a consensus check:\n\n"
                    f"{error_summary}\n\n"
                    "Review your output, apply these corrections, and identify "
                    "any similar errors you might have missed.\n\n"
                    "Original content:\n"
                    f"{original_content.strip()}\n\n"
                    "Return the corrected content only."
                ),
                state_manager=self.state_manager,
            )
        except Exception as exc:
            raise RuntimeError(
                "Grammar training-feedback call failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(corrected, str) or not corrected.strip():
            raise RuntimeError("Grammar training feedback returned no content.")
        return corrected.strip()

    # -- consensus machinery ------------------------------------------------

    def _resolve_provider_targets(self) -> None:
        """Select the configured direct-call provider/model combinations."""
        targets: List[Dict[str, Any]] = []
        zen = getattr(self.orchestrator, "zen", None)
        zen_model = getattr(zen, "model", None)
        if (
            getattr(zen, "enabled", False)
            and getattr(zen, "client", None) is not None
            and isinstance(zen_model, str)
            and zen_model.strip()
        ):
            targets.append(
                {"provider": "zen", "model": zen_model.strip(), "kind": "zen"}
            )

        openrouter_client = getattr(self.orchestrator, "client", None)
        openrouter_model = getattr(self.orchestrator, "model", None)
        if openrouter_client is not None and isinstance(
            openrouter_model, str
        ) and openrouter_model.strip():
            targets.append(
                {
                    "provider": "openrouter",
                    "model": openrouter_model.strip(),
                    "kind": "openrouter",
                }
            )

        nim_key = (
            os.getenv("NVIDIA_NIM_API_KEY")
            or os.getenv("NVIDIA_API_KEY")
            or ""
        ).strip()
        if nim_key:
            self.nim_client = OpenAI(
                api_key=nim_key,
                base_url=NIM_GRAMMAR_BASE_URL,
                timeout=NIM_GRAMMAR_TIMEOUT,
            )
            targets.append(
                {"provider": "nim", "model": self.nim_model, "kind": "nim"}
            )
            _console.print(
                "[green]Grammar NIM judge ready[/green] — "
                f"[bold]{self.nim_model}[/bold] (dedicated language client)"
            )
        else:
            _console.print(
                "[yellow]No NIM key — grammar consensus will use the "
                "remaining configured providers.[/yellow]"
            )

        if not targets:
            raise EnvironmentError(
                "GrammarAgent requires at least one configured provider: "
                "set OPENCODE_ZEN_API_KEY, OPENROUTER_API_KEY, or a NIM key."
            )
        self._provider_targets = targets
        if len(targets) < self.min_agreement:
            _console.print(
                "[yellow]Fewer providers are configured than the requested "
                f"agreement threshold ({self.min_agreement}); consensus "
                "errors will be impossible unless more providers respond.[/yellow]"
            )

    def _run_consensus_check(
        self, chunks: List[str], content_type: str
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Run every provider over every chunk; return usable results."""
        provider_results: List[Dict[str, Any]] = []
        successful_labels = set()
        for target in self._provider_targets:
            errors: List[Dict[str, Any]] = []
            status = "ok"
            failure: Optional[str] = None
            for index, chunk in enumerate(chunks):
                label = (
                    f"chunk {index + 1}/{len(chunks)}"
                    if len(chunks) > 1
                    else "full text"
                )
                last_error: Optional[BaseException] = None
                for attempt in range(1, MAX_PROVIDER_ATTEMPTS + 1):
                    try:
                        response = self._complete_with_provider(
                            target, chunk, content_type, label
                        )
                        errors.extend(
                            self._parse_errors(
                                response,
                                target["provider"],
                                target["model"],
                                label,
                                index,
                            )
                        )
                        last_error = None
                        break
                    except Exception as exc:
                        last_error = exc
                        if attempt < MAX_PROVIDER_ATTEMPTS:
                            _console.print(
                                f"[yellow]Grammar judge "
                                f"{target['provider']} {label} attempt "
                                f"{attempt} failed "
                                f"({type(exc).__name__}); retrying…[/yellow]"
                            )
                if last_error is not None:
                    status = "failed"
                    failure = (
                        f"{type(last_error).__name__}: {last_error}"
                    )
                    logger.warning(
                        "Grammar check failed on %s (%s): %s",
                        target["provider"],
                        target["model"],
                        last_error,
                    )
                    _console.print(
                        f"[yellow]Grammar check failed on "
                        f"{target['provider']}: {last_error}[/yellow]"
                    )
                    break
            if status == "ok":
                successful_labels.add(target["provider"])
            provider_results.append(
                {
                    "provider": target["provider"],
                    "model": target["model"],
                    "status": status,
                    "error": failure,
                    "errors": errors,
                }
            )
        return provider_results, len(successful_labels)

    def _complete_with_provider(
        self,
        target: Dict[str, Any],
        chunk: str,
        content_type: str,
        label: str,
    ) -> str:
        """Ask one provider/model directly and track successful usage."""
        messages = [
            {"role": "system", "content": GRAMMAR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _grammar_user_prompt(chunk, content_type, label),
            },
        ]
        kind = target["kind"]
        if kind == "zen":
            response = self.orchestrator.zen.stream_once(
                messages, f"grammar-{target['provider']}"
            )
        elif kind == "openrouter":
            response = _extract_chat_text(
                self.orchestrator.client.chat.completions.create(
                    model=target["model"],
                    messages=messages,
                    temperature=DIRECT_TEMPERATURE,
                    max_tokens=MAX_RESPONSE_TOKENS,
                    timeout=self.orchestrator.timeout,
                )
            )
        elif kind == "nim":
            if self.nim_client is None:
                raise RuntimeError("NIM grammar client is unavailable.")
            response = _extract_chat_text(
                self.nim_client.chat.completions.create(
                    model=target["model"],
                    messages=messages,
                    temperature=DIRECT_TEMPERATURE,
                    max_tokens=MAX_RESPONSE_TOKENS,
                    timeout=NIM_GRAMMAR_TIMEOUT,
                )
            )
        else:
            raise RuntimeError(f"Unknown grammar provider {kind!r}.")
        if not isinstance(response, str) or not response.strip():
            raise RuntimeError(
                f"Grammar judge {target['provider']} returned no content."
            )
        try:
            self.state_manager.log_api_call()
        except Exception as exc:
            raise RuntimeError(
                "Grammar judgment succeeded but usage logging failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return response.strip()

    def _parse_errors(
        self,
        response: str,
        provider: str,
        model: str,
        chunk_label: str,
        chunk_index: int,
    ) -> List[Dict[str, Any]]:
        """Parse one provider response into normalized candidate findings."""
        payload = _extract_json_payload(response)
        items = _coerce_error_list(payload)
        candidates: List[Dict[str, Any]] = []
        seen = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            error_type = str(item.get("type", "")).strip().lower()
            if error_type not in ERROR_TYPES:
                continue
            location = str(
                item.get("location", "") or chunk_label
            ).strip() or chunk_label
            if chunk_label != "full text":
                location = f"{location} [{chunk_label}]"
            original = str(item.get("original", "")).strip()
            suggestion = str(item.get("suggestion", "")).strip()
            explanation = str(item.get("explanation", "")).strip()
            if not original and not suggestion:
                continue
            key = (
                error_type,
                _normalize_text(original),
                _normalize_text(suggestion),
            )
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "type": error_type,
                    "location": location[:160],
                    "original_text": original[:MAX_ERROR_TEXT_CHARS],
                    "suggestion": suggestion[:MAX_ERROR_TEXT_CHARS],
                    "explanation": explanation[:MAX_ERROR_TEXT_CHARS],
                    "provider": provider,
                    "model": model,
                    "chunk": chunk_label,
                    "chunk_index": chunk_index,
                }
            )
            if len(candidates) >= MAX_ERRORS_PER_RESPONSE:
                break
        return candidates

    def _find_consensus(
        self, all_results: List[Dict[str, Any]], successful_providers: int
    ) -> List[Dict[str, Any]]:
        """Group near-duplicate findings and keep multi-model agreements."""
        groups: List[List[Dict[str, Any]]] = []
        for result in all_results:
            if result.get("status") != "ok":
                continue
            for candidate in result.get("errors", []):
                placed = False
                for group in groups:
                    if _errors_match(candidate, group[0]):
                        group.append(candidate)
                        placed = True
                        break
                if not placed:
                    groups.append([candidate])

        consensus: List[Dict[str, Any]] = []
        order = [target["provider"] for target in self._provider_targets]
        for group in groups:
            providers = sorted(
                {item["provider"] for item in group},
                key=lambda name: order.index(name) if name in order else 99,
            )
            if len(providers) < self.min_agreement or successful_providers < 1:
                continue
            error_type = _majority_value(
                [item["type"] for item in group], order=[]
            )
            original = max(group, key=lambda item: len(item["original_text"]))
            suggestion = max(group, key=lambda item: len(item["suggestion"]))
            explanation = max(group, key=lambda item: len(item["explanation"]))
            locations = sorted({item["location"] for item in group})
            chunk_index = min(item["chunk_index"] for item in group)
            consensus.append(
                {
                    "type": error_type,
                    "location": "; ".join(locations)[:240],
                    "original_text": original["original_text"],
                    "suggestion": suggestion["suggestion"],
                    "explanation": explanation["explanation"],
                    "models_agreed": len(providers),
                    "providers": providers,
                    "confidence": round(
                        len(providers) / successful_providers, 3
                    ),
                    "chunk_index": chunk_index,
                }
            )
        consensus.sort(key=lambda item: (item.pop("chunk_index"), item["type"]))
        for position, item in enumerate(consensus, start=1):
            item["id"] = f"grammar-{position:03d}"
        return consensus

    def _summarise(
        self,
        consensus_errors: List[Dict[str, Any]],
        provider_results: List[Dict[str, Any]],
        successful_providers: int,
        content_type: str,
        chunk_count: int,
    ) -> Dict[str, Any]:
        """Build the requested summary plus provider/chunk metadata."""
        counts = {error_type: 0 for error_type in ERROR_TYPES}
        for error in consensus_errors:
            counts[error["type"]] += 1
        consensus_rate = (
            round(
                sum(error["confidence"] for error in consensus_errors)
                / len(consensus_errors),
                3,
            )
            if consensus_errors
            else 1.0
        )
        providers = [
            {
                "provider": result["provider"],
                "model": result["model"],
                "status": result["status"],
                "error_count": len(result.get("errors", [])),
                "error": result.get("error"),
            }
            for result in provider_results
        ]
        eligible_judges = sorted(
            result["provider"]
            for result in provider_results
            if result.get("status") == "ok"
        )
        if successful_providers < 1:
            judgment = "no_judges"
        elif successful_providers < self.min_agreement:
            judgment = "insufficient_judges"
        else:
            judgment = "consensus"
        return {
            "content_type": content_type,
            "chunks": chunk_count,
            "providers_available": len(self._provider_targets),
            "providers_succeeded": successful_providers,
            "eligible_judges": eligible_judges,
            "judgment": judgment,
            "min_agreement": self.min_agreement,
            "total_errors": len(consensus_errors),
            "spelling_errors": counts["spelling"],
            "grammar_errors": counts["grammar"],
            "style_errors": counts["style"],
            "clarity_errors": counts["clarity"],
            "consensus_rate": consensus_rate,
            "providers": providers,
        }


def _grammar_user_prompt(chunk: str, content_type: str, label: str) -> str:
    """Build the judging prompt for one independent model call."""
    return (
        "You are a professional English grammar and style checker.\n\n"
        "Review the following content for:\n"
        "1. SPELLING errors\n"
        "2. GRAMMAR errors (subject-verb agreement, tense consistency)\n"
        "3. STYLE issues (awkward phrasing, unclear sentences)\n"
        "4. CLARITY problems (ambiguous statements)\n\n"
        "For each error found, return JSON:\n"
        "{\n"
        '  "type": "spelling|grammar|style|clarity",\n'
        '  "location": "where in the text",\n'
        '  "original": "the problematic text",\n'
        '  "suggestion": "how to fix it",\n'
        '  "explanation": "why this is wrong"\n'
        "}\n\n"
        "Respond with VALID JSON ONLY, no commentary: either a JSON array of "
        "those objects, or an object shaped like {\"errors\": [...]}. Quote "
        "exact substrings from the content. If there are no errors, return [].\n\n"
        f"Content type: {content_type}\n"
        f"Content scope: {label}\n\n"
        "Content to check:\n"
        f"{chunk.strip()}"
    )


def _chunk_content(content: str) -> List[str]:
    """Split long content into independently checkable paragraph chunks."""
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", content.strip())
        if paragraph.strip()
    ]
    chunks: List[str] = []
    current: List[str] = []
    current_length = 0
    for paragraph in paragraphs:
        pieces = [paragraph] if len(paragraph) <= MAX_CHUNK_CHARS else (
            _split_long_paragraph(paragraph)
        )
        for piece in pieces:
            if current and current_length + len(piece) + 2 > MAX_CHUNK_CHARS:
                chunks.append("\n\n".join(current))
                current = []
                current_length = 0
            current.append(piece)
            current_length += len(piece) + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks or [content.strip()]


def _split_long_paragraph(paragraph: str) -> List[str]:
    """Split one oversized paragraph at sentence, then word, boundaries."""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph.strip())
        if sentence.strip()
    ]
    pieces: List[str] = []
    current = ""
    for sentence in sentences or [paragraph.strip()]:
        words = sentence.split()
        text = " ".join(words)
        if len(text) <= MAX_CHUNK_CHARS:
            candidate = f"{current} {text}".strip()
            if len(candidate) <= MAX_CHUNK_CHARS:
                current = candidate
                continue
            if current:
                pieces.append(current)
            current = text
            continue
        if current:
            pieces.append(current)
            current = ""
        words = text.split()
        for start in range(0, len(words), 200):
            pieces.append(" ".join(words[start:start + 200]))
    if current:
        pieces.append(current)
    return pieces or [paragraph.strip()]


def _extract_chat_text(response: Any) -> str:
    """Return message text from an OpenAI-compatible chat response."""
    choices = getattr(response, "choices", None) or []
    message = getattr(choices[0], "message", None) if choices else None
    return (getattr(message, "content", None) or "").strip()


def _extract_json_payload(raw: str) -> Any:
    """Extract the first complete JSON array/object from model output."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("Empty model response.")
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    decoder = json.JSONDecoder()
    for start, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[start:])
        except ValueError:
            continue
        return payload
    try:
        return json.loads(text)
    except ValueError as exc:
        raise ValueError(f"Model response was not valid JSON: {exc}") from exc


def _coerce_error_list(payload: Any) -> List[Any]:
    """Accept either a JSON error array or an object containing one."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("errors", "findings", "issues", "corrections", "results"):
            if isinstance(payload.get(key), list):
                return payload[key]
        if any(key in payload for key in ("original", "suggestion", "type")):
            return [payload]
    return []


def _normalize_text(value: str) -> str:
    """Normalize quoted findings for duplicate/similarity comparison."""
    text = value.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^\w\s]", "", text)


def _errors_match(first: Dict[str, Any], second: Dict[str, Any]) -> bool:
    """True when two candidates describe substantially the same finding."""
    if first.get("provider") == second.get("provider"):
        return False
    if first.get("chunk") != second.get("chunk"):
        return False
    first_type = first.get("type", "")
    second_type = second.get("type", "")
    if first_type and second_type and first_type != second_type:
        return False
    first_text = _normalize_text(first.get("original_text", ""))
    second_text = _normalize_text(second.get("original_text", ""))
    if first_text and second_text:
        if len(first_text) >= 16 and (
            first_text in second_text or second_text in first_text
        ):
            return True
        return (
            difflib.SequenceMatcher(None, first_text, second_text).ratio()
            >= SIMILARITY_THRESHOLD
        )
    first_fix = _normalize_text(first.get("suggestion", ""))
    second_fix = _normalize_text(second.get("suggestion", ""))
    if first_fix and second_fix:
        return (
            difflib.SequenceMatcher(None, first_fix, second_fix).ratio()
            >= SIMILARITY_THRESHOLD
        )
    return False


def _majority_value(values: List[str], order: List[str]) -> str:
    """Return the most common value, preserving provider order on ties."""
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    ranked = sorted(
        counts.items(),
        key=lambda item: (
            -item[1],
            order.index(item[0]) if item[0] in order else 99,
            item[0],
        ),
    )
    return ranked[0][0]
