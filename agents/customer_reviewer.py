"""CustomerReviewerAgent: evaluates products through a buyer's eyes.

Where :class:`ReviewerAgent` checks structural quality, the customer
reviewer role-plays a skeptical, non-technical buyer deciding whether to
spend money. It scores first impression, clarity, completeness, value for
money, and usability, then renders a purchase verdict: ``BUY``, ``PASS``,
or ``NEEDS IMPROVEMENT``.

The review runs through the standard provider cascade (Zen first,
then OpenRouter, then NIM fallback), so no single provider outage can kill
the customer gate the way the hard NIM preference did in Maiden Voyage 012.
The result records which model answered.

Example:
    ```python
    from agents.customer_reviewer import CustomerReviewerAgent
    from core.orchestrator import Orchestrator
    from core.state_manager import StateManager

    reviewer = CustomerReviewerAgent(Orchestrator(), StateManager("demo"))
    result = reviewer.review_as_customer(product_code, "My Planner", 15.0)
    print(result["verdict"], result["scores"]["overall"])
    ```
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from rich.console import Console

from core.orchestrator import Orchestrator
from core.state_manager import StateManager

__all__ = ["CustomerReviewerAgent", "CUSTOMER_SYSTEM_PROMPT", "VALID_VERDICTS"]

#: Role instructions positioning the model as a buyer, never a fixer.
CUSTOMER_SYSTEM_PROMPT: str = (
    "You are a skeptical but fair customer who is considering buying a "
    "digital product. You have no technical knowledge - you just want "
    "something that works and delivers value. Evaluate the product honestly. "
    "You are NOT a developer or reviewer who fixes things. You are a BUYER "
    "deciding whether to purchase. You are evaluating this as a BUYER. "
    "After your assessment, you must state exactly how much you would pay "
    "for this product if you saw it in a store. Be honest - if it's "
    "mediocre, say a lower price. If it's excellent, say a higher price."
)

#: Allowed verdicts (normalised to upper case).
VALID_VERDICTS: tuple = ("BUY", "PASS", "NEEDS IMPROVEMENT")

_console = Console()


class CustomerReviewerAgent:
    """Review products as a paying customer would.

    Args:
        orchestrator: The swarm's :class:`Orchestrator`. The review runs
            through the full provider cascade; ``model_used`` records which
            provider actually answered.
        state_manager: The project's :class:`StateManager` (usage is
            tracked automatically on every model call).
    """

    def __init__(
        self, orchestrator: Orchestrator, state_manager: StateManager
    ) -> None:
        # Duck-typed checks so test doubles work (swarm-wide convention).
        for name in ("ask_agent", "ask_agent_with_fallback", "ask_fallback_agent",
                     "build_context_prompt", "get_model_info"):
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

    def review_as_customer(
        self, product_content: str, product_name: str, price: float
    ) -> Dict[str, Any]:
        """Evaluate a product as a buyer and return verdict + scores.

        Args:
            product_content: Full template/code text under review.
            product_name: Human-readable product name.
            price: Price in euros the customer would pay.

        Returns:
            Dict with ``verdict`` (``BUY``/``PASS``/``NEEDS IMPROVEMENT``),
            ``scores`` (first_impression, clarity, completeness,
            value_for_money, usability, overall — ints 1–10),
            ``customer_feedback`` (honest 3–5 sentence review),
            ``would_recommend`` (bool), ``key_concerns`` and
            ``what_works`` (lists), ``willingness_to_pay`` (float euros,
            clamped to 5.0–25.0), plus ``model_used``.

        Raises:
            ValueError: On empty content/name or invalid price.
            RuntimeError: If the model call or response parsing fails.
        """
        if not isinstance(product_content, str) or not product_content.strip():
            raise ValueError("product_content must be a non-empty string.")
        cleaned_name = product_name.strip() if isinstance(product_name, str) else ""
        if not cleaned_name:
            raise ValueError("product_name must be a non-empty string.")
        if isinstance(price, bool) or not isinstance(price, (int, float)):
            raise ValueError(f"price must be a number, got {price!r}.")
        _console.print(
            f"[cyan]Customer review:[/cyan] [bold]{cleaned_name}[/bold] "
            f"(EUR {float(price):g})…"
        )

        _console.print(
            "[cyan]Customer review via provider cascade "
            "(Zen, then OpenRouter, then NIM).[/cyan]"
        )
        # FIX 1 + FIX 2: the customer sees the WHOLE product through the
        # cascade. The old code hard-preferred NIM (single point of failure
        # that killed MV012) and truncated to 12,000 chars mislabelled as
        # "Full product content" (same blindness fixed in the technical
        # reviewer). A ~27k-char product is ~7k tokens — comfortably inside
        # every cascade provider's context, so no excerpt is needed.
        content = product_content.strip()
        user_prompt = (
            f"Product name: {cleaned_name}\n"
            f"Price: EUR {float(price):g}\n\n"
            f"Full product content:\n{content}\n\n"
            "Evaluate as a buyer on these criteria:\n"
            "1. First Impression (1-10): Does it look professional?\n"
            "2. Clarity (1-10): Do I understand what this does within 30 seconds?\n"
            "3. Completeness (1-10): Does it deliver what was promised?\n"
            "4. Value for Money (1-10): "
            f"Would I pay EUR {float(price):g} for this?\n"
            "5. Usability (1-10): Can I use this immediately without extra help?\n"
            "6. Overall Verdict: BUY / PASS / NEEDS IMPROVEMENT\n"
            "7. Customer Feedback: 3-5 sentences of honest feedback as if "
            "writing a review.\n"
            "8. Willingness To Pay: answer this question - If you were a real "
            "customer looking at this product right now, how much would you "
            "be willing to pay for it in euros? Be realistic - consider the "
            "quality, completeness, and value it provides. Give a specific "
            "number between \u20ac5 and \u20ac25.\n\n"
            "Respond with VALID JSON ONLY, no commentary: "
            '{"verdict": "BUY", '
            '"scores": {"first_impression": 8, "clarity": 8, '
            '"completeness": 7, "value_for_money": 7, "usability": 8}, '
            '"customer_feedback": "...", "would_recommend": true, '
            '"key_concerns": ["..."], "what_works": ["..."], '
            '"willingness_to_pay": 12.5}'
        )
        try:
            # Cascade: Zen → OpenRouter → NIM. model_used records who
            # actually answered (see _normalise).
            raw = self.orchestrator.ask_agent_with_fallback(
                system_prompt=CUSTOMER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                state_manager=self.state_manager,
            )
            if isinstance(raw, dict):
                model_used = str(raw.get("model_used", "cascade"))
                raw = raw.get("content", "")
            else:
                model_used = "cascade"
            # Annotate with the concrete cascade provider (opencode-zen /
            # openrouter / nvidia-nim) so telemetry shows who served.
            provider = getattr(self.orchestrator, "last_provider", None)
            if provider:
                model_used = f"{model_used} ({provider})"
        except Exception as exc:
            raise RuntimeError(
                f"Customer review call failed for {cleaned_name!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        result = self._normalise(raw, model_used)
        _console.print(
            f"[{'green' if result['verdict'] == 'BUY' else 'yellow'}]"
            f"Customer verdict: {result['verdict']} "
            f"(overall {result['scores']['overall']}/10, via {model_used})[/]"
        )
        return result

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _normalise(raw: str, model_used: str) -> Dict[str, Any]:
        """Parse + normalise the model JSON into the result schema."""
        payload = _parse_json(raw)
        if not isinstance(payload, dict):
            raise RuntimeError("Customer review response was not a JSON object.")
        verdict = str(payload.get("verdict", "")).strip().upper()
        if verdict not in VALID_VERDICTS:
            raise RuntimeError(
                f"Customer review verdict {verdict!r} not in {list(VALID_VERDICTS)}."
            )
        raw_scores = payload.get("scores", {})
        scores = {
            key: _clamp_score((raw_scores or {}).get(key, 5))
            for key in ("first_impression", "clarity", "completeness",
                        "value_for_money", "usability")
        }
        scores["overall"] = round(sum(scores.values()) / len(scores))
        feedback = str(payload.get("customer_feedback", "")).strip()
        if not feedback:
            raise RuntimeError("Customer review provided no feedback text.")
        return {
            "verdict": verdict,
            "scores": scores,
            "customer_feedback": feedback[:2000],
            "would_recommend": verdict == "BUY" or bool(payload.get("would_recommend", False)),
            "key_concerns": _str_list(payload.get("key_concerns", [])),
            "what_works": _str_list(payload.get("what_works", [])),
            "willingness_to_pay": _clamp_price(payload.get("willingness_to_pay")),
            "model_used": model_used,
        }


def _clamp_price(value: Any, low: float = 5.0, high: float = 25.0) -> float:
    """Coerce to float euros and clamp into [low, high] (default 15.0)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 15.0
    if number != number:  # NaN guard (NaN never equals itself).
        return 15.0
    return round(max(low, min(high, number)), 2)


def _clamp_score(value: Any) -> int:
    """Coerce to int and clamp into 1–10 (default 5)."""
    try:
        return max(1, min(10, int(float(value))))
    except (TypeError, ValueError):
        return 5


def _str_list(values: Any) -> List[str]:
    """Coerce to a list of non-empty strings (caps at 20)."""
    if not isinstance(values, list):
        return []
    return [str(v).strip() for v in values if str(v).strip()][:20]


def _parse_json(raw: str) -> Any:
    """Parse model JSON tolerantly (strips code fences first)."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("Empty model response.")
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
