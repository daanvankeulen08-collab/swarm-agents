"""Telegram Notifier: build alerts for the agent swarm.

Sends build lifecycle notifications (started, complete, review needed,
publish-ready) to a Telegram chat via the ``python-telegram-bot`` library.
Designed for graceful degradation: missing credentials, a missing library,
or a Telegram outage only ever yield ``False`` — never an exception — so
pipelines keep running without notifications.

Credentials resolve in order: explicit constructor args, then the
``TELEGRAM_BOT_TOKEN`` / ``TELEGRAM_CHAT_ID`` environment variables
(``.env`` supported via python-dotenv).

Example:
    ```python
    from core.notifier import Notifier

    notifier = Notifier()  # reads TELEGRAM_* from env/.env
    notifier.notify_build_started("proj_1", "My Planner")
    ```
"""

from __future__ import annotations

import asyncio
import html
import threading
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from rich.console import Console

try:
    from telegram import Bot as _TelegramBot

    _TELEGRAM_INSTALLED = True
except ImportError:  # Library optional; Notifier degrades gracefully.
    _TelegramBot = None  # type: ignore[assignment]
    _TELEGRAM_INSTALLED = False

__all__ = ["Notifier", "TELEGRAM_AVAILABLE"]

#: True when python-telegram-bot is importable.
TELEGRAM_AVAILABLE: bool = _TELEGRAM_INSTALLED

_console = Console()


class Notifier:
    """Send Telegram build notifications with graceful degradation.

    Args:
        bot_token: Bot API token. Falls back to ``TELEGRAM_BOT_TOKEN``.
        chat_id: Target chat ID. Falls back to ``TELEGRAM_CHAT_ID``.

    When credentials or the library are missing, a warning is printed once
    and every method returns ``False``. Use :attr:`enabled` to check.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> None:
        load_dotenv()
        import os as _os

        self.bot_token: str = (bot_token or _os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
        raw_chat = chat_id if chat_id is not None else _os.getenv("TELEGRAM_CHAT_ID")
        self.chat_id: str = str(raw_chat).strip() if raw_chat is not None else ""
        self.enabled: bool = bool(self.bot_token and self.chat_id and _TELEGRAM_INSTALLED)
        if self.enabled:
            _console.print("[green]Telegram Notifier enabled.[/green]")
        else:
            reasons = []
            if not _TELEGRAM_INSTALLED:
                reasons.append("python-telegram-bot not installed")
            if not self.bot_token:
                reasons.append("TELEGRAM_BOT_TOKEN missing")
            if not self.chat_id:
                reasons.append("TELEGRAM_CHAT_ID missing")
            _console.print(
                f"[yellow]Telegram Notifier disabled ({'; '.join(reasons)}). "
                "Pipeline will run without notifications.[/yellow]"
            )

    # -- core send ----------------------------------------------------------

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send ``text`` to the configured chat.

        Args:
            text: Message body (already formatted by the caller).
            parse_mode: Telegram parse mode (default ``"HTML"``).

        Returns:
            True on delivery, False otherwise (disabled, network error,
            API error). Never raises.

        Fallback: pipeline messages interpolate exception text and model
        output that can contain ``<...>`` sequences (e.g. a charmap error
        literally says ``character maps to <undefined>``). If Telegram
        rejects the HTML parse, the message is retried as plain text so a
        formatting problem can never silence a status update.
        """
        if not self.enabled or not text:
            return False
        try:
            return self._deliver(text, parse_mode)
        except Exception as exc:
            if parse_mode and "parse entities" in str(exc).lower():
                _console.print(
                    "[yellow]Telegram HTML parse failed; "
                    "retrying as plain text.[/yellow]"
                )
                try:
                    return self._deliver(text, None)  # type: ignore[arg-type]
                except Exception as retry_exc:
                    _console.print(
                        f"[yellow]Telegram send failed: {retry_exc}[/yellow]"
                    )
                    return False
            _console.print(f"[yellow]Telegram send failed: {exc}[/yellow]")
            return False

    def _deliver(self, text: str, parse_mode: str) -> bool:
        """Deliver via python-telegram-bot (sync wrapper around async API)."""
        async def _send() -> bool:
            bot = _TelegramBot(token=self.bot_token)  # type: ignore[operator]
            try:
                await bot.send_message(
                    chat_id=self.chat_id, text=text, parse_mode=parse_mode
                )
                return True
            finally:
                try:
                    await bot.shutdown()
                except Exception:
                    pass

        try:
            return asyncio.run(_send())
        except RuntimeError:
            # An event loop is already running here: deliver on a fresh
            # loop in a helper thread instead of crashing.
            outcome: Dict[str, Any] = {}

            def _target() -> None:
                try:
                    outcome["ok"] = asyncio.run(_send())
                except Exception as exc:
                    outcome["error"] = exc

            thread = threading.Thread(target=_target, daemon=True)
            thread.start()
            thread.join(timeout=60)
            if "ok" in outcome:
                return bool(outcome["ok"])
            raise RuntimeError(outcome.get("error", "delivery thread timed out"))

    # -- lifecycle notifications ----------------------------------------------

    def notify_build_started(self, project_id: str, product_name: str) -> bool:
        """Notify that a build started. Returns True on delivery."""
        return self.send_message(
            "🚀 <b>BUILD STARTED</b>\n"
            f"Project: {html.escape(project_id)}\n"
            f"Product: {html.escape(product_name)}\n"
            "I'll notify you when it's done."
        )

    def notify_build_complete(
        self,
        project_id: str,
        product_name: str,
        review_status: str,
        quality_score: int,
        zip_path: str,
    ) -> bool:
        """Notify that a build finished (with review outcome)."""
        return self.send_message(
            "🤖 <b>BUILD COMPLETE</b>\n"
            f"Project: {html.escape(project_id)}\n"
            f"Product: {html.escape(product_name)}\n"
            f"Status: {html.escape(review_status)}\n"
            f"Quality Score: {int(quality_score)}/10\n"
            "\n"
            f"📦 Package: {html.escape(zip_path)}\n"
            "\n"
            "Next step: Review the package and approve for publishing."
        )

    def notify_review_needed(
        self,
        project_id: str,
        product_name: str,
        zip_path: str,
        issues: List[str] = None,
    ) -> bool:
        """Notify that human review is needed, listing issues if given."""
        lines = [
            "⚠️ <b>REVIEW NEEDED</b>",
            f"Project: {html.escape(project_id)}",
            f"Product: {html.escape(product_name)}",
        ]
        if issues:
            lines += ["", "Issues found:"]
            lines += [f"- {html.escape(str(issue))}" for issue in issues[:10]]
        lines += [
            "",
            f"Package location: {html.escape(zip_path)}",
            "",
            "Please review and let me know if we should rebuild or publish.",
        ]
        return self.send_message("\n".join(lines))

    def notify_publish_ready(
        self,
        project_id: str,
        product_name: str,
        suggested_price: float,
        publish_path: str,
        price_note: Optional[str] = None,
    ) -> bool:
        """Notify that an approved product is ready for publishing.

        Args:
            price_note: Optional context appended to the price line
                (e.g. ``"based on customer willingness-to-pay"``).
        """
        price_line = f"Suggested Price: €{float(suggested_price):g}"
        if price_note and str(price_note).strip():
            price_line += f" ({html.escape(str(price_note).strip())})"
        return self.send_message(
            "✅ <b>READY TO PUBLISH</b>\n"
            f"Project: {html.escape(project_id)}\n"
            f"Product: {html.escape(product_name)}\n"
            f"{price_line}\n"
            "\n"
            f"Package location: {html.escape(publish_path)}\n"
            "\n"
            "Would you like me to publish this to Gumroad/Payhip?"
        )
