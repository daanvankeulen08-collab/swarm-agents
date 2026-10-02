"""WebSearch: DuckDuckGo-powered real-time market data for research.

:class:`WebSearch` wraps the ``duckduckgo_search`` package (no API key, no
login) and returns normalised results plus a market-data synthesiser used
by :meth:`SubNicheFinder.find_subniches` to validate competition counts
against live web evidence.

All network failures (timeouts, rate limits, blocks) degrade to empty
results with a logged warning — scraping never crashes a pipeline. A small
delay between queries keeps request rates polite.

Example:
    ```python
    from research.web_search import WebSearch

    ws = WebSearch()
    results = ws.search("freelance medical interpreter notion template",
                        max_results=5)
    print(results[0]["title"], results[0]["url"])
    data = ws.search_market_data("weekly planner")
    print(data["total_results"], data["price_mentions"])
    ```
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List
from urllib.parse import urlparse

from rich.console import Console

try:  # Canonical package name first (no deprecation warning).
    from ddgs import DDGS as _DDGS

    _SEARCH_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS as _DDGS  # type: ignore[no-redef]

        _SEARCH_AVAILABLE = True
    except ImportError:
        _DDGS = None  # type: ignore[assignment]
        _SEARCH_AVAILABLE = False

__all__ = ["WebSearch", "SEARCH_AVAILABLE", "DEFAULT_DELAY"]

#: True when a DDG client library is importable.
SEARCH_AVAILABLE: bool = _SEARCH_AVAILABLE

#: Polite pause (seconds) between search queries.
DEFAULT_DELAY: float = 1.0

_console = Console()


class WebSearch:
    """DuckDuckGo web search with normalised results and market synthesis.

    Args:
        delay: Seconds to wait between queries (politeness throttle).
        timeout: Per-query timeout in seconds.

    Example:
        ```python
        ws = WebSearch()
        hits = ws.search("notion template", max_results=5)
        ```
    """

    def __init__(self, delay: float = DEFAULT_DELAY, timeout: int = 20) -> None:
        if delay < 0:
            raise ValueError("delay must be >= 0.")
        self.delay: float = float(delay)
        self.timeout: int = timeout
        self._last_query_at: float = 0.0

    def search(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """Run one DuckDuckGo text search.

        Args:
            query: Search keywords (``site:`` operators allowed).
            max_results: Maximum results to return (1–30).

        Returns:
            List of dicts with ``title``, ``url``, and ``snippet`` keys.
            Empty list on any failure (network, rate limit, missing
            library) — never raises for remote problems.

        Raises:
            ValueError: On empty query or out-of-range max_results.
        """
        cleaned = query.strip() if isinstance(query, str) else ""
        if not cleaned:
            raise ValueError("query must be a non-empty string.")
        if not 1 <= int(max_results) <= 30:
            raise ValueError("max_results must be between 1 and 30.")
        if not _SEARCH_AVAILABLE or _DDGS is None:
            _console.print(
                "[yellow]DuckDuckGo library missing; returning no results.[/yellow]"
            )
            return []
        self._throttle()
        try:
            raw = list(
                _DDGS().text(cleaned, max_results=int(max_results))
            )
        except Exception as exc:
            _console.print(
                f"[yellow]Web search failed for {cleaned!r}: "
                f"{type(exc).__name__}: {exc}[/yellow]"
            )
            return []
        results = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            url = str(item.get("href", "") or item.get("url", "")).strip()
            if not title or not url:
                continue
            results.append({
                "title": title[:200],
                "url": url[:500],
                "snippet": str(item.get("body", "") or "").strip()[:500],
            })
        return results

    def search_market_data(self, niche: str) -> Dict[str, Any]:
        """Gather market evidence for ``niche`` from live search results.

        Searches ``"{niche} notion template"`` and ``"{niche} digital
        product"``, then synthesises competition and pricing signals.

        Args:
            niche: Market niche, e.g. ``"weekly planner"``.

        Returns:
            Dict with ``niche``, ``total_results``, ``unique_domains``,
            ``top_sources`` (up to 8 domains), ``price_mentions`` (e.g.
            ``["$12", ...]``), ``sample_listings`` (title/url/snippet), and
            ``notes``. Never raises for remote problems.

        Raises:
            ValueError: On empty niche.
        """
        cleaned = niche.strip() if isinstance(niche, str) else ""
        if not cleaned:
            raise ValueError("niche must be a non-empty string.")
        _console.print(f"[cyan]Gathering market data for {cleaned!r}…[/cyan]")
        hits: List[Dict[str, str]] = []
        for query in (f"{cleaned} notion template", f"{cleaned} digital product"):
            hits.extend(self.search(query, max_results=10))
        seen, unique = set(), []
        for hit in hits:
            if hit["url"] not in seen:
                seen.add(hit["url"])
                unique.append(hit)
        domains = sorted({urlparse(h["url"]).netloc for h in unique if h["url"]})
        prices = sorted(
            {m for h in unique for m in re.findall(r"[$€]\s?\d[\d,]*(?:\.\d{1,2})?",
                                                   h["snippet"] + " " + h["title"])},
            key=lambda p: (p[0], len(p)),
        )[:10]
        data = {
            "niche": cleaned,
            "total_results": len(unique),
            "unique_domains": len(domains),
            "top_sources": domains[:8],
            "price_mentions": prices,
            "sample_listings": unique[:10],
            "notes": (
                f"{len(unique)} unique results across {len(domains)} domains."
                if unique
                else "No live results retrieved (network block or empty niche)."
            ),
        }
        _console.print(f"[green]Market data: {data['notes']}[/green]")
        return data

    # -- internals ----------------------------------------------------------

    def _throttle(self) -> None:
        """Sleep until ``delay`` seconds passed since the last query."""
        elapsed = time.monotonic() - self._last_query_at
        wait = self.delay - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_query_at = time.monotonic()
