"""SearXNG search client using public instances.

Uses public SearXNG JSON endpoints, so no API key or account is required.
Instances are tried in fallback order: the first instance that returns
usable results wins. Requests to the same instance are spaced at least six
seconds apart (ten requests per minute), and every failure degrades to an
empty result so search problems never crash a caller.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List

import requests

__all__ = ["SearchClient"]

logger = logging.getLogger(__name__)

INSTANCES = [
    "https://search.lumy.live",
    "https://priv.au",
    "https://search.sapti.me",
    "https://searx.be",
    "https://paulgo.io",
]
_MIN_INTERVAL_SECONDS = 6.0
_MAX_RESULTS = 10
_USER_AGENT = "Mozilla/5.0 (compatible; swarm-agents/1.0)"


class SearchClient:
    """SearXNG search client using public instances."""

    def __init__(self, timeout: int = 10) -> None:
        """Create a client with a shared HTTP session.

        Args:
            timeout: Per-request timeout in seconds.
        """
        self.timeout = max(1, int(timeout))
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": _USER_AGENT})
        self.last_request_time: Dict[str, float] = {}

    def _rate_limit(self, instance_url: str) -> None:
        """Enforce at least six seconds between requests to one instance."""
        now = time.monotonic()
        wait = _MIN_INTERVAL_SECONDS - (
            now - self.last_request_time.get(instance_url, 0.0)
        )
        if wait > 0:
            time.sleep(wait)
        self.last_request_time[instance_url] = time.monotonic()

    def search(
        self, query: str, max_results: int = 5
    ) -> List[Dict[str, str | int]]:
        """Search via the SearXNG JSON API.

        Args:
            query: Search query.
            max_results: Maximum results retained, clamped to 1-10.

        Returns:
            List of dicts with ``title``, ``url``, ``snippet``,
            1-based ``position``, and ``engine`` keys. Returns an empty
            list for blank queries and when every instance fails.
        """
        cleaned = query.strip() if isinstance(query, str) else ""
        if not cleaned:
            return []
        limit = max(1, min(int(max_results or 0), _MAX_RESULTS))
        for instance_url in INSTANCES:
            try:
                self._rate_limit(instance_url)
                response = self.session.get(
                    f"{instance_url}/search",
                    params={"q": cleaned, "format": "json", "pageno": 1},
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                logger.warning(
                    "SearXNG request to %s failed for %r: %s",
                    instance_url, cleaned, exc,
                )
                continue
            if response.status_code != 200:
                logger.warning(
                    "SearXNG instance %s returned HTTP %s",
                    instance_url, response.status_code,
                )
                continue
            try:
                payload = response.json()
            except ValueError as exc:
                logger.warning(
                    "SearXNG instance %s returned invalid JSON: %s",
                    instance_url, exc,
                )
                continue
            results = self._parse(payload, limit)
            if results:
                logger.info(
                    "SearXNG search succeeded via %s", instance_url
                )
                return results
            logger.warning(
                "SearXNG instance %s returned no usable results", instance_url
            )
        logger.warning("All SearXNG instances failed for %r", cleaned)
        return []

    def search_multiple(
        self, queries: List[str], max_results: int = 5
    ) -> Dict[str, List[Dict[str, str | int]]]:
        """Execute multiple searches and return results keyed by query."""
        grouped: Dict[str, List[Dict[str, str | int]]] = {}
        for query in queries or []:
            key = query if isinstance(query, str) else ""
            grouped[key] = self.search(key, max_results)
        return grouped

    @classmethod
    def _parse(cls, payload: object, limit: int) -> List[Dict[str, str | int]]:
        """Parse one SearXNG JSON payload into result records."""
        raw = []
        if isinstance(payload, dict):
            results = payload.get("results")
            if isinstance(results, list):
                raw = results
        parsed: List[Dict[str, str | int]] = []
        for item in raw:
            if len(parsed) >= limit:
                break
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            if not title or not url:
                continue
            snippet = str(item.get("content") or "").strip()
            parsed.append({
                "title": title[:200],
                "url": url[:500],
                "snippet": snippet[:500],
                "position": len(parsed) + 1,
                "engine": str(item.get("engine") or "unknown")[:80],
            })
        return parsed
