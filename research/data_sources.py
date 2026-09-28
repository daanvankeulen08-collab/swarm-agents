"""Ethical data collectors for the revenue-stream research framework.

Three public-data-only scrapers back the
:class:`~research.research_agent.ResearchAgent`:

* :class:`RedditScraper` — top posts from chosen subreddits (public JSON).
* :class:`MarketplaceAnalyzer` — best-effort listing extraction from Gumroad,
  Etsy, and Creative Market search pages.
* :class:`YouTubeResearcher` — video metadata from public search pages.

Ground rules every scraper follows:

* Only public pages, no login, no private data, no API keys.
* ``robots.txt`` is consulted before fetching (fail-open with a recorded
  note when it cannot be retrieved — documented per fetch).
* Minimum 2-second delay between HTTP requests (configurable).
* All failures are captured as structured notes; scrapers never raise for
  remote problems (bad input still raises :class:`ValueError`).
* Every fetch is journaled via :meth:`BaseScraper.get_raw_data` so results
  stay verifiable.

Example:
    ```python
    from research.data_sources import RedditScraper

    scraper = RedditScraper()
    posts = scraper.research(["SideProject", "Entrepreneur"],
                             ["side hustle", "digital product"])
    print(posts[0]["post_title"], posts[0]["upvotes"])
    print(scraper.get_raw_data()[0]["url"])
    ```
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib import robotparser
from urllib.parse import quote_plus, urljoin, urlparse

import requests
from rich.console import Console

__all__ = [
    "BaseScraper",
    "RedditScraper",
    "MarketplaceAnalyzer",
    "YouTubeResearcher",
    "DEFAULT_USER_AGENT",
    "DEFAULT_MIN_DELAY",
]

#: Descriptive User-Agent identifying the research bot and its purpose.
DEFAULT_USER_AGENT: str = (
    "AgentSwarmResearchBot/1.0 (+https://example.com/bot; "
    "public-data research; contact: research@example.com)"
)

#: Minimum seconds between HTTP requests (ethical rate limiting).
DEFAULT_MIN_DELAY: float = 2.0

#: Commercial keywords used to spot product/service mentions in post text.
COMMERCIAL_KEYWORDS: tuple = (
    "buy", "sell", "purchase", "product", "tool", "template", "course",
    "ebook", "e-book", "subscription", "shop", "store", "gumroad", "etsy",
    "download", "pricing", "customer", "revenue",
)

_console = Console()


def _utcnow_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


class BaseScraper:
    """Shared ethical-fetch machinery: throttle, robots, journal, no-raise.

    Args:
        min_delay: Minimum seconds between HTTP requests (>= 0).
        timeout: Per-request timeout in seconds.
        user_agent: HTTP User-Agent header value.
        respect_robots: Consult ``robots.txt`` before fetching. When the
            file itself cannot be retrieved, the fetch proceeds and the
            incident is journaled (fail-open, documented).

    Raw fetch journal entries (see :meth:`get_raw_data`) hold ``url``,
    ``status`` (HTTP code or ``None``), ``fetched_at`` (ISO), ``bytes``,
    ``robots`` (``"allowed"`` / ``"disallowed"`` / ``"unreachable-allowed"``
    / ``"check-skipped"``) and, on failure, ``"error"``.
    """

    def __init__(
        self,
        min_delay: float = DEFAULT_MIN_DELAY,
        timeout: int = 15,
        user_agent: str = DEFAULT_USER_AGENT,
        respect_robots: bool = True,
    ) -> None:
        if min_delay < 0:
            raise ValueError("min_delay must be >= 0.")
        self.min_delay: float = float(min_delay)
        self.timeout: int = timeout
        self.user_agent: str = user_agent
        self.respect_robots: bool = respect_robots
        self._last_request_at: float = 0.0
        self._raw_data: List[Dict[str, Any]] = []
        self._robots_cache: Dict[str, robotparser.RobotFileParser] = {}

    # -- public journal API -------------------------------------------------

    def get_raw_data(self) -> List[Dict[str, Any]]:
        """Return a copy of the raw fetch journal (verifiable audit trail)."""
        return [dict(entry) for entry in self._raw_data]

    def clear_raw_data(self) -> None:
        """Empty the raw fetch journal."""
        self._raw_data.clear()

    # -- fetch machinery ----------------------------------------------------

    def _throttle(self) -> None:
        """Sleep until ``min_delay`` seconds passed since the last request."""
        elapsed = time.monotonic() - self._last_request_at
        wait = self.min_delay - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()

    def _robots_status(self, url: str) -> str:
        """Check ``robots.txt``; fail-open with note when unreachable."""
        if not self.respect_robots:
            return "check-skipped"
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        try:
            parser = self._robots_cache.get(origin)
            if parser is None:
                parser = robotparser.RobotFileParser()
                parser.set_url(urljoin(origin, "/robots.txt"))
                parser.read()  # Network I/O; may raise.
                self._robots_cache[origin] = parser
            return "allowed" if parser.can_fetch(self.user_agent, url) else "disallowed"
        except Exception as exc:
            return f"unreachable-allowed (robots fetch failed: {exc})"

    def fetch(
        self, url: str, params: Optional[Dict[str, Any]] = None
    ) -> Optional[requests.Response]:
        """GET ``url`` ethically: throttled, robots-checked, never raising.

        Args:
            url: Absolute HTTP(S) URL.
            params: Optional query parameters.

        Returns:
            The :class:`requests.Response` on HTTP success, else None. Every
            outcome (including robots blocks and exceptions) is journaled.
        """
        robots = self._robots_status(url)
        if robots == "disallowed":
            self._journal(url, None, robots, "Blocked by robots.txt.")
            _console.print(f"[yellow]Skipping (robots.txt): {url}[/yellow]")
            return None
        self._throttle()
        try:
            response = requests.get(
                url,
                params=params,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                },
                timeout=self.timeout,
            )
            self._journal(url, response.status_code, robots, None,
                          len(response.content))
            if response.status_code != 200:
                _console.print(
                    f"[yellow]HTTP {response.status_code} for {url}[/yellow]"
                )
                return None
            return response
        except requests.RequestException as exc:
            self._journal(url, None, robots, f"Request failed: {exc}")
            _console.print(f"[yellow]Fetch failed for {url}: {exc}[/yellow]")
            return None

    def _journal(
        self,
        url: str,
        status: Optional[int],
        robots: str,
        error: Optional[str],
        size: int = 0,
    ) -> None:
        """Append one entry to the raw fetch journal."""
        entry: Dict[str, Any] = {
            "url": url,
            "status": status,
            "fetched_at": _utcnow_iso(),
            "bytes": size,
            "robots": robots,
        }
        if error:
            entry["error"] = error
        self._raw_data.append(entry)


class RedditScraper(BaseScraper):
    """Collect top posts from subreddits via Reddit's public JSON listings.

    No API key or login is used — only the same public ``.json`` endpoints
    any browser can open.

    Example:
        ```python
        posts = RedditScraper().research(["SideProject"], ["side hustle"])
        ```
    """

    BASE_URL: str = "https://www.reddit.com"

    def research(
        self,
        subreddits: List[str],
        topics: List[str],
        limit_per_sub: int = 10,
        time_filter: str = "month",
    ) -> List[Dict[str, Any]]:
        """Scrape top posts from ``subreddits`` filtered by ``topics``.

        Args:
            subreddits: Subreddit names without the ``r/`` prefix.
            topics: Interest keywords (e.g. ``["side hustle"]``). Posts
                matching any topic are flagged ``relevant=True``; others
                are still returned with ``relevant=False`` for context.
            limit_per_sub: Posts to pull per subreddit (1–25).
            time_filter: Reddit listing window: hour/day/week/month/year/all.

        Returns:
            List of dicts with ``subreddit``, ``post_title``, ``upvotes``,
            ``num_comments``, ``key_insights``, ``mentioned_products``,
            ``source`` (permalink), and ``relevant``.

        Raises:
            ValueError: On empty/invalid inputs.
        """
        subs = self._clean_str_list(subreddits, "subreddits")
        tps = self._clean_str_list(topics, "topics")
        if not 1 <= int(limit_per_sub) <= 25:
            raise ValueError("limit_per_sub must be between 1 and 25.")
        if time_filter not in ("hour", "day", "week", "month", "year", "all"):
            raise ValueError(f"Invalid time_filter {time_filter!r}.")
        lowered_topics = [t.lower() for t in tps]

        posts: List[Dict[str, Any]] = []
        for sub in subs:
            if not re.fullmatch(r"[A-Za-z0-9_]+", sub):
                _console.print(f"[yellow]Skipping invalid subreddit {sub!r}.[/yellow]")
                continue
            _console.print(f"[cyan]Scraping r/{sub} (top/{time_filter})…[/cyan]")
            url = f"{self.BASE_URL}/r/{sub}/top/.json"
            response = self.fetch(
                url, params={"limit": limit_per_sub, "t": time_filter}
            )
            if response is None:
                continue
            try:
                payload = response.json()
            except ValueError:
                _console.print(f"[yellow]Non-JSON listing for r/{sub}.[/yellow]")
                continue
            for child in payload.get("data", {}).get("children", []):
                data = child.get("data", {})
                posts.append(self._shape_post(sub, data, lowered_topics))
        _console.print(f"[green]Collected {len(posts)} Reddit posts.[/green]")
        # Most-upvoted, relevant-first ordering for downstream analysis.
        posts.sort(key=lambda p: (p["relevant"], p["upvotes"]), reverse=True)
        return posts

    @staticmethod
    def _clean_str_list(values: Any, name: str) -> List[str]:
        """Validate a list-of-strings argument."""
        if not isinstance(values, list) or not values:
            raise ValueError(f"{name} must be a non-empty list of strings.")
        cleaned = [v.strip() for v in values if isinstance(v, str) and v.strip()]
        if not cleaned:
            raise ValueError(f"{name} must be a non-empty list of strings.")
        return cleaned

    def _shape_post(
        self, sub: str, data: Dict[str, Any], topics: List[str]
    ) -> Dict[str, Any]:
        """Convert one Reddit listing child into the result schema."""
        title = str(data.get("title", "")).strip()
        selftext = str(data.get("selftext", "") or "")
        haystack = f"{title}\n{selftext}".lower()
        matched = [t for t in topics if t in haystack]
        return {
            "subreddit": sub,
            "post_title": title,
            "upvotes": int(data.get("score", 0) or 0),
            "num_comments": int(data.get("num_comments", 0) or 0),
            "key_insights": self._key_insights(selftext, topics),
            "mentioned_products": self._product_mentions(f"{title}\n{selftext}"),
            "source": urljoin(self.BASE_URL, str(data.get("permalink", "") or "")),
            "relevant": bool(matched),
            "matched_topics": matched,
        }

    @staticmethod
    def _key_insights(selftext: str, topics: List[str]) -> List[str]:
        """Pull up to 3 insight sentences (topic-matching preferred)."""
        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", selftext.strip()) if s.strip()
        ]
        if not sentences:
            return []
        on_topic = [s for s in sentences if any(t in s.lower() for t in topics)]
        picks = (on_topic + [s for s in sentences if s not in on_topic])[:3]
        return [p[:300] for p in picks]

    @staticmethod
    def _product_mentions(text: str) -> List[str]:
        """Heuristically extract commercial mentions + prices from text."""
        mentions: List[str] = []
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            low = sentence.strip().lower()
            if low and any(k in low for k in COMMERCIAL_KEYWORDS):
                mentions.append(sentence.strip()[:200])
                if len(mentions) >= 3:
                    break
        for price in sorted(set(re.findall(r"\$\s?\d[\d,]*(?:\.\d{1,2})?", text)))[:5]:
            token = f"Price mentioned: {price}"
            if token not in mentions:
                mentions.append(token)
        return mentions


class MarketplaceAnalyzer(BaseScraper):
    """Best-effort listing extraction from marketplace search pages.

    Supported ``platform`` values: ``"gumroad"``, ``"etsy"``,
    ``"creative market"`` (alias ``"creativemarket"``). Pages are JavaScript
    heavy and change often, so extraction is heuristic and every shortfall is
    reported in ``fetch_notes`` — listings are never fabricated.

    Example:
        ```python
        result = MarketplaceAnalyzer().analyze("gumroad", "notion template", (5, 20))
        print(result["listings"], result["fetch_notes"])
        ```
    """

    SEARCH_URLS: Dict[str, str] = {
        "gumroad": "https://gumroad.com/discover?query={q}",
        "etsy": "https://www.etsy.com/search?q={q}",
        "creativemarket": "https://creativemarket.com/search?q={q}",
    }

    def analyze(
        self,
        platform: str,
        category: str,
        price_range: tuple,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Fetch and parse marketplace listings for ``category``.

        Args:
            platform: One of gumroad / etsy / creative market.
            category: Search keywords, e.g. ``"notion template"``.
            price_range: ``(min, max)`` tuple used to flag in-range items.
            limit: Maximum listings to return.

        Returns:
            Dict with ``platform``, ``category``, ``listings`` (each with
            ``title``, ``price``, ``reviews``, ``seller``, ``url``,
            ``in_price_range``), ``patterns`` (price stats + recurring
            title keywords), and ``fetch_notes`` describing coverage gaps.

        Raises:
            ValueError: On unknown platform or invalid inputs.
        """
        key = (platform or "").strip().lower().replace(" ", "")
        if key == "creativemarket":
            key = "creativemarket"
        if key not in self.SEARCH_URLS:
            raise ValueError(
                f"Unsupported platform {platform!r}; expected one of: "
                "gumroad, etsy, creative market."
            )
        if not isinstance(category, str) or not category.strip():
            raise ValueError("category must be a non-empty string.")
        low, high = self._parse_price_range(price_range)
        url = self.SEARCH_URLS[key].format(q=quote_plus(category.strip()))
        _console.print(f"[cyan]Analyzing {platform} for {category!r}…[/cyan]")

        response = self.fetch(url)
        listings: List[Dict[str, Any]] = []
        notes: List[str] = []
        if response is None:
            notes.append(f"Could not retrieve {platform} search page; 0 listings.")
        else:
            listings = self._extract_listings(response.text, url, limit)
            if not listings:
                notes.append(
                    f"Parsed {platform} page but extracted 0 listings "
                    "(page structure not recognised); 0 listings."
                )
        for item in listings:
            price = item.get("price")
            item["in_price_range"] = (
                isinstance(price, (int, float)) and low <= float(price) <= high
            )
        result = {
            "platform": platform.strip(),
            "category": category.strip(),
            "price_range": [low, high],
            "listings": listings[:limit],
            "patterns": self._patterns(listings),
            "fetch_notes": notes,
        }
        _console.print(
            f"[green]Marketplace scan done: {len(result['listings'])} listings "
            f"on {platform}.[/green]"
        )
        return result

    @staticmethod
    def _parse_price_range(price_range: tuple) -> tuple:
        """Validate a ``(min, max)`` tuple into floats."""
        try:
            low_raw, high_raw = price_range
            low, high = float(low_raw), float(high_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"price_range must be a (min, max) numeric tuple, got {price_range!r}."
            ) from exc
        if low > high:
            raise ValueError(f"price_range min exceeds max: {price_range!r}.")
        return (low, high)

    def _extract_listings(
        self, html: str, page_url: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Heuristically pull product anchors + nearby prices from HTML."""
        listings: List[Dict[str, Any]] = []
        seen = set()
        # Product-ish links: /l/<slug>, /listing/<id>, /products/<slug>.
        for match in re.finditer(
            r'<a[^>]+href="([^"]*(?:/l/|/listing/|/products/)[^"]*)"[^>]*>(.*?)</a>',
            html,
            re.IGNORECASE | re.DOTALL,
        ):
            href, inner = match.group(1), match.group(2)
            title = re.sub(r"<[^>]+>", " ", inner)
            title = re.sub(r"\s+", " ", title).strip()
            if len(title) < 4:
                continue
            absolute = urljoin(page_url, href.split("?")[0])
            if absolute in seen:
                continue
            seen.add(absolute)
            # Price: nearest $ amount within a small window after the anchor.
            window = html[match.end(): match.end() + 600]
            price_match = re.search(r"\$\s?(\d[\d,]*\.?\d*)", window)
            price: Optional[float] = None
            if price_match:
                try:
                    price = float(price_match.group(1).replace(",", ""))
                except ValueError:
                    price = None
            listings.append({
                "title": title[:160],
                "price": price,
                "reviews": None,  # Public pages rarely expose counts reliably.
                "seller": None,
                "url": absolute,
            })
            if len(listings) >= limit:
                break
        return listings

    @staticmethod
    def _patterns(listings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarise price stats + recurring title keywords."""
        prices = [float(i["price"]) for i in listings
                  if isinstance(i.get("price"), (int, float))]
        words: Dict[str, int] = {}
        for item in listings:
            for word in re.findall(r"[a-z]{4,}", str(item.get("title", "")).lower()):
                words[word] = words.get(word, 0) + 1
        keywords = sorted(words, key=lambda w: (-words[w], w))[:10]
        return {
            "listing_count": len(listings),
            "priced_count": len(prices),
            "min_price": min(prices) if prices else None,
            "max_price": max(prices) if prices else None,
            "avg_price": round(sum(prices) / len(prices), 2) if prices else None,
            "common_title_keywords": keywords,
        }


class YouTubeResearcher(BaseScraper):
    """Collect public video metadata for a niche from YouTube search pages.

    Only the public search page is read (no API key, no login). Metadata is
    parsed from the embedded ``ytInitialData`` payload; revenue figures are
    rough niche-level estimates, never per-channel facts.

    Example:
        ```python
        result = YouTubeResearcher().research("faceless finance channels")
        print(result["videos"][0]["title"])
        ```
    """

    SEARCH_URL: str = "https://www.youtube.com/results?search_query={q}"

    def research(self, niche: str, max_results: int = 10) -> Dict[str, Any]:
        """Search YouTube for ``niche`` and extract video metadata.

        Args:
            niche: Search keywords, e.g. ``"faceless finance channels"``.
            max_results: Maximum videos to return (1–25).

        Returns:
            Dict with ``niche``, ``videos`` (each with ``title``,
            ``channel``, ``views``, ``url``), and ``fetch_notes``.

        Raises:
            ValueError: On empty niche or out-of-range max_results.
        """
        if not isinstance(niche, str) or not niche.strip():
            raise ValueError("niche must be a non-empty string.")
        if not 1 <= int(max_results) <= 25:
            raise ValueError("max_results must be between 1 and 25.")
        url = self.SEARCH_URL.format(q=quote_plus(niche.strip()))
        _console.print(f"[cyan]Researching YouTube for {niche!r}…[/cyan]")

        response = self.fetch(url)
        videos: List[Dict[str, Any]] = []
        notes: List[str] = []
        if response is None:
            notes.append("Could not retrieve YouTube search page; 0 videos.")
        else:
            videos = self._extract_videos(response.text, int(max_results))
            if not videos:
                notes.append(
                    "Parsed YouTube page but extracted 0 videos "
                    "(page structure not recognised); 0 videos."
                )
        _console.print(f"[green]YouTube scan done: {len(videos)} videos.[/green]")
        return {"niche": niche.strip(), "videos": videos, "fetch_notes": notes}

    def _extract_videos(self, html: str, limit: int) -> List[Dict[str, Any]]:
        """Parse videoRenderer items from the embedded ytInitialData JSON."""
        match = re.search(r"var ytInitialData = (\{.*?});\s*</script>",
                          html, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(1))
        except ValueError:
            return []
        videos: List[Dict[str, Any]] = []
        self._walk(data, videos, limit)
        return videos[:limit]

    def _walk(self, node: Any, out: List[Dict[str, Any]], limit: int) -> None:
        """Recursively collect videoRenderer dicts (depth-safe via limit)."""
        if len(out) >= limit or not isinstance(node, (dict, list)):
            return
        items = node.values() if isinstance(node, dict) else node
        for value in items:
            if len(out) >= limit:
                return
            if isinstance(value, dict) and "videoRenderer" in value:
                renderer = value["videoRenderer"]
                video_id = str(renderer.get("videoId", ""))
                if not video_id:
                    continue
                title_runs = renderer.get("title", {}).get("runs", [{}])
                channel_runs = renderer.get("ownerText", {}).get("runs", [{}])
                out.append({
                    "title": str(title_runs[0].get("text", "")).strip()[:200],
                    "channel": str(channel_runs[0].get("text", "")).strip()[:120],
                    "views": str(renderer.get("viewCountText", {})
                                  .get("simpleText", "")).strip()[:40],
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                })
            else:
                self._walk(value, out, limit)
