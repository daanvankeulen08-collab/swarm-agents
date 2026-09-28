"""ResearchAgent: finds safe, automatable micro-revenue streams.

The ResearchAgent is the swarm's scouting foundation — it decides *which*
revenue streams are worth pursuing before the builder pipeline spends a cent.
It combines ethical public-data collection
(:mod:`research.data_sources`), legal screening
(:mod:`research.legal_analyzer`), and AI synthesis via the
:class:`~core.orchestrator.Orchestrator`. All progress and findings live on
``state.research_findings``, the single source of truth.

Pipeline:
    1. :meth:`research_reddit` — demand signals from subreddit top posts.
    2. :meth:`research_marketplace` — competitive reality from Gumroad / Etsy
       / Creative Market listings.
    3. :meth:`analyze_youtube_opportunities` — faceless-video potential with
       copyright-risk eyes open.
    4. :meth:`legal_risk_assessment` — per-stream legal screening.
    5. :meth:`generate_opportunity_report` — ranked, schema-validated report.

Example:
    ```python
    from core.orchestrator import Orchestrator
    from core.state_manager import StateManager
    from research.research_agent import ResearchAgent

    state_manager = StateManager(project_id="demo")
    agent = ResearchAgent(Orchestrator(), state_manager)

    agent.research_reddit(["SideProject", "Entrepreneur"], ["side hustle"])
    agent.research_marketplace("gumroad", "notion template", (5, 20))
    agent.analyze_youtube_opportunities("faceless finance channels")
    agent.legal_risk_assessment("Selling original Notion templates on Gumroad")
    report = agent.generate_opportunity_report()
    print(report["recommendations"])
    ```

Ethics: public data only, robots.txt respected, 2s rate limits, no private
data, disclaimers on legal output. Windows-compatible (pathlib, UTF-8) with
:mod:`rich` progress output throughout.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from rich.console import Console

from core.orchestrator import Orchestrator
from core.state_manager import StateManager
from research.data_sources import (
    MarketplaceAnalyzer,
    RedditScraper,
    YouTubeResearcher,
)
from research.legal_analyzer import DISCLAIMER, LegalAnalyzer

__all__ = ["ResearchAgent", "REPORT_SCHEMA_HINT", "SUPPORTED_PLATFORMS"]

#: Platforms accepted by research_marketplace().
SUPPORTED_PLATFORMS: tuple = ("gumroad", "etsy", "creative market")

#: Schema hint appended to AI prompts that must return opportunities.
REPORT_SCHEMA_HINT: str = (
    'Respond with VALID JSON ONLY, no commentary, using exactly this shape: '
    '{"opportunities": [{"name": "...", "category": "...", '
    '"description": "...", "price_range": [5, 20], '
    '"feasibility_score": 8, "risk_level": "low", '
    '"risk_factors": ["..."], "revenue_potential": "...", '
    '"technical_requirements": ["..."], '
    '"legal_considerations": ["..."], "automation_level": "partial", '
    '"sources": ["..."]}], '
    '"recommendations": ["..."], "avoid": ["... with reason ..."]}'
)

_console = Console()


def _utcnow_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _skeleton_findings() -> Dict[str, Any]:
    """Empty research_findings structure stored on the state."""
    return {
        "reddit": [],
        "reddit_themes": {},
        "marketplace": [],
        "youtube": {},
        "legal": [],
        "report": None,
        "updated_at": _utcnow_iso(),
    }


class ResearchAgent:
    """Investigate revenue opportunities across Reddit, marketplaces, YouTube.

    Args:
        orchestrator: The swarm's :class:`Orchestrator` for AI analysis.
        state_manager: The project's :class:`StateManager`; research
            progress is tracked via phase ``"research"`` and findings via
            ``state.research_findings``.
        reddit_scraper: Optional pre-built scraper (dependency injection
            for tests; a default is created otherwise).
        marketplace_analyzer: Optional pre-built analyzer (as above).
        youtube_researcher: Optional pre-built researcher (as above).
        legal_analyzer: Optional pre-built analyzer (as above).
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        state_manager: StateManager,
        reddit_scraper: Optional[RedditScraper] = None,
        marketplace_analyzer: Optional[MarketplaceAnalyzer] = None,
        youtube_researcher: Optional[YouTubeResearcher] = None,
        legal_analyzer: Optional[LegalAnalyzer] = None,
    ) -> None:
        # Duck-typed checks so test doubles work (swarm-wide convention).
        for name in ("ask_agent", "build_context_prompt", "get_model_info"):
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
        self.reddit = reddit_scraper or RedditScraper()
        self.marketplaces = marketplace_analyzer or MarketplaceAnalyzer()
        self.youtube = youtube_researcher or YouTubeResearcher()
        self.legal = legal_analyzer or LegalAnalyzer(orchestrator, state_manager)

    # -- collection methods ---------------------------------------------------

    def research_reddit(
        self, subreddits: List[str], topics: List[str]
    ) -> List[Dict[str, Any]]:
        """Scrape subreddit top posts and AI-distil demand signals.

        Focus: side hustles, digital products, automation opportunities.

        Args:
            subreddits: Subreddit names without ``r/`` (e.g. ``["SideProject"]``).
            topics: Interest keywords (e.g. ``["side hustle"]``).

        Returns:
            List of post dicts with ``post_title`` (post title), ``upvotes``,
            ``key_insights``, ``mentioned_products`` (mentioned
            products/services), ``source``, ``subreddit``, ``relevant``.

        Raises:
            ValueError: On invalid inputs.
            RuntimeError: If AI theme analysis fails (scraped posts are still
                persisted before the error is raised).
        """
        self._mark_researching()
        posts = self.reddit.research(subreddits, topics)
        _console.print(
            f"[cyan]Analysing {len(posts)} Reddit posts with AI…[/cyan]"
        )
        themes: Dict[str, Any] = {"themes": [], "product_gaps": []}
        if posts:
            digest = "\n".join(
                f"- [{p['subreddit']}] {p['post_title']} "
                f"({p['upvotes']} upvotes): "
                f"{' | '.join(p['key_insights'][:2])}"
                for p in posts[:25]
            )
            try:
                themes = self._ask_json(
                    "You are a market researcher specialising in side hustles "
                    "and digital micro-products. Distil the Reddit posts below "
                    "into demand themes and unmet product needs. "
                    "Respond with VALID JSON ONLY: "
                    '{"themes": ["..."], "product_gaps": ["..."]}',
                    f"Subreddits: {', '.join(subreddits)}\n"
                    f"Topics: {', '.join(topics)}\n\nPosts:\n{digest}",
                )
            except RuntimeError as exc:
                self._update_findings(reddit=posts, reddit_themes=themes)
                raise RuntimeError(f"Reddit AI analysis failed: {exc}") from exc
            if not isinstance(themes, dict):
                themes = {"themes": [], "product_gaps": []}
        self._update_findings(reddit=posts, reddit_themes=themes)
        _console.print(
            f"[green]Reddit research done: {len(posts)} posts, "
            f"{len(themes.get('themes', []))} themes.[/green]"
        )
        return posts

    def research_marketplace(
        self, platform: str, category: str, price_range: tuple
    ) -> List[Dict[str, Any]]:
        """Analyse marketplace listings and AI-summarise competitive patterns.

        Args:
            platform: ``"gumroad"``, ``"etsy"``, or ``"creative market"``.
            category: Search keywords, e.g. ``"notion template"``.
            price_range: ``(min, max)`` tuple flagging in-range items.

        Returns:
            List of listing dicts with top-selling-style products, prices,
            reviews, seller info (where publicly exposed), source URL, plus
            an ``ai_patterns`` entry context. Competitive patterns are
            persisted alongside.

        Raises:
            ValueError: On unknown platform or invalid inputs.
            RuntimeError: If AI pattern analysis fails (listings are still
                persisted before the error is raised).
        """
        key = (platform or "").strip().lower()
        if key not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Unsupported platform {platform!r}; expected one of: "
                f"{list(SUPPORTED_PLATFORMS)}."
            )
        self._mark_researching()
        scan = self.marketplaces.analyze(platform.strip(), category, price_range)
        listings = scan.get("listings", [])
        patterns: Dict[str, Any] = dict(scan.get("patterns", {}))
        if listings:
            _console.print(
                f"[cyan]Analysing {len(listings)} {platform} listings with AI…[/cyan]"
            )
            digest = "\n".join(
                f"- {item.get('title', '?')} "
                f"(${item.get('price', '?')}) {item.get('url', '')}"
                for item in listings[:20]
            )
            try:
                ai_notes = self._ask_json(
                    "You are a competitive analyst for digital micro-products. "
                    "Identify patterns in what makes these listings successful "
                    "(pricing, titles, positioning). Respond with VALID JSON "
                    "ONLY: {\"success_patterns\": [\"...\"], "
                    "\"pricing_advice\": \"...\", \"gaps\": [\"...\"]}",
                    f"Platform: {platform}\nCategory: {category}\n\n"
                    f"Listings:\n{digest}",
                )
            except RuntimeError as exc:
                self._append_marketplace(scan, patterns)
                raise RuntimeError(
                    f"Marketplace AI analysis failed: {exc}"
                ) from exc
            if isinstance(ai_notes, dict):
                patterns["ai_analysis"] = ai_notes
        self._append_marketplace(scan, patterns)
        _console.print(
            f"[green]Marketplace research done: {len(listings)} listings "
            f"on {platform}.[/green]"
        )
        return listings

    def analyze_youtube_opportunities(self, niche: str) -> Dict[str, Any]:
        """Research faceless-video potential for ``niche`` with risk eyes open.

        Covers faceless channel formats, clipper channels (copyright-safe
        content only), revenue estimates, content types, tools used, and a
        copyright risk assessment.

        Args:
            niche: Search keywords, e.g. ``"faceless finance channels"``.

        Returns:
            Dict with ``niche``, ``videos`` (raw metadata), ``formats``,
            ``revenue_estimates``, ``content_types``, ``tools_used``,
            ``copyright_risk`` (risk assessment dict), and ``sources``.

        Raises:
            ValueError: On empty niche.
            RuntimeError: If AI analysis fails (raw videos are still
                persisted before the error is raised).
        """
        if not isinstance(niche, str) or not niche.strip():
            raise ValueError("niche must be a non-empty string.")
        self._mark_researching()
        scan = self.youtube.research(niche.strip())
        videos = scan.get("videos", [])
        _console.print(
            f"[cyan]Analysing {len(videos)} YouTube videos with AI…[/cyan]"
        )
        digest = "\n".join(
            f"- {v.get('title', '?')} [{v.get('channel', '?')}] "
            f"{v.get('views', '')} {v.get('url', '')}"
            for v in videos[:20]
        ) or "(no videos retrieved)"
        try:
            analysis = self._ask_json(
                "You are a YouTube strategist specialising in FACELESS channels "
                "and copyright-safe production (original narration, licensed "
                "stock, transformative commentary). Never recommend re-uploading "
                "other creators' content. Respond with VALID JSON ONLY: "
                '{"formats": ["..."], "revenue_estimates": "...", '
                '"content_types": ["..."], "tools_used": ["..."], '
                '"copyright_risk": {"risk_level": "low|medium|high", '
                '"risk_factors": ["..."], "safe_practices": ["..."]}}',
                f"Niche: {niche}\n\nTop videos:\n{digest}",
            )
        except RuntimeError as exc:
            self._update_findings(youtube={**scan, "analysis": {}})
            raise RuntimeError(f"YouTube AI analysis failed: {exc}") from exc
        if not isinstance(analysis, dict):
            analysis = {}
        result = {
            "niche": niche.strip(),
            "videos": videos,
            "formats": analysis.get("formats", []),
            "revenue_estimates": analysis.get("revenue_estimates", ""),
            "content_types": analysis.get("content_types", []),
            "tools_used": analysis.get("tools_used", []),
            "copyright_risk": analysis.get("copyright_risk", {}),
            "sources": [v["url"] for v in videos if v.get("url")],
            "fetch_notes": scan.get("fetch_notes", []),
        }
        self._update_findings(youtube=result)
        _console.print("[green]YouTube research done.[/green]")
        return result

    def legal_risk_assessment(self, revenue_stream: str) -> Dict[str, Any]:
        """Screen one revenue stream: copyright, ToS, legal exposure.

        Args:
            revenue_stream: Plain-language description of the stream.

        Returns:
            Structured assessment with copyright implications, platform
            terms-of-service compliance, potential legal issues, and
            ``risk_level`` (low/medium/high) with explanation — plus the
            standard legal-information disclaimer.

        Raises:
            ValueError: If the description is empty.
            RuntimeError: If the AI review fails.
        """
        self._mark_researching()
        assessment = self.legal.analyze(revenue_stream)
        findings = self._load_findings()
        findings["legal"].append(assessment)
        self._store_findings(findings)
        return assessment

    def generate_opportunity_report(self) -> Dict[str, Any]:
        """Combine all research into a ranked opportunity report.

        Opportunities are ranked by feasibility, risk, and revenue potential;
        each carries pros/cons-style fields plus recommendations and an
        explicit ``avoid`` list with reasons.

        Returns:
            Dict with ``research_date`` (ISO), ``opportunities`` (each with
            name, category, description, price_range, feasibility_score 1–10,
            risk_level, risk_factors, revenue_potential,
            technical_requirements, legal_considerations, automation_level,
            sources), ``recommendations``, and ``avoid``. The report is
            saved to ``state.research_findings["report"]``.

        Raises:
            ValueError: If no research has been collected yet, or the AI
                response cannot be normalised to the schema.
            RuntimeError: If the AI ranking call fails.
        """
        findings = self._load_findings()
        if not self._has_any_research(findings):
            raise ValueError(
                "No research collected yet — run research_reddit(), "
                "research_marketplace(), analyze_youtube_opportunities() or "
                "legal_risk_assessment() before generating a report."
            )
        _console.print("[cyan]Generating ranked opportunity report with AI…[/cyan]")
        digest = self._findings_digest(findings)
        try:
            raw_report = self._ask_json(
                "You are a conservative analyst ranking micro-revenue streams "
                "for a solo builder. Favour original, low-risk, automatable "
                "products (€5–€20 digital goods). Penalise copyright-grey "
                "tactics (re-uploads, clipper channels using others' footage). "
                + REPORT_SCHEMA_HINT,
                "Evidence collected:\n\n" + digest,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"Report generation failed: {exc}") from exc
        report = self._normalise_report(raw_report)
        findings["report"] = report
        self._store_findings(findings)
        _console.print(
            f"[green]Report ready: {len(report['opportunities'])} ranked "
            f"opportunities, {len(report['avoid'])} explicit avoids.[/green]"
        )
        return report

    # -- state plumbing ---------------------------------------------------------

    def _mark_researching(self) -> None:
        """Set phase to research (best effort — never masks real errors)."""
        try:
            self.state_manager.update_phase("research")
        except Exception as exc:
            _console.print(f"[yellow]Could not set research phase: {exc}[/yellow]")

    def _load_findings(self) -> Dict[str, Any]:
        """Load research_findings, falling back to a skeleton on any problem."""
        try:
            current = self.state_manager.get_state().research_findings
        except Exception:
            current = None
        if not isinstance(current, dict):
            return _skeleton_findings()
        merged = _skeleton_findings()
        for key, value in current.items():
            if key in merged:
                merged[key] = value
        return merged

    def _store_findings(self, findings: Dict[str, Any]) -> None:
        """Persist findings with a fresh timestamp."""
        findings["updated_at"] = _utcnow_iso()
        state = self.state_manager.get_state()
        state.research_findings = findings
        self.state_manager.save(state)

    def _update_findings(self, **parts: Any) -> None:
        """Merge ``parts`` into stored findings and persist."""
        findings = self._load_findings()
        findings.update(parts)
        self._store_findings(findings)

    def _append_marketplace(
        self, scan: Dict[str, Any], patterns: Dict[str, Any]
    ) -> None:
        """Append one marketplace scan (+ AI patterns) to stored findings."""
        findings = self._load_findings()
        record = dict(scan)
        record["patterns"] = patterns
        findings["marketplace"].append(record)
        self._store_findings(findings)

    # -- AI plumbing ------------------------------------------------------------

    def _ask_json(self, system_prompt: str, user_prompt: str) -> Any:
        """Call the model and parse a JSON response (errors → RuntimeError)."""
        try:
            raw = self.orchestrator.ask_agent(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                state_manager=self.state_manager,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Orchestrator call failed: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            return _parse_json(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            snippet = (raw or "").strip()[:200]
            raise RuntimeError(
                f"Model response was not valid JSON ({exc}); "
                f"started with: {snippet!r}."
            ) from exc

    # -- report normalisation -----------------------------------------------------

    @staticmethod
    def _has_any_research(findings: Dict[str, Any]) -> bool:
        """True when at least one collection step produced data."""
        return bool(
            findings.get("reddit")
            or findings.get("marketplace")
            or findings.get("youtube", {}).get("videos")
            or findings.get("legal")
        )

    @staticmethod
    def _findings_digest(findings: Dict[str, Any]) -> str:
        """Compress stored findings into an AI-sized evidence brief."""
        lines: List[str] = []
        reddit = findings.get("reddit", [])
        if reddit:
            lines.append(f"REDDIT ({len(reddit)} posts):")
            for post in reddit[:15]:
                lines.append(
                    f"- {post.get('post_title', '?')} "
                    f"({post.get('upvotes', 0)} upvotes, "
                    f"r/{post.get('subreddit', '?')})"
                )
            themes = findings.get("reddit_themes", {})
            if themes.get("themes"):
                lines.append("Themes: " + "; ".join(themes["themes"][:8]))
        for scan in findings.get("marketplace", [])[:5]:
            listings = scan.get("listings", [])[:10]
            lines.append(
                f"MARKETPLACE {scan.get('platform', '?')} "
                f"({scan.get('category', '?')}, {len(listings)} listings):"
            )
            for item in listings:
                lines.append(
                    f"- {item.get('title', '?')} (${item.get('price', '?')})"
                )
        youtube = findings.get("youtube", {})
        if youtube.get("videos"):
            lines.append(f"YOUTUBE ({youtube.get('niche', '?')}):")
            for video in youtube["videos"][:10]:
                lines.append(
                    f"- {video.get('title', '?')} [{video.get('channel', '?')}]"
                )
        for assessment in findings.get("legal", [])[:10]:
            lines.append(
                f"LEGAL [{assessment.get('risk_level', '?')}] "
                f"{assessment.get('revenue_stream', '?')}"
            )
        return "\n".join(lines)[:12000] or "(no evidence summarised)"

    def _normalise_report(self, raw: Any) -> Dict[str, Any]:
        """Coerce AI output to the exact report schema (or raise ValueError)."""
        if not isinstance(raw, dict) or not isinstance(
            raw.get("opportunities"), list
        ):
            raise ValueError(
                "Report response must be an object with an 'opportunities' list."
            )
        opportunities = [
            self._normalise_opportunity(item, index)
            for index, item in enumerate(raw["opportunities"])
        ]
        if not opportunities:
            raise ValueError("Report contained zero opportunities.")
        return {
            "research_date": _utcnow_iso(),
            "opportunities": opportunities,
            "recommendations": _str_list(raw.get("recommendations", [])),
            "avoid": _str_list(raw.get("avoid", [])),
            "legal_disclaimer": DISCLAIMER,
        }

    @staticmethod
    def _normalise_opportunity(item: Any, index: int) -> Dict[str, Any]:
        """Coerce one opportunity dict; fills safe defaults, clamps ranges."""
        if not isinstance(item, dict):
            raise ValueError(f"Opportunity #{index} is not an object.")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError(f"Opportunity #{index} is missing a name.")
        price = item.get("price_range", [5, 20])
        try:
            low, high = float(price[0]), float(price[1])
        except (TypeError, ValueError, IndexError):
            low, high = 5.0, 20.0
        try:
            score = int(float(item.get("feasibility_score", 5)))
        except (TypeError, ValueError):
            score = 5
        risk = str(item.get("risk_level", "medium")).strip().lower()
        if risk not in ("low", "medium", "high"):
            risk = "medium"
        automation = str(item.get("automation_level", "partial")).strip().lower()
        if automation not in ("full", "partial", "manual"):
            automation = "partial"
        return {
            "name": name[:120],
            "category": str(item.get("category", "digital product")).strip()[:80],
            "description": str(item.get("description", "")).strip()[:1000],
            "price_range": [low, high],
            "feasibility_score": max(1, min(10, score)),
            "risk_level": risk,
            "risk_factors": _str_list(item.get("risk_factors", [])),
            "revenue_potential": str(item.get("revenue_potential", "")).strip()[:500],
            "technical_requirements": _str_list(
                item.get("technical_requirements", [])
            ),
            "legal_considerations": _str_list(
                item.get("legal_considerations", [])
            ),
            "automation_level": automation,
            "sources": _str_list(item.get("sources", [])),
        }


def _str_list(values: Any) -> List[str]:
    """Coerce to a list of non-empty strings (caps at 20 items)."""
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
