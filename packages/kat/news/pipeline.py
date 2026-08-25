"""The news pipeline, end to end.

::

    sources -> ingest -> normalise -> dedupe -> extract -> match -> impact
            -> research signals -> persistence

Two properties hold throughout.

**Point-in-time integrity.** Every article carries the moment *this process*
could first have acted on it, stamped once at normalisation. :meth:`NewsPipeline.
signals_at` filters on that and nothing else, so a backtest asking "what did we
know at T" gets exactly what we knew, not what had been published by then.

**Failure isolation.** One dead feed, one malformed item, one regex that throws
must not stop the others. Every stage catches per-item, counts the failure, and
carries on — a news outage should degrade the research signal, not halt trading.

The pipeline is a background component. Nothing on the order path waits for it,
and a strategy that finds no news signal simply trades on everything else.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import httpx

from kat.common.enums import MarketType
from kat.common.timeutil import Clock
from kat.config.settings import NewsSettings
from kat.db.writer import AsyncWriter
from kat.news.dedupe import Deduplicator
from kat.news.extract import Extractor
from kat.news.impact import ImpactEstimate, ImpactEstimator
from kat.news.matching import MarketMatch, MarketMatcher, MarketProfile
from kat.news.normalise import normalise
from kat.news.sources import NewsSource, load_sources
from kat.news.types import Article, DataQuality
from kat.observability.logging import get_logger
from kat.observability.metrics import metrics

log = get_logger(__name__)

__all__ = ["NewsPipeline", "PipelineResult"]


@dataclass(slots=True)
class PipelineResult:
    """What one poll cycle did."""

    fetched: int = 0
    normalised: int = 0
    unique: int = 0
    matched: int = 0
    signals: int = 0
    dropped: int = 0
    future_dated: int = 0
    source_failures: list[str] = field(default_factory=list)
    item_failures: int = 0
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "fetched": self.fetched,
            "normalised": self.normalised,
            "unique": self.unique,
            "matched": self.matched,
            "signals": self.signals,
            "dropped": self.dropped,
            "future_dated": self.future_dated,
            "source_failures": list(self.source_failures),
            "item_failures": self.item_failures,
            "duration_ms": round(self.duration_ms, 1),
        }


@dataclass(slots=True)
class StoredSignal:
    """A research signal with the article it came from, kept for replay."""

    ticker: str
    market_type: MarketType
    ingested_ms: int
    payload: dict[str, Any]


def _interleave(batches: list[list], cap: int) -> list:
    """Round-robin across sources, up to ``cap`` items in total.

    Concatenating every source's items and truncating looks equivalent and is
    not: with twenty sources and a cap that binds, the sources at the end of the
    list are cut every single cycle and are effectively never ingested at all.
    Taking one item from each source in turn spends the budget evenly, and
    because each source orders its own feed newest-first, what survives the cap
    is the newest items from everywhere rather than everything from somewhere.
    """
    out: list = []
    if cap <= 0:
        return out
    index = 0
    while len(out) < cap:
        progressed = False
        for batch in batches:
            if index < len(batch):
                out.append(batch[index])
                progressed = True
                if len(out) >= cap:
                    break
        if not progressed:
            break
        index += 1
    return out


class NewsPipeline:
    """Polls free sources and turns them into per-market research signals."""

    def __init__(
        self,
        settings: NewsSettings,
        clock: Clock,
        *,
        writer: AsyncWriter | None = None,
        sources: list[NewsSource] | None = None,
        retention: int = 4_000,
    ) -> None:
        self.settings = settings
        self.clock = clock
        self.writer = writer
        self.sources = sources if sources is not None else []
        self.deduplicator = Deduplicator()
        self.extractor = Extractor()
        self.matcher = MarketMatcher()
        self.estimator = ImpactEstimator()

        #: Signals are kept in memory, newest last, so a strategy can ask for
        #: what was knowable at a past instant without a database round-trip.
        self._signals: deque[StoredSignal] = deque(maxlen=retention)
        self._articles: deque[Article] = deque(maxlen=retention)

        self.cycles = 0
        self.last_result: PipelineResult | None = None
        self._client: httpx.AsyncClient | None = None

    # -- lifecycle -----------------------------------------------------------

    def load_sources(self) -> int:
        """Load feed definitions from configuration.

        A malformed configuration disables news rather than stopping start-up:
        the platform trades without news, it just trades with less information.
        """
        try:
            self.sources = load_sources(self.settings.feeds_file)
        except Exception as exc:
            log.error("news.sources_unloadable", error=str(exc), file=self.settings.feeds_file)
            self.sources = []
        log.info("news.sources_loaded", count=len(self.sources))
        return len(self.sources)

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.settings.request_timeout_s, connect=5.0),
                headers={"User-Agent": self.settings.user_agent},
                follow_redirects=True,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def set_markets(self, profiles: list[MarketProfile]) -> None:
        self.matcher.set_markets(profiles)

    # -- the cycle -----------------------------------------------------------

    async def poll(self) -> PipelineResult:
        """One full pass over every source."""
        import time

        started = time.monotonic()
        result = PipelineResult()
        self.cycles += 1

        if not self.sources:
            result.duration_ms = (time.monotonic() - started) * 1000
            self.last_result = result
            return result

        client = self._http()
        # Concurrent, and every failure isolated to its own source.
        fetched = await asyncio.gather(
            *(self._fetch(source, result) for source in self.sources),
            return_exceptions=True,
        )

        batches = [b for b in fetched if isinstance(b, list)]
        result.fetched = sum(len(b) for b in batches)
        raw_items = _interleave(batches, self.settings.max_articles_per_poll)
        result.dropped = result.fetched - len(raw_items)
        if result.dropped:
            # Never drop silently: a cap that quietly discards a fifth of the
            # feed looks identical in the logs to a cap that never binds.
            log.warning(
                "news.poll_capped",
                fetched=result.fetched,
                kept=len(raw_items),
                dropped=result.dropped,
                cap=self.settings.max_articles_per_poll,
            )

        articles: list[Article] = []
        for item in raw_items:
            try:
                article = normalise(item, self.clock)
            except Exception as exc:  # one bad item, not the batch
                result.item_failures += 1
                log.warning("news.normalise_failed", source=item.source_id, error=str(exc))
                continue
            if article is not None:
                articles.append(article)
        result.normalised = len(articles)

        # One line per affected source per cycle, rather than one per article.
        # A source whose clock or timezone is wrong is worth knowing about once;
        # it is not worth thirty warnings every five minutes forever.
        future_dated: dict[str, int] = {}
        for article in articles:
            if (
                article.quality is DataQuality.SUSPECT
                and article.published_ms
                and article.published_ms > article.ingested_ms
            ):
                future_dated[article.source] = future_dated.get(article.source, 0) + 1
        for source, count in future_dated.items():
            worst = max(
                a.published_ms - a.ingested_ms
                for a in articles
                if a.source == source and a.published_ms and a.published_ms > a.ingested_ms
            )
            log.warning(
                "news.source_clock_ahead",
                source=source,
                articles=count,
                worst_ahead_ms=worst,
                note=(
                    "the source timestamps items after we received them, which is "
                    "impossible; they are kept and flagged, and nothing downstream "
                    "reads published time for decisions"
                ),
            )
        result.future_dated = sum(future_dated.values())

        unique = self.deduplicator.process(articles)
        result.unique = len(unique)

        enriched: list[Article] = []
        for article in unique:
            try:
                enriched.append(self.extractor.enrich(article))
            except Exception as exc:
                result.item_failures += 1
                log.warning(
                    "news.extract_failed", article_id=article.article_id, error=str(exc)
                )
                enriched.append(article)

        now_ms = self.clock.now_ms()
        for article in enriched:
            try:
                matches = self.matcher.match(article)
            except Exception as exc:
                result.item_failures += 1
                log.warning("news.match_failed", article_id=article.article_id, error=str(exc))
                matches = []

            top = max((m.relevance for m in matches), default=None)
            stored = article.evolve(relevance=top)
            self._articles.append(stored)

            if self.writer is not None:
                self.writer.enqueue(stored.to_orm())

            if matches:
                result.matched += 1
            for match in matches:
                signal = self._build_signal(stored, match, now_ms)
                if signal is not None:
                    result.signals += 1

        result.duration_ms = (time.monotonic() - started) * 1000
        self.last_result = result

        metrics.increment("news.articles_ingested", result.unique)
        metrics.increment("news.signals_generated", result.signals)
        log.info("news.cycle", **result.to_dict())
        _ = client  # the session is reused across cycles
        return result

    async def _fetch(self, source: NewsSource, result: PipelineResult) -> list:
        try:
            return await source.fetch(self._http())
        except Exception as exc:  # one dead feed must not stop the rest
            result.source_failures.append(f"{source.source_id}: {type(exc).__name__}")
            log.warning("news.source_failed", source=source.source_id, error=str(exc))
            return []

    def _build_signal(
        self, article: Article, match: MarketMatch, now_ms: int
    ) -> ImpactEstimate | None:
        profile = self.matcher.profile(match.ticker)
        estimate = self.estimator.estimate(
            article,
            ticker=match.ticker,
            market_title=profile.title if profile else "",
            market_rules=profile.rules if profile else "",
            relevance=match.relevance,
            now_ms=now_ms,
        )

        if self.writer is not None:
            self.writer.enqueue(
                match.to_orm(
                    direction=estimate.direction,
                    impact=estimate.magnitude,
                    confidence=estimate.confidence,
                )
            )

        # Stored even without a resolved direction: "there is news here but we
        # cannot tell which way" is information a strategy may act on by standing
        # aside, and it belongs in the record either way.
        self._signals.append(
            StoredSignal(
                ticker=match.ticker,
                market_type=match.market_type,
                ingested_ms=article.ingested_ms,
                payload=self.estimator.to_research_signal(estimate, article),
            )
        )
        return estimate

    # -- consumption ---------------------------------------------------------

    def signals_at(self, now_ms: int, *, max_age_ms: int = 6 * 3_600_000) -> dict[str, dict]:
        """Research signals knowable at ``now_ms``, keyed by ticker.

        The filter is on **ingestion** time. This is the function a backtest
        calls, and it is why the whole pipeline is careful about which timestamp
        it stamps where.
        """
        cutoff = now_ms - max_age_ms
        by_ticker: dict[str, list[dict]] = {}
        for signal in self._signals:
            if signal.ingested_ms > now_ms or signal.ingested_ms < cutoff:
                continue
            by_ticker.setdefault(signal.ticker, []).append(signal.payload)

        return {
            ticker: {"news": sorted(items, key=lambda p: -(p.get("ingested_ms") or 0))}
            for ticker, items in by_ticker.items()
        }

    def recent_articles(self, limit: int = 50) -> list[Article]:
        return list(self._articles)[-limit:][::-1]

    # -- component wiring ----------------------------------------------------

    async def run(self) -> None:
        """Poll forever. Registered as a supervised component."""
        if not self.sources:
            self.load_sources()
        log.info(
            "news.pipeline_started",
            sources=len(self.sources),
            interval_s=self.settings.poll_interval_s,
        )
        while True:
            try:
                await self.poll()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # the loop outlives a bad cycle
                log.error("news.cycle_failed", error=str(exc), exc_info=True)
            await asyncio.sleep(self.settings.poll_interval_s)

    def healthy(self) -> bool:
        """Healthy when at least one source produced something recently.

        A pipeline with no sources configured is reported healthy: news is
        optional, and "switched off" is not "broken".
        """
        if not self.sources:
            return True
        if self.last_result is None:
            return True  # not yet run
        return len(self.last_result.source_failures) < len(self.sources)

    def stats(self) -> dict[str, Any]:
        return {
            "sources": len(self.sources),
            "cycles": self.cycles,
            "articles_retained": len(self._articles),
            "signals_retained": len(self._signals),
            "last_cycle": self.last_result.to_dict() if self.last_result else None,
            "dedupe": self.deduplicator.stats(),
            "extractor": self.extractor.stats(),
            "matcher": self.matcher.stats(),
            "impact": self.estimator.stats(),
        }
