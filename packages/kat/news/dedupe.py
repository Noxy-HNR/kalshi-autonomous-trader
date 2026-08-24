"""Deduplication, and why it is a trading concern rather than tidiness.

A news feed is mostly the same stories repeated. One wire report is reprinted by
a dozen outlets, each with its own headline and a paragraph of local context. If
those count as a dozen articles then "how many independent sources report this?"
-- which is the only real handle on whether a story is confirmed -- becomes a
count of how many outlets subscribe to Reuters.

So the pipeline collapses them, and keeps the count. Twelve copies of one wire
story become one article with eleven corroborations, which is the honest reading.

Two mechanisms
--------------
**Exact.** Same content hash means the same normalised text. Trivially a
duplicate.

**Near.** Simhash within a Hamming threshold. This is the case that dominates:
same story, different headline, one added sentence. The default threshold of 3
bits out of 64 is deliberately tight -- a loose threshold merges genuinely
different stories about the same subject, and losing a real story is worse than
counting one twice.

The window
----------
Deduplication is against a bounded recent window, not all history. Two reports of
"Fed holds rates" a month apart are two different events, and a system that
merged them would silently discard the second. The window is time-based rather
than count-based so that a quiet period does not widen it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from kat.news.normalise import simhash_distance
from kat.news.types import Article
from kat.observability.logging import get_logger

log = get_logger(__name__)

__all__ = ["DeduplicationResult", "Deduplicator"]

#: Hamming distance below which two articles are the same story. Tight on
#: purpose: merging two real stories loses one of them permanently.
DEFAULT_SIMHASH_THRESHOLD = 3

#: How far back to look. Beyond this, a repeat is a new event, not a reprint.
DEFAULT_WINDOW_MS = 6 * 3_600_000  # six hours


@dataclass(slots=True)
class DeduplicationResult:
    """What happened to one article."""

    article: Article
    is_duplicate: bool
    canonical_id: str | None = None
    distance: int | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "article_id": self.article.article_id,
            "is_duplicate": self.is_duplicate,
            "canonical_id": self.canonical_id,
            "distance": self.distance,
            "reason": self.reason,
        }


@dataclass(slots=True)
class _Seen:
    article_id: str
    content_hash: str
    simhash: int
    ingested_ms: int
    sources: set[str] = field(default_factory=set)


class Deduplicator:
    """Collapses reprints of one story, counting them as corroboration."""

    def __init__(
        self,
        *,
        simhash_threshold: int = DEFAULT_SIMHASH_THRESHOLD,
        window_ms: int = DEFAULT_WINDOW_MS,
        max_tracked: int = 5_000,
    ) -> None:
        self.simhash_threshold = simhash_threshold
        self.window_ms = window_ms
        self.max_tracked = max_tracked
        self._recent: deque[_Seen] = deque(maxlen=max_tracked)
        self._by_hash: dict[str, _Seen] = {}
        self.exact_duplicates = 0
        self.near_duplicates = 0
        self.unique_articles = 0

    def _evict(self, now_ms: int) -> None:
        """Drop entries older than the window from both structures."""
        cutoff = now_ms - self.window_ms
        while self._recent and self._recent[0].ingested_ms < cutoff:
            stale = self._recent.popleft()
            if self._by_hash.get(stale.content_hash) is stale:
                del self._by_hash[stale.content_hash]

    def check(self, article: Article) -> DeduplicationResult:
        """Classify one article against the recent window.

        Does not mutate the window; call :meth:`accept` for that. Separated so a
        caller can decide what to do with a duplicate before it becomes part of
        the history everything else is compared against.
        """
        self._evict(article.ingested_ms)

        exact = self._by_hash.get(article.content_hash)
        if exact is not None and exact.article_id != article.article_id:
            return DeduplicationResult(
                article=article,
                is_duplicate=True,
                canonical_id=exact.article_id,
                distance=0,
                reason="identical normalised text",
            )
        if exact is not None:
            # Same article id: we have literally seen this item before, from a
            # re-poll of the same feed.
            return DeduplicationResult(
                article=article,
                is_duplicate=True,
                canonical_id=exact.article_id,
                distance=0,
                reason="already ingested",
            )

        if article.simhash:
            best: _Seen | None = None
            best_distance = self.simhash_threshold + 1
            for seen in self._recent:
                if not seen.simhash:
                    continue
                distance = simhash_distance(article.simhash, seen.simhash)
                if distance < best_distance:
                    best, best_distance = seen, distance
            if best is not None and best_distance <= self.simhash_threshold:
                return DeduplicationResult(
                    article=article,
                    is_duplicate=True,
                    canonical_id=best.article_id,
                    distance=best_distance,
                    reason=f"near-duplicate within {best_distance} bits",
                )

        return DeduplicationResult(article=article, is_duplicate=False, reason="unique")

    def accept(self, result: DeduplicationResult) -> Article:
        """Record the outcome and return the article to carry forward.

        A duplicate returns the *original* article marked with ``duplicate_of``
        so the caller can drop it, while the canonical entry gains a
        corroboration. A unique article joins the window.
        """
        article = result.article
        if result.is_duplicate:
            canonical = next(
                (s for s in self._recent if s.article_id == result.canonical_id), None
            )
            if canonical is not None and article.source not in canonical.sources:
                # A reprint by a *different* outlet is corroboration; the same
                # outlet re-emitting its own story is not.
                canonical.sources.add(article.source)
            if result.distance == 0:
                self.exact_duplicates += 1
            else:
                self.near_duplicates += 1
            log.debug(
                "news.duplicate",
                article_id=article.article_id,
                canonical=result.canonical_id,
                distance=result.distance,
                source=article.source,
            )
            return article.evolve(duplicate_of=result.canonical_id)

        self._recent.append(
            _Seen(
                article_id=article.article_id,
                content_hash=article.content_hash,
                simhash=article.simhash,
                ingested_ms=article.ingested_ms,
                sources={article.source},
            )
        )
        self._by_hash[article.content_hash] = self._recent[-1]
        self.unique_articles += 1
        return article

    def process(self, articles: list[Article]) -> list[Article]:
        """Deduplicate a batch, returning only the unique articles.

        Corroboration counts are attached to the survivors, so a story reported
        by six outlets emerges once with five corroborations rather than six
        times with none.
        """
        unique: list[Article] = []
        for article in articles:
            result = self.check(article)
            carried = self.accept(result)
            if not result.is_duplicate:
                unique.append(carried)

        # Fold the corroboration counts back in now that the whole batch is known.
        counts = {s.article_id: max(0, len(s.sources) - 1) for s in self._recent}
        return [
            a.evolve(corroborations=counts.get(a.article_id, a.corroborations))
            for a in unique
        ]

    def corroborations(self, article_id: str) -> int:
        """How many *distinct other* sources carried this story."""
        for seen in self._recent:
            if seen.article_id == article_id:
                return max(0, len(seen.sources) - 1)
        return 0

    def stats(self) -> dict[str, Any]:
        total = self.unique_articles + self.exact_duplicates + self.near_duplicates
        return {
            "tracked": len(self._recent),
            "unique": self.unique_articles,
            "exact_duplicates": self.exact_duplicates,
            "near_duplicates": self.near_duplicates,
            "duplicate_rate": (
                round((self.exact_duplicates + self.near_duplicates) / total, 4)
                if total
                else None
            ),
        }
