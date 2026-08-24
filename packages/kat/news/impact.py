"""Estimating what a story means for a market.

This is the stage that most invites overclaiming, so it is worth being blunt
about what it produces: a **direction**, a **magnitude in probability points**,
and a **confidence** — all three coarse, all three frequently wrong, and all
three designed so that the honest answer "no view" is easy to return and often
returned.

Direction is the hard part
--------------------------
An extracted event carries a polarity relative to its *subject*: "wins" is +1 for
whoever won. Turning that into a market direction requires knowing which side of
the market the subject sits on, and that is genuinely often impossible from text.
"Smith wins primary" is +1 for Smith — but a market titled "Will Jones win the
nomination?" needs the opposite sign, and nothing in the sentence says so.

So direction is only asserted when the market's own text names the subject in a
way that resolves the polarity. Otherwise the result is ``None``, which
downstream treats as "there is news here, but we do not know which way", and
which a strategy is expected to decline to trade on.

Magnitude
---------
A blend of three things that can each be measured: how confident the extraction
was, how relevant the article is to the market, and how reliable and corroborated
the source is. The output is capped well below what a genuine surprise can do to
a price, because the failure mode of over-estimating impact is a large position
taken on a misread headline.

Decay
-----
News is perishable. Impact decays from **ingestion** time, never from publication
time, on a half-life. This is the same distinction that matters everywhere else
in the pipeline: a back-dated government feed item is fresh to us the moment we
receive it, and a strategy must not treat it as already stale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from kat.news.types import Article, EventType
from kat.observability.logging import get_logger

log = get_logger(__name__)

__all__ = ["ImpactEstimate", "ImpactEstimator"]

#: How quickly a story stops mattering. Half its impact is gone after this.
DEFAULT_HALF_LIFE_MS = 45 * 60_000  # 45 minutes

#: Beyond this an article is treated as having no remaining impact at all,
#: rather than an exponentially small one that still nudges a size.
DEFAULT_MAX_AGE_MS = 6 * 3_600_000

#: Ceiling on the probability shift any single story is allowed to imply.
#: Over-estimating impact is how a misread headline becomes a large position.
MAX_IMPACT = 0.25

#: Base impact by event type, before confidence, relevance and decay. These are
#: judgement calls, chosen so that a routine scheduled release moves less than a
#: genuine surprise.
_BASE_IMPACT: dict[EventType, float] = {
    EventType.ELECTION_RESULT: 0.30,
    EventType.RESIGNATION: 0.22,
    EventType.APPOINTMENT: 0.15,
    EventType.LEGAL_ACTION: 0.15,
    EventType.MONETARY_POLICY: 0.22,
    EventType.ECONOMIC_RELEASE: 0.12,
    EventType.LEGISLATION: 0.16,
    EventType.REGULATION: 0.10,
    EventType.POLL_RELEASE: 0.08,
    EventType.EARNINGS: 0.12,
    EventType.MERGER: 0.18,
    EventType.CONFLICT: 0.20,
    EventType.DIPLOMACY: 0.16,
    EventType.DISASTER: 0.18,
    EventType.WEATHER_ALERT: 0.14,
    EventType.HEALTH: 0.12,
    EventType.SPORTS_RESULT: 0.30,
    EventType.LAUNCH: 0.20,
    EventType.DEATH: 0.20,
    EventType.UNKNOWN: 0.05,
}


@dataclass(slots=True)
class ImpactEstimate:
    """What one article implies for one market."""

    article_id: str
    ticker: str
    direction: str | None
    magnitude: float
    confidence: float
    freshness: float
    event_types: tuple[str, ...] = ()
    reason: str = ""

    @property
    def has_view(self) -> bool:
        """Whether this is actionable at all.

        A direction of ``None`` means "something happened and we cannot tell
        which way", which is a legitimate and common answer.
        """
        return self.direction is not None and self.magnitude > 0 and self.confidence > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "article_id": self.article_id,
            "ticker": self.ticker,
            "direction": self.direction,
            "magnitude": round(self.magnitude, 4),
            "confidence": round(self.confidence, 4),
            "freshness": round(self.freshness, 4),
            "event_types": list(self.event_types),
            "has_view": self.has_view,
            "reason": self.reason,
        }


def freshness(article: Article, now_ms: int, *, half_life_ms: int = DEFAULT_HALF_LIFE_MS,
              max_age_ms: int = DEFAULT_MAX_AGE_MS) -> float:
    """Remaining weight in [0, 1], decayed from **ingestion**.

    Measured from when we could first have acted, not from what the source
    claims. A feed that back-dates its items by six hours would otherwise
    deliver everything already expired.
    """
    age = article.age_ms(now_ms)
    if age >= max_age_ms:
        return 0.0
    return math.pow(0.5, age / half_life_ms)


class ImpactEstimator:
    """Turns matched articles into directional views, or into honest silence."""

    def __init__(
        self,
        *,
        half_life_ms: int = DEFAULT_HALF_LIFE_MS,
        max_age_ms: int = DEFAULT_MAX_AGE_MS,
        min_confidence: float = 0.25,
    ) -> None:
        self.half_life_ms = half_life_ms
        self.max_age_ms = max_age_ms
        self.min_confidence = min_confidence
        self.estimated = 0
        self.with_direction = 0

    def _resolve_direction(
        self, article: Article, market_title: str, market_rules: str
    ) -> tuple[str | None, str]:
        """Work out which side of the market the news favours.

        Returns ``(direction, reason)`` where direction is ``"yes"``, ``"no"`` or
        ``None``. ``None`` is returned whenever the subject of the event cannot
        be tied to the market's own wording, which is most of the time.
        """
        market_text = f"{market_title} {market_rules}".lower()

        signed = [e for e in article.events if e.polarity != 0 and e.subject]
        if not signed:
            return None, "no event with both a polarity and a subject"

        best = max(signed, key=lambda e: e.confidence)
        subject_surface = best.subject.replace("_", " ")

        if subject_surface not in market_text:
            # The event has a subject, but the market does not name it, so we
            # cannot tell whether the market is asking about that subject or
            # about its opponent.
            return None, (
                f"event subject '{subject_surface}' does not appear in the market text; "
                "direction is unresolvable"
            )

        # The market names the subject, so the subject's polarity carries over.
        direction = "yes" if best.polarity > 0 else "no"
        return direction, (
            f"'{best.trigger}' with polarity {best.polarity:+d} for "
            f"'{subject_surface}', which the market names"
        )

    def estimate(
        self,
        article: Article,
        *,
        ticker: str,
        market_title: str,
        market_rules: str = "",
        relevance: float,
        now_ms: int,
    ) -> ImpactEstimate:
        """Estimate impact for one article-market pairing."""
        self.estimated += 1

        current_freshness = freshness(
            article, now_ms, half_life_ms=self.half_life_ms, max_age_ms=self.max_age_ms
        )
        event_types = tuple(dict.fromkeys(e.event_type.value for e in article.events))

        if current_freshness <= 0:
            return ImpactEstimate(
                article_id=article.article_id, ticker=ticker, direction=None,
                magnitude=0.0, confidence=0.0, freshness=0.0, event_types=event_types,
                reason="article is older than the maximum useful age",
            )

        if not article.events:
            return ImpactEstimate(
                article_id=article.article_id, ticker=ticker, direction=None,
                magnitude=0.0, confidence=0.0, freshness=current_freshness,
                event_types=(), reason="no typed event was extracted",
            )

        strongest = max(article.events, key=lambda e: e.confidence)
        base = _BASE_IMPACT.get(strongest.event_type, 0.05)

        # Corroboration: independent outlets carrying the same story is the
        # cheapest available evidence that it is real. Saturating, because the
        # tenth outlet adds nothing the third did not.
        corroboration = min(1.0, article.corroborations / 3.0)

        confidence = (
            0.40 * strongest.confidence
            + 0.25 * relevance
            + 0.20 * article.source_reliability
            + 0.15 * corroboration
        ) * current_freshness

        magnitude = min(MAX_IMPACT, base * relevance * strongest.confidence * current_freshness)

        direction, reason = self._resolve_direction(article, market_title, market_rules)

        if confidence < self.min_confidence:
            direction = None
            reason = f"confidence {confidence:.3f} is below the {self.min_confidence} floor"

        if direction is not None:
            self.with_direction += 1

        return ImpactEstimate(
            article_id=article.article_id,
            ticker=ticker,
            direction=direction,
            magnitude=magnitude,
            confidence=confidence,
            freshness=current_freshness,
            event_types=event_types,
            reason=reason,
        )

    def to_research_signal(
        self, estimate: ImpactEstimate, article: Article
    ) -> dict[str, Any]:
        """The shape ``StrategyContext.research["news"]`` entries take.

        Both timestamps travel with the signal so a strategy applying its own
        staleness cutoff uses ingestion time, as it must.
        """
        return {
            "article_id": article.article_id,
            "headline": article.title,
            "direction": estimate.direction,
            "estimated_impact": estimate.magnitude,
            "confidence": estimate.confidence,
            "relevance": article.relevance or 0.0,
            "published_ms": article.published_ms,
            "ingested_ms": article.ingested_ms,
            "source": article.source,
            "source_reliability": article.source_reliability,
            "corroborations": article.corroborations,
            "event_types": list(estimate.event_types),
        }

    def stats(self) -> dict[str, Any]:
        return {
            "estimated": self.estimated,
            "with_direction": self.with_direction,
            "directional_rate": (
                round(self.with_direction / self.estimated, 4) if self.estimated else None
            ),
        }
