"""Matching articles to Kalshi markets.

The hard part is not finding *a* connection -- it is refusing the many weak ones.
An article mentioning "the Federal Reserve" is topically related to every
interest-rate market on the exchange, and acting on that relationship would mean
trading every rate market on every Fed story. Relevance has to be scored, and the
bar has to be high enough that most articles match nothing.

Scoring
-------
Four signals, combined into a relevance in [0, 1]:

**Ticker match.** The market's ticker or series appears in the text. Rare and
almost always decisive when it happens.

**Entity overlap.** Entities the extractor found that also appear in the market's
title or rules, weighted by the entity's salience in the article. An entity in
the headline counts for much more than one mentioned once.

**Keyword overlap.** TF-IDF keywords against the market's title. This is the
statistical half, and it is what catches a relationship the gazetteer has never
heard of.

**Event-type affinity.** A monetary-policy story is more likely to matter to a
rate market than a sports market. Coarse, and only ever a multiplier -- it can
strengthen or weaken a match that already exists on other evidence, never create
one on its own.

Why the bar is high
-------------------
The default threshold rejects most pairings. That is the intended behaviour: the
downstream consumer sizes a position on this number, and a system that finds
every article "somewhat relevant" to forty markets will trade forty times on one
piece of news. An article matching nothing is the common case and is not a
failure.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from kat.common.enums import MarketType
from kat.common.timeutil import utcnow_aware
from kat.db.models import NewsMarketLink
from kat.news.normalise import tokenise
from kat.news.types import Article, EntityKind, EventType
from kat.observability.logging import get_logger

log = get_logger(__name__)

__all__ = ["MarketMatch", "MarketMatcher", "MarketProfile"]

#: Below this, a pairing is noise. Deliberately high; see the module docstring.
#:
#: Calibrated against the live exchange rather than chosen by feel. Scoring 192
#: real articles against 500 real markets produced 20,578 non-zero pairs, and
#: 5,728 of them landed on exactly 0.367 -- the value produced by *one* headline
#: entity matching with no keyword corroboration at all. That spike is the
#: "topically related to every rate market" failure the module docstring warns
#: about, and an example from it is instructive: "76 percent in Canada say
#: ending US trade negotiations was right" matched "US nominal GDP growth in
#: 2036" on the single entity `us` and the single keyword `percent`.
#:
#: A threshold of 0.40 sits just above that spike. It cut the surviving pairs
#: from 10,119 to 4,190 while still keeping 84 of the 107 articles that matched
#: anything, so it discards the degenerate single-entity tier without costing
#: much genuine coverage.
DEFAULT_RELEVANCE_THRESHOLD = 0.40

#: Distinct *events* one article may produce signals for. The cap has to be on
#: events rather than markets because a market is one strike on a ladder: a
#: single Fed story legitimately bears on every rung of "Fed funds rate at end
#: of 2036", and a market-level cap of five would spend the whole budget on one
#: ladder and leave every other event with nothing.
DEFAULT_MAX_EVENTS_PER_ARTICLE = 3

#: Hard ceiling regardless of how the events divide, so a pathologically wide
#: ladder cannot flood the signal set.
DEFAULT_MAX_MATCHES_PER_ARTICLE = 15

#: Event types that plausibly move a market whose text mentions these themes.
#: Only ever a multiplier on evidence that already exists.
_EVENT_AFFINITY: dict[EventType, tuple[str, ...]] = {
    EventType.MONETARY_POLICY: ("rate", "fed", "fomc", "inflation", "interest", "cpi"),
    EventType.ECONOMIC_RELEASE: ("cpi", "gdp", "inflation", "unemployment", "jobs", "payroll"),
    EventType.ELECTION_RESULT: ("election", "win", "president", "senate", "governor", "primary"),
    EventType.POLL_RELEASE: ("election", "win", "president", "senate", "governor", "poll"),
    EventType.WEATHER_ALERT: ("temperature", "high", "low", "rain", "snow", "hurricane", "storm"),
    EventType.DISASTER: ("hurricane", "storm", "earthquake", "wildfire", "flood"),
    EventType.CONFLICT: ("war", "strike", "invasion", "ceasefire", "military"),
    EventType.DIPLOMACY: ("ceasefire", "peace", "treaty", "summit", "deal"),
    EventType.LEGISLATION: ("bill", "law", "act", "pass", "congress", "shutdown"),
    EventType.EARNINGS: ("earnings", "revenue", "profit", "eps"),
    EventType.SPORTS_RESULT: ("win", "beat", "game", "match", "championship", "series"),
    EventType.LAUNCH: ("launch", "rocket", "orbit", "spacex", "starship"),
}


#: Above this many distinct tokens, a market's text is broad enough that overlap
#: starts happening by accident. Chosen from the observed distribution of real
#: Kalshi markets: an ordinary market's title and rules run 20-80 distinct
#: tokens, so the penalty is inert for almost every real market and bites only
#: on the outliers.
_BREADTH_FREE_TOKENS = 90


def _breadth_penalty(token_count: int) -> float:
    """Discount for markets whose text is broad enough to match by accident.

    Normalising overlap against the *article* alone is only half the problem. A
    market whose title is a 200-token concatenation -- Kalshi's multi-leg parlay
    markets list every leg, every team and every player in one title -- has far
    more chances to collide with an arbitrary story than a market titled "Will
    the Fed cut in September?". Without this, those markets match essentially
    every article and crowd out the markets the story is genuinely about, which
    is exactly what they did the first time this ran against the live exchange.

    The penalty is ``sqrt`` rather than linear so that a market with modestly
    more text is barely affected, while one with an order of magnitude more has
    to clear a proportionally higher bar of actual overlap.
    """
    if token_count <= _BREADTH_FREE_TOKENS:
        return 1.0
    return math.sqrt(_BREADTH_FREE_TOKENS / token_count)


#: Vocabulary that appears in Kalshi resolution rules regardless of subject.
#:
#: Every market's rules close with some variant of "...then the market resolves
#: to Yes", and most name a source and a settlement time. Those tokens are
#: therefore shared by the entire exchange and carry no information about what a
#: market is *about* -- but the scorer cannot tell them from real content, so an
#: article containing the word "market" or "yes" earned keyword credit against
#: every market listed. That is how "Yes, yields are important, but don't forget
#: about Nvidia" scored 0.70 against a 2036 fed funds strike.
#:
#: This is domain stopwording: the same idea as :mod:`kat.news.normalise`'s
#: stopword list, applied to the market side of the comparison.
_RULES_BOILERPLATE = frozenset(
    """market markets resolve resolves resolved resolution yes no then if above
    below between shall will would according source official officially publish
    published publishes website site report reported determined determination
    settle settled settlement expire expires expiration close closes closing
    time date day et pm am utc value values announced announcement per
    accordance case event events series ticker contract contracts""".split()  # noqa: SIM905 - a prose word list stays readable; a list literal does not
)


@dataclass(slots=True)
class MarketProfile:
    """The text of a market, prepared once for repeated matching.

    Built from the market's title, subtitle and resolution rules. The rules text
    matters more than it looks: it is where the specific named entities usually
    live, and it is the difference between matching "will inflation exceed 3%"
    on the word "inflation" and matching it on "CPI year-over-year".
    """

    ticker: str
    market_type: MarketType
    title: str
    event_ticker: str = ""
    series_ticker: str = ""
    rules: str = ""
    close_ms: int | None = None
    tokens: frozenset[str] = field(default_factory=frozenset)
    ticker_terms: frozenset[str] = field(default_factory=frozenset)
    #: Tokenised market text, space-delimited and space-padded, so a multi-word
    #: entity can be located with ``" phrase " in haystack`` -- a substring test
    #: that cannot match inside a longer word the way a raw ``in`` on the title
    #: can. Without this, the entity "us" matched "because", "trust" and
    #: "industry", and scored real relevance on every one of them.
    haystack: str = ""

    @classmethod
    def build(
        cls,
        *,
        ticker: str,
        market_type: MarketType,
        title: str,
        event_ticker: str = "",
        series_ticker: str = "",
        rules: str = "",
        close_ms: int | None = None,
    ) -> MarketProfile:
        text = f"{title} {rules}"
        # Ticker segments are searched for literally, so a market whose ticker
        # names its subject can match on that alone.
        segments = {
            part.lower()
            for raw in (ticker, event_ticker, series_ticker)
            for part in re.split(r"[-_]", raw)
            if len(part) > 3 and not part.isdigit()
        }
        return cls(
            ticker=ticker,
            market_type=market_type,
            title=title,
            event_ticker=event_ticker,
            series_ticker=series_ticker,
            rules=rules,
            close_ms=close_ms,
            tokens=frozenset(tokenise(text)) - _RULES_BOILERPLATE,
            ticker_terms=frozenset(segments),
            haystack=f" {' '.join(tokenise(text))} ",
        )


@dataclass(slots=True)
class MarketMatch:
    """One article-to-market pairing, with the evidence behind it."""

    article_id: str
    ticker: str
    market_type: MarketType
    relevance: float
    #: Carried so the cap can group a strike ladder as one event; see
    #: :meth:`MarketMatcher._limit`.
    event_ticker: str = ""
    ticker_hit: bool = False
    entity_overlap: tuple[str, ...] = ()
    keyword_overlap: tuple[str, ...] = ()
    event_types: tuple[str, ...] = ()
    matched_by: str = "combined"

    def to_orm(self, *, direction: str | None = None, impact: float | None = None,
               confidence: float | None = None) -> NewsMarketLink:
        return NewsMarketLink(
            article_id=self.article_id,
            ticker=self.ticker,
            market_type=self.market_type.value,
            relevance=self.relevance,
            estimated_impact=impact,
            direction=direction,
            confidence=confidence,
            matched_by=self.matched_by,
            ts=utcnow_aware(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "article_id": self.article_id,
            "ticker": self.ticker,
            "market_type": self.market_type.value,
            "event_ticker": self.event_ticker,
            "relevance": round(self.relevance, 4),
            "ticker_hit": self.ticker_hit,
            "entity_overlap": list(self.entity_overlap),
            "keyword_overlap": list(self.keyword_overlap),
            "event_types": list(self.event_types),
            "matched_by": self.matched_by,
        }


class MarketMatcher:
    """Scores articles against a set of market profiles."""

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
        max_events_per_article: int = DEFAULT_MAX_EVENTS_PER_ARTICLE,
        max_matches_per_article: int = DEFAULT_MAX_MATCHES_PER_ARTICLE,
    ) -> None:
        self.threshold = threshold
        #: One story should not trade forty unrelated events. A cap on top of
        #: the threshold, because a very general story clears the bar broadly.
        self.max_events_per_article = max_events_per_article
        self.max_matches_per_article = max_matches_per_article
        self._profiles: dict[str, MarketProfile] = {}
        self.scored = 0
        self.matched = 0

    def set_markets(self, profiles: list[MarketProfile]) -> None:
        self._profiles = {p.ticker: p for p in profiles}

    def add_market(self, profile: MarketProfile) -> None:
        self._profiles[profile.ticker] = profile

    def profile(self, ticker: str) -> MarketProfile | None:
        """The profile behind a match, for a caller that needs the market text."""
        return self._profiles.get(ticker)

    def __len__(self) -> int:
        return len(self._profiles)

    def score(self, article: Article, profile: MarketProfile) -> MarketMatch | None:
        """Score one pairing, or ``None`` when there is no evidence at all."""
        text = article.text.lower()

        ticker_hit = any(term in text for term in profile.ticker_terms)

        entity_hits: list[tuple[str, float]] = []
        for entity in article.entities:
            if entity.kind in (EntityKind.NUMBER, EntityKind.DATE, EntityKind.MONEY):
                continue
            surface = " ".join(tokenise(entity.text))
            normalised = entity.normalised.replace("_", " ")
            if surface and f" {surface} " in profile.haystack:
                entity_hits.append((entity.normalised, entity.salience))
            elif normalised and f" {normalised} " in profile.haystack:
                entity_hits.append((entity.normalised, entity.salience * 0.8))

        keyword_hits = [
            (term, weight) for term, weight in article.keywords if term in profile.tokens
        ]

        if not (ticker_hit or entity_hits or keyword_hits):
            return None

        # Entity component: salience-weighted, saturating. Two strong entities is
        # much better than one; six is not much better than three.
        entity_score = min(1.0, sum(salience for _, salience in entity_hits) / 1.5)

        # Keyword component: normalised against the article's own top weight so a
        # long article does not automatically outscore a short one.
        top_weight = max((w for _, w in article.keywords), default=0.0) or 1.0
        keyword_score = min(1.0, sum(w for _, w in keyword_hits) / (top_weight * 2.5))

        base = 0.55 * entity_score + 0.45 * keyword_score
        base *= _breadth_penalty(len(profile.tokens))
        if ticker_hit:
            # A literal ticker match is close to decisive on its own.
            base = max(base, 0.75)

        # Event affinity multiplies evidence that already exists; it never
        # creates a match by itself.
        affinity = 1.0
        event_types: list[str] = []
        for event in article.events:
            themes = _EVENT_AFFINITY.get(event.event_type)
            if not themes:
                continue
            if any(theme in profile.tokens for theme in themes):
                affinity = max(affinity, 1.0 + 0.25 * event.confidence)
                event_types.append(event.event_type.value)

        relevance = min(1.0, base * affinity)
        if relevance <= 0:
            return None

        entity_hits.sort(key=lambda kv: -kv[1])
        keyword_hits.sort(key=lambda kv: -kv[1])
        return MarketMatch(
            article_id=article.article_id,
            ticker=profile.ticker,
            market_type=profile.market_type,
            relevance=relevance,
            event_ticker=profile.event_ticker,
            ticker_hit=ticker_hit,
            entity_overlap=tuple(name for name, _ in entity_hits[:6]),
            keyword_overlap=tuple(term for term, _ in keyword_hits[:6]),
            event_types=tuple(dict.fromkeys(event_types)),
            matched_by=(
                "ticker"
                if ticker_hit
                else "entity"
                if entity_score >= keyword_score
                else "keyword"
            ),
        )

    def match(self, article: Article) -> list[MarketMatch]:
        """Every market this article is relevant to, best first.

        Returning an empty list is the common and correct outcome.
        """
        self.scored += 1
        matches = []
        for profile in self._profiles.values():
            # A market that has already closed cannot be traded on this news.
            if profile.close_ms is not None and profile.close_ms <= article.ingested_ms:
                continue
            scored = self.score(article, profile)
            if scored is not None and scored.relevance >= self.threshold:
                matches.append(scored)

        matches.sort(key=lambda m: -m.relevance)
        matches = self._limit(matches)
        if matches:
            self.matched += 1
            log.debug(
                "news.matched",
                article_id=article.article_id,
                markets=[m.ticker for m in matches],
                best=round(matches[0].relevance, 3),
            )
        return matches

    def _limit(self, matches: list[MarketMatch]) -> list[MarketMatch]:
        """Cap by event first, then by absolute count.

        Ranking purely by relevance and truncating would keep fifteen rungs of
        the single best-matching ladder and drop every other event, which is the
        opposite of what a signal set should look like. Grouping by event first
        keeps the ladders intact for the few events that matter.
        """
        by_event: dict[str, list[MarketMatch]] = {}
        for match in matches:
            # Markets without an event ticker are their own group, so an
            # unlabelled market never merges with an unrelated one.
            key = match.event_ticker or match.ticker
            by_event.setdefault(key, []).append(match)

        # Events ranked by their strongest market, since that is the evidence
        # that the story bears on the event at all.
        ranked = sorted(by_event.values(), key=lambda g: -g[0].relevance)
        kept: list[MarketMatch] = []
        for group in ranked[: self.max_events_per_article]:
            kept.extend(group)
        kept.sort(key=lambda m: -m.relevance)
        return kept[: self.max_matches_per_article]

    def stats(self) -> dict[str, Any]:
        return {
            "markets": len(self._profiles),
            "articles_scored": self.scored,
            "articles_matched": self.matched,
            "match_rate": round(self.matched / self.scored, 4) if self.scored else None,
            "threshold": self.threshold,
        }
