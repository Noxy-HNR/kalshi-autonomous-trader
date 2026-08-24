"""Act on a fresh, relevant news signal -- and only on the part not yet priced.

The contract with the news pipeline
-----------------------------------
This strategy reads ``ctx.research["news"]``, a list of dictionaries with the
shape the news pipeline produces (one entry per article/market link, mirroring
:class:`~kat.db.models.NewsMarketLink` plus the article's ingest stamp)::

    {
        "article_id":       str,
        "ingested_ms":      int,    # platform clock, never the publisher's stamp
        "relevance":        float,  # [0, 1]
        "direction":        str,    # yes|up|bullish  /  no|down|bearish
        "estimated_impact": float,  # magnitude as a fraction of $1.00, [0, 1]
        "confidence":       float,  # [0, 1]
    }

Anything malformed is dropped and counted rather than raising: this input comes
from a pipeline that parses the open internet, and a strategy that throws on a
missing key is a strategy that stops trading the day a feed changes its schema.

``ingested_ms`` is load-bearing and the reason ``published_ms`` is not accepted
here at all. Several government feeds back-date items by hours because they
publish the document's issue date; filtering on that stamp would hand a backtest
several hours of hindsight and produce a beautiful, fictitious equity curve.
:mod:`kat.news.types` makes the same point at greater length.

The part that stops this from being a losing strategy
-----------------------------------------------------
News is the most competitive signal on the venue. By the time a headline has
been ingested, matched to a market and scored, the price has usually already
moved -- and buying the move after it has happened is the single most reliable
way to lose money on news. So the estimated impact is netted against the move
the market has *already* made in the same direction, and only the residual is
traded. When the market has moved further than the estimate, there is no trade:
an overshoot is a mean-reversion signal and belongs to a different strategy.

Two further discounts are applied on principle. Only ``impact_fraction`` of the
pipeline's magnitude estimate is claimed, because that estimate comes from a
rule-based scorer with no model behind it, and the weight of each item decays
linearly to zero at the staleness cutoff so that there is no cliff at the
boundary where a signal is worth full value one millisecond and nothing the
next.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kat.common.enums import MarketType, OutcomeSide
from kat.common.fixedpoint import CC_PER_DOLLAR
from kat.risk.types import OrderIntent
from kat.strategies.base import StrategyContext, StrategyParam
from kat.strategies.common import (
    F_MID_CHANGE_5M_CC,
    TradingStrategy,
    conservative_fair_value_cc,
)
from kat.strategies.proposal import (
    EV_PARAM_SPACE,
    LIQUIDITY_PARAM_SPACE,
    net_ev_floor_micros,
    propose_event_trade,
    screen_event_market,
    screen_freshness,
)

__all__ = ["NewsReactionStrategy"]

_SKIP_NO_NEWS = "no_news_supplied"
_SKIP_ALL_FILTERED = "all_news_stale_or_irrelevant"
_SKIP_ALREADY_PRICED = "impact_already_in_the_price"
_SKIP_PRICE_BAND = "price_outside_band"

_BULLISH = frozenset({"yes", "up", "bullish", "positive"})
_BEARISH = frozenset({"no", "down", "bearish", "negative"})

#: Only the five-minute move feature is needed, and only to measure what the
#: market has already done with the news.
_WARMUP_MS = 300_000


@dataclass(frozen=True, slots=True)
class _NewsItem:
    """One validated news-to-market link."""

    article_id: str
    ingested_ms: int
    relevance: float
    confidence: float
    impact: float
    sign: int

    @classmethod
    def parse(cls, raw: Any) -> _NewsItem | None:
        """Validate one entry, returning ``None`` for anything unusable."""
        if not isinstance(raw, dict):
            return None
        direction = str(raw.get("direction", "")).strip().lower()
        if direction in _BULLISH:
            sign = 1
        elif direction in _BEARISH:
            sign = -1
        else:
            return None
        try:
            ingested_ms = int(raw["ingested_ms"])
            relevance = float(raw.get("relevance", 0.0))
            confidence = float(raw.get("confidence", 0.0))
            impact = float(raw.get("estimated_impact", 0.0))
        except (KeyError, TypeError, ValueError):
            return None
        if impact <= 0.0:
            return None
        return cls(
            article_id=str(raw.get("article_id", "")),
            ingested_ms=ingested_ms,
            relevance=min(1.0, max(0.0, relevance)),
            confidence=min(1.0, max(0.0, confidence)),
            impact=min(1.0, max(0.0, impact)),
            sign=sign,
        )


class NewsReactionStrategy(TradingStrategy):
    """Trade the portion of a fresh news impact the market has not priced yet."""

    FAMILY = "news_reaction"
    NAME = "News Reaction"
    VERSION = "1.0.0"
    MARKET_TYPE = MarketType.EVENT
    DESCRIPTION = """
    Aggregates fresh, relevant news links for this market into a signed impact
    estimate, discounts it, nets off the move the market has already made in the
    same direction, and trades only the residual.
    """
    PARAM_SPACE = (
        *LIQUIDITY_PARAM_SPACE,
        *EV_PARAM_SPACE,
        StrategyParam(
            "max_news_age_ms", 600_000, low=30_000, high=3_600_000, step=30_000,
            description="hard staleness cutoff measured from ingest, not from publication",
        ),
        StrategyParam(
            "min_relevance", 0.5, low=0.0, high=1.0, step=0.05,
            description="article/market relevance below which the item is ignored",
        ),
        StrategyParam(
            "min_confidence", 0.4, low=0.0, high=1.0, step=0.05,
            description="pipeline confidence below which the item is ignored",
        ),
        StrategyParam(
            "impact_fraction", 0.5, low=0.05, high=1.0, step=0.05,
            description="fraction of the pipeline's magnitude estimate actually claimed",
        ),
        StrategyParam(
            "max_impact_cc", 1_500, low=100, high=5_000, step=100,
            description="cap on the aggregate impact, however many articles agree",
        ),
        StrategyParam(
            "min_residual_cc", 100, low=25, high=2_000, step=25,
            description="unpriced residual required before an order is considered",
        ),
        StrategyParam(
            "min_price_cc", 300, low=100, high=4_000, step=100,
            description="cheapest midpoint this strategy will trade",
        ),
        StrategyParam(
            "max_price_cc", 9_700, low=6_000, high=9_900, step=100,
            description="dearest midpoint this strategy will trade",
        ),
    )
    REQUIRES_FEATURES = (F_MID_CHANGE_5M_CC,)
    REQUIRES_BOOK = True
    MAX_DATA_AGE_MS = 5_000

    def warmup_ms(self) -> int:
        return _WARMUP_MS

    def generate(self, ctx: StrategyContext) -> list[OrderIntent]:
        self.begin(ctx)

        raw_items = ctx.research.get("news")
        if not isinstance(raw_items, list) or not raw_items:
            return self.skip(_SKIP_NO_NEWS)

        stale = screen_freshness(ctx, self.metadata.max_data_age_ms)
        if stale is not None:
            return self.skip(stale)

        impact_cc, used, weight = self._aggregate_impact_cc(ctx, raw_items)
        if used == 0:
            return self.skip(_SKIP_ALL_FILTERED)

        quote, reason = screen_event_market(
            ctx,
            max_spread_cc=self.int_param("max_spread_cc"),
            min_depth_qc=self.int_param("min_depth_qc"),
        )
        if quote is None or reason is not None:
            return self.skip(reason or "no_quote")

        mid_cc = quote.mid_cc
        if not self.int_param("min_price_cc") <= mid_cc <= self.int_param("max_price_cc"):
            return self.skip(_SKIP_PRICE_BAND)

        residual_cc = self._residual_cc(ctx, impact_cc)
        if abs(residual_cc) < self.int_param("min_residual_cc"):
            return self.skip(_SKIP_ALREADY_PRICED)

        fair_value_cc = conservative_fair_value_cc(mid_cc + residual_cc, mid_cc)
        side = OutcomeSide.YES if residual_cc > 0 else OutcomeSide.NO

        book = ctx.market.book
        if book is None:
            return self.skip("no_usable_book")

        proposal, reason = propose_event_trade(
            self,
            ctx,
            book=book,
            quote=quote,
            side=side,
            fair_value_cc=fair_value_cc,
            max_position_qc=self.int_param("max_position_qc"),
            min_net_ev_micros=net_ev_floor_micros(self.float_param("min_net_ev_usd")),
        )
        if proposal is None:
            return self.skip(reason or "no_proposal")

        intent = self.event_intent(
            ctx,
            side=proposal.side,
            qty_qc=proposal.qty_qc,
            limit_price_cc=proposal.limit_price_cc,
            fair_value_cc=proposal.fair_value_cc,
            confidence=self._confidence(residual_cc, weight),
            costing=proposal.costing,
            reason="news_reaction",
            extra={
                "articles_used": used,
                "aggregate_impact_cc": round(impact_cc, 1),
                "unpriced_residual_cc": round(residual_cc, 1),
                "already_moved_cc": round(ctx.feature(F_MID_CHANGE_5M_CC), 1),
                "market_mid_cc": mid_cc,
            },
        )
        return self.emit([intent])

    # -- internals -----------------------------------------------------------

    def _aggregate_impact_cc(
        self, ctx: StrategyContext, raw_items: list[Any]
    ) -> tuple[float, int, float]:
        """Signed impact in centicents, how many items produced it, total weight.

        Items are weighted by relevance, pipeline confidence and a linear decay
        to zero at the staleness cutoff. Future-stamped items are dropped
        outright: in a replay that means a clock or ordering bug, and acting on
        one would be acting on information we did not have.
        """
        max_age_ms = self.int_param("max_news_age_ms")
        min_relevance = self.float_param("min_relevance")
        min_confidence = self.float_param("min_confidence")
        fraction = self.float_param("impact_fraction")

        total_cc = 0.0
        total_weight = 0.0
        used = 0
        for raw in raw_items:
            item = _NewsItem.parse(raw)
            if item is None:
                continue
            age_ms = ctx.now_ms - item.ingested_ms
            if age_ms < 0 or age_ms >= max_age_ms:
                continue
            if item.relevance < min_relevance or item.confidence < min_confidence:
                continue
            decay = 1.0 - (age_ms / max_age_ms)
            weight = item.relevance * item.confidence * decay
            total_cc += item.sign * item.impact * fraction * CC_PER_DOLLAR * weight
            total_weight += weight
            used += 1

        cap_cc = float(self.int_param("max_impact_cc"))
        return max(-cap_cc, min(cap_cc, total_cc)), used, total_weight

    def _residual_cc(self, ctx: StrategyContext, impact_cc: float) -> float:
        """Impact minus the part the market has already moved in that direction.

        Only aligned movement is netted off. A market that moved *against* the
        news does not increase the opportunity -- it is evidence the impact
        estimate is wrong, so it is treated as zero rather than as extra edge.
        """
        moved_cc = ctx.feature(F_MID_CHANGE_5M_CC)
        if impact_cc >= 0.0:
            aligned = max(0.0, moved_cc)
            return max(0.0, impact_cc - aligned)
        aligned = min(0.0, moved_cc)
        return min(0.0, impact_cc - aligned)

    def _confidence(self, residual_cc: float, weight: float) -> float:
        """Needs both a sizeable residual and well-supported news behind it."""
        floor_cc = max(1, self.int_param("min_residual_cc"))
        size_strength = min(1.0, abs(residual_cc) / (2.0 * floor_cc))
        support = min(1.0, weight)
        return min(size_strength, support)
