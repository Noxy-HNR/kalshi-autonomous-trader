"""Follow a move only when the tape says the move is real.

Momentum in event contracts is not the same phenomenon as momentum in equities.
An event contract has a terminal value of exactly zero or exactly one dollar, so
a trend is not a drift in a price level -- it is the market's probability
estimate being revised, and a revision that is genuinely happening tends to
continue for as long as the information behind it is being absorbed.

That gives a usable distinction between a move worth following and a move worth
ignoring, and it is the entire content of this strategy: **a real revision
prints trades**. A price that has drifted three cents because the resting bid
was pulled has moved without anybody transacting, and it will drift back the
moment the bid returns. So the move must be corroborated twice over -- by volume
above the market's own baseline, and by a trade imbalance pointing the same way
as the move -- before any order is considered. Both filters are lower bounds,
which is precisely the mirror of :mod:`kat.strategies.event.mean_reversion`,
and the two strategies therefore never both fire on the same move.

The forecast is deliberately timid. ``continuation_fraction`` defaults to well
under half of the observed move and is capped in absolute centicents, because
the honest claim is "some of this continues", not "this repeats". Extrapolating
a five-minute move at full size is how a strategy buys the last contract of a
repricing at the price the repricing ended at.

The price band excludes the extremes for a different reason than it does in the
mean-reversion strategy: near a dollar there is very little room left to
continue into, and the trade becomes a few cents of upside against the whole
price as downside.
"""

from __future__ import annotations

import math

from kat.common.enums import MarketType, OutcomeSide
from kat.risk.types import OrderIntent
from kat.strategies.base import StrategyContext, StrategyParam
from kat.strategies.common import (
    F_MID_CHANGE_1M_CC,
    F_MID_CHANGE_5M_CC,
    F_REALIZED_VOL_5M_CC,
    F_TRADE_IMBALANCE_5M,
    F_VOLUME_ZSCORE_5M,
    TradingStrategy,
    clamp_price_cc,
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

__all__ = ["MomentumStrategy"]

_SKIP_NO_VOL = "realized_vol_unavailable"
_SKIP_MOVE_TOO_SMALL = "move_below_threshold"
_SKIP_NO_VOLUME = "move_not_volume_confirmed"
_SKIP_NO_IMBALANCE = "trade_imbalance_does_not_confirm"
_SKIP_REVERSING = "move_already_reversing"
_SKIP_PRICE_BAND = "price_outside_band"

_FIVE_MINUTE_SCALE = math.sqrt(5.0)
_WARMUP_MS = 900_000


class MomentumStrategy(TradingStrategy):
    """Buy the side a volume-confirmed repricing is moving toward."""

    FAMILY = "momentum"
    NAME = "Volume-Confirmed Momentum"
    VERSION = "1.0.0"
    MARKET_TYPE = MarketType.EVENT
    DESCRIPTION = """
    Follows five-minute moves that are large relative to the market's realised
    volatility, confirmed by volume above the market's own baseline and by a
    trade imbalance in the same direction, and still extending. Forecasts a
    fraction of the observed move, capped in absolute centicents.
    """
    PARAM_SPACE = (
        *LIQUIDITY_PARAM_SPACE,
        *EV_PARAM_SPACE,
        StrategyParam(
            "entry_z", 1.5, low=0.5, high=6.0, step=0.1,
            description="how many standard deviations the five-minute move must exceed",
        ),
        StrategyParam(
            "min_move_cc", 100, low=25, high=2_000, step=25,
            description="absolute floor on the move, so a quiet market cannot manufacture a z-score",
        ),
        StrategyParam(
            "min_volume_zscore", 1.5, low=0.0, high=6.0, step=0.25,
            description="volume must exceed this many standard deviations of its own baseline",
        ),
        StrategyParam(
            "min_trade_imbalance", 0.25, low=0.0, high=0.95, step=0.05,
            description="signed buy/sell pressure required, in the direction of the move",
        ),
        StrategyParam(
            "continuation_fraction", 0.3, low=0.05, high=1.0, step=0.05,
            description="fraction of the observed move expected to continue",
        ),
        StrategyParam(
            "max_continuation_cc", 300, low=25, high=3_000, step=25,
            description="hard cap on how far fair value may be displaced from the midpoint",
        ),
        StrategyParam(
            "min_price_cc", 700, low=100, high=4_000, step=100,
            description="cheapest midpoint worth chasing",
        ),
        StrategyParam(
            "max_price_cc", 9_300, low=6_000, high=9_900, step=100,
            description="dearest midpoint worth chasing; above it there is little room left",
        ),
    )
    REQUIRES_FEATURES = (
        F_MID_CHANGE_1M_CC,
        F_MID_CHANGE_5M_CC,
        F_REALIZED_VOL_5M_CC,
        F_VOLUME_ZSCORE_5M,
        F_TRADE_IMBALANCE_5M,
    )
    REQUIRES_BOOK = True
    MAX_DATA_AGE_MS = 5_000

    def warmup_ms(self) -> int:
        return _WARMUP_MS

    def generate(self, ctx: StrategyContext) -> list[OrderIntent]:
        self.begin(ctx)

        stale = screen_freshness(ctx, self.metadata.max_data_age_ms)
        if stale is not None:
            return self.skip(stale)

        vol_cc = ctx.feature(F_REALIZED_VOL_5M_CC)
        if not vol_cc > 0.0:
            return self.skip(_SKIP_NO_VOL)

        move_5m_cc = ctx.feature(F_MID_CHANGE_5M_CC)
        move_1m_cc = ctx.feature(F_MID_CHANGE_1M_CC)
        z_score = move_5m_cc / (vol_cc * _FIVE_MINUTE_SCALE)
        if abs(z_score) < self.float_param("entry_z") or abs(move_5m_cc) < self.int_param("min_move_cc"):
            return self.skip(_SKIP_MOVE_TOO_SMALL)

        direction = 1.0 if move_5m_cc > 0 else -1.0
        if ctx.feature(F_VOLUME_ZSCORE_5M) < self.float_param("min_volume_zscore"):
            return self.skip(_SKIP_NO_VOLUME)

        imbalance = ctx.feature(F_TRADE_IMBALANCE_5M)
        if imbalance * direction < self.float_param("min_trade_imbalance"):
            return self.skip(_SKIP_NO_IMBALANCE)
        # The final minute must still be pointing the same way. A move that has
        # already turned over is a move somebody else is now fading.
        if move_1m_cc * direction < 0:
            return self.skip(_SKIP_REVERSING)

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

        displacement = self.float_param("continuation_fraction") * move_5m_cc
        cap_cc = float(self.int_param("max_continuation_cc"))
        displacement = max(-cap_cc, min(cap_cc, displacement))
        fair_value_cc = conservative_fair_value_cc(mid_cc + displacement, mid_cc)
        if fair_value_cc == clamp_price_cc(mid_cc):
            return self.skip(_SKIP_MOVE_TOO_SMALL)

        side = OutcomeSide.YES if move_5m_cc > 0 else OutcomeSide.NO

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
            confidence=self._confidence(z_score, imbalance * direction),
            costing=proposal.costing,
            reason="volume_confirmed_momentum",
            extra={
                "move_5m_cc": round(move_5m_cc, 2),
                "move_1m_cc": round(move_1m_cc, 2),
                "z_score": round(z_score, 3),
                "volume_zscore": round(ctx.feature(F_VOLUME_ZSCORE_5M), 3),
                "trade_imbalance": round(imbalance, 3),
                "market_mid_cc": mid_cc,
            },
        )
        return self.emit([intent])

    # -- internals -----------------------------------------------------------

    def _confidence(self, z_score: float, aligned_imbalance: float) -> float:
        """Conviction is the weaker of the two confirmations, never their sum."""
        entry_z = max(0.1, self.float_param("entry_z"))
        move_strength = min(1.0, abs(z_score) / (2.0 * entry_z))
        flow_strength = min(1.0, max(0.0, aligned_imbalance))
        return min(move_strength, flow_strength)
