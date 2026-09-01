"""Fade a short-term overreaction, but only when it looks like noise.

The trade is old and the failure mode is older: most sharp moves in an event
contract are not overreactions, they are news. Selling every spike is a strategy
that wins small amounts often and loses its account the day a market moves from
forty cents to ninety and never comes back.

Three filters separate "noise" from "news" here, and all three must agree.

**Size relative to this market's own volatility.** The move is standardised by
the realised volatility of the market itself, not compared with a fixed number
of cents. A three-cent move in a market that moves three cents an hour is an
event; the same move in a market that moves three cents a minute is Tuesday.

**Volume.** A large move on ordinary volume is somebody's order walking a thin
book. A large move on extraordinary volume is the market repricing, and this
strategy stands aside for it. The volume filter is an upper bound, which is the
opposite of the one :mod:`kat.strategies.event.momentum` uses, and that is the
honest relationship between the two: they are mutually exclusive by
construction rather than competing for the same signal.

**Deceleration.** A move still extending in its final minute has not finished.
The strategy requires the last minute to be materially smaller than the five
before it, which costs some of the reversion but removes the trades taken in
front of the rest of the move.

Even with all three the reversion captured is a fraction of the move, never the
whole of it: ``reversion_fraction`` defaults well below one because the
alternative -- assuming the price returns exactly to where it started -- is an
assumption with no evidence behind it that happens to maximise apparent edge.
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
    F_REALIZED_VOL_60M_CC,
    F_VOLUME_ZSCORE_5M,
    TradingStrategy,
    clamp_price_cc,
    conservative_fair_value_cc,
    per_bar_volatility_cc,
)
from kat.strategies.proposal import (
    EV_PARAM_SPACE,
    LIQUIDITY_PARAM_SPACE,
    net_ev_floor_micros,
    propose_event_trade,
    screen_event_market,
    screen_freshness,
)

__all__ = ["MeanReversionStrategy"]

_SKIP_NO_VOL = "realized_vol_unavailable"
_SKIP_MOVE_TOO_SMALL = "move_below_threshold"
_SKIP_INFORMED_MOVE = "move_confirmed_by_volume"
_SKIP_STILL_EXTENDING = "move_still_extending"
_SKIP_PRICE_BAND = "price_outside_band"

#: The realised-volatility feature is the standard deviation of *one-minute* mid
#: changes (see :data:`~kat.strategies.common.F_REALIZED_VOL_5M_CC`), so scaling
#: it to a five-minute horizon is a square-root-of-time conversion.
_FIVE_MINUTE_SCALE = math.sqrt(5.0)

#: Fifteen minutes of one-minute observations stand behind the volatility
#: estimate; nothing this strategy does is meaningful before then.
_WARMUP_MS = 900_000


class MeanReversionStrategy(TradingStrategy):
    """Buy the side a decelerating, unconfirmed move has made cheap."""

    FAMILY = "mean_reversion"
    NAME = "Overreaction Fade"
    VERSION = "1.0.0"
    MARKET_TYPE = MarketType.EVENT
    DESCRIPTION = """
    Fades five-minute moves that are large relative to the market's own realised
    volatility, unconfirmed by volume, and already decelerating. Targets a
    fraction of the move rather than a full retracement, and refuses the trade
    entirely near the price extremes where a spike is usually resolution.
    """
    PARAM_SPACE = (
        *LIQUIDITY_PARAM_SPACE,
        *EV_PARAM_SPACE,
        StrategyParam(
            "entry_z", 2.0, low=1.0, high=6.0, step=0.1,
            description="how many standard deviations the five-minute move must exceed",
        ),
        StrategyParam(
            "min_move_cc", 100, low=25, high=2_000, step=25,
            description="absolute floor on the move, so a quiet market cannot manufacture a z-score",
        ),
        StrategyParam(
            "max_volume_zscore", 2.0, low=0.0, high=6.0, step=0.25,
            description="above this the move is treated as information and is not faded",
        ),
        StrategyParam(
            "stall_fraction", 0.5, low=0.05, high=1.0, step=0.05,
            description="last minute must be at most this fraction of the five-minute move",
        ),
        StrategyParam(
            "reversion_fraction", 0.35, low=0.05, high=0.9, step=0.05,
            description="fraction of the move expected to retrace",
        ),
        StrategyParam(
            "max_reversion_cc", 400, low=25, high=3_000, step=25,
            description="hard cap on how far fair value may be displaced from the midpoint",
        ),
        StrategyParam(
            "min_price_cc", 700, low=100, high=4_000, step=100,
            description="below this a spike is usually resolution, not noise",
        ),
        StrategyParam(
            "max_price_cc", 9_300, low=6_000, high=9_900, step=100,
            description="above this a spike is usually resolution, not noise",
        ),
    )
    REQUIRES_FEATURES = (
        F_MID_CHANGE_1M_CC,
        F_MID_CHANGE_5M_CC,
        F_REALIZED_VOL_5M_CC,
        F_REALIZED_VOL_60M_CC,
        F_VOLUME_ZSCORE_5M,
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

        vol_cc = per_bar_volatility_cc(ctx)
        if vol_cc is None:
            # A zero volatility is not a calm market, it is a missing estimate:
            # dividing by it would make every move infinitely significant.
            return self.skip(_SKIP_NO_VOL)

        move_5m_cc = ctx.feature(F_MID_CHANGE_5M_CC)
        move_1m_cc = ctx.feature(F_MID_CHANGE_1M_CC)
        z_score = move_5m_cc / (vol_cc * _FIVE_MINUTE_SCALE)
        if abs(z_score) < self.float_param("entry_z"):
            return self.skip(_SKIP_MOVE_TOO_SMALL)
        if abs(move_5m_cc) < self.int_param("min_move_cc"):
            return self.skip(_SKIP_MOVE_TOO_SMALL)
        if ctx.feature(F_VOLUME_ZSCORE_5M) > self.float_param("max_volume_zscore"):
            return self.skip(_SKIP_INFORMED_MOVE)
        if abs(move_1m_cc) > self.float_param("stall_fraction") * abs(move_5m_cc):
            return self.skip(_SKIP_STILL_EXTENDING)

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

        # Fair value sits back along the path the move came down, never past its
        # origin, and never further than the configured cap.
        displacement = -self.float_param("reversion_fraction") * move_5m_cc
        cap_cc = float(self.int_param("max_reversion_cc"))
        displacement = max(-cap_cc, min(cap_cc, displacement))
        fair_value_cc = conservative_fair_value_cc(mid_cc + displacement, mid_cc)
        if fair_value_cc == clamp_price_cc(mid_cc):
            return self.skip(_SKIP_MOVE_TOO_SMALL)

        # A move up leaves YES rich, so the cheap side is NO, and vice versa.
        side = OutcomeSide.NO if move_5m_cc > 0 else OutcomeSide.YES

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
            confidence=self._confidence(z_score),
            costing=proposal.costing,
            reason="overreaction_fade",
            extra={
                "move_5m_cc": round(move_5m_cc, 2),
                "move_1m_cc": round(move_1m_cc, 2),
                "realized_vol_cc": round(vol_cc, 4),
                "z_score": round(z_score, 3),
                "volume_zscore": round(ctx.feature(F_VOLUME_ZSCORE_5M), 3),
                "market_mid_cc": mid_cc,
            },
        )
        return self.emit([intent])

    # -- internals -----------------------------------------------------------

    def _confidence(self, z_score: float) -> float:
        """Half conviction at the entry threshold, full at twice it."""
        entry_z = max(0.1, self.float_param("entry_z"))
        return min(1.0, abs(z_score) / (2.0 * entry_z))
