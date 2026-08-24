"""Take the side of a perpetual that is paid funding, when the carry clears costs.

The trade has no price view at all. It expects the mark to be exactly where it
is now when the position closes, and its entire expected value is the funding
received over the holding period minus the fees and the spread paid to get in
and out. That is why every number in the arithmetic has to be real: there is no
directional edge to hide an error in.

Sign convention
---------------
A positive funding rate means longs pay shorts, which is the convention the rest
of the platform uses -- see ``funding_rate_bps_per_hour`` on
:meth:`kat.risk.perp_risk.PerpRiskEngine.evaluate` and
:func:`kat.strategies.common.funding_benefit_micros`. So a large positive rate
is a reason to be short and a large negative rate a reason to be long. Getting
this backwards produces a strategy that pays the carry it thinks it is
collecting, and does so quietly, because the position itself looks correct.

What makes this survivable
--------------------------
* **A hard cap on the holding period.** ``max_hold_hours`` bounds both the
  expected-value horizon and, through ``max_hold_ms`` on the intent metadata,
  what the position manager is told about the intended life of the position. A
  carry trade with no time limit is a short volatility position that nobody
  decided to take.
* **The rate is discounted, never extrapolated at face value.** Funding is a
  snapshot of a number that moves. ``rate_persistence`` shrinks the assumed
  future rate, so a trade needs the carry to clear costs with room to spare
  rather than exactly.
* **At least one payment must fall inside the window.** When the exchange
  publishes the next funding time, a position that would be closed before it is
  a position that pays the spread and collects nothing.
* **The exit is priced now.** ``propose_perp_trade`` charges the measured
  round-trip crossing cost against the forecast, which for this strategy is the
  current midpoint. So the carry has to beat the spread, which on a wide
  perpetual book is most of the reason a rate that looks generous is not.

The 1.00x exposure ceiling applies as it does everywhere: this strategy asks for
a size and :mod:`kat.risk.invariants` decides what it may actually have. Carry
trades are the classic place leverage creeps in, which is precisely why there is
no leverage input anywhere in this module.
"""

from __future__ import annotations

from kat.common.enums import MarketType
from kat.marketdata.state import funding_rate_bps_per_hour
from kat.risk.types import OrderIntent
from kat.strategies.base import StrategyContext, StrategyParam
from kat.strategies.common import TradingStrategy
from kat.strategies.proposal import (
    EV_PARAM_SPACE,
    LIQUIDITY_PARAM_SPACE,
    conservative_funding_rate,
    net_ev_floor_micros,
    propose_perp_trade,
    screen_freshness,
    screen_perp_market,
)

__all__ = ["FundingCarryStrategy"]

_SKIP_NO_RATE = "funding_rate_unknown"
_SKIP_RATE_TOO_SMALL = "funding_rate_below_threshold"
_SKIP_NO_PAYMENT_IN_WINDOW = "no_funding_payment_inside_hold_window"
_SKIP_SCHEDULE_STALE = "funding_schedule_in_the_past"
_SKIP_NO_MID = "no_midpoint"

_MS_PER_HOUR = 3_600_000.0


class FundingCarryStrategy(TradingStrategy):
    """Hold the receiving side of a perpetual while the carry pays for itself."""

    FAMILY = "funding_carry"
    NAME = "Perpetual Funding Carry"
    VERSION = "1.0.0"
    MARKET_TYPE = MarketType.PERP
    DESCRIPTION = """
    Takes the side that receives funding when the discounted rate, integrated
    over a hard-capped holding period, exceeds the round trip of fees and spread
    measured against the current book. Carries no price view.
    """
    PARAM_SPACE = (
        *LIQUIDITY_PARAM_SPACE,
        *EV_PARAM_SPACE,
        StrategyParam(
            "min_rate_bps_per_hour", 1.0, low=0.1, high=100.0, step=0.1,
            description="funding rate magnitude below which the carry is not worth the spread",
        ),
        StrategyParam(
            "max_hold_hours", 8.0, low=1.0, high=72.0, step=1.0,
            description="hard cap on the intended holding period, in hours",
        ),
        StrategyParam(
            "rate_persistence", 0.5, low=0.05, high=1.0, step=0.05,
            description="fraction of the current rate assumed to persist over the holding period",
        ),
        StrategyParam(
            "require_scheduled_payment", True, choices=(True, False),
            description="refuse when the next published funding time falls outside the hold window",
        ),
    )
    # Funding and the book both arrive on the market snapshot, so nothing is
    # required of the feature engine.
    REQUIRES_FEATURES = ()
    REQUIRES_BOOK = True
    MAX_DATA_AGE_MS = 5_000

    def warmup_ms(self) -> int:
        """No history: the rate and the book are both point-in-time."""
        return 0

    def generate(self, ctx: StrategyContext) -> list[OrderIntent]:
        self.begin(ctx)

        stale = screen_freshness(ctx, self.metadata.max_data_age_ms)
        if stale is not None:
            return self.skip(stale)

        # ``market.funding_rate`` is the raw fraction Kalshi publishes, charged
        # over one funding interval. Everything below this line works in basis
        # points per hour, so the conversion happens once, here, through the
        # single shared helper.
        rate = funding_rate_bps_per_hour(
            ctx.market.funding_rate, ctx.market.funding_interval_hours
        )
        if rate is None:
            return self.skip(_SKIP_NO_RATE)
        if abs(rate) < self.float_param("min_rate_bps_per_hour"):
            return self.skip(_SKIP_RATE_TOO_SMALL)

        # Positive rate: longs pay shorts, so the receiving side is short.
        is_long = rate < 0.0
        hold_hours = self.float_param("max_hold_hours")

        schedule_reason = self._check_funding_schedule(ctx, hold_hours)
        if schedule_reason is not None:
            return self.skip(schedule_reason)

        book, reason = screen_perp_market(
            ctx,
            max_spread_cc=self.int_param("max_spread_cc"),
            min_depth_qc=self.int_param("min_depth_qc"),
        )
        if book is None or reason is not None:
            return self.skip(reason or "no_book")

        mid_cc = book.mid_cc
        if mid_cc is None:
            return self.skip(_SKIP_NO_MID)

        assumed_rate = conservative_funding_rate(
            rate, is_long=is_long, persistence=self.float_param("rate_persistence")
        )

        proposal, reason = propose_perp_trade(
            self,
            ctx,
            book=book,
            is_long=is_long,
            # No price view whatsoever: the midpoint is where we expect to close.
            forecast_mid_cc=mid_cc,
            max_position_qc=self.int_param("max_position_qc"),
            min_net_ev_micros=net_ev_floor_micros(self.float_param("min_net_ev_usd")),
            funding_rate_bps_per_hour=assumed_rate,
            hold_hours=hold_hours,
        )
        if proposal is None:
            return self.skip(reason or "no_proposal")

        intent = self.perp_intent(
            ctx,
            is_buy=proposal.is_long,
            qty_qc=proposal.qty_qc,
            limit_price_cc=proposal.limit_price_cc,
            fair_value_cc=proposal.fair_value_cc,
            confidence=self._confidence(rate),
            costing=proposal.costing,
            reason="funding_carry",
            extra={
                "funding_rate_bps_per_hour": round(rate, 6),
                "assumed_rate_bps_per_hour": round(assumed_rate, 6),
                "hold_hours": hold_hours,
                # The position manager reads this to enforce the cap; the
                # strategy itself has no way to close anything.
                "max_hold_ms": int(hold_hours * _MS_PER_HOUR),
                "next_funding_ms": ctx.market.next_funding_ms,
                "crossing_cc": proposal.round_trip.crossing_cc,
                "market_mid_cc": mid_cc,
            },
        )
        return self.emit([intent])

    # -- internals -----------------------------------------------------------

    def _check_funding_schedule(self, ctx: StrategyContext, hold_hours: float) -> str | None:
        """Refuse when no published payment falls inside the holding window.

        A missing schedule is accepted -- some venues accrue funding
        continuously and publish no next-payment time -- but a schedule whose
        next payment is already in the past is treated as broken data rather
        than as an imminent payment, because the two are indistinguishable from
        here and only one of them is safe to assume.
        """
        if not bool(self.param("require_scheduled_payment", True)):
            return None
        next_ms = ctx.market.next_funding_ms
        if next_ms is None:
            return None
        hours_away = (next_ms - ctx.now_ms) / _MS_PER_HOUR
        if hours_away < 0.0:
            return _SKIP_SCHEDULE_STALE
        if hours_away > hold_hours:
            return _SKIP_NO_PAYMENT_IN_WINDOW
        return None

    def _confidence(self, rate: float) -> float:
        """Half conviction at the entry threshold, full at twice it."""
        threshold = max(1e-6, self.float_param("min_rate_bps_per_hour"))
        return min(1.0, abs(rate) / (2.0 * threshold))
