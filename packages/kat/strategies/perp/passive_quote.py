"""Earn the spread on a perpetual, where the spread is actually worth earning.

The event-contract version of this idea was measured and it failed, twice over.
The touch on the markets carrying the volume held a median of 124,000 contracts,
so a resting quote was never reached except adversely; and where the queue was
short there was no flow to be reached by. Both halves of that failure were
properties of the instrument, not of the idea:

===================  ==========================  ====================
                     event contract              perpetual
===================  ==========================  ====================
fee                  175cc at 50c = 350bp        2bp maker, 4bp taker
tick                 100cc = 3,333bp at 3c       1cc = 0.8bp at $1.26
maker round trip     negative after real costs   +2.5 to +19.5bp
touch queue          median 124,000 contracts    $49-$100 of notional
===================  ==========================  ====================

A one-tick spread on a three-cent contract is a third of its notional, which is
why event-contract spreads are wide enough to attract every quoter on the
exchange and why the queue behind them is hopeless. A perpetual ticks at 0.8
basis points, so the spread is a real, small number and the queue at it is a few
hundred dollars.

What this does not assume
-------------------------
That thin queues make it work. Queue length was the binding constraint on event
contracts and it is checked here for the same reason: fills go in time order, so
everything already resting at our price trades before we do, and a quote joining
a long queue collects only the fills nobody else wanted. ``max_queue_qc`` is the
first-class parameter it deserves to be.

That a wide spread is a gift. The widest perpetual spreads sit on the markets
with the least flow, exactly as on event contracts, and being paid 19 basis
points by nobody is worth nothing. The screen is on both.

That funding is free. A perpetual position pays or receives funding every four
hours, and a quote that keeps getting hit on one side accumulates inventory that
carries a running cost. Inventory is reduced before it is added to, and the
funding on the position we would end up holding is charged against the edge
rather than ignored.
"""

from __future__ import annotations

from kat.common.enums import MarketType
from kat.kalshi.fees import PerpFeeSchedule, perp_fee_micros
from kat.marketdata.state import funding_rate_bps_per_hour
from kat.risk.types import OrderIntent
from kat.strategies.base import StrategyContext, StrategyParam
from kat.strategies.common import (
    MIN_ORDER_QC,
    TradingStrategy,
    funding_benefit_micros,
    perp_maker_costing,
)
from kat.strategies.proposal import (
    EV_PARAM_SPACE,
    screen_freshness,
    screen_perp_market,
)

__all__ = ["PerpPassiveQuoteStrategy"]

_SKIP_SPREAD_TOO_TIGHT = "spread_does_not_cover_two_fees"
_SKIP_QUEUE_TOO_LONG = "queue_ahead_too_long"
_SKIP_ALREADY_QUOTING = "already_resting_in_this_market"
_SKIP_INVENTORY_FULL = "inventory_at_limit"
_SKIP_SIZE_BELOW_MIN = "size_below_minimum"

#: Hours of funding charged against a quote's edge. One four-hour period is the
#: shortest a filled position can be held and still cross a funding stamp, so it
#: is the least this can cost and the right thing to underwrite.
_FUNDING_HOURS = 4.0


class PerpPassiveQuoteStrategy(TradingStrategy):
    """Rests quotes on a perpetual and is paid the spread for being there."""

    FAMILY = "passive_quote"
    NAME = "Perp Passive Quote"
    VERSION = "1.0.0"
    MARKET_TYPE = MarketType.PERP
    DESCRIPTION = """
    Rests a quote at the touch on a perpetual rather than crossing to it, so a
    fill earns the spread instead of paying it. Quotes only where the spread
    covers both fees at the perpetual's own rates, where the queue at our price
    is short enough to reach the front of, and where funding on the resulting
    inventory does not eat the edge.
    """
    PARAM_SPACE = (
        *EV_PARAM_SPACE,
        StrategyParam(
            "max_spread_bps", 200, low=10, high=1_000, step=10,
            description="widest spread worth quoting, in bps of mid; a spread "
                        "this wide usually means nobody is trading here",
        ),
        StrategyParam(
            "min_depth_qc", 100, low=100, high=1_000_000, step=100,
            description="minimum resting size on BOTH sides across five levels",
        ),
        StrategyParam(
            "max_queue_qc", 50_000, low=100, high=10_000_000, step=100,
            description="longest queue at our own price worth joining, in "
                        "centi-contracts; this is the binding constraint",
        ),
        StrategyParam(
            "fee_cover_bps", 15_000, low=10_000, high=50_000, step=1_000,
            description="spread required as a multiple of two fees; 15000 is 1.5x",
        ),
        StrategyParam(
            "quote_notional_micros", 15_000_000, low=1_000_000,
            high=1_000_000_000, step=1_000_000,
            description="notional to rest on each quote, in micros",
        ),
        StrategyParam(
            "max_inventory_micros", 60_000_000, low=1_000_000,
            high=10_000_000_000, step=1_000_000,
            description="largest absolute position notional before quoting only "
                        "reduces it",
        ),
    )
    REQUIRES_FEATURES = ()
    REQUIRES_BOOK = True
    MAX_DATA_AGE_MS = 5_000

    def warmup_ms(self) -> int:
        """Nothing to learn: the edge is the spread, which is visible now."""
        return 0

    def generate(self, ctx: StrategyContext) -> list[OrderIntent]:
        self.begin(ctx)

        stale = screen_freshness(ctx, self.metadata.max_data_age_ms)
        if stale is not None:
            return self.skip(stale)

        book, reason = screen_perp_market(
            ctx,
            # screen_perp_market takes an absolute spread; converting from bps
            # needs the mid, which needs the book, so the wide-spread test is
            # done below instead and this one is left permissive.
            max_spread_cc=1_000_000,
            min_depth_qc=self.int_param("min_depth_qc"),
        )
        if book is None or reason is not None:
            return self.skip(reason or "no_book")

        bid_cc, ask_cc = book.best_bid_cc, book.best_ask_cc
        if bid_cc is None or ask_cc is None or bid_cc >= ask_cc:
            return self.skip("no_book")
        mid_cc = (bid_cc + ask_cc) // 2
        spread_cc = ask_cc - bid_cc
        if mid_cc <= 0:
            return self.skip("no_book")

        # A spread this wide is not generosity, it is absence. The widest
        # perpetual books belong to the markets with the least flow, and being
        # paid twenty basis points by nobody is worth nothing.
        if spread_cc * 10_000 // mid_cc > self.int_param("max_spread_bps"):
            return self.skip("spread_too_wide")

        # One quote at a time, or the fills accumulate one large position
        # instead of cycling small ones.
        if ctx.resting_qty_qc > 0:
            return self.skip(_SKIP_ALREADY_QUOTING)

        schedule = self._schedule(ctx)
        position_qc = ctx.position_qc

        # Reduce before adding, so a quote that only ever fills on one side does
        # not drift into a directional position nothing here has a view on.
        limit_micros = self.int_param("max_inventory_micros")
        position_micros = abs(position_qc) * mid_cc
        if position_qc > 0:
            is_buy = False
        elif position_qc < 0:
            is_buy = True
        else:
            # Flat: quote the side funding pays us to hold. A positive rate
            # means longs pay shorts, so a short is the side being paid.
            rate = funding_rate_bps_per_hour(
                ctx.market.funding_rate, ctx.market.funding_interval_hours
            )
            is_buy = not (rate is not None and rate > 0)

        price_cc = bid_cc if is_buy else ask_cc

        # The queue at our own price. This, not the spread, decided the
        # event-contract version: fills go in time order, so everything already
        # resting here trades before we do.
        queue_qc = book.best_bid_size_qc if is_buy else book.best_ask_size_qc
        if queue_qc > self.int_param("max_queue_qc"):
            return self.skip(_SKIP_QUEUE_TOO_LONG)

        qty_qc = self._size_qc(ctx, price_cc, position_qc, position_micros, limit_micros)
        if qty_qc < MIN_ORDER_QC:
            return self.skip(
                _SKIP_INVENTORY_FULL
                if position_micros >= limit_micros
                else _SKIP_SIZE_BELOW_MIN
            )

        # Both fees at the perpetual's own rates, not the event contract's
        # quadratic one, which at these prices would be about a hundred times
        # too large and would refuse every quote on the wrong arithmetic.
        fee_micros = perp_fee_micros(
            price_cc=price_cc, qty_qc=qty_qc, schedule=schedule, is_taker=False
        ) + perp_fee_micros(
            price_cc=price_cc, qty_qc=qty_qc, schedule=schedule, is_taker=True
        )
        required_micros = fee_micros * self.int_param("fee_cover_bps") // 10_000
        # The *full* spread, not half of it. A round trip that rests on both
        # sides buys at the bid and sells at the ask, so it earns bid-to-ask
        # once -- the half-spread is what one leg earns against the mid, and
        # testing a two-leg cost against a one-leg revenue rejects economics
        # that are genuinely positive. The exit is still costed as a taker
        # above, which is the conservatism that belongs here.
        if spread_cc * qty_qc < required_micros:
            return self.skip(_SKIP_SPREAD_TOO_TIGHT)

        # What funding would cost on the inventory this fill leaves us holding.
        # A quote that earns the spread and then pays it back in carry has not
        # earned anything.
        rate = funding_rate_bps_per_hour(
            ctx.market.funding_rate, ctx.market.funding_interval_hours
        )
        funding_micros = (
            funding_benefit_micros(
                rate_bps_per_hour=rate,
                notional_micros=qty_qc * price_cc,
                hours=_FUNDING_HOURS,
                is_long=is_buy,
            )
            if rate is not None
            else 0
        )

        intent = self.perp_intent(
            ctx,
            is_buy=is_buy,
            qty_qc=qty_qc,
            limit_price_cc=price_cc,
            fair_value_cc=mid_cc,
            confidence=0.5,
            costing=perp_maker_costing(
                is_long=is_buy,
                entry_cc=price_cc,
                fair_cc=mid_cc,
                qty_qc=qty_qc,
                schedule=schedule,
                funding_micros=funding_micros,
            ),
            reason="perp_passive_quote",
            extra={
                "spread_cc": spread_cc,
                "spread_bps": spread_cc * 10_000 // mid_cc,
                "queue_ahead_qc": queue_qc,
                "fee_micros": fee_micros,
                "funding_micros": funding_micros,
                "position_qc": position_qc,
            },
        )
        # The whole point: crossing would pay the spread this exists to earn.
        intent.post_only = True
        return self.emit([intent])

    # -- helpers -------------------------------------------------------------

    def _schedule(self, ctx: StrategyContext) -> PerpFeeSchedule:
        """The market's own rates when the context carries them.

        Defaults charge the taker rate on both legs, which understates a maker's
        edge rather than inventing a discount it was not quoted.
        """
        return ctx.perp_fee_schedule or PerpFeeSchedule()

    def _size_qc(
        self, ctx: StrategyContext, price_cc: int,
        position_qc: int, position_micros: int, limit_micros: int,
    ) -> int:
        """Contracts to rest, floored to whole contracts and capped by inventory."""
        if price_cc <= 0:
            return 0
        wanted_qc = self.int_param("quote_notional_micros") // price_cc
        if position_micros >= limit_micros:
            # At the cap the only permissible quote is one that reduces, and it
            # may not reduce by more than we hold.
            wanted_qc = min(wanted_qc, abs(position_qc))
        return (wanted_qc // MIN_ORDER_QC) * MIN_ORDER_QC
