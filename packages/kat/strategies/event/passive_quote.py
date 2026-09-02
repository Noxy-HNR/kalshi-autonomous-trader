"""Earn the spread instead of paying it.

Every strategy here so far crosses the spread to get in and crosses it again to
get out, and then pays two fees on top. Measured over a week of banked data that
is a losing structure on this venue whatever the forecast: a directional model
with 0.949 walk-forward AUC produced ninety completed round trips and not one
winner, because a thirty-minute move on these markets is smaller than the toll
to capture it.

A resting quote inverts the sign of the largest term. Posting at the touch and
being lifted earns the half-spread rather than paying it, so the round trip is::

    +spread - 2 * fee        instead of        -spread - 2 * fee

At 95 cents the fee is 33 centicents a side, so any spread above 66 pays for
itself before a single opinion is formed. That is the whole idea, and it is the
only structure measured here that is positive before prediction.

What it is not
--------------
This is not free money, and the two reasons it is not are the two things that
decide whether it works.

*Adverse selection.* A resting bid is lifted precisely when someone wants to
sell, which is disproportionately when the price is about to fall. The spread is
payment for that risk, not a gift, and a simulator that fills a resting order the
moment the mid touches it will report profits nobody can trade.
:mod:`kat.simulation.fills` is built for exactly this: it assumes we are behind
*all* the size already resting at our price, and fills only when the market
trades through.

*Inventory.* Earning the spread requires going flat again. A quote that is only
ever lifted on one side accumulates a position and becomes a directional bet
with none of the analysis a directional bet deserves, so this quotes the side
that reduces inventory before the side that adds to it.
"""

from __future__ import annotations

from kat.common.enums import MarketType, OutcomeSide
from kat.common.fixedpoint import CC_PER_DOLLAR, QC_PER_CONTRACT
from kat.kalshi.fees import realised_fee_micros
from kat.risk.types import OrderIntent
from kat.strategies.base import StrategyContext, StrategyParam
from kat.strategies.common import (
    TradingStrategy,
    event_costing,
    fee_schedule_for,
)
from kat.strategies.proposal import (
    EV_PARAM_SPACE,
    screen_event_market,
    screen_freshness,
)

__all__ = ["PassiveQuoteStrategy"]

_SKIP_SPREAD_TOO_TIGHT = "spread_does_not_cover_two_fees"
_SKIP_PRICE_BAND = "price_outside_band"
_SKIP_INVENTORY_FULL = "inventory_at_limit"
_SKIP_NO_QUOTE = "no_quotable_side"
_SKIP_QUEUE_TOO_LONG = "queue_ahead_too_long"
_SKIP_ALREADY_QUOTING = "already_resting_in_this_market"


class PassiveQuoteStrategy(TradingStrategy):
    """Posts resting quotes and is paid the spread for being there."""

    FAMILY = "passive_quote"
    NAME = "Passive Quote"
    VERSION = "1.0.0"
    MARKET_TYPE = MarketType.EVENT
    DESCRIPTION = """
    Rests a quote at the touch rather than crossing to it, so a fill earns the
    half-spread instead of paying it. Quotes only where the spread covers two
    fees, and quotes the side that reduces inventory first so the position does
    not drift into an unexamined directional bet.
    """
    PARAM_SPACE = (
        # Deliberately *not* LIQUIDITY_PARAM_SPACE. Three of its guards invert
        # for a maker, and inheriting them pointed this strategy at the markets
        # it is worst on while forbidding the ones it exists for:
        #
        #   max_spread_cc  a taker pays the spread, so a wide one is a cost. A
        #                  maker is paid it. The default of 200 rejected every
        #                  market whose spread was worth quoting.
        #   min_depth_qc   a taker needs size to fill against. A maker needs the
        #                  queue *ahead* to be short, because it fills in order.
        #                  As a minimum it excluded precisely the reachable
        #                  books; the real constraint is max_queue_qc below.
        #   price band     [5c, 95c] excluded 73% of the exchange's liquid
        #                  volume, which sits under 10c. The fee is quadratic,
        #                  so it collapses at the extremes: 20cc a side at 3c
        #                  against 175cc at 50c.
        StrategyParam(
            "max_spread_cc", 1_500, low=50, high=5_000, step=25,
            description="widest spread worth quoting; wide is good for a maker",
        ),
        StrategyParam(
            "min_depth_qc", 100, low=100, high=100_000, step=100,
            description="minimum resting size on BOTH sides across five levels, in centi-contracts",
        ),
        StrategyParam(
            "max_queue_qc", 5_000, low=100, high=1_000_000, step=100,
            description="longest queue at our own price worth joining, in centi-contracts",
        ),
        *EV_PARAM_SPACE,
        StrategyParam(
            "fee_cover_bps", 15_000, low=10_000, high=50_000, step=1_000,
            description="spread required as a multiple of two fees, in bps; 15000 is 1.5x",
        ),
        StrategyParam(
            "quote_qc", 200, low=100, high=10_000, step=100,
            description="size of each resting quote, in centi-contracts",
        ),
        StrategyParam(
            "max_inventory_qc", 1_000, low=100, high=50_000, step=100,
            description="largest absolute position before quoting only reduces it",
        ),
        StrategyParam(
            "min_price_cc", 100, low=100, high=5_000, step=100,
            description="cheapest midpoint worth quoting",
        ),
        StrategyParam(
            "max_price_cc", 9_900, low=5_000, high=9_900, step=100,
            description="dearest midpoint worth quoting",
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

        # The spread has to pay for both fees with room to spare, or resting is
        # just a slower way of losing. The fee is quadratic in price, so this
        # bar moves with the market rather than being a constant.
        fee_cc = realised_fee_micros(price_cc=mid_cc, qty_qc=QC_PER_CONTRACT) // 100
        required_cc = 2 * fee_cc * self.int_param("fee_cover_bps") // 10_000
        if quote.spread_cc < required_cc:
            return self.skip(_SKIP_SPREAD_TOO_TIGHT)

        # One quote at a time. Without this the strategy re-quotes every
        # cycle and stacks hundreds of resting orders into a single market: the
        # fills then accumulate one large position instead of cycling in and out
        # of small ones, and a week produced five distinct round trips where it
        # should produce hundreds. A market maker that cannot see its own working
        # orders is not quoting, it is accumulating.
        if ctx.resting_qty_qc > 0:
            return self.skip(_SKIP_ALREADY_QUOTING)

        position_qc = ctx.position_qc
        limit_qc = self.int_param("max_inventory_qc")

        # Reduce before adding. A quote that only ever fills on one side turns
        # into a directional position that nothing here has an opinion about.
        if position_qc >= limit_qc:
            side = OutcomeSide.NO
        elif position_qc <= -limit_qc:
            side = OutcomeSide.YES
        elif position_qc > 0:
            side = OutcomeSide.NO
        elif position_qc < 0:
            side = OutcomeSide.YES
        else:
            # Flat: quote the cheaper side, which is the one whose fee is
            # smallest and whose payoff per contract is largest.
            side = (
                OutcomeSide.YES
                if quote.yes_bid_cc <= CC_PER_DOLLAR - quote.yes_ask_cc
                else OutcomeSide.NO
            )

        # Rest *at* the touch on our own side. Joining the queue rather than
        # improving it: improving would earn less spread and still be behind
        # whoever improves next.
        # The book is YES-denominated, so a passive NO quote rests where the YES
        # *ask* is: buying NO at 1 - yes_ask is the same order as offering YES at
        # yes_ask. Using no_ask_cc here would be crossing, not resting -- the
        # exact confusion EventQuote exists to keep in one place.
        price_cc = (
            quote.yes_bid_cc
            if side is OutcomeSide.YES
            else CC_PER_DOLLAR - quote.yes_ask_cc
        )
        if not 0 < price_cc < CC_PER_DOLLAR:
            return self.skip(_SKIP_NO_QUOTE)

        # How much is already resting where we are about to rest. This, not the
        # spread, is what decides whether a passive quote is ever reached: fills
        # go in time order, so everything ahead has to trade before we do. On
        # the exchange's busiest markets the touch carries a median of 124,000
        # contracts, and joining the back of that queue is not market making --
        # the only fills it produces are the ones nobody else wanted, which is
        # adverse selection with the spread given away for free.
        queue_qc = (
            quote.yes_bid_size_qc
            if side is OutcomeSide.YES
            else quote.yes_ask_size_qc
        )
        if queue_qc > self.int_param("max_queue_qc"):
            return self.skip(_SKIP_QUEUE_TOO_LONG)

        qty_qc = self.int_param("quote_qc")
        if abs(position_qc) >= limit_qc:
            qty_qc = min(qty_qc, abs(position_qc))
        if qty_qc <= 0:
            return self.skip(_SKIP_INVENTORY_FULL)

        # Valued at the mid: a resting buy that fills at the bid is immediately
        # worth the mid, and the half-spread between them is the entire edge.
        intent = self.event_intent(
            ctx,
            side=side,
            qty_qc=qty_qc,
            limit_price_cc=price_cc,
            fair_value_cc=mid_cc,
            confidence=0.5,
            costing=event_costing(
                side=side,
                cost_cc=price_cc,
                fair_value_cc=mid_cc,
                qty_qc=qty_qc,
                schedule=fee_schedule_for(ctx),
                passive=True,
            ),
            reason="passive_quote",
            extra={
                "spread_cc": quote.spread_cc,
                "required_spread_cc": required_cc,
                "fee_cc": fee_cc,
                "position_qc": position_qc,
                "resting_at_cc": price_cc,
                "queue_ahead_qc": queue_qc,
            },
        )
        # The whole point. Crossing would pay the spread this exists to earn,
        # and the execution engine rests a post-only order rather than lifting.
        intent.post_only = True
        return self.emit([intent])
