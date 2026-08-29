"""Turning an opinion into a costed proposal, identically for every strategy.

A concrete strategy in this package does exactly one interesting thing: it forms
a view on what a market is worth. Everything either side of that view -- is the
data fresh, is the book wide, is there room under the position cap, how many
contracts does the budget buy, what does walking three levels actually cost, is
the expected value still positive once a round trip of fees is charged -- is
identical across all eight of them.

Written eight times it would be written eight slightly different ways, and the
differences would all drift in the same direction, because every one of these
steps has an optimistic variant that is easier to write than the correct one.
So it is written once, here, and a strategy that wants to trade must go through
:func:`propose_event_trade` or :func:`propose_perp_trade` to do it.

Two decisions in here are worth stating explicitly.

**Liquidity is screened on both sides.** A one-sided book is not a tradeable
market even when the side you want to hit is deep, because the position has to
be closed as well as opened, and the exit is the leg that happens when the
market has moved against you.

**The exit is priced, not assumed.** For perpetuals the fair value handed in is
a forecast *midpoint*; :func:`propose_perp_trade` converts it into a realisable
exit by charging half the measured round-trip crossing cost against it, so the
spread is paid for in the expected value rather than discovered afterwards. The
event path gets the same treatment for free, because
:func:`~kat.strategies.common.event_costing` charges a round trip of fees and
the fair value is denominated in settlement terms.

Refusals are returned as reason strings rather than raised. Declining to trade
is the normal outcome -- see :meth:`~kat.strategies.common.TradingStrategy.skip`
-- and an exception per quiet market would be an exception per tick.
"""

from __future__ import annotations

from dataclasses import dataclass

from kat.common.enums import OutcomeSide
from kat.common.fixedpoint import QC_PER_CONTRACT, Rounding, parse_dollars_micros
from kat.marketdata.book import OrderBook
from kat.strategies.base import StrategyContext, StrategyParam
from kat.strategies.common import (
    MIN_ORDER_QC,
    Costing,
    EventQuote,
    ExecutionCost,
    TradingStrategy,
    clamp_price_cc,
    contracts_for_budget_qc,
    event_costing,
    event_execution_cost,
    fee_schedule_for,
    funding_benefit_micros,
    perp_costing,
    perp_execution_cost,
)

__all__ = [
    "EV_PARAM_SPACE",
    "LIQUIDITY_PARAM_SPACE",
    "SKIP_NEGATIVE_EV",
    "SKIP_NO_BOOK",
    "SKIP_NO_LIQUIDITY",
    "SKIP_NO_TIMESTAMP",
    "SKIP_POSITION_CAP",
    "SKIP_SIZE_BELOW_MIN",
    "SKIP_STALE",
    "SKIP_THIN_BOOK",
    "SKIP_WIDE_SPREAD",
    "EventProposal",
    "PerpProposal",
    "RoundTrip",
    "conservative_funding_rate",
    "data_age_ms",
    "floor_contracts_qc",
    "headroom_qc",
    "net_ev_floor_micros",
    "perp_round_trip",
    "propose_event_trade",
    "propose_perp_trade",
    "screen_event_market",
    "screen_freshness",
    "screen_perp_market",
]

# Stable skip-reason codes. They are counted per strategy by ``TradingStrategy``
# and rendered on the dashboard, so they are constants rather than literals: a
# renamed reason silently splits one counter into two.
SKIP_NO_BOOK = "no_usable_book"
SKIP_NO_TIMESTAMP = "market_timestamp_unknown"
SKIP_STALE = "stale_market_data"
SKIP_WIDE_SPREAD = "spread_too_wide"
SKIP_THIN_BOOK = "book_too_thin"
SKIP_POSITION_CAP = "position_cap_reached"
SKIP_SIZE_BELOW_MIN = "size_below_one_contract"
SKIP_NO_LIQUIDITY = "insufficient_depth_for_size"
SKIP_NEGATIVE_EV = "expected_value_below_floor"


#: Market-quality parameters every concrete strategy carries. They are in the
#: search space because "how wide is too wide" is genuinely market-dependent,
#: but the bounds keep the optimiser inside markets that can be exited.
LIQUIDITY_PARAM_SPACE: tuple[StrategyParam, ...] = (
    StrategyParam(
        "max_spread_cc", 200, low=50, high=1_500, step=25,
        description="widest bid/ask spread this strategy will trade, in centicents",
    ),
    StrategyParam(
        "min_depth_qc", 2_000, low=200, high=100_000, step=100,
        description="minimum resting size on BOTH sides across five levels, in centi-contracts",
    ),
    StrategyParam(
        "max_position_qc", 5_000, low=100, high=200_000, step=100,
        description="largest absolute position this strategy will hold in one market",
    ),
)

#: The floor on expected value. Separate from the liquidity space because it is
#: the one parameter that decides whether a marginal trade is worth doing at all.
EV_PARAM_SPACE: tuple[StrategyParam, ...] = (
    StrategyParam(
        "min_net_ev_usd", 0.05, low=0.0, high=10.0, step=0.01,
        description="net expected value, after a round trip of fees, below which no order is sent",
    ),
)


def net_ev_floor_micros(min_net_ev_usd: float) -> int:
    """The EV floor in micro-dollars, rounded up so the floor is never softened."""
    return parse_dollars_micros(min_net_ev_usd, rounding=Rounding.CEIL)


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------


def data_age_ms(ctx: StrategyContext) -> int | None:
    """Age of the snapshot at ``ctx.now_ms``, or ``None`` when it has no stamp.

    An unstamped snapshot is treated as unknown rather than as fresh. Defaulting
    the other way would make a strategy trade hardest in exactly the situation
    where the feed has stopped and the timestamp was never set.
    """
    stamped = ctx.market.updated_ms or ctx.market.book_updated_ms
    if stamped <= 0:
        return None
    return ctx.now_ms - stamped


def screen_freshness(ctx: StrategyContext, max_age_ms: int) -> str | None:
    """``None`` when the snapshot is fresh enough to act on, else a skip reason.

    A market older than ``max_age_ms`` is still actable when the feed is known
    to be delivering, because this is a delta feed: the exchange sends a message
    when the book *changes*, so silence on one market usually means our copy is
    still exact. That judgement is not made here -- ``ctx.feed_live`` carries it
    from the component that owns the connection and can see every market at
    once, which is the only place the two kinds of silence are distinguishable.

    When it is not supplied the age check stands alone, so a backtest building
    contexts by hand keeps the older, stricter behaviour.
    """
    age = data_age_ms(ctx)
    if age is None:
        return SKIP_NO_TIMESTAMP
    if age > max_age_ms and not ctx.feed_live:
        return SKIP_STALE
    return None


def screen_event_market(
    ctx: StrategyContext, *, max_spread_cc: int, min_depth_qc: int
) -> tuple[EventQuote | None, str | None]:
    """Quote for a tradeable event market, or the reason it is not one.

    Exactly one element of the pair is non-``None``.
    """
    quote = EventQuote.from_book(ctx.market.book)
    if quote is None:
        return None, SKIP_NO_BOOK
    if quote.spread_cc > max_spread_cc:
        return None, SKIP_WIDE_SPREAD
    if min(quote.bid_depth_qc, quote.ask_depth_qc) < min_depth_qc:
        return None, SKIP_THIN_BOOK
    return quote, None


def screen_perp_market(
    ctx: StrategyContext, *, max_spread_cc: int, min_depth_qc: int
) -> tuple[OrderBook | None, str | None]:
    """Book for a tradeable perpetual market, or the reason it is not one."""
    book = ctx.market.book
    if book is None or book.is_crossed:
        return None, SKIP_NO_BOOK
    bid, ask = book.best_bid_cc, book.best_ask_cc
    if bid is None or ask is None:
        return None, SKIP_NO_BOOK
    if ask - bid > max_spread_cc:
        return None, SKIP_WIDE_SPREAD
    if min(book.depth_qc(is_bid=True, levels=5), book.depth_qc(is_bid=False, levels=5)) < min_depth_qc:
        return None, SKIP_THIN_BOOK
    return book, None


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def floor_contracts_qc(qty_qc: int) -> int:
    """Round a size down to whole contracts; Kalshi does not trade fractions."""
    if qty_qc <= 0:
        return 0
    return (qty_qc // QC_PER_CONTRACT) * QC_PER_CONTRACT


def headroom_qc(position_qc: int, max_position_qc: int, *, increases: bool) -> int:
    """Size still permitted under the position cap, in whole contracts.

    ``position_qc`` is signed: positive is long YES for an event contract and
    long the perpetual otherwise. An order that moves toward flat gets headroom
    on the far side of zero as well, which is what allows a reversing signal to
    close an existing position and open the opposite one in a single intent.
    """
    cap = abs(max_position_qc)
    room = (cap - position_qc) if increases else (cap + position_qc)
    return floor_contracts_qc(max(0, room))


# ---------------------------------------------------------------------------
# Event proposals
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EventProposal:
    """A sized, priced, costed event-contract trade that is worth doing."""

    side: OutcomeSide
    qty_qc: int
    limit_price_cc: int
    fair_value_cc: int
    costing: Costing
    execution: ExecutionCost

    @property
    def edge_cc(self) -> int:
        return self.costing.edge_cc


def propose_event_trade(
    strategy: TradingStrategy,
    ctx: StrategyContext,
    *,
    book: OrderBook,
    quote: EventQuote,
    side: OutcomeSide,
    fair_value_cc: int,
    max_position_qc: int,
    min_net_ev_micros: int,
) -> tuple[EventProposal | None, str | None]:
    """Size and cost a buy of ``side``, or explain why there is nothing to do.

    ``fair_value_cc`` is YES-denominated on the way in and on the way out, which
    is what the risk engine expects; the NO valuation is derived inside
    :func:`~kat.strategies.common.event_costing`.

    The size is derived from the touch price, then re-derived from the achieved
    VWAP if walking the book pushed the notional past the budget. Sizing once
    from the touch and calling it done is how an order that was meant to spend
    twenty-five dollars spends twenty-nine.
    """
    budget_micros = strategy.budget_micros()
    if budget_micros <= 0:
        return None, SKIP_SIZE_BELOW_MIN

    room_qc = headroom_qc(ctx.position_qc, max_position_qc, increases=side is OutcomeSide.YES)
    if room_qc < MIN_ORDER_QC:
        return None, SKIP_POSITION_CAP

    wanted_qc = min(
        contracts_for_budget_qc(budget_micros=budget_micros, price_cc=quote.cost_cc(side)),
        room_qc,
    )
    if wanted_qc < MIN_ORDER_QC:
        return None, SKIP_SIZE_BELOW_MIN

    levels = strategy.int_param("max_book_levels")
    execution = event_execution_cost(book, side, wanted_qc, levels=levels)
    if execution is None:
        return None, SKIP_NO_LIQUIDITY

    if execution.vwap_cc * execution.qty_qc > budget_micros:
        # Walking cost more than the touch implied. Re-size against what the
        # order actually costs rather than trimming the budget after the fact.
        affordable_qc = contracts_for_budget_qc(
            budget_micros=budget_micros, price_cc=execution.vwap_cc
        )
        if affordable_qc < MIN_ORDER_QC:
            return None, SKIP_SIZE_BELOW_MIN
        execution = event_execution_cost(book, side, affordable_qc, levels=levels)
        if execution is None:
            return None, SKIP_NO_LIQUIDITY

    costing = event_costing(
        side=side,
        cost_cc=execution.vwap_cc,
        fair_value_cc=fair_value_cc,
        qty_qc=execution.qty_qc,
        schedule=fee_schedule_for(ctx),
    )
    if costing.net_micros < min_net_ev_micros:
        return None, SKIP_NEGATIVE_EV

    return (
        EventProposal(
            side=side,
            qty_qc=execution.qty_qc,
            limit_price_cc=execution.worst_cc,
            fair_value_cc=fair_value_cc,
            costing=costing,
            execution=execution,
        ),
        None,
    )


# ---------------------------------------------------------------------------
# Perpetual proposals
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RoundTrip:
    """What it costs to open a perpetual position *and close it again*.

    Both legs are priced against the book that exists now. The close will really
    happen against a different book, so this is an estimate -- but it is an
    estimate of the right quantity, whereas pricing only the open is an exact
    measurement of the wrong one.
    """

    entry: ExecutionCost
    exit_leg: ExecutionCost

    @property
    def qty_qc(self) -> int:
        return self.entry.qty_qc

    @property
    def crossing_cc(self) -> int:
        """Full cost of crossing the spread twice, per contract."""
        return abs(self.entry.vwap_cc - self.exit_leg.vwap_cc)

    @property
    def half_crossing_cc(self) -> int:
        """Crossing cost attributable to one leg, rounded up against us."""
        return -(-self.crossing_cc // 2)


def perp_round_trip(
    book: OrderBook, *, is_long: bool, qty_qc: int, levels: int
) -> RoundTrip | None:
    """Price both legs at the largest size the book supports on both sides."""
    entry = perp_execution_cost(book, is_buy=is_long, wanted_qc=qty_qc, levels=levels)
    exit_leg = perp_execution_cost(book, is_buy=not is_long, wanted_qc=qty_qc, levels=levels)
    if entry is None or exit_leg is None:
        return None

    tradeable_qc = min(entry.qty_qc, exit_leg.qty_qc)
    if tradeable_qc < MIN_ORDER_QC:
        return None
    if entry.qty_qc != tradeable_qc:
        entry = perp_execution_cost(book, is_buy=is_long, wanted_qc=tradeable_qc, levels=levels)
    if exit_leg.qty_qc != tradeable_qc:
        exit_leg = perp_execution_cost(book, is_buy=not is_long, wanted_qc=tradeable_qc, levels=levels)
    if entry is None or exit_leg is None:
        return None
    return RoundTrip(entry=entry, exit_leg=exit_leg)


def conservative_funding_rate(
    rate_bps_per_hour: float | None, *, is_long: bool, persistence: float
) -> float:
    """Discount a funding rate we would *receive*; leave one we would pay alone.

    Kalshi's sign convention is that a positive rate means longs pay shorts, so
    which side of the payment we are on depends on the direction of the trade.
    The rate is a snapshot and it moves; assuming it persists is optimistic when
    it is income and pessimistic when it is a cost, so the discount is applied
    to income only. Applying it symmetrically would quietly shrink a cost, which
    is the same error in the other direction.
    """
    if rate_bps_per_hour is None or rate_bps_per_hour == 0.0:
        return 0.0
    receiving = (rate_bps_per_hour > 0.0) != is_long
    if not receiving:
        return rate_bps_per_hour
    return rate_bps_per_hour * min(1.0, max(0.0, persistence))


@dataclass(frozen=True, slots=True)
class PerpProposal:
    """A sized, priced, costed perpetual trade that is worth doing."""

    is_long: bool
    qty_qc: int
    limit_price_cc: int
    fair_value_cc: int
    costing: Costing
    round_trip: RoundTrip

    @property
    def edge_cc(self) -> int:
        return self.costing.edge_cc


def propose_perp_trade(
    strategy: TradingStrategy,
    ctx: StrategyContext,
    *,
    book: OrderBook,
    is_long: bool,
    forecast_mid_cc: int,
    max_position_qc: int,
    min_net_ev_micros: int,
    funding_rate_bps_per_hour: float | None = None,
    hold_hours: float = 0.0,
) -> tuple[PerpProposal | None, str | None]:
    """Size and cost a perpetual position, or explain why there is nothing to do.

    ``forecast_mid_cc`` is where the strategy expects the *midpoint* to be when
    it closes. It is converted here into a realisable exit price by charging one
    leg of the measured crossing cost against it, so a forecast that only just
    clears the spread produces no trade instead of a small guaranteed loss.

    Funding is folded into the expected value rather than checked separately: on
    a perpetual it is part of the edge, and for a carry trade it is the whole of
    it. The size requested is a *preference*; the 1.00x exposure ceiling in
    :mod:`kat.risk.invariants` will resize or refuse it, and this function has
    no visibility of account equity by design.
    """
    budget_micros = strategy.budget_micros()
    if budget_micros <= 0:
        return None, SKIP_SIZE_BELOW_MIN

    room_qc = headroom_qc(ctx.position_qc, max_position_qc, increases=is_long)
    if room_qc < MIN_ORDER_QC:
        return None, SKIP_POSITION_CAP

    touch_cc = book.best_ask_cc if is_long else book.best_bid_cc
    if touch_cc is None:
        return None, SKIP_NO_BOOK

    levels = strategy.int_param("max_book_levels")
    wanted_qc = min(contracts_for_budget_qc(budget_micros=budget_micros, price_cc=touch_cc), room_qc)
    if wanted_qc < MIN_ORDER_QC:
        return None, SKIP_SIZE_BELOW_MIN

    trip = perp_round_trip(book, is_long=is_long, qty_qc=wanted_qc, levels=levels)
    if trip is None:
        return None, SKIP_NO_LIQUIDITY

    if trip.entry.vwap_cc * trip.qty_qc > budget_micros:
        affordable_qc = contracts_for_budget_qc(
            budget_micros=budget_micros, price_cc=trip.entry.vwap_cc
        )
        if affordable_qc < MIN_ORDER_QC:
            return None, SKIP_SIZE_BELOW_MIN
        trip = perp_round_trip(book, is_long=is_long, qty_qc=affordable_qc, levels=levels)
        if trip is None:
            return None, SKIP_NO_LIQUIDITY

    half = trip.half_crossing_cc
    realisable_exit_cc = clamp_price_cc(forecast_mid_cc - half if is_long else forecast_mid_cc + half)

    funding_micros = 0
    if funding_rate_bps_per_hour is not None and hold_hours > 0.0:
        funding_micros = funding_benefit_micros(
            rate_bps_per_hour=funding_rate_bps_per_hour,
            notional_micros=trip.entry.vwap_cc * trip.qty_qc,
            hours=hold_hours,
            is_long=is_long,
        )

    costing = perp_costing(
        is_long=is_long,
        entry_cc=trip.entry.vwap_cc,
        fair_cc=realisable_exit_cc,
        qty_qc=trip.qty_qc,
        schedule=fee_schedule_for(ctx),
        funding_micros=funding_micros,
    )
    if costing.net_micros < min_net_ev_micros:
        return None, SKIP_NEGATIVE_EV

    return (
        PerpProposal(
            is_long=is_long,
            qty_qc=trip.qty_qc,
            limit_price_cc=trip.entry.worst_cc,
            fair_value_cc=realisable_exit_cc,
            costing=costing,
            round_trip=trip,
        ),
        None,
    )
