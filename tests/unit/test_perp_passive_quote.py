"""The perpetual quoter's guards, and the arithmetic that decides them.

Two mistakes made this strategy refuse every market it was built for, and both
were about charging one instrument's costs to another.

The first was the fee. The simulator had a single fee path and it was the
quadratic event-contract one -- 350 basis points on a fifty-cent contract
against a perpetual maker's two. Charged that, no perpetual spread on the
exchange covers its costs and the strategy declines everything.

The second was the spread. A round trip that rests on both sides buys at the
bid and sells at the ask, so it earns bid-to-ask *once*: the full spread. The
half-spread is what a single leg earns against the mid, and testing a two-leg
cost against a one-leg revenue rejects economics that are genuinely positive.

The queue guard is inherited deliberately. It was the binding constraint on
event contracts -- fills go in time order, so everything already resting at our
price trades first -- and nothing about a perpetual repeals that.
"""

from __future__ import annotations

from decimal import Decimal

from kat.common.enums import MarketType
from kat.kalshi.fees import PerpFeeSchedule
from kat.marketdata.book import OrderBook
from kat.marketdata.state import MarketSnapshot
from kat.strategies.base import StrategyContext
from kat.strategies.perp.passive_quote import PerpPassiveQuoteStrategy

NOW_MS = 1_000_000
RATES = PerpFeeSchedule(
    maker_rate=Decimal("0.0002"), taker_rate=Decimal("0.0004"), exact=True
)


def _ctx(
    *,
    bid_cc: int = 71_549,
    ask_cc: int = 71_580,
    size_qc: int = 5_000,
    position_qc: int = 0,
    resting_qty_qc: int = 0,
    funding_rate: float | None = None,
) -> StrategyContext:
    book = OrderBook.from_two_sided_levels(
        "KXSUIPERP",
        bid_levels=[(bid_cc, size_qc)],
        ask_levels=[(ask_cc, size_qc)],
    )
    snapshot = MarketSnapshot(
        ticker="KXSUIPERP", market_type=MarketType.PERP, book=book, updated_ms=NOW_MS
    )
    if funding_rate is not None:
        snapshot.funding_rate = funding_rate
        snapshot.funding_interval_hours = 4.0
    ctx = StrategyContext(
        now_ms=NOW_MS,
        market=snapshot,
        features={},
        position_qc=position_qc,
        resting_qty_qc=resting_qty_qc,
        feed_live=True,
        perp_fee_schedule=RATES,
    )
    return ctx


def _skip(strategy) -> str | None:
    return strategy.health()["last_skip"]


class TestItQuotesWhereTheSpreadPays:
    def test_a_spread_covering_both_fees_is_quoted(self):
        """31cc on a $7.15 perpetual is 4.3bp against 6bp of maker-plus-taker
        fees on a round trip -- which the full spread covers and the half does
        not."""
        s = PerpPassiveQuoteStrategy()
        intents = s.generate(_ctx(bid_cc=71_549, ask_cc=71_800))
        assert len(intents) == 1
        assert intents[0].post_only is True

    def test_it_rests_rather_than_crosses(self):
        s = PerpPassiveQuoteStrategy()
        intent = s.generate(_ctx(bid_cc=71_549, ask_cc=71_800))[0]
        assert intent.limit_price_cc in (71_549, 71_800)

    def test_a_one_tick_spread_does_not_pay(self):
        """A single tick is 0.14bp here, well under the fees."""
        s = PerpPassiveQuoteStrategy()
        assert s.generate(_ctx(bid_cc=71_549, ask_cc=71_550)) == []
        assert _skip(s) == "spread_does_not_cover_two_fees"


class TestTheQueueIsStillTheConstraint:
    def test_a_long_queue_is_refused(self):
        """Same spread, same edge, unreachable: everything resting at our price
        trades before we do."""
        s = PerpPassiveQuoteStrategy()
        assert s.generate(
            _ctx(bid_cc=71_549, ask_cc=71_800, size_qc=50_000_000)
        ) == []
        assert _skip(s) == "queue_ahead_too_long"

    def test_the_queue_reported_is_the_side_we_join(self):
        s = PerpPassiveQuoteStrategy()
        intent = s.generate(_ctx(bid_cc=71_549, ask_cc=71_800, size_qc=5_000))[0]
        assert intent.metadata["queue_ahead_qc"] == 5_000

    def test_it_does_not_stack_quotes(self):
        s = PerpPassiveQuoteStrategy()
        assert s.generate(
            _ctx(bid_cc=71_549, ask_cc=71_800, resting_qty_qc=100)
        ) == []
        assert _skip(s) == "already_resting_in_this_market"

    def test_a_spread_nobody_trades_at_is_refused(self):
        """The widest perpetual books belong to the markets with no flow, and
        being paid twenty basis points by nobody is worth nothing."""
        s = PerpPassiveQuoteStrategy()
        assert s.generate(_ctx(bid_cc=50_000, ask_cc=95_000)) == []
        assert _skip(s) == "spread_too_wide"


class TestInventoryAndFunding:
    def test_a_long_position_is_quoted_down(self):
        s = PerpPassiveQuoteStrategy()
        intent = s.generate(_ctx(bid_cc=71_549, ask_cc=71_800, position_qc=500))[0]
        assert intent.limit_price_cc == 71_800  # resting on the offer reduces

    def test_a_short_position_is_quoted_up(self):
        s = PerpPassiveQuoteStrategy()
        intent = s.generate(_ctx(bid_cc=71_549, ask_cc=71_800, position_qc=-500))[0]
        assert intent.limit_price_cc == 71_549

    def test_flat_it_takes_the_side_funding_pays(self):
        """A positive rate means longs pay shorts, so the short is paid."""
        s = PerpPassiveQuoteStrategy()
        intent = s.generate(
            _ctx(bid_cc=71_549, ask_cc=71_800, funding_rate=0.0004)
        )[0]
        assert intent.limit_price_cc == 71_800  # resting on the offer goes short

    def test_funding_is_charged_against_the_edge(self):
        """A quote that earns the spread and pays it back in carry has earned
        nothing, so the carry is part of the arithmetic and not a footnote."""
        s = PerpPassiveQuoteStrategy()
        intent = s.generate(
            _ctx(bid_cc=71_549, ask_cc=71_800, funding_rate=0.0004)
        )[0]
        assert intent.metadata["funding_micros"] > 0

    def test_a_quote_is_whole_contracts(self):
        s = PerpPassiveQuoteStrategy()
        intent = s.generate(_ctx(bid_cc=71_549, ask_cc=71_800))[0]
        assert intent.qty_qc % 100 == 0


class TestItUsesPerpetualFeesNotEventFees:
    def test_without_a_schedule_it_charges_taker_on_both_legs(self):
        """Defaulting to the maker discount would be inventing a rate we were
        never quoted."""
        s = PerpPassiveQuoteStrategy()
        ctx = _ctx(bid_cc=71_549, ask_cc=71_800)
        ctx.perp_fee_schedule = None
        intents = s.generate(ctx)
        # Still quotable at this spread, but costed dearer than with the discount.
        assert len(intents) == 1
        with_rates = PerpPassiveQuoteStrategy().generate(_ctx(bid_cc=71_549, ask_cc=71_800))
        assert intents[0].metadata["fee_micros"] > with_rates[0].metadata["fee_micros"]

    def test_the_event_quadratic_fee_would_refuse_this_market(self):
        """The gap that made every perpetual look unprofitable: at these prices
        the event fee is about a hundred times the perpetual one."""
        from kat.kalshi.fees import estimate_fee_micros, perp_fee_micros

        # At a price both instruments can express -- fifty cents -- the event
        # fee is 350bp of notional and the perpetual taker fee is 4bp. Above a
        # dollar the quadratic formula goes negative, which is its own reason a
        # perpetual must never be priced through it.
        event = estimate_fee_micros(price_cc=5_000, qty_qc=2_000).fee_micros
        perp = perp_fee_micros(price_cc=5_000, qty_qc=2_000, schedule=RATES)
        assert event > perp * 50
