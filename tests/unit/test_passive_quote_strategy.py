"""A maker's guards are not a taker's guards.

This strategy was written to earn the spread rather than pay it, and then given
the liquidity screen every other strategy uses. Three of those guards invert
when the sign of the spread flips, and inheriting them aimed the strategy at the
markets it is worst on while forbidding the ones it exists for:

* a wide spread is a cost to a taker and the entire product to a maker;
* depth is what a taker fills against and what a maker queues behind;
* the [5c, 95c] band excludes 73% of the exchange's liquid volume, which sits
  under 10c, where the quadratic fee has collapsed to a fifth of its peak.

The fourth guard is the one that decides whether any of it works. Fills go in
time order, so a resting quote is reached only after everything already at its
price has traded. Measured against real books, the touch on the busiest markets
carries a median of 124,000 contracts; joining that queue produces only the
fills nobody else wanted.
"""

from __future__ import annotations

from kat.common.enums import MarketType
from kat.marketdata.book import OrderBook
from kat.marketdata.state import MarketSnapshot
from kat.strategies.base import StrategyContext
from kat.strategies.event.passive_quote import PassiveQuoteStrategy

NOW_MS = 1_000_000


def _ctx(
    *,
    bid_cc: int,
    ask_cc: int,
    size_qc: int = 100,
    position_qc: int = 0,
    resting_qty_qc: int = 0,
) -> StrategyContext:
    book = OrderBook.from_event_contract_levels(
        "X",
        yes_levels=[(bid_cc, size_qc)],
        no_levels=[(10_000 - ask_cc, size_qc)],
    )
    snapshot = MarketSnapshot(
        ticker="X", market_type=MarketType.EVENT, book=book, updated_ms=NOW_MS
    )
    return StrategyContext(
        now_ms=NOW_MS,
        market=snapshot,
        features={},
        position_qc=position_qc,
        resting_qty_qc=resting_qty_qc,
        feed_live=True,
    )


def _skip(strategy) -> str | None:
    return strategy.health()["last_skip"]


class TestTheQueueIsTheConstraint:
    def test_a_short_queue_is_quoted(self):
        """Four cents wide at five cents: the tick is 100cc and both fees come
        to about 47cc, so the round trip pays before any view is taken. Four
        rather than two, because a spread under four cents is now refused for
        adverse selection whatever the fee arithmetic says."""
        s = PassiveQuoteStrategy()
        intents = s.generate(_ctx(bid_cc=300, ask_cc=700, size_qc=100))
        assert len(intents) == 1
        assert intents[0].post_only is True
        assert intents[0].metadata["queue_ahead_qc"] == 100

    def test_a_long_queue_is_refused(self):
        """Same market, same spread, same edge -- and unreachable. Everything
        resting at our price trades before we do, so the only difference that
        matters here is the one the old screen could not see."""
        s = PassiveQuoteStrategy()
        assert s.generate(_ctx(bid_cc=300, ask_cc=700, size_qc=5_000_000)) == []
        assert _skip(s) == "queue_ahead_too_long"

    def test_the_queue_checked_is_the_one_we_join(self):
        """A NO quote rests where the YES *ask* is, so it queues behind the ask.
        Checking the bid there would read the wrong side of the book and let a
        quote join a queue it was meant to refuse."""
        s = PassiveQuoteStrategy()
        book = OrderBook.from_event_contract_levels(
            "X",
            yes_levels=[(300, 5_000_000)],       # deep bid
            no_levels=[(10_000 - 700, 100)],     # thin ask
        )
        snapshot = MarketSnapshot(
            ticker="X", market_type=MarketType.EVENT, book=book, updated_ms=NOW_MS
        )
        ctx = StrategyContext(
            now_ms=NOW_MS, market=snapshot, features={},
            position_qc=500, feed_live=True,     # long, so it quotes NO to reduce
        )
        intents = s.generate(ctx)
        assert len(intents) == 1
        assert intents[0].metadata["queue_ahead_qc"] == 100


class TestItNoLongerInheritsTakerGuards:
    def test_a_wide_spread_is_what_it_wants(self):
        """The old default rejected anything above 200cc, which is every spread
        worth resting in."""
        s = PassiveQuoteStrategy()
        assert s.int_param("max_spread_cc") >= 1_000
        assert s.generate(_ctx(bid_cc=7_000, ask_cc=8_000, size_qc=100)) != []

    def test_cheap_contracts_are_in_range(self):
        """Where 73% of the exchange's liquid volume trades, and where the
        quadratic fee is smallest."""
        s = PassiveQuoteStrategy()
        assert s.int_param("min_price_cc") <= 200
        assert s.generate(_ctx(bid_cc=100, ask_cc=500, size_qc=100)) != []

    def test_a_thin_book_is_not_a_disqualification(self):
        """min_depth_qc as a *minimum* excluded precisely the reachable books."""
        s = PassiveQuoteStrategy()
        assert s.int_param("min_depth_qc") <= 200


class TestItStillRefusesWhatItShould:
    def test_a_spread_that_cannot_cover_two_fees_is_refused(self):
        """At fifty cents the fee is 175cc a side, so one tick is a loss."""
        s = PassiveQuoteStrategy()
        assert s.generate(_ctx(bid_cc=5_000, ask_cc=5_100, size_qc=100)) == []
        assert _skip(s) == "spread_does_not_cover_two_fees"

    def test_it_does_not_stack_quotes(self):
        """A maker that cannot see its own working orders accumulates a position
        instead of cycling one."""
        s = PassiveQuoteStrategy()
        assert s.generate(
            _ctx(bid_cc=300, ask_cc=700, size_qc=100, resting_qty_qc=200)
        ) == []
        assert _skip(s) == "already_resting_in_this_market"


class TestTheSpreadHasToPayForAdverseSelection:
    """Fees are not the only cost of resting, and on one venue they are income.

    Measured over 201 fills a continuous touch quote would have taken on
    Polymarket US, the five-minute mark-out from our own fill price -- the
    captured half-spread already in it -- runs -221 +- 107 centicents where the
    spread is under two cents and turns positive, +175 +- 86, only above four.
    The fee arithmetic cannot see that: on a venue that pays the maker, two fees
    are a negative requirement that any spread clears.
    """

    def test_a_tight_spread_is_refused_even_where_fees_are_a_rebate(self):
        from kat.polymarket.fees import PolymarketFeeSchedule

        s = PassiveQuoteStrategy()
        ctx = _ctx(bid_cc=4_900, ask_cc=5_000, size_qc=100)
        ctx.polymarket_fee_schedule = PolymarketFeeSchedule()

        assert s.generate(ctx) == []
        assert _skip(s) == "spread_does_not_cover_two_fees"

    def test_four_cents_is_quoted(self):
        from kat.polymarket.fees import PolymarketFeeSchedule

        s = PassiveQuoteStrategy()
        ctx = _ctx(bid_cc=4_800, ask_cc=5_200, size_qc=100)
        ctx.polymarket_fee_schedule = PolymarketFeeSchedule()

        assert len(s.generate(ctx)) == 1

    def test_the_floor_can_be_tuned_away(self):
        """It is a measurement, not a law, and the search space has to reach it."""
        s = PassiveQuoteStrategy({"min_spread_cc": 0})
        assert s.int_param("min_spread_cc") == 0


class TestRiskDoesNotUseTakerGuardsOnAnOrderThatRests:
    """The same inversion, one layer up, and it made the platform contradictory.

    The strategy requires four cents of spread to cover adverse selection; the
    risk engine's market-quality check refused anything above three. Between
    them they approved nothing at all: 34,846 intents in a day of replay and not
    one order. A wide spread is what a maker is paid, far-side depth is what a
    taker fills against, and slippage is the cost of crossing -- which a
    post-only order cannot do.
    """

    def _verdict(self, *, post_only: bool):
        from kat.common.enums import MarketType, OutcomeSide
        from kat.config.settings import EventRiskSettings
        from kat.risk.event_risk import EventAccountState, EventRiskEngine
        from kat.risk.types import OrderIntent

        engine = EventRiskEngine(EventRiskSettings())
        book = OrderBook.from_event_contract_levels(
            "X",
            yes_levels=[(4_800, 100)],           # four cents wide: past max_spread_cc
            no_levels=[(10_000 - 5_200, 100)],   # and thin, as a maker's book is
        )
        intent = OrderIntent(
            market_type=MarketType.EVENT, ticker="X", qty_qc=100,
            limit_price_cc=4_800, outcome_side=OutcomeSide.YES,
            fair_value_cc=5_000, strategy_id="test",
        )
        intent.post_only = post_only
        return engine.evaluate(
            intent,
            EventAccountState(balance_micros=200_000_000, updated_at_ms=NOW_MS),
            book=book, now_ms=NOW_MS,
        )

    def test_a_resting_order_is_not_refused_for_a_wide_thin_book(self):
        assert self._verdict(post_only=True).reason_code != "MARKET_QUALITY"

    def test_a_crossing_order_still_is(self):
        """The guards are right for the order they were written for."""
        assert self._verdict(post_only=False).reason_code == "MARKET_QUALITY"
