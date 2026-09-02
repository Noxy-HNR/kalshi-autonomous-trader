"""Resting an entry on a forecast: what it must do, and what it must refuse.

The strategy exists to test one arithmetic claim. Crossing both legs of a
perpetual round trip, a directional model needs 72.6% accuracy to break even and
reaches 65.1%. Resting the entry removes the spread -- the largest term in the
cost -- and drops the bar to 63.5%, which the model clears. Everything here
protects that claim from being tested dishonestly.

Two of these tests matter more than the rest. The entry must actually *rest*,
because a strategy that crosses is measuring the bar it was built to avoid. And
the probability must be an answer to this question: fed a settlement probability
or another horizon's forecast, the number still reads as a direction and
produces confidently wrong trades rather than no trades.
"""

from __future__ import annotations

from decimal import Decimal

from kat.common.enums import MarketType
from kat.kalshi.fees import PerpFeeSchedule
from kat.marketdata.book import OrderBook
from kat.marketdata.state import MarketSnapshot
from kat.strategies.base import StrategyContext
from kat.strategies.perp.model_quote import TARGET, PerpModelQuoteStrategy

NOW_MS = 10_000_000
RATES = PerpFeeSchedule(
    maker_rate=Decimal("0.0002"), taker_rate=Decimal("0.0004"), exact=True
)


def _ctx(
    *,
    probability: float | None = 0.70,
    target: str | None = TARGET,
    bid_cc: int = 71_549,
    ask_cc: int = 71_580,
    size_qc: int = 5_000,
    position_qc: int = 0,
    position_age_ms: int | None = None,
    resting_qty_qc: int = 0,
    feed_live: bool = True,
) -> StrategyContext:
    book = OrderBook.from_two_sided_levels(
        "KXSUIPERP",
        bid_levels=[(bid_cc, size_qc)],
        ask_levels=[(ask_cc, size_qc)],
    )
    snapshot = MarketSnapshot(
        ticker="KXSUIPERP", market_type=MarketType.PERP, book=book, updated_ms=NOW_MS
    )
    return StrategyContext(
        now_ms=NOW_MS,
        market=snapshot,
        features={},
        model_probability=probability,
        model_target=target,
        position_qc=position_qc,
        position_age_ms=position_age_ms,
        resting_qty_qc=resting_qty_qc,
        feed_live=feed_live,
        perp_fee_schedule=RATES,
    )


def _skip(strategy) -> str | None:
    return strategy.health()["last_skip"]


class TestTheEntryRests:
    def test_a_confident_up_call_rests_on_the_bid(self):
        """The whole hypothesis. Crossing to the ask would pay the spread this
        exists to earn and put the break-even bar back out of reach."""
        s = PerpModelQuoteStrategy()
        intent = s.generate(_ctx(probability=0.70))[0]
        assert intent.post_only is True
        assert intent.limit_price_cc == 71_549

    def test_a_confident_down_call_rests_on_the_ask(self):
        s = PerpModelQuoteStrategy()
        intent = s.generate(_ctx(probability=0.30))[0]
        assert intent.post_only is True
        assert intent.limit_price_cc == 71_580

    def test_the_probability_travels_with_the_intent(self):
        s = PerpModelQuoteStrategy()
        intent = s.generate(_ctx(probability=0.70))[0]
        assert intent.metadata["probability"] == 0.70

    def test_a_quote_is_whole_contracts(self):
        s = PerpModelQuoteStrategy()
        assert s.generate(_ctx())[0].qty_qc % 100 == 0


class TestItOnlyReadsForecastsItUnderstands:
    def test_no_probability_means_no_trade(self):
        s = PerpModelQuoteStrategy()
        assert s.generate(_ctx(probability=None)) == []
        assert _skip(s) == "no_model_probability"

    def test_a_settlement_probability_is_refused(self):
        """Read as a direction it is not wrong, it is meaningless -- and a
        meaningless number above 0.5 still buys."""
        s = PerpModelQuoteStrategy()
        assert s.generate(_ctx(probability=0.9, target="settlement_yes")) == []
        assert _skip(s) == "model_answers_a_different_question"

    def test_another_horizon_is_refused(self):
        s = PerpModelQuoteStrategy()
        assert s.generate(_ctx(probability=0.9, target="perp_up_60m")) == []
        assert _skip(s) == "model_answers_a_different_question"

    def test_an_even_probability_is_not_a_forecast(self):
        s = PerpModelQuoteStrategy()
        assert s.generate(_ctx(probability=0.52)) == []
        assert _skip(s) == "confidence_below_threshold"


class TestTheHorizonGovernsTheExit:
    def test_inside_the_horizon_it_holds(self):
        s = PerpModelQuoteStrategy()
        assert s.generate(_ctx(position_qc=200, position_age_ms=60_000)) == []
        assert _skip(s) == "holding_within_horizon"

    def test_past_the_horizon_it_crosses_out(self):
        """A fifteen-minute forecast has nothing to say at minute sixteen, and
        waiting passively for a good exit turns it into a position nobody
        analysed."""
        s = PerpModelQuoteStrategy()
        intent = s.generate(
            _ctx(position_qc=200, position_age_ms=16 * 60_000)
        )[0]
        assert intent.reduce_only is True
        assert intent.post_only is False
        assert intent.qty_qc == 200

    def test_the_exit_closes_a_short_by_buying(self):
        s = PerpModelQuoteStrategy()
        intent = s.generate(
            _ctx(position_qc=-200, position_age_ms=16 * 60_000)
        )[0]
        assert intent.reduce_only is True
        assert intent.limit_price_cc == 71_580  # crossing to the ask

    def test_an_unknown_age_is_not_treated_as_expired(self):
        """A missing value is never a favourable one: unknown age means hold,
        not liquidate."""
        s = PerpModelQuoteStrategy()
        assert s.generate(_ctx(position_qc=200, position_age_ms=None)) == []
        assert _skip(s) == "holding_within_horizon"


class TestItRefusesWhatItCannotReach:
    def test_a_long_queue_is_refused(self):
        """Fills go in time order, and this is the constraint that decided
        both passive structures measured before it."""
        s = PerpModelQuoteStrategy()
        assert s.generate(_ctx(size_qc=50_000_000)) == []
        assert _skip(s) == "queue_ahead_too_long"

    def test_it_does_not_stack_quotes(self):
        s = PerpModelQuoteStrategy()
        assert s.generate(_ctx(resting_qty_qc=100)) == []
        assert _skip(s) == "already_resting_in_this_market"

    def test_a_stale_book_stops_it_when_the_feed_is_not_known_live(self):
        """The age check is only the whole story when nothing else vouches for
        the feed. On a delta feed silence usually means the book is unchanged,
        which is why feed_live exists and why it is set by the component that
        can see every market at once rather than guessed here."""
        s = PerpModelQuoteStrategy()
        ctx = _ctx(feed_live=False)
        ctx.market.updated_ms = NOW_MS - 10 * 60_000
        assert s.generate(ctx) == []

    def test_a_live_feed_vouches_for_a_quiet_market(self):
        s = PerpModelQuoteStrategy()
        ctx = _ctx(feed_live=True)
        ctx.market.updated_ms = NOW_MS - 10 * 60_000
        assert s.generate(ctx) != []
