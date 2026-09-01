"""A probability is only usable by something that knows what it is a probability of.

``MispricingStrategy`` turns ``model_probability`` into a fair value for the
contract and compares that to the ask. That arithmetic is only meaningful if the
number is the chance the contract *resolves YES*.

The deployed model's target was ``forward_move_30m``: the chance the mid travels
50 centicents within half an hour. That is a small number for almost every
market and has nothing to do with what the contract is worth. Read as a fair
value it made every expensive market look wildly overpriced -- a low
move-probability becoming a low fair value -- and the platform bought NO on ten
markets in a row at prices from $0.54 to $0.91, each justified by a number that
was never about settlement.

Nothing was broken in the model, the strategy or the risk engine. The units were
wrong, and the units were the one thing nothing carried.
"""

from __future__ import annotations

from kat.common.enums import MarketType
from kat.marketdata.book import OrderBook
from kat.marketdata.state import MarketSnapshot
from kat.strategies.base import StrategyContext
from kat.strategies.event.mispricing import MispricingStrategy

NOW_MS = 1_000_000


def _ctx(*, target: str | None, probability: float = 0.10) -> StrategyContext:
    book = OrderBook.from_event_contract_levels(
        "X", yes_levels=[(1_500, 500_000)], no_levels=[(8_400, 500_000)]
    )
    snapshot = MarketSnapshot(
        ticker="X", market_type=MarketType.EVENT, book=book, updated_ms=NOW_MS
    )
    return StrategyContext(
        now_ms=NOW_MS,
        market=snapshot,
        features={},
        model_probability=probability,
        model_target=target,
        model_id="m",
        model_version="v",
        feed_live=True,
    )


class TestTheStrategyChecksWhatItIsReading:
    def test_a_forward_move_model_is_refused(self):
        """The model that was actually deployed, and the trades it produced."""
        strategy = MispricingStrategy()
        intents = strategy.generate(_ctx(target="forward_move_30m"))

        assert intents == []
        assert strategy.health()["last_skip"] == "model_answers_a_different_question"

    def test_no_target_at_all_is_refused(self):
        """Absent units are not permission to assume the convenient ones."""
        strategy = MispricingStrategy()
        assert strategy.generate(_ctx(target=None)) == []
        assert strategy.health()["last_skip"] == "model_answers_a_different_question"

    def test_a_settlement_model_is_accepted(self):
        """The target the strategy is written for gets past the check."""
        strategy = MispricingStrategy()
        strategy.generate(_ctx(target="settlement_yes"))

        assert strategy.health()["last_skip"] != "model_answers_a_different_question"

    def test_the_refusal_comes_before_any_sizing(self):
        """It must not be reachable by tuning: a wrong unit is not a small edge."""
        strategy = MispricingStrategy()
        # A probability that would otherwise scream "buy NO".
        assert strategy.generate(_ctx(target="forward_move_30m", probability=0.01)) == []
