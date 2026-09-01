"""The consumer a forward-move model deserves.

The platform could train a model and had nothing that could use one. The only
model-consuming strategy needs the probability a contract *resolves YES*; the
trainer produced the target it could label, whether the mid rises by more than a
threshold. Fed to the wrong consumer that number was read as a fair value and
produced systematically wrong trades.

This strategy reads it as what it is: an expected drift, using the threshold the
model was trained on, taken from the model's own target.
"""

from __future__ import annotations

import pytest

from kat.common.enums import MarketType
from kat.marketdata.book import OrderBook
from kat.marketdata.state import MarketSnapshot
from kat.strategies.base import StrategyContext
from kat.strategies.event.model_drift import ModelDriftStrategy

NOW_MS = 1_000_000


def _ctx(*, probability: float, target: str | None, mid_cc: int = 5_000) -> StrategyContext:
    book = OrderBook.from_event_contract_levels(
        "X",
        yes_levels=[(mid_cc - 50, 500_000)],
        no_levels=[(10_000 - (mid_cc + 50), 500_000)],
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


def _skip(strategy) -> str | None:
    return strategy.health()["last_skip"]


class TestItOnlyReadsTargetsItUnderstands:
    def test_a_settlement_model_is_refused(self):
        """That target belongs to the mispricing strategy, not this one."""
        s = ModelDriftStrategy()
        assert s.generate(_ctx(probability=0.9, target="settlement_yes")) == []
        assert _skip(s) == "model_answers_a_different_question"

    def test_a_forward_move_target_without_a_threshold_is_refused(self):
        """The old target name. Assuming a threshold would misprice every trade
        by the ratio between the assumed one and the real one."""
        s = ModelDriftStrategy()
        assert s.generate(_ctx(probability=0.9, target="forward_move_30m")) == []
        assert _skip(s) == "model_answers_a_different_question"

    def test_no_target_at_all_is_refused(self):
        s = ModelDriftStrategy()
        assert s.generate(_ctx(probability=0.9, target=None)) == []
        assert _skip(s) == "model_answers_a_different_question"

    @pytest.mark.parametrize(
        ("target", "threshold"),
        [("forward_move_30m_200cc", 200), ("forward_move_5m_50cc", 50)],
    )
    def test_the_threshold_comes_from_the_target(self, target, threshold):
        assert ModelDriftStrategy._threshold_cc(target) == threshold

    @pytest.mark.parametrize(
        "target",
        ["settlement_yes", "forward_move_30m", "forward_move_30m_0cc", "", None],
    )
    def test_an_unreadable_target_has_no_threshold(self, target):
        assert ModelDriftStrategy._threshold_cc(target) is None


class TestTheDriftIsSignedAndConservative:
    def test_a_coin_flip_produces_no_view(self):
        """p = 0.5 is no information, and must not become a trade."""
        s = ModelDriftStrategy()
        assert s.generate(_ctx(probability=0.5, target="forward_move_30m_200cc")) == []
        assert _skip(s) == "model_has_no_directional_view"

    def test_a_weak_view_is_refused(self):
        s = ModelDriftStrategy()
        assert s.generate(_ctx(probability=0.55, target="forward_move_30m_200cc")) == []
        assert _skip(s) == "model_has_no_directional_view"

    def test_a_confident_upward_view_buys_yes(self):
        from kat.common.enums import OutcomeSide

        s = ModelDriftStrategy()
        intents = s.generate(_ctx(probability=0.97, target="forward_move_30m_900cc"))

        assert intents, f"expected an intent, skipped with {_skip(s)}"
        assert intents[0].outcome_side is OutcomeSide.YES

    def test_a_confident_downward_view_buys_no(self):
        """A low probability of rising is a view that it falls."""
        from kat.common.enums import OutcomeSide

        s = ModelDriftStrategy()
        intents = s.generate(_ctx(probability=0.03, target="forward_move_30m_900cc"))

        assert intents, f"expected an intent, skipped with {_skip(s)}"
        assert intents[0].outcome_side is OutcomeSide.NO

    def test_the_drift_never_exceeds_the_threshold(self):
        """drift = T(2p-1), so even total conviction moves fair value by T."""
        s = ModelDriftStrategy()
        intents = s.generate(_ctx(probability=0.999, target="forward_move_30m_900cc"))

        assert intents
        fair = intents[0].fair_value_cc
        assert fair is not None
        assert fair - 5_000 <= 900

    def test_a_small_threshold_cannot_manufacture_an_edge(self):
        """The 50cc model the platform actually deployed: too small to pay for a
        round trip, so no amount of conviction should produce a trade."""
        s = ModelDriftStrategy()
        assert s.generate(_ctx(probability=0.99, target="forward_move_30m_50cc")) == []
        assert _skip(s) == "drift_below_threshold"
