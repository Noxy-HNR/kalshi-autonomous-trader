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


def _ctx(
    *,
    probability: float,
    target: str | None,
    mid_cc: int = 5_000,
    base_rate: float = 0.024,
) -> StrategyContext:
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
        model_base_rate=base_rate,
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
    def test_the_base_rate_produces_no_view(self):
        """Saying exactly what is already known is not information.

        A half is neutral only for a balanced target. This model's event happens
        2.4% of the time, so reading it against 0.5 made every output a strong
        sell -- 37 round trips in a week, all short, zero winners.
        """
        s = ModelDriftStrategy()
        assert s.generate(_ctx(probability=0.024, target="forward_move_30m_200cc")) == []
        assert _skip(s) == "model_has_no_directional_view"

    def test_a_coin_flip_is_bullish_against_a_rare_event(self):
        """p = 0.5 against a 2.4% base rate is a very strong upward view, and
        must not be read as neutral."""
        s = ModelDriftStrategy()
        s.generate(_ctx(probability=0.5, target="forward_move_30m_900cc", mid_cc=5_000))
        assert _skip(s) != "model_has_no_directional_view"

    def test_no_base_rate_is_refused(self):
        """Without it the probability cannot be read directionally at all."""
        s = ModelDriftStrategy()
        ctx = _ctx(probability=0.9, target="forward_move_30m_200cc")
        ctx.model_base_rate = None
        assert s.generate(ctx) == []
        assert _skip(s) == "model_reports_no_base_rate"

    def test_a_weak_view_is_refused(self):
        s = ModelDriftStrategy()
        assert s.generate(_ctx(probability=0.03, target="forward_move_30m_200cc")) == []
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
        intents = s.generate(_ctx(probability=0.0005, target="forward_move_30m_900cc"))

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


class TestTheEdgeMustCoverWhatCapturingItCosts:
    """The fee is quadratic, so the bar is a function of price.

    A round trip runs about 450 centicents mid-board and about 130 at the
    extremes, against a drift this model can never push past its own threshold.
    A fixed floor is wrong in both directions: mid-board it admits trades the
    fees eat, at the edges it blocks trades that comfortably pay.
    """

    def test_the_cost_collapses_at_the_extremes(self):
        middle = ModelDriftStrategy._round_trip_cc(5_000, 100)
        edge = ModelDriftStrategy._round_trip_cc(9_500, 100)

        assert middle > 350
        assert edge < middle / 2

    def test_the_entry_spread_is_not_counted_twice(self):
        """The edge is fair minus ask, so entry is already paid before this
        floor applies. Only the exit half and both fees remain."""
        from kat.common.fixedpoint import QC_PER_CONTRACT
        from kat.kalshi.fees import realised_fee_micros

        fee_cc = realised_fee_micros(price_cc=5_000, qty_qc=QC_PER_CONTRACT) // 100
        assert ModelDriftStrategy._round_trip_cc(5_000, 200) == 100 + 2 * fee_cc

    def test_a_mid_priced_market_cannot_pay_for_a_small_drift(self):
        """The arithmetic that decides where this model is tradeable at all."""
        s = ModelDriftStrategy()
        # Total conviction, 200cc threshold: the largest drift possible is 200,
        # and a round trip at $0.50 costs more than twice that.
        assert s.generate(
            _ctx(probability=0.999, target="forward_move_30m_200cc", mid_cc=5_000)
        ) == []
        assert _skip(s) == "drift_below_threshold"

    def test_the_same_conviction_pays_at_the_extremes(self):
        """Where the fee is small, the same drift clears its cost."""
        s = ModelDriftStrategy()
        intents = s.generate(
            _ctx(probability=0.999, target="forward_move_30m_900cc", mid_cc=7_800)
        )
        assert intents, f"expected an intent, skipped with {_skip(s)}"

    def test_the_floor_is_never_below_the_configured_minimum(self):
        """Cost-awareness raises the bar; it must not lower it."""
        s = ModelDriftStrategy()
        # A market so cheap to trade that cost < min_edge_cc; min_edge still binds.
        cheap = ModelDriftStrategy._round_trip_cc(9_900, 10)
        assert cheap < s.int_param("min_edge_cc")
