"""Two risk inputs that only the execution engine knows must reach it.

``EventAccountState.orders_this_minute`` and
``EventPositionState.pending_exposure_micros`` are both read by the event risk
engine and were both written only by the backtester. In the live and paper engine
they sat at their defaults of zero on every evaluation, so:

* ``max_orders_per_minute`` compared ``0 < 30`` and could not fire, though its
  rejection calls itself "runaway-order protection engaged"; and
* capital already promised to resting orders counted for nothing against the
  available-cash and per-market concentration limits.

Both defaults read as "nothing submitted, nothing pending" -- the most permissive
answer either field has, and the opposite of this platform's posture everywhere
else, where what cannot be established is refused rather than assumed favourable.
The backtester filled them in, so it enforced limits the live engine did not and
a backtest measured a stricter platform than the one that would trade.

Observed on the running engine: 55 good-till-cancelled orders against a single
market in 55 seconds, roughly twice the configured rate limit, none filling, none
counted against capital and none refused.
"""

from __future__ import annotations

from datetime import UTC, datetime

from kat.common.enums import (
    MarketType,
    OrderState,
    OutcomeSide,
    RiskDecision,
    TradingEnvironment,
)
from kat.common.timeutil import SimulatedClock
from kat.config.settings import EventRiskSettings
from kat.engine.trading import TradingLoop
from kat.execution.event_exec import EventExecutionEngine
from kat.execution.order import ManagedOrder
from kat.marketdata.state import MarketState
from kat.risk.event_risk import EventAccountState, EventRiskEngine
from kat.risk.types import OrderIntent

NOW_MS = 10_000_000_000
TICKER = "X"


def _engine() -> EventExecutionEngine:
    return EventExecutionEngine(
        rest=None,
        market_state=MarketState(MarketType.EVENT),
        clock=SimulatedClock(NOW_MS),
        environment=TradingEnvironment.PAPER,
    )


def _order(*, ticker: str = TICKER, qty_qc: int = 100, price_cc: int = 5_000,
           age_s: float = 0.0, state: OrderState = OrderState.ACKNOWLEDGED) -> ManagedOrder:
    order = ManagedOrder(
        market_type=MarketType.EVENT,
        ticker=ticker,
        requested_qty_qc=qty_qc,
        limit_price_cc=price_cc,
        outcome_side=OutcomeSide.YES,
        strategy_id="s",
    )
    order.state = state
    order.submitted_ts = datetime.fromtimestamp((NOW_MS / 1000.0) - age_s, tz=UTC)
    return order


class TestTheEngineCanCountItsOwnOrders:
    def test_recent_orders_are_counted(self):
        engine = _engine()
        for i in range(5):
            o = _order(age_s=i * 5)
            engine.orders[o.order_id] = o

        assert engine.orders_in_last_minute(now_ms=NOW_MS) == 5

    def test_orders_older_than_a_minute_fall_out(self):
        engine = _engine()
        recent = _order(age_s=10)
        stale = _order(age_s=120)
        engine.orders[recent.order_id] = recent
        engine.orders[stale.order_id] = stale

        assert engine.orders_in_last_minute(now_ms=NOW_MS) == 1

    def test_an_unsubmitted_order_is_not_counted(self):
        engine = _engine()
        o = _order()
        o.submitted_ts = None
        engine.orders[o.order_id] = o

        assert engine.orders_in_last_minute(now_ms=NOW_MS) == 0

    def test_no_orders_is_zero(self):
        assert _engine().orders_in_last_minute(now_ms=NOW_MS) == 0


class TestTheEngineCanValueItsWorkingOrders:
    def test_working_orders_commit_capital(self):
        engine = _engine()
        o = _order(qty_qc=461, price_cc=8_500)
        engine.orders[o.order_id] = o

        # qty_qc * price_cc is already micros: 1e-2 contracts * 1e-4 USD.
        assert engine.working_exposure_micros() == {TICKER: 461 * 8_500}

    def test_orders_in_the_same_market_add_up(self):
        """The observed case: identical resting orders, once per cycle."""
        engine = _engine()
        for _ in range(55):
            o = _order(qty_qc=461, price_cc=8_500)
            engine.orders[o.order_id] = o

        committed = engine.working_exposure_micros()[TICKER]
        assert committed == 55 * 461 * 8_500
        assert committed > 200_000_000  # more than the whole $200 account

    def test_a_filled_order_commits_nothing_further(self):
        engine = _engine()
        o = _order(state=OrderState.FILLED)
        o.filled_qty_qc = o.requested_qty_qc
        engine.orders[o.order_id] = o

        assert engine.working_exposure_micros() == {}

    def test_only_the_unfilled_remainder_is_committed(self):
        engine = _engine()
        o = _order(qty_qc=100, price_cc=5_000)
        o.filled_qty_qc = 40
        engine.orders[o.order_id] = o

        assert engine.working_exposure_micros() == {TICKER: 60 * 5_000}


class TestTheLoopCarriesThemToTheAccountState:
    @staticmethod
    def _loop(execution) -> TradingLoop:
        loop = TradingLoop(
            market_type=MarketType.EVENT,
            market_state=MarketState(MarketType.EVENT),
            risk_engine=EventRiskEngine(EventRiskSettings()),
            execution=execution,
            recorder=None,  # type: ignore[arg-type]
            clock=SimulatedClock(NOW_MS),
        )
        loop.account_state = EventAccountState(
            balance_micros=200_000_000, updated_at_ms=NOW_MS
        )
        return loop

    def test_the_order_rate_reaches_the_account_state(self):
        execution = _engine()
        for i in range(40):
            o = _order(age_s=i * 0.5)
            execution.orders[o.order_id] = o
        loop = self._loop(execution)

        loop._refresh_from_execution(NOW_MS)

        assert loop.account_state.orders_this_minute == 40

    def test_pending_capital_reaches_a_market_with_no_position(self):
        """The market that ran away had 55 resting orders and no position."""
        execution = _engine()
        for _ in range(55):
            o = _order(qty_qc=461, price_cc=8_500)
            execution.orders[o.order_id] = o
        loop = self._loop(execution)

        loop._refresh_from_execution(NOW_MS)

        position = loop.account_state.positions[TICKER]
        assert position.pending_exposure_micros == 55 * 461 * 8_500
        assert position.position_qc == 0
        # A placeholder for pending capital is not an open position.
        assert loop.account_state.total_pending_micros() > 0

    def test_pending_capital_is_cleared_when_the_orders_go(self):
        execution = _engine()
        o = _order()
        execution.orders[o.order_id] = o
        loop = self._loop(execution)
        loop._refresh_from_execution(NOW_MS)
        assert loop.account_state.positions[TICKER].pending_exposure_micros > 0

        o.state = OrderState.FILLED
        o.filled_qty_qc = o.requested_qty_qc
        loop._refresh_from_execution(NOW_MS)

        # The account held nothing in this market but the order, so with the
        # order gone there is nothing left to carry: the entry goes too. Keeping
        # it grew the dict with every market ever quoted and exempted the market
        # from the open-position cap.
        assert TICKER not in loop.account_state.positions

    def test_the_rate_limit_now_actually_fires(self):
        """End to end: the limit exists to stop this, and now it does."""
        execution = _engine()
        settings = EventRiskSettings()
        for i in range(settings.max_orders_per_minute + 10):
            o = _order(age_s=i * 0.5)
            execution.orders[o.order_id] = o
        loop = self._loop(execution)
        loop._refresh_from_execution(NOW_MS)

        verdict = loop.risk_engine.evaluate(
            OrderIntent(
                market_type=MarketType.EVENT,
                ticker="OTHER",
                qty_qc=100,
                limit_price_cc=5_000,
                outcome_side=OutcomeSide.YES,
                expected_value_micros=1_000_000,
            ),
            loop.account_state,
            book=None,
            now_ms=NOW_MS,
            estimated_fees_micros=0,
        )

        assert verdict.decision is RiskDecision.REJECTED
        assert verdict.reason_code == "ORDER_RATE_LIMIT"

    def test_pending_capital_is_spent_capital(self):
        """Cash promised to resting orders cannot also fund a new position."""
        execution = _engine()
        # $190 of the $200 account committed to resting orders elsewhere.
        o = _order(ticker="ELSEWHERE", qty_qc=38_000, price_cc=5_000)
        execution.orders[o.order_id] = o
        loop = self._loop(execution)
        loop._refresh_from_execution(NOW_MS)

        verdict = loop.risk_engine.evaluate(
            OrderIntent(
                market_type=MarketType.EVENT,
                ticker="NEW",
                qty_qc=10_000,  # $50 of new exposure
                limit_price_cc=5_000,
                outcome_side=OutcomeSide.YES,
                expected_value_micros=1_000_000,
            ),
            loop.account_state,
            book=None,
            now_ms=NOW_MS,
            estimated_fees_micros=0,
        )

        assert verdict.approved_qty_qc < 10_000
