"""A market that only ever held an order is not a position.

The trading loop gives a market with working orders an entry in the account's
positions, so its committed capital reaches the risk totals. Nothing removed the
entry once the orders were gone, so the dict grew with every market ever quoted
-- 600 left behind after ten watchlist rotations with nothing held -- and every
cycle walks it several times over, about 107ms a cycle at 20,000 entries.

Worse, the open-position cap tested whether the market had an entry rather than
whether the order would open a position, so a market that had merely held a quote
was exempt from the cap for good. Review reproduced it: an account at the cap
rejected a new market and approved the same order on a once-quoted one.
"""

from __future__ import annotations

from datetime import UTC, datetime

from kat.common.enums import MarketType, OrderState, OutcomeSide, TradingEnvironment
from kat.common.timeutil import SimulatedClock
from kat.config.settings import EventRiskSettings
from kat.engine.trading import TradingLoop
from kat.execution.base import _TERMINAL_RETENTION_MS
from kat.execution.event_exec import EventExecutionEngine
from kat.execution.order import ManagedOrder
from kat.marketdata.state import MarketState
from kat.risk.event_risk import EventAccountState, EventPositionState, EventRiskEngine
from kat.risk.types import OrderIntent

NOW_MS = 10_000_000_000


def _at(age_ms: float) -> datetime:
    return datetime.fromtimestamp((NOW_MS - age_ms) / 1000.0, tz=UTC)


def _loop() -> tuple[TradingLoop, EventExecutionEngine]:
    engine = EventExecutionEngine(
        rest=None, market_state=MarketState(MarketType.EVENT),
        clock=SimulatedClock(NOW_MS), environment=TradingEnvironment.PAPER,
    )
    loop = TradingLoop(
        market_type=MarketType.EVENT, market_state=MarketState(MarketType.EVENT),
        risk_engine=EventRiskEngine(EventRiskSettings()), execution=engine,
        recorder=None,  # type: ignore[arg-type]
        clock=SimulatedClock(NOW_MS),
    )
    loop.paper_fills = True
    loop.account_state = EventAccountState(balance_micros=200_000_000, updated_at_ms=NOW_MS)
    return loop, engine


def _rest(engine: EventExecutionEngine, ticker: str) -> ManagedOrder:
    order = ManagedOrder(
        market_type=MarketType.EVENT, ticker=ticker, requested_qty_qc=100,
        limit_price_cc=5_000, outcome_side=OutcomeSide.YES, strategy_id="s",
    )
    order.state = OrderState.ACKNOWLEDGED
    order.submitted_ts = _at(_TERMINAL_RETENTION_MS * 3)
    engine.orders[order.order_id] = order
    return order


def _quote_then_forget(loop: TradingLoop, engine: EventExecutionEngine, tickers: list[str]) -> None:
    orders = [_rest(engine, t) for t in tickers]
    loop._refresh_from_execution(NOW_MS)
    for order in orders:
        order.state = OrderState.CANCELLED
        order.terminal_ts = _at(_TERMINAL_RETENTION_MS * 2)
    engine.prune_terminal_orders(now_ms=NOW_MS)
    loop._refresh_from_execution(NOW_MS)


class TestEntriesDoNotOutliveTheirOrders:
    def test_ten_watchlist_rotations_leave_nothing_behind(self):
        loop, engine = _loop()
        for rotation in range(10):
            _quote_then_forget(loop, engine, [f"MKT-{rotation}-{i}" for i in range(60)])

        assert engine.working_orders() == []
        assert loop.account_state.positions == {}

    def test_a_market_with_a_working_order_keeps_its_entry(self):
        """Its commitment still has to reach the risk totals."""
        loop, engine = _loop()
        _rest(engine, "BUSY")
        loop._refresh_from_execution(NOW_MS)

        entry = loop.account_state.positions["BUSY"]
        assert entry.pending_exposure_micros > 0

    def test_a_real_position_is_never_removed(self):
        loop, _ = _loop()
        loop.account_state.positions["HELD"] = EventPositionState(
            ticker="HELD", position_qc=100, avg_entry_cc=4_000, exposure_micros=400_000,
        )
        loop._refresh_from_execution(NOW_MS)

        assert "HELD" in loop.account_state.positions


def _intent(ticker: str) -> OrderIntent:
    return OrderIntent(
        market_type=MarketType.EVENT, ticker=ticker, qty_qc=100,
        limit_price_cc=5_000, outcome_side=OutcomeSide.YES, strategy_id="s",
    )


class TestTheOpenPositionCapCountsPositionsNotEntries:
    def _at_the_cap(self) -> tuple[EventRiskEngine, EventAccountState]:
        risk = EventRiskEngine(EventRiskSettings())
        state = EventAccountState(balance_micros=200_000_000, updated_at_ms=NOW_MS)
        for i in range(risk.settings.max_open_positions):
            state.positions[f"HELD-{i}"] = EventPositionState(
                ticker=f"HELD-{i}", position_qc=100, avg_entry_cc=100, exposure_micros=10_000,
            )
        return risk, state

    def test_a_new_market_is_refused_at_the_cap(self):
        risk, state = self._at_the_cap()
        verdict = risk.evaluate(_intent("NEVER-QUOTED"), state, now_ms=NOW_MS)
        assert verdict.reason_code == "MAX_OPEN_POSITIONS"

    def test_an_empty_entry_is_not_an_exemption(self):
        """The loophole: present in the dict, holding nothing."""
        risk, state = self._at_the_cap()
        state.positions["ONCE-QUOTED"] = EventPositionState(ticker="ONCE-QUOTED")

        verdict = risk.evaluate(_intent("ONCE-QUOTED"), state, now_ms=NOW_MS)
        assert verdict.reason_code == "MAX_OPEN_POSITIONS"

    def test_adding_to_a_position_already_held_is_not_a_new_one(self):
        """The cap limits how many positions are open, not trading in open ones."""
        risk, state = self._at_the_cap()
        verdict = risk.evaluate(_intent("HELD-0"), state, now_ms=NOW_MS)
        assert verdict.reason_code != "MAX_OPEN_POSITIONS"
