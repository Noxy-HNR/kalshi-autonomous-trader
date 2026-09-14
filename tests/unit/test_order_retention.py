"""A finished order belongs in the database, not in the trading loop's memory.

The execution engine kept every order it ever placed, and several scans over that
dict run every trading cycle -- one of them once per strategy per market, about
five hundred times a cycle. On 2026-09-13 the dict held 103,455 orders, 103,423
of them cancelled, and the cycle grew in step with it: 23ms at 730 orders, half
a second at 30,000, 3.6 seconds at 103,000. The loop stopped yielding, the
database writer starved, and trading flapped in and out of DEGRADED 189 times.
The same growth had produced the 1.7-second cycles two days earlier.

What makes forgetting safe is what it must never forget: anything still able to
consume exposure, anything reconciliation has yet to resolve, and anything the
runaway-order limit still has to count.
"""

from __future__ import annotations

from datetime import UTC, datetime

from kat.common.enums import MarketType, OrderState, OutcomeSide, TradingEnvironment
from kat.common.timeutil import SimulatedClock
from kat.config.settings import EventRiskSettings
from kat.engine.trading import TradingLoop
from kat.execution.base import _RATE_WINDOW_MS, _TERMINAL_RETENTION_MS
from kat.execution.event_exec import EventExecutionEngine
from kat.execution.order import ManagedOrder
from kat.marketdata.state import MarketState
from kat.risk.event_risk import EventRiskEngine

NOW_MS = 10_000_000_000
TICKER = "X"


def _engine() -> EventExecutionEngine:
    return EventExecutionEngine(
        rest=None,
        market_state=MarketState(MarketType.EVENT),
        clock=SimulatedClock(NOW_MS),
        environment=TradingEnvironment.PAPER,
    )


def _at(age_ms: float) -> datetime:
    return datetime.fromtimestamp((NOW_MS - age_ms) / 1000.0, tz=UTC)


def _order(
    engine: EventExecutionEngine,
    *,
    state: OrderState,
    submitted_age_ms: float | None,
    terminal_age_ms: float | None,
    ticker: str = TICKER,
    qty_qc: int = 100,
) -> ManagedOrder:
    order = ManagedOrder(
        market_type=MarketType.EVENT,
        ticker=ticker,
        requested_qty_qc=qty_qc,
        limit_price_cc=5_000,
        outcome_side=OutcomeSide.YES,
        strategy_id="s",
    )
    order.state = state
    order.submitted_ts = None if submitted_age_ms is None else _at(submitted_age_ms)
    order.terminal_ts = None if terminal_age_ms is None else _at(terminal_age_ms)
    engine.orders[order.order_id] = order
    engine._by_client_id[order.client_order_id] = order
    return order


LONG_AGO = _TERMINAL_RETENTION_MS + 60_000


class TestWhatIsForgotten:
    def test_an_order_finished_long_ago_is_forgotten(self):
        engine = _engine()
        order = _order(
            engine, state=OrderState.CANCELLED,
            submitted_age_ms=LONG_AGO + 60_000, terminal_age_ms=LONG_AGO,
        )

        assert engine.prune_terminal_orders(now_ms=NOW_MS) == 1
        assert order.order_id not in engine.orders
        assert engine.find_by_client_order_id(order.client_order_id) is None

    def test_every_terminal_state_is_a_candidate(self):
        engine = _engine()
        for state in (OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED,
                      OrderState.RISK_REJECTED, OrderState.EXPIRED):
            _order(engine, state=state, submitted_age_ms=LONG_AGO + 1_000,
                   terminal_age_ms=LONG_AGO)

        assert engine.prune_terminal_orders(now_ms=NOW_MS) == 5
        assert engine.orders == {}

    def test_the_stats_say_memory_is_bounded(self):
        engine = _engine()
        for _ in range(3):
            _order(engine, state=OrderState.CANCELLED,
                   submitted_age_ms=LONG_AGO + 1_000, terminal_age_ms=LONG_AGO)
        engine.prune_terminal_orders(now_ms=NOW_MS)

        stats = engine.stats()
        assert stats["pruned_orders"] == 3
        assert stats["tracked_orders"] == 0


class TestWhatIsNeverForgotten:
    def test_a_working_order_is_kept_however_old(self):
        """It can still be filled, so its exposure still counts."""
        engine = _engine()
        for state in (OrderState.SUBMITTED, OrderState.ACKNOWLEDGED,
                      OrderState.PARTIALLY_FILLED, OrderState.CANCEL_REQUESTED):
            _order(engine, state=state, submitted_age_ms=86_400_000, terminal_age_ms=None)

        assert engine.prune_terminal_orders(now_ms=NOW_MS) == 0
        assert len(engine.orders) == 4

    def test_an_unknown_order_is_kept_for_reconciliation(self):
        """Reconciliation resolves UNKNOWN orders from this dict and nowhere else."""
        engine = _engine()
        _order(engine, state=OrderState.UNKNOWN, submitted_age_ms=86_400_000,
               terminal_age_ms=None)

        assert engine.prune_terminal_orders(now_ms=NOW_MS) == 0

    def test_a_recently_finished_order_is_kept(self):
        """An exchange message can arrive after we believed an order was done."""
        engine = _engine()
        _order(engine, state=OrderState.CANCELLED,
               submitted_age_ms=_TERMINAL_RETENTION_MS, terminal_age_ms=60_000)

        assert engine.prune_terminal_orders(now_ms=NOW_MS) == 0

    def test_a_terminal_order_with_no_finish_time_is_kept(self):
        """When it finished is unknown, and an unknown age is not an old one."""
        engine = _engine()
        _order(engine, state=OrderState.CANCELLED, submitted_age_ms=LONG_AGO,
               terminal_age_ms=None)

        assert engine.prune_terminal_orders(now_ms=NOW_MS) == 0


class TestTheRateLimitStillSeesEverythingItCounts:
    def test_an_order_submitted_inside_the_window_is_kept_whatever_its_finish_time(self):
        """Forgetting it would loosen the runaway-order limit without a word."""
        engine = _engine()
        _order(engine, state=OrderState.CANCELLED,
               submitted_age_ms=_RATE_WINDOW_MS // 2, terminal_age_ms=LONG_AGO)

        assert engine.prune_terminal_orders(now_ms=NOW_MS) == 0

    def test_pruning_never_changes_the_count(self):
        engine = _engine()
        for age in (5_000, 30_000, 59_000):
            _order(engine, state=OrderState.CANCELLED, submitted_age_ms=age,
                   terminal_age_ms=age - 1_000)
        for _ in range(10):
            _order(engine, state=OrderState.CANCELLED,
                   submitted_age_ms=LONG_AGO + 1_000, terminal_age_ms=LONG_AGO)

        before = engine.orders_in_last_minute(now_ms=NOW_MS)
        engine.prune_terminal_orders(now_ms=NOW_MS)
        assert engine.orders_in_last_minute(now_ms=NOW_MS) == before == 3

    def test_retention_is_never_shorter_than_the_rate_window(self):
        assert _TERMINAL_RETENTION_MS >= _RATE_WINDOW_MS


class TestAtTheScaleThatBrokeIt:
    def test_twenty_thousand_dead_orders_leave_only_the_working_one(self):
        engine = _engine()
        for _ in range(20_000):
            _order(engine, state=OrderState.CANCELLED,
                   submitted_age_ms=LONG_AGO + 1_000, terminal_age_ms=LONG_AGO)
        live = _order(engine, state=OrderState.ACKNOWLEDGED, submitted_age_ms=5_000,
                      terminal_age_ms=None)

        assert engine.prune_terminal_orders(now_ms=NOW_MS) == 20_000
        assert list(engine.orders) == [live.order_id]
        assert engine.working_orders(ticker=TICKER) == [live]


def _loop(engine: EventExecutionEngine) -> TradingLoop:
    return TradingLoop(
        market_type=MarketType.EVENT,
        market_state=MarketState(MarketType.EVENT),
        risk_engine=EventRiskEngine(EventRiskSettings()),
        execution=engine,
        recorder=None,  # type: ignore[arg-type]
        clock=SimulatedClock(NOW_MS),
    )


class TestTheSweepAsksOnceNotFiveHundredTimes:
    def test_the_snapshot_matches_asking_market_by_market(self):
        engine = _engine()
        _order(engine, state=OrderState.ACKNOWLEDGED, submitted_age_ms=1_000,
               terminal_age_ms=None, ticker="A", qty_qc=300)
        _order(engine, state=OrderState.PARTIALLY_FILLED, submitted_age_ms=1_000,
               terminal_age_ms=None, ticker="A", qty_qc=200)
        _order(engine, state=OrderState.ACKNOWLEDGED, submitted_age_ms=1_000,
               terminal_age_ms=None, ticker="B", qty_qc=100)
        _order(engine, state=OrderState.CANCELLED, submitted_age_ms=1_000,
               terminal_age_ms=500, ticker="B", qty_qc=900)
        loop = _loop(engine)

        per_market = {t: loop._resting_qty_qc(t) for t in ("A", "B", "C")}
        loop._resting_by_ticker = loop._resting_snapshot()
        from_snapshot = {t: loop._resting_qty_qc(t) for t in ("A", "B", "C")}

        assert from_snapshot == per_market == {"A": 500, "B": 100, "C": 0}

    def test_inside_a_sweep_the_engine_is_not_asked_again(self):
        engine = _engine()
        loop = _loop(engine)
        loop._resting_by_ticker = {"A": 700}

        def refuse(**_):
            raise AssertionError("scanned the order book inside a sweep")

        engine.working_orders = refuse  # type: ignore[method-assign]
        assert loop._resting_qty_qc("A") == 700
        assert loop._resting_qty_qc("Z") == 0

    def test_an_engine_that_cannot_answer_falls_back_rather_than_reading_as_empty(self):
        """A failed snapshot must not report every market as having nothing working."""
        engine = _engine()
        loop = _loop(engine)

        def broken(**_):
            raise RuntimeError("unavailable")

        engine.working_orders = broken  # type: ignore[method-assign]
        assert loop._resting_snapshot() is None


class TestEveryCycleForgets:
    async def test_a_cycle_prunes_before_anything_scans(self):
        """Wired where it runs every cycle, not merely available to be called."""
        from kat.risk.event_risk import EventAccountState

        engine = _engine()
        dead = _order(engine, state=OrderState.CANCELLED,
                      submitted_age_ms=LONG_AGO + 1_000, terminal_age_ms=LONG_AGO)
        live = _order(engine, state=OrderState.ACKNOWLEDGED, submitted_age_ms=5_000,
                      terminal_age_ms=None)
        loop = _loop(engine)
        loop.trading_permitted = True
        loop.account_state = EventAccountState(balance_micros=200_000_000, updated_at_ms=NOW_MS)

        await loop.run_cycle()

        assert dead.order_id not in engine.orders
        assert live.order_id in engine.orders

    async def test_the_snapshot_does_not_outlive_the_sweep(self):
        """Outside a sweep a context must ask the engine, not read a stale map."""
        from kat.risk.event_risk import EventAccountState

        engine = _engine()
        loop = _loop(engine)
        loop.trading_permitted = True
        loop.account_state = EventAccountState(balance_micros=200_000_000, updated_at_ms=NOW_MS)

        await loop.run_cycle()
        assert loop._resting_by_ticker is None
