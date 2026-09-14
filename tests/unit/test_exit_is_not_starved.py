"""An exit belongs to the strategy that holds the position, and nobody else's orders
may hold it back.

The first fix for model_drift's stacked exits skipped an exit whenever *any* order
was working in the market. Review drove the real trading cycle with the real risk
engine, paper execution engine and a second strategy quoting the same market, and
found that second strategy's staggered orders kept the market-wide count above
zero for 540 cycles -- nine TTLs -- while a fillable exit waited. On the live path
an order stuck in CANCEL_REQUESTED would have held it until restart.

These run the same cycle and assert the exit completes where it did before the
guard existed, while model_drift still never stacks an exit on its own.
"""

from __future__ import annotations

from datetime import UTC, datetime

from kat.common.enums import (
    MarketType,
    OperatingMode,
    OrderState,
    OutcomeSide,
    StrategyState,
    TradingEnvironment,
)
from kat.common.timeutil import SimulatedClock
from kat.config.settings import EventRiskSettings
from kat.engine.trading import TradingLoop
from kat.execution.event_exec import EventExecutionEngine
from kat.execution.order import ManagedOrder
from kat.marketdata.book import OrderBook
from kat.marketdata.state import MarketState
from kat.risk.event_risk import EventAccountState, EventRiskEngine
from kat.strategies.event.model_drift import ModelDriftStrategy
from kat.strategies.event.news_reaction import NewsReactionStrategy

T = "MKT"
START = 1_760_000_000_000


class _NullRecorder:
    def build(self, verdict, **kw):
        return type("D", (), {"verdict": verdict, "decision_id": "d"})()

    def record(self, decision):
        return decision

    def attach_execution(self, decision, **kw):
        pass


def _book() -> OrderBook:
    # A 40-cent market, one cent wide, deep enough for both strategies' screens.
    return OrderBook.from_event_contract_levels(
        T, yes_levels=[(3950, 300_000), (3900, 300_000)],
        no_levels=[(10_000 - 4050, 300_000), (10_000 - 4100, 300_000)],
    )


def _build(monkeypatch, *, with_news: bool):
    clock = SimulatedClock(START)
    # Order timestamps follow the simulated clock, so the TTL means something.
    monkeypatch.setattr(
        "kat.execution.order.utcnow_aware",
        lambda: datetime.fromtimestamp(clock.now_ms() / 1000.0, tz=UTC),
    )
    state = MarketState(MarketType.EVENT)
    state.apply_snapshot(T, _book(), now_ms=clock.now_ms())
    engine = EventExecutionEngine(
        rest=None, market_state=state, clock=clock, environment=TradingEnvironment.PAPER
    )
    loop = TradingLoop(
        market_type=MarketType.EVENT, market_state=state,
        risk_engine=EventRiskEngine(EventRiskSettings()), execution=engine,
        recorder=_NullRecorder(), clock=clock,  # type: ignore[arg-type]
    )
    account = EventAccountState(balance_micros=200_000_000, updated_at_ms=clock.now_ms())
    # Hold five NO contracts, opened a minute ago -- inside the model's horizon.
    account.apply_fill(ticker=T, qty_qc=500, price_cc=4000, is_buy=False,
                       now_ms=clock.now_ms() - 60_000)
    account.peak_equity_micros = account.equity_micros
    loop.account_state = account
    loop.paper_fills = True
    loop.trading_permitted = True
    loop.operating_mode = OperatingMode.FULL_AUTONOMOUS
    loop.model_probabilities = {T: (0.024, "m", "v")}  # at the base rate: no view yet
    loop.model_target = "forward_move_30m_200cc"
    loop.model_base_rate = 0.024
    loop.features = {T: {"mid_change_5m_cc": 0.0}}
    drift = ModelDriftStrategy()
    drift.state = StrategyState.PAPER
    loop.add_strategy(drift)
    news = None
    if with_news:
        news = NewsReactionStrategy()
        news.state = StrategyState.PAPER
        loop.add_strategy(news)
    return clock, state, engine, loop, drift, news


async def _cycle(clock, state, loop, *, news_on: bool):
    clock.advance(1_000)
    state.apply_snapshot(T, _book(), now_ms=clock.now_ms())
    loop.account_state.updated_at_ms = clock.now_ms()
    loop.research = (
        {T: {"news": [{"article_id": "a", "ingested_ms": clock.now_ms() - 1_000,
                        "relevance": 1.0, "direction": "no",
                        "estimated_impact": 0.2, "confidence": 1.0}]}}
        if news_on else {}
    )
    return await loop.run_cycle()


def _position(loop) -> int:
    held = loop.account_state.positions.get(T)
    return held.position_qc if held is not None else 0


class TestAnotherStrategyCannotHoldTheExitBack:
    async def test_another_strategys_working_orders_do_not_hold_a_fillable_exit(self, monkeypatch):
        """Through the real cycle, with the market busy with somebody else's orders.

        Review built this with news_reaction's NO bids resting behind a paper
        simulator that could not fill a NO buy below fifty cents. With that fixed
        those bids fill, so the other strategy's working orders are placed here
        directly: what matters is that they do not count, however they got there.
        Under the market-wide guard this exit was skipped for as long as they
        stayed.
        """
        clock, state, engine, loop, drift, _ = _build(monkeypatch, with_news=False)
        await _cycle(clock, state, loop, news_on=False)
        assert _position(loop) == -500

        for _ in range(5):
            other = _order(strategy_id="event.news_reaction", state=OrderState.ACKNOWLEDGED,
                           side=OutcomeSide.NO, reduce_only=False)
            other.submitted_ts = datetime.fromtimestamp(clock.now_ms() / 1000.0, tz=UTC)
            engine.orders[other.order_id] = other
        assert loop._resting_qty_qc(T) > 0, "the market must be busy for this to mean anything"

        loop.model_probabilities = {T: (0.9, "m", "v")}  # the view reverses now
        await _cycle(clock, state, loop, news_on=False)

        assert _position(loop) == 0, drift.health().get("skips")


def _order(*, strategy_id: str, state: OrderState, side: OutcomeSide,
           reduce_only: bool, qty_qc: int = 500) -> ManagedOrder:
    order = ManagedOrder(
        market_type=MarketType.EVENT, ticker=T, requested_qty_qc=qty_qc,
        limit_price_cc=4_050, outcome_side=side, strategy_id=strategy_id,
        reduce_only=reduce_only,
    )
    order.state = state
    return order


class TestWhatCountsAsThisStrategysExit:
    def _loop_holding_no(self, monkeypatch, *orders: ManagedOrder) -> TradingLoop:
        _, _, engine, loop, _, _ = _build(monkeypatch, with_news=False)
        for order in orders:
            engine.orders[order.order_id] = order
        return loop

    def test_its_own_working_exit_counts(self, monkeypatch):
        loop = self._loop_holding_no(monkeypatch, _order(
            strategy_id="event.model_drift", state=OrderState.ACKNOWLEDGED,
            side=OutcomeSide.YES, reduce_only=True))
        assert loop._resting_exit_qty_qc("event.model_drift", T) == 500

    def test_another_strategys_order_does_not(self, monkeypatch):
        loop = self._loop_holding_no(monkeypatch, _order(
            strategy_id="event.news_reaction", state=OrderState.ACKNOWLEDGED,
            side=OutcomeSide.YES, reduce_only=True))
        assert loop._resting_exit_qty_qc("event.model_drift", T) == 0

    def test_an_order_on_the_side_that_adds_to_the_position_does_not(self, monkeypatch):
        loop = self._loop_holding_no(monkeypatch, _order(
            strategy_id="event.model_drift", state=OrderState.ACKNOWLEDGED,
            side=OutcomeSide.NO, reduce_only=True))
        assert loop._resting_exit_qty_qc("event.model_drift", T) == 0

    def test_an_order_whose_cancel_is_stuck_does_not(self, monkeypatch):
        """A 404 on cancel leaves CANCEL_REQUESTED indefinitely on the live path."""
        loop = self._loop_holding_no(monkeypatch, _order(
            strategy_id="event.model_drift", state=OrderState.CANCEL_REQUESTED,
            side=OutcomeSide.YES, reduce_only=True))
        assert loop._resting_exit_qty_qc("event.model_drift", T) == 0

    def test_a_plain_order_that_is_not_reduce_only_does_not(self, monkeypatch):
        loop = self._loop_holding_no(monkeypatch, _order(
            strategy_id="event.model_drift", state=OrderState.ACKNOWLEDGED,
            side=OutcomeSide.YES, reduce_only=False))
        assert loop._resting_exit_qty_qc("event.model_drift", T) == 0

    def test_the_snapshot_agrees_with_asking_directly(self, monkeypatch):
        loop = self._loop_holding_no(
            monkeypatch,
            _order(strategy_id="event.model_drift", state=OrderState.ACKNOWLEDGED,
                   side=OutcomeSide.YES, reduce_only=True, qty_qc=300),
            _order(strategy_id="event.model_drift", state=OrderState.PARTIALLY_FILLED,
                   side=OutcomeSide.YES, reduce_only=True, qty_qc=200),
            _order(strategy_id="event.news_reaction", state=OrderState.ACKNOWLEDGED,
                   side=OutcomeSide.YES, reduce_only=True, qty_qc=900),
        )
        direct = loop._resting_exit_qty_qc("event.model_drift", T)
        loop._resting_exits = loop._resting_exit_snapshot()
        assert loop._resting_exit_qty_qc("event.model_drift", T) == direct == 500
