"""Two strategies with opposite views must not trade a market back and forth.

Reproduced from the running engine on 2026-09-14, through the real trading cycle,
the real risk engine and the paper execution engine: news_reaction reads bullish
news and buys YES; model_drift's forecast is bearish, so it exits that position
in the next cycle; news_reaction buys it back. Before the re-entry cooldown the
pair did this once a second and paid the spread every time.

The control disables the cooldown and shows the harness does see the churn, so
the assertion that it is gone is not passing by accident.
"""

from __future__ import annotations

from datetime import UTC, datetime

from kat.common.enums import MarketType, OperatingMode, StrategyState, TradingEnvironment
from kat.common.timeutil import SimulatedClock
from kat.config.settings import EventRiskSettings
from kat.engine.trading import TradingLoop
from kat.execution.event_exec import EventExecutionEngine
from kat.marketdata.book import OrderBook
from kat.marketdata.state import MarketState
from kat.risk.event_risk import EventAccountState, EventRiskEngine
from kat.strategies.event.model_drift import ModelDriftStrategy
from kat.strategies.event.news_reaction import NewsReactionStrategy

T = "MKT"
START = 1_760_000_000_000
CYCLES = 30


class _NullRecorder:
    def build(self, verdict, **kw):
        return type("D", (), {"verdict": verdict, "decision_id": "d"})()

    def record(self, decision):
        return decision

    def attach_execution(self, decision, **kw):
        pass


def _book() -> OrderBook:
    return OrderBook.from_event_contract_levels(
        T, yes_levels=[(3950, 300_000), (3900, 300_000)],
        no_levels=[(10_000 - 4050, 300_000), (10_000 - 4100, 300_000)],
    )


async def _run(monkeypatch) -> tuple[EventExecutionEngine, int]:
    clock = SimulatedClock(START)
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
    account.peak_equity_micros = account.equity_micros
    loop.account_state = account
    loop.paper_fills = True
    loop.trading_permitted = True
    loop.operating_mode = OperatingMode.FULL_AUTONOMOUS
    loop.model_probabilities = {T: (0.001, "m", "v")}  # far below the base rate: bearish
    loop.model_target = "forward_move_30m_200cc"
    loop.model_base_rate = 0.024
    loop.features = {T: {"mid_change_5m_cc": 0.0}}
    for strategy in (ModelDriftStrategy(), NewsReactionStrategy()):
        strategy.state = StrategyState.PAPER
        loop.add_strategy(strategy)

    for _ in range(CYCLES):
        clock.advance(1_000)
        state.apply_snapshot(T, _book(), now_ms=clock.now_ms())
        account.updated_at_ms = clock.now_ms()
        loop.research = {T: {"news": [{
            "article_id": "a", "ingested_ms": clock.now_ms() - 1_000, "relevance": 1.0,
            "direction": "yes", "estimated_impact": 0.2, "confidence": 1.0,
        }]}}
        await loop.run_cycle()

    entries = sum(
        1 for o in engine.orders.values()
        if o.strategy_id == "event.news_reaction" and o.filled_qty_qc > 0
    )
    return engine, entries


def _exits(engine: EventExecutionEngine) -> int:
    return sum(
        1 for o in engine.orders.values()
        if o.strategy_id == "event.model_drift" and o.reduce_only and o.filled_qty_qc > 0
    )


class TestOppositeViewsDoNotChurn:
    async def test_without_the_cooldown_the_pair_trades_back_and_forth(self, monkeypatch):
        """The control: the harness reproduces what the engine did."""
        monkeypatch.setattr(EventAccountState, "exited_within", lambda self, *a, **k: None)
        engine, entries = await _run(monkeypatch)

        assert entries >= 5, f"control did not reproduce the churn: {entries} entries"
        assert _exits(engine) >= 5

    async def test_with_the_cooldown_one_round_trip_is_the_end_of_it(self, monkeypatch):
        engine, entries = await _run(monkeypatch)

        assert entries >= 1, "news_reaction never entered, so this proves nothing"
        assert _exits(engine) == 1
        assert entries == 1
