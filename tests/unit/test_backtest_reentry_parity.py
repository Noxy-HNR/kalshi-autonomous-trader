"""A backtest refuses the same re-entries the engine refuses.

The backtester builds a fresh account state for every decision from its own
ledger. Unless the ledger keeps the exits, that state never carries one, the
re-entry cooldown never fires, and a backtest measures a looser platform than the
one that would trade -- the mismatch this codebase has been caught by before.
"""

from __future__ import annotations

from kat.common.enums import MarketType, OutcomeSide
from kat.config.settings import AppSettings, EventRiskSettings
from kat.risk.types import OrderIntent
from kat.simulation.engine import BacktestConfig, BacktestEngine, EventFeed, book_event
from kat.simulation.fills import SimFill, SimOrder
from kat.strategies.event.model_drift import ModelDriftStrategy

NOW = 1_760_000_000_000


def _engine() -> BacktestEngine:
    return BacktestEngine(
        # One book: the engine refuses an empty feed, and nothing here runs it.
        feed=EventFeed(
            [book_event("X", info_ts_ms=NOW, bids=[(3_950, 300_000)], asks=[(4_050, 300_000)], seq=1)],
            name="one book",
        ),
        strategies=[ModelDriftStrategy()],  # required; the feed is empty, so it never runs
        config=BacktestConfig(market_type=MarketType.EVENT, initial_capital_micros=200_000_000),
        settings=AppSettings(),
        event_risk_settings=EventRiskSettings(),
    )


def _fill(engine: BacktestEngine, *, order_id: str, is_buy: bool, reduce_only: bool) -> SimFill:
    order = SimOrder(
        order_id=order_id, ticker="X", market_type=MarketType.EVENT, is_buy=is_buy, qty_qc=100,
        limit_price_cc=4_000, decision_ms=NOW, arrival_ms=NOW, reduce_only=reduce_only,
    )
    engine.fills.orders[order_id] = order
    return SimFill(
        order_id=order_id, ticker="X", market_type=MarketType.EVENT, ts_ms=NOW, is_buy=is_buy,
        qty_qc=100, price_cc=4_000, notional_micros=400_000, fee_micros=0, is_taker=True,
        fee_price_cc=4_000, slippage_cc=0, latency_ms=0, decision_ms=NOW,
    )


def _open_then_exit(engine: BacktestEngine) -> None:
    """Buy YES, then sell it back with a reduce-only exit: the market ends flat."""
    engine._apply_fill(_fill(engine, order_id="in", is_buy=True, reduce_only=False))
    engine._apply_fill(_fill(engine, order_id="out", is_buy=False, reduce_only=True))


def _state(engine: BacktestEngine, now_ms: int):
    return engine.event_ledger.account_state(
        market=engine.market, fills=engine.fills, now_ms=now_ms, orders_this_minute=0
    )


class TestTheBacktestRemembersExits:
    def test_a_filled_exit_is_recorded_on_the_ledger(self):
        engine = _engine()
        _open_then_exit(engine)

        assert engine.event_ledger.recent_exits_ms == {"X": NOW}

    def test_a_filled_entry_is_not(self):
        engine = _engine()
        engine._apply_fill(_fill(engine, order_id="in", is_buy=True, reduce_only=False))

        assert engine.event_ledger.recent_exits_ms == {}

    def test_the_risk_engine_refuses_a_re_entry_in_the_backtest(self):
        """The same refusal the engine makes: flat, just exited, and trying to get back in."""
        engine = _engine()
        _open_then_exit(engine)
        state = _state(engine, NOW + 1_000)
        held = state.positions.get("X")
        assert held is None or held.position_qc == 0, "the exit must have left the market flat"

        intent = OrderIntent(
            market_type=MarketType.EVENT, ticker="X", qty_qc=100,
            limit_price_cc=5_000, outcome_side=OutcomeSide.YES, strategy_id="s",
        )
        verdict = engine.event_risk.evaluate(intent, state, now_ms=NOW + 1_000)
        assert verdict.reason_code == "REENTRY_COOLDOWN"
