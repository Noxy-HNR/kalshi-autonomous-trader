"""A resting order that nobody cancels is a standing offer to be picked off.

The platform submits good-till-cancelled limit orders and nothing ever cancelled
them. Two consequences, both bad.

A resting order holds its market's position headroom for as long as it rests,
because pending exposure counts against the concentration cap exactly as a
holding does. Observed on the running engine: ``POSITION_LIMIT`` rejections on
markets holding no position at all, while five sixths of the account's cash sat
unused. Every market the platform touched became permanently unavailable to it.

The second is worse than idleness. A stale limit does not merely fail to fill --
it fills precisely when the market crosses it, which is when the price has moved
against the opinion that placed it. Left resting, a cancelled idea becomes free
optionality for whoever is on the other side.

A strategy re-prices every cycle, so an order still resting after the TTL has
been superseded by its own author.
"""

from __future__ import annotations

from datetime import UTC, datetime

from kat.common.enums import MarketType, OrderState, OutcomeSide, TradingEnvironment
from kat.common.timeutil import SimulatedClock
from kat.execution.event_exec import EventExecutionEngine
from kat.execution.order import ManagedOrder
from kat.marketdata.state import MarketState

NOW_MS = 10_000_000_000
TTL_MS = 60_000


def _engine() -> EventExecutionEngine:
    return EventExecutionEngine(
        rest=None,
        market_state=MarketState(MarketType.EVENT),
        clock=SimulatedClock(NOW_MS),
        environment=TradingEnvironment.PAPER,
    )


def _resting(engine, *, ticker: str = "X", age_s: float, qty_qc: int = 461):
    order = ManagedOrder(
        market_type=MarketType.EVENT,
        ticker=ticker,
        requested_qty_qc=qty_qc,
        limit_price_cc=8_500,
        outcome_side=OutcomeSide.YES,
        strategy_id="s",
    )
    order.state = OrderState.ACKNOWLEDGED
    order.submitted_ts = datetime.fromtimestamp((NOW_MS / 1000.0) - age_s, tz=UTC)
    engine.orders[order.order_id] = order
    return order


class TestExpiry:
    async def test_an_order_past_the_ttl_is_cancelled(self):
        engine = _engine()
        order = _resting(engine, age_s=120)

        cancelled = await engine.expire_stale_orders(now_ms=NOW_MS, ttl_ms=TTL_MS)

        assert cancelled == 1
        assert order.state is OrderState.CANCELLED

    async def test_a_young_order_is_left_alone(self):
        engine = _engine()
        order = _resting(engine, age_s=10)

        assert await engine.expire_stale_orders(now_ms=NOW_MS, ttl_ms=TTL_MS) == 0
        assert order.state is OrderState.ACKNOWLEDGED

    async def test_only_the_stale_ones_go(self):
        engine = _engine()
        young = _resting(engine, ticker="YOUNG", age_s=5)
        old = _resting(engine, ticker="OLD", age_s=600)

        assert await engine.expire_stale_orders(now_ms=NOW_MS, ttl_ms=TTL_MS) == 1
        assert young.state is OrderState.ACKNOWLEDGED
        assert old.state is OrderState.CANCELLED

    async def test_a_filled_order_is_not_touched(self):
        engine = _engine()
        order = _resting(engine, age_s=600)
        order.state = OrderState.FILLED
        order.filled_qty_qc = order.requested_qty_qc

        assert await engine.expire_stale_orders(now_ms=NOW_MS, ttl_ms=TTL_MS) == 0
        assert order.state is OrderState.FILLED

    async def test_expiry_is_idempotent(self):
        engine = _engine()
        _resting(engine, age_s=600)

        assert await engine.expire_stale_orders(now_ms=NOW_MS, ttl_ms=TTL_MS) == 1
        assert await engine.expire_stale_orders(now_ms=NOW_MS, ttl_ms=TTL_MS) == 0

    async def test_nothing_resting_costs_nothing(self):
        assert await _engine().expire_stale_orders(now_ms=NOW_MS, ttl_ms=TTL_MS) == 0


class TestExpiryReleasesTheMarket:
    async def test_cancelling_frees_the_committed_capital(self):
        """The headroom that was never coming back."""
        engine = _engine()
        for _ in range(18):
            _resting(engine, age_s=600)

        committed = engine.working_exposure_micros()
        assert committed["X"] == 18 * 461 * 8_500

        await engine.expire_stale_orders(now_ms=NOW_MS, ttl_ms=TTL_MS)

        assert engine.working_exposure_micros() == {}

    async def test_a_market_blocked_only_by_resting_orders_becomes_tradeable(self):
        """End to end: pending exposure was the whole reason it was refused."""
        from kat.config.settings import EventRiskSettings
        from kat.engine.trading import TradingLoop
        from kat.risk.event_risk import EventAccountState, EventRiskEngine
        from kat.risk.types import OrderIntent

        engine = _engine()
        loop = TradingLoop(
            market_type=MarketType.EVENT,
            market_state=MarketState(MarketType.EVENT),
            risk_engine=EventRiskEngine(EventRiskSettings()),
            execution=engine,
            recorder=None,  # type: ignore[arg-type]
            clock=SimulatedClock(NOW_MS),
        )
        loop.account_state = EventAccountState(
            balance_micros=200_000_000, updated_at_ms=NOW_MS
        )
        for _ in range(18):
            _resting(engine, age_s=600)

        intent = OrderIntent(
            market_type=MarketType.EVENT,
            ticker="X",
            qty_qc=100,
            limit_price_cc=8_500,
            outcome_side=OutcomeSide.YES,
            expected_value_micros=1_000_000,
        )

        loop._refresh_from_execution(NOW_MS)
        blocked = loop.risk_engine.evaluate(
            intent, loop.account_state, book=None, now_ms=NOW_MS, estimated_fees_micros=0
        )
        assert blocked.reason_code == "POSITION_LIMIT"

        await engine.expire_stale_orders(now_ms=NOW_MS, ttl_ms=TTL_MS)
        loop._refresh_from_execution(NOW_MS)
        freed = loop.risk_engine.evaluate(
            intent, loop.account_state, book=None, now_ms=NOW_MS, estimated_fees_micros=0
        )
        assert freed.approved
