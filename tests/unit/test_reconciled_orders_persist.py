"""What reconciliation learns about an order has to reach the database.

``resolve_unknown`` asks the exchange what happened to an order whose outcome we
never saw, and moves it to that state in memory. Nothing on that path wrote the
result, so the order's row kept whatever was last persisted -- typically the
ambiguous state it was in before reconciliation settled it -- and the audit trail
disagreed with memory from then on. Review found it while checking whether
forgetting finished orders could lose anything; it could not, but the premise
that a finished order is already in the database was false for this path.
"""

from __future__ import annotations

from types import SimpleNamespace

from kat.common.enums import MarketType, OrderState, OutcomeSide, TradingEnvironment
from kat.common.timeutil import SimulatedClock
from kat.execution.event_exec import EventExecutionEngine
from kat.execution.order import ManagedOrder
from kat.marketdata.state import MarketState

NOW = 1_760_000_000_000


class _Writer:
    def __init__(self) -> None:
        self.rows: list[object] = []

    def enqueue(self, obj, *, critical: bool = False) -> bool:
        self.rows.append(obj)
        return True


class _Rest:
    def __init__(self, listing: list[object]) -> None:
        self.listing = listing

    async def get_orders(self, *, ticker: str | None = None, status: str | None = None):
        return self.listing


def _engine(listing: list[object]) -> tuple[EventExecutionEngine, _Writer]:
    engine = EventExecutionEngine(
        rest=_Rest(listing), market_state=MarketState(MarketType.EVENT),
        clock=SimulatedClock(NOW), environment=TradingEnvironment.DEMO,
    )
    writer = _Writer()
    engine.writer = writer  # type: ignore[assignment]
    return engine, writer


def _submitted() -> ManagedOrder:
    order = ManagedOrder(
        market_type=MarketType.EVENT, ticker="X", requested_qty_qc=100,
        limit_price_cc=5_000, outcome_side=OutcomeSide.YES, strategy_id="s",
    )
    order.state = OrderState.SUBMITTED
    return order


def _order_rows(writer: _Writer) -> list[object]:
    return [row for row in writer.rows if getattr(row, "__tablename__", "") == "orders"]


class TestReconciliationIsWrittenDown:
    async def test_an_order_the_exchange_never_had_is_persisted_as_rejected(self):
        engine, writer = _engine(listing=[])
        order = _submitted()

        assert await engine.resolve_unknown(order) is OrderState.REJECTED

        rows = _order_rows(writer)
        assert len(rows) == 1
        assert rows[0].state == "rejected"
        assert rows[0].order_id == order.order_id

    async def test_an_order_the_exchange_has_is_persisted_in_its_state(self):
        order = _submitted()
        match = SimpleNamespace(
            client_order_id=order.client_order_id, order_id="exch-1", filled_qc=0,
            yes_price_cc=5_000, no_price_cc=5_000,
            to_internal_state=lambda: OrderState.ACKNOWLEDGED,
        )
        engine, writer = _engine(listing=[match])

        assert await engine.resolve_unknown(order) is OrderState.ACKNOWLEDGED

        rows = _order_rows(writer)
        assert len(rows) == 1
        assert rows[0].state == "acknowledged"
        assert rows[0].exchange_order_id == "exch-1"

    async def test_an_unreachable_exchange_writes_nothing(self):
        """No answer is not a state; the order stays as it was and so does its row."""
        class _Down:
            async def get_orders(self, **_):
                raise RuntimeError("exchange unreachable")

        engine = EventExecutionEngine(
            rest=_Down(), market_state=MarketState(MarketType.EVENT),
            clock=SimulatedClock(NOW), environment=TradingEnvironment.DEMO,
        )
        writer = _Writer()
        engine.writer = writer  # type: ignore[assignment]
        order = _submitted()

        assert await engine.resolve_unknown(order) is OrderState.SUBMITTED
        assert _order_rows(writer) == []
