"""The whole chain, from a market tick to an order leaving the process.

Every piece of this had unit tests. The chain did not, and in five hours against
the live demo exchange it never once executed: the strategies declined every
market for reasons that were each individually correct, so nothing proved that a
market which *should* trade actually produces an order.

That gap matters more than it sounds. A platform whose order path has never run
is a platform whose order path is untested no matter how good its parts are, and
"it declined everything" is indistinguishable from "it is broken" until one case
goes all the way through.

So this builds the market the demo exchange does not have -- tight spread, deep
book, a mid that actually moves -- and asserts an order comes out the far end,
having passed through the real strategy, the real risk engine and the real
execution engine. Only the exchange itself is a stub.
"""

from __future__ import annotations

from typing import Any

import pytest

from kat.common.enums import MarketType, OperatingMode, OrderState, StrategyState
from kat.common.timeutil import SimulatedClock
from kat.config.settings import EventRiskSettings
from kat.engine.trading import TradingLoop
from kat.execution.order import ManagedOrder
from kat.marketdata.book import OrderBook
from kat.marketdata.state import MarketState
from kat.risk.event_risk import EventAccountState, EventRiskEngine
from kat.risk.types import RiskVerdict
from kat.strategies.event.mean_reversion import MeanReversionStrategy

TICKER = "TEST-EVENT-MKT"
BAR_MS = 60_000


class _RecordingExecution:
    """Stands in for the exchange, and only for the exchange.

    It accepts a ``RiskVerdict`` exactly as the real engines do, so anything that
    reaches it has already passed every gate ahead of it.
    """

    market_type = MarketType.EVENT

    def __init__(self) -> None:
        self.submitted: list[RiskVerdict] = []
        self.orders: list[ManagedOrder] = []

    async def submit(self, verdict: RiskVerdict) -> Any:
        assert verdict.approved, "an unapproved verdict must never reach execution"
        assert verdict.approved_qty_qc > 0
        self.submitted.append(verdict)
        # A real ManagedOrder, not a stand-in with the attributes this test
        # happens to need. The loop reads several of its fields after submission,
        # and a duck-typed stub would let a change to those fields pass here and
        # fail in production.
        order = ManagedOrder(
            market_type=verdict.intent.market_type,
            ticker=verdict.intent.ticker,
            requested_qty_qc=verdict.approved_qty_qc,
            limit_price_cc=verdict.intent.limit_price_cc,
            outcome_side=verdict.intent.outcome_side,
            book_side=verdict.intent.book_side,
            strategy_id=verdict.intent.strategy_id,
        )
        order.state = OrderState.SUBMITTED
        self.orders.append(order)
        return order

    def working_orders(self) -> list[Any]:
        return []

    async def expire_stale_orders(self, *, now_ms: int, ttl_ms: int) -> int:
        """Nothing rests here: every order this double accepts fills at once."""
        return 0

    # The loop refreshes the account state from these every cycle, because the
    # risk engine's order-rate limit and pending-capital accounting have no other
    # source. Answered from the real orders above rather than hard-coded: a stub
    # returning a constant zero is precisely the bug those two fields had.
    def orders_in_last_minute(self, *, now_ms: int) -> int:
        cutoff = now_ms - 60_000
        return sum(
            1
            for o in self.orders
            if o.submitted_ts is not None and o.submitted_ts.timestamp() * 1000.0 >= cutoff
        )

    def working_exposure_micros(self) -> dict[str, int]:
        committed: dict[str, int] = {}
        for o in self.orders:
            if o.exposure_qty_qc > 0 and o.limit_price_cc:
                committed[o.ticker] = (
                    committed.get(o.ticker, 0) + o.exposure_qty_qc * o.limit_price_cc
                )
        return committed


class _NullRecorder:
    """The decision record, reduced to what the loop calls."""

    def __init__(self) -> None:
        self.decisions: list[Any] = []
        self.attached: list[tuple[Any, Any]] = []

    def build(self, verdict: RiskVerdict, **kwargs: Any) -> Any:
        return type("Decision", (), {"verdict": verdict, "decision_id": "d1", **kwargs})()

    def record(self, decision: Any) -> Any:
        self.decisions.append(decision)
        return decision

    def attach_execution(self, decision: Any, **fields: Any) -> None:
        self.attached.append((decision, fields))


def _book(mid_cc: int) -> OrderBook:
    """A one-cent-wide, deep book centred on ``mid_cc``.

    The demo exchange offers 10-97 cent spreads on static books, which is why
    nothing trades there. This is what a liquid market looks like.
    """
    yes = mid_cc - 50
    no = 10_000 - (mid_cc + 50)
    return OrderBook.from_event_contract_levels(
        TICKER,
        yes_levels=[(yes, 50_000), (yes - 100, 50_000)],
        no_levels=[(no, 50_000), (no - 100, 50_000)],
    )


def _dislocated() -> dict[str, float]:
    """A market that has moved a long way and then stopped moving.

    Both halves matter. Mean reversion needs the dislocation to fade, and it
    needs the move to have *finished*: a large five-minute drop that is still
    dropping this minute is a falling knife, and the strategy declines it with
    ``move_still_extending``. Writing this test taught me that, which is the
    behaviour working.

    The price the path lands on matters too, and for a reason worth recording.
    Kalshi's fee is proportional to ``price * (1 - price)``, so it peaks at 50c
    and falls away toward either end. The reversion cap bounds the edge at four
    cents wherever the market is, so a 50c market pays the most fee for the
    least edge and cannot clear the floor -- while the same dislocation at 12c
    clears it comfortably. That is the strategy being right about fees, not a
    threshold that needed loosening.
    """
    return {
        "realized_vol_5m_cc": 200.0,
        "mid_change_5m_cc": -1500.0,  # a fifteen-cent drop over five minutes
        "mid_change_1m_cc": -30.0,    # ...which has now flattened out
        "volume_zscore_5m": 0.0,
    }


@pytest.fixture
def loop() -> tuple[TradingLoop, _RecordingExecution, MarketState, SimulatedClock]:
    clock = SimulatedClock(1_700_000_000_000)
    state = MarketState(MarketType.EVENT)
    execution = _RecordingExecution()

    trading = TradingLoop(
        market_type=MarketType.EVENT,
        market_state=state,
        risk_engine=EventRiskEngine(EventRiskSettings()),
        execution=execution,  # type: ignore[arg-type]
        recorder=_NullRecorder(),  # type: ignore[arg-type]
        clock=clock,
    )
    trading.account_state = EventAccountState(
        balance_micros=200_000_000, updated_at_ms=clock.now_ms()
    )
    trading.account_state.peak_equity_micros = trading.account_state.equity_micros
    trading.trading_permitted = True
    trading.operating_mode = OperatingMode.FULL_AUTONOMOUS

    strategy = MeanReversionStrategy()
    strategy.state = StrategyState.LIVE
    trading.add_strategy(strategy)
    return trading, execution, state, clock


def _drive(state: MarketState, clock: SimulatedClock, mids: list[int]) -> None:
    """Feed a price path in, one bar apart, so volatility becomes computable."""
    for mid in mids:
        clock.advance(BAR_MS)
        state.apply_snapshot(TICKER, _book(mid), now_ms=clock.now_ms())


async def _run(trading: TradingLoop) -> Any:
    """Run one cycle with the account state freshly reconciled.

    The price path advances the clock ten minutes, which leaves the seeded
    account state older than the risk engine's 30-second budget -- and it
    rejects on STALE_ACCOUNT_STATE, correctly. In the running platform the
    reconciler keeps that stamp current; here it has to be done by hand, and
    doing it is part of representing the real system rather than working around
    a check.
    """
    trading.account_state.updated_at_ms = trading.clock.now_ms()
    return await trading.run_cycle()


@pytest.mark.asyncio
async def test_a_liquid_moving_market_produces_an_order(
    loop: tuple[TradingLoop, _RecordingExecution, MarketState, SimulatedClock],
) -> None:
    """The case the demo exchange never supplies: a real dislocation."""
    trading, execution, state, clock = loop

    # A market oscillating around 50c, then dislocated sharply away from it.
    # Mean reversion exists for exactly this and should want to buy it back.
    path = [2700, 2800, 2600, 2750, 2650, 2700, 2800, 2600, 2700, 1200]
    _drive(state, clock, path)
    trading.features = {TICKER: _dislocated()}

    result = await _run(trading)

    assert result.markets_examined == 1, "the market must be looked at"
    assert result.markets_skipped_stale == 0, "freshly applied data must not read as stale"
    assert result.intents_generated >= 1, (
        f"a liquid, dislocated market produced no intent; strategy skips: "
        f"{trading.strategies[0].health()['skips']}"
    )
    assert execution.submitted, (
        f"an approved intent never reached the exchange; risk rejections: "
        f"{result.reasons} approved={result.intents_approved} rejected={result.intents_rejected}"
    )

    verdict = execution.submitted[0]
    assert verdict.approved_qty_qc > 0
    assert verdict.intent.ticker == TICKER
    assert verdict.intent.market_type is MarketType.EVENT


@pytest.mark.asyncio
async def test_the_order_is_sized_within_the_account(
    loop: tuple[TradingLoop, _RecordingExecution, MarketState, SimulatedClock],
) -> None:
    """A $200 account must not produce a $2,000 order."""
    trading, execution, state, clock = loop
    _drive(state, clock, [2700, 2800, 2600, 2750, 2650, 2700, 2800, 2600, 2700, 1200])
    trading.features = {TICKER: _dislocated()}

    await _run(trading)
    assert execution.submitted

    verdict = execution.submitted[0]
    equity = trading.account_state.equity_micros
    assert 0 < verdict.approved_notional_micros <= equity, (
        "an order larger than the whole account escaped sizing"
    )


@pytest.mark.asyncio
async def test_a_wide_market_produces_nothing(
    loop: tuple[TradingLoop, _RecordingExecution, MarketState, SimulatedClock],
) -> None:
    """The demo exchange's actual conditions: a 40-cent spread.

    The counterpart to the first test. Together they show the loop distinguishes
    a tradeable market from an untradeable one, rather than declining everything
    or accepting everything.
    """
    trading, execution, state, clock = loop
    clock.advance(BAR_MS)
    wide = OrderBook.from_event_contract_levels(
        TICKER,
        yes_levels=[(3000, 50_000)],
        no_levels=[(3000, 50_000)],  # reflects to a 70c YES ask: 40c wide
    )
    state.apply_snapshot(TICKER, wide, now_ms=clock.now_ms())
    trading.features = {TICKER: _dislocated()}

    result = await _run(trading)
    assert result.orders_submitted == 0
    assert not execution.submitted, "a 40-cent spread must never reach the exchange"


@pytest.mark.asyncio
async def test_a_halted_platform_submits_nothing(
    loop: tuple[TradingLoop, _RecordingExecution, MarketState, SimulatedClock],
) -> None:
    """The gate that matters most, on the same market that does trade."""
    trading, execution, state, clock = loop
    _drive(state, clock, [2700, 2800, 2600, 2750, 2650, 2700, 2800, 2600, 2700, 1200])
    trading.features = {TICKER: _dislocated()}
    trading.trading_permitted = False

    result = await _run(trading)
    assert result.orders_submitted == 0
    assert not execution.submitted
