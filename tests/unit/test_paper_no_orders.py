"""A NO order is priced on the YES book the way the backtester has always priced it.

The paper simulator compared a NO limit directly against the YES bid. The book is
YES-denominated, and a NO buy at p is a YES sell at 1 - p, so that comparison was
wrong on both halves of the board. Below fifty cents it refused marketable orders:
a NO buy at the NO ask with three thousand contracts of depth rested unfilled,
which is why model_drift's exits never filled in paper and piled up. Above fifty
cents it filled orders that were not marketable, at a price past their own limit.
"""

from __future__ import annotations

from kat.common.enums import (
    MarketType,
    OrderState,
    OutcomeSide,
    RiskDecision,
    TradingEnvironment,
)
from kat.common.fixedpoint import CC_PER_DOLLAR
from kat.common.timeutil import SimulatedClock
from kat.execution.event_exec import EventExecutionEngine
from kat.marketdata.book import OrderBook
from kat.marketdata.state import MarketState
from kat.risk.types import OrderIntent, RiskVerdict

NOW = 1_760_000_000_000


def _engine(*, yes_bid: int, yes_ask: int, depth_qc: int = 300_000) -> EventExecutionEngine:
    market = MarketState(MarketType.EVENT)
    market.apply_snapshot(
        "X",
        OrderBook.from_event_contract_levels(
            "X", yes_levels=[(yes_bid, depth_qc)], no_levels=[(CC_PER_DOLLAR - yes_ask, depth_qc)]
        ),
        now_ms=NOW,
    )
    return EventExecutionEngine(
        rest=None, market_state=market, clock=SimulatedClock(NOW),
        environment=TradingEnvironment.PAPER,
    )


async def _submit(engine, *, side: OutcomeSide, limit_cc: int, qty_qc: int = 500):
    intent = OrderIntent(
        market_type=MarketType.EVENT, ticker="X", qty_qc=qty_qc,
        limit_price_cc=limit_cc, outcome_side=side, strategy_id="s",
    )
    return await engine.submit(
        RiskVerdict(decision=RiskDecision.APPROVED, intent=intent,
                    approved_qty_qc=qty_qc, reason_code="APPROVED", reason="ok")
    )


class TestANoBuyBelowFiftyCents:
    async def test_at_the_no_ask_it_fills(self):
        """YES 39.5 / 40.5, so NO is offered at 60.5."""
        engine = _engine(yes_bid=3_950, yes_ask=4_050)
        order = await _submit(engine, side=OutcomeSide.NO, limit_cc=CC_PER_DOLLAR - 3_950)

        assert order.filled_qty_qc == 500, order.state_history[-1]["reason"]

    async def test_below_the_no_ask_it_rests(self):
        engine = _engine(yes_bid=3_950, yes_ask=4_050)
        order = await _submit(engine, side=OutcomeSide.NO, limit_cc=5_000)

        assert order.filled_qty_qc == 0
        assert order.state is OrderState.ACKNOWLEDGED


class TestANoBuyAboveFiftyCents:
    async def test_below_the_no_ask_it_rests_rather_than_filling_past_its_limit(self):
        """YES 70.5 / 71.5, so NO is offered at 29.5. A NO bid at 20 is not marketable.

        The old comparison, 20 against the YES bid of 70.5, called it marketable and
        filled it at 29.5 -- half again what the order was willing to pay.
        """
        engine = _engine(yes_bid=7_050, yes_ask=7_150)
        order = await _submit(engine, side=OutcomeSide.NO, limit_cc=2_000)

        assert order.filled_qty_qc == 0
        assert order.state is OrderState.ACKNOWLEDGED

    async def test_at_the_no_ask_it_fills_no_worse_than_its_limit(self):
        engine = _engine(yes_bid=7_050, yes_ask=7_150)
        order = await _submit(engine, side=OutcomeSide.NO, limit_cc=3_000)

        assert order.filled_qty_qc == 500
        assert order.avg_fill_price_cc <= 3_000  # in NO terms, as live records it


class TestYesOrdersAreUnchanged:
    async def test_a_yes_buy_at_the_ask_fills(self):
        engine = _engine(yes_bid=3_950, yes_ask=4_050)
        order = await _submit(engine, side=OutcomeSide.YES, limit_cc=4_050)
        assert order.filled_qty_qc == 500

    async def test_a_yes_buy_below_the_ask_rests(self):
        engine = _engine(yes_bid=3_950, yes_ask=4_050)
        order = await _submit(engine, side=OutcomeSide.YES, limit_cc=4_000)
        assert order.filled_qty_qc == 0


def _loop_over(engine: EventExecutionEngine, balance_micros: int):
    from kat.config.settings import EventRiskSettings
    from kat.engine.trading import TradingLoop
    from kat.risk.event_risk import EventAccountState, EventRiskEngine

    loop = TradingLoop(
        market_type=MarketType.EVENT, market_state=engine.market_state,
        risk_engine=EventRiskEngine(EventRiskSettings()), execution=engine,
        recorder=None,  # type: ignore[arg-type]
        clock=SimulatedClock(NOW),
    )
    loop.paper_fills = True
    loop.account_state = EventAccountState(balance_micros=balance_micros, updated_at_ms=NOW)
    return loop


async def _trade(engine, loop, *, side: OutcomeSide, limit_cc: int, qty_qc: int = 500):
    intent = OrderIntent(
        market_type=MarketType.EVENT, ticker="X", qty_qc=qty_qc,
        limit_price_cc=limit_cc, outcome_side=side, strategy_id="s",
    )
    order = await engine.submit(
        RiskVerdict(decision=RiskDecision.APPROVED, intent=intent,
                    approved_qty_qc=qty_qc, reason_code="APPROVED", reason="ok")
    )
    assert order.filled_qty_qc == qty_qc, order.state_history[-1]["reason"]
    loop._apply_paper_fill(intent, order)
    return order


class TestTheAccountIsRightInDollars:
    """Whatever scale a fill price is recorded on, the account must come out exact.

    A round trip across a one-cent spread on five contracts costs five cents before
    fees, whichever outcome it opens and whichever it closes with.
    """

    START = 200_000_000

    async def test_opening_no_and_closing_with_a_yes_buy(self):
        engine = _engine(yes_bid=3_950, yes_ask=4_050, depth_qc=10_000_000)
        loop = _loop_over(engine, self.START)
        account = loop.account_state

        entry = await _trade(engine, loop, side=OutcomeSide.NO, limit_cc=CC_PER_DOLLAR - 3_950)
        held = account.positions["X"]
        assert held.position_qc == -500
        assert held.avg_entry_cc == 6_050  # NO bought at 60.5 cents
        assert account.balance_micros == self.START - 6_050 * 500 - entry.fees_micros

        exit_ = await _trade(engine, loop, side=OutcomeSide.YES, limit_cc=4_050)
        assert "X" not in account.positions
        assert account.realized_pnl_today_micros == -50_000
        assert account.balance_micros == (
            self.START - 50_000 - entry.fees_micros - exit_.fees_micros
        )

    async def test_opening_yes_and_closing_with_a_no_buy_above_fifty_cents(self):
        engine = _engine(yes_bid=7_050, yes_ask=7_150, depth_qc=10_000_000)
        loop = _loop_over(engine, self.START)
        account = loop.account_state

        entry = await _trade(engine, loop, side=OutcomeSide.YES, limit_cc=7_150)
        assert account.positions["X"].position_qc == 500

        exit_ = await _trade(engine, loop, side=OutcomeSide.NO, limit_cc=CC_PER_DOLLAR - 7_050)
        assert "X" not in account.positions
        assert account.realized_pnl_today_micros == -50_000
        assert account.balance_micros == (
            self.START - 50_000 - entry.fees_micros - exit_.fees_micros
        )


class TestAFillIsRecordedInItsOwnOutcomesTerms:
    """As the live path records it, from the exchange's no_price.

    Recording the YES side of the walk put a NO order's fill and its limit on
    different scales, so a NO buy filled exactly on its limit reported -4,100cc of
    slippage into the decision record, the orders table and the dashboard.
    """

    async def test_a_no_buy_at_the_no_ask_records_the_no_price(self):
        engine = _engine(yes_bid=3_950, yes_ask=4_050)
        order = await _submit(engine, side=OutcomeSide.NO, limit_cc=CC_PER_DOLLAR - 3_950)

        assert order.avg_fill_price_cc == 6_050
        assert order.slippage_cc() == 0

    async def test_a_yes_buy_records_the_yes_price(self):
        engine = _engine(yes_bid=3_950, yes_ask=4_050)
        order = await _submit(engine, side=OutcomeSide.YES, limit_cc=4_050)

        assert order.avg_fill_price_cc == 4_050
        assert order.slippage_cc() == 0
