"""A fee is charged for the trade that happened, not the trade that was proposed.

These tests exist because of a real, measured loss. Over one paper session the
platform recorded 157 fills carrying $101.17 of fees against $2,092.01 of
notional -- 4.84%, where the same book costs 1.74% once the fee follows the fill.
Nothing was broken in the fee model: ``estimate_fee_micros`` was being asked for
the quantity the *strategy* requested, and that figure was then billed against
the much smaller quantity the *risk engine* approved. One fill of 4.32 contracts
was charged the fee for 29 -- 366,700 micros where the model says 54,700.

The damage is not merely cosmetic. A fee that large swamps every edge a strategy
could find, so paper P&L stops measuring the strategies at all -- and paper P&L
is what the models are trained and gated on.
"""

from __future__ import annotations

import pytest

from kat.common.enums import (
    MarketType,
    OrderState,
    OutcomeSide,
    RiskDecision,
    TradingEnvironment,
)
from kat.common.timeutil import SimulatedClock
from kat.execution.event_exec import EventExecutionEngine
from kat.kalshi.fees import (
    INEXACT_SAFETY_MARGIN,
    estimate_fee_micros,
    prorate_fee_micros,
    realised_fee_micros,
)
from kat.marketdata.book import OrderBook
from kat.marketdata.state import MarketState
from kat.risk.types import OrderIntent, RiskVerdict

TICKER = "KXPRIMARYMOV-SENATEMAD26-EMAR-P60"

# The live case, exactly: the strategy asked for 29 contracts, risk approved
# 4.32, and the fee estimated for 29 was charged against the 4.32 that filled.
REQUESTED_QC = 2_900
APPROVED_QC = 432
PRICE_CC = 8_600


def _engine(*, depth_qc: int = 1_000_000) -> EventExecutionEngine:
    state = MarketState(MarketType.EVENT)
    book = OrderBook.from_event_contract_levels(
        TICKER,
        # A YES ask at PRICE_CC, expressed as the NO bid that Kalshi actually
        # publishes, so a YES buy at the limit crosses and fills.
        yes_levels=[(PRICE_CC - 100, depth_qc)],
        no_levels=[(10_000 - PRICE_CC, depth_qc)],
    )
    state.apply_snapshot(TICKER, book, now_ms=0)
    return EventExecutionEngine(
        rest=None,
        market_state=state,
        clock=SimulatedClock(0),
        environment=TradingEnvironment.PAPER,
    )


def _verdict(*, approved_qc: int, estimated_fees_micros: int) -> RiskVerdict:
    intent = OrderIntent(
        market_type=MarketType.EVENT,
        ticker=TICKER,
        qty_qc=REQUESTED_QC,
        limit_price_cc=PRICE_CC,
        outcome_side=OutcomeSide.YES,
    )
    return RiskVerdict(
        decision=RiskDecision.RESIZED,
        intent=intent,
        approved_qty_qc=approved_qc,
        reason_code="APPROVED",
        reason="resized",
        estimated_fees_micros=estimated_fees_micros,
    )


class TestThePaperFillIsChargedForWhatFilled:
    async def test_a_resized_order_is_not_charged_the_full_orders_fee(self):
        """The regression. Without the fix this bills 366_700 for 4.32 contracts."""
        engine = _engine()
        stale_estimate = estimate_fee_micros(
            price_cc=PRICE_CC, qty_qc=REQUESTED_QC
        ).fee_micros
        assert stale_estimate == 366_700, "fixture drifted from the observed case"

        order = await engine.submit(
            _verdict(approved_qc=APPROVED_QC, estimated_fees_micros=stale_estimate)
        )

        assert order.filled_qty_qc == APPROVED_QC
        expected = realised_fee_micros(
            price_cc=order.avg_fill_price_cc, qty_qc=APPROVED_QC
        )
        assert order.fees_micros == expected
        assert order.fees_micros < stale_estimate / 6

    async def test_the_fee_never_exceeds_the_notional_it_was_charged_on(self):
        """The property that would have caught this in production.

        A fee larger than the money at risk is never a real fee; it is always an
        accounting error somewhere upstream.
        """
        engine = _engine()
        order = await engine.submit(
            _verdict(approved_qc=APPROVED_QC, estimated_fees_micros=10_000_000)
        )
        notional_micros = order.filled_qty_qc * (order.avg_fill_price_cc or 0)
        assert 0 < order.fees_micros < notional_micros

    async def test_the_persisted_fill_carries_the_same_fee_as_the_order(self):
        """The dashboard reads fills; the account reads the order. They must agree."""
        rows: list[object] = []

        class _Writer:
            def enqueue(self, row: object, *, critical: bool = False) -> None:
                rows.append(row)

        engine = _engine()
        engine.writer = _Writer()  # type: ignore[assignment]
        order = await engine.submit(
            _verdict(approved_qc=APPROVED_QC, estimated_fees_micros=366_700)
        )

        fills = [r for r in rows if getattr(r, "count_qc", None) is not None]
        assert len(fills) == 1
        assert fills[0].fee_micros == order.fees_micros
        assert fills[0].count_qc == order.filled_qty_qc

    async def test_a_partial_fill_is_charged_only_for_the_part_that_filled(self):
        """Thin depth fills less than was approved; the fee must follow the fill."""
        engine = _engine(depth_qc=200)
        order = await engine.submit(
            _verdict(approved_qc=APPROVED_QC, estimated_fees_micros=366_700)
        )

        assert 0 < order.filled_qty_qc < APPROVED_QC
        expected = realised_fee_micros(
            price_cc=order.avg_fill_price_cc, qty_qc=order.filled_qty_qc
        )
        assert order.fees_micros == expected

    async def test_an_unmarketable_order_rests_and_is_charged_nothing(self):
        engine = _engine()
        verdict = _verdict(approved_qc=APPROVED_QC, estimated_fees_micros=366_700)
        verdict.intent.limit_price_cc = 100  # far below the ask
        order = await engine.submit(verdict)

        assert order.state is OrderState.ACKNOWLEDGED
        assert order.fees_micros == 0


class TestTheRiskVerdictReportsTheFeeForTheSizeItApproved:
    def test_prorating_is_exact_for_a_linear_schedule(self):
        """The assumption the helper rests on, checked against the real model."""
        full = estimate_fee_micros(price_cc=PRICE_CC, qty_qc=REQUESTED_QC).fee_micros
        direct = estimate_fee_micros(price_cc=PRICE_CC, qty_qc=APPROVED_QC).fee_micros
        prorated = prorate_fee_micros(
            full, from_qty_qc=REQUESTED_QC, to_qty_qc=APPROVED_QC
        )
        # Both round up to a centicent, so they agree to within one.
        assert abs(prorated - direct) <= 100

    @pytest.mark.parametrize(
        ("fee", "frm", "to", "expected"),
        [
            (1_000, 100, 50, 500),
            (1_000, 100, 0, 0),  # nothing approved, nothing charged
            (1_000, 100, 100, 1_000),
            (1_000, 100, 200, 1_000),  # never scales a fee *up*
            (1_000, 0, 50, 1_000),  # unknowable ratio: keep the estimate
            (0, 100, 50, 0),
            (1_000, 3, 1, 334),  # rounds up, never invents a cheaper trade
        ],
    )
    def test_prorating_edge_cases(self, fee: int, frm: int, to: int, expected: int):
        assert prorate_fee_micros(fee, from_qty_qc=frm, to_qty_qc=to) == expected

    def test_a_resizing_risk_engine_restates_the_fee(self):
        """End to end through the real event risk engine."""
        from kat.config.settings import EventRiskSettings
        from kat.risk.event_risk import EventAccountState, EventRiskEngine

        settings = EventRiskSettings()
        engine = EventRiskEngine(settings)
        state = EventAccountState(balance_micros=20_000_000, updated_at_ms=0)

        intent = OrderIntent(
            market_type=MarketType.EVENT,
            ticker=TICKER,
            qty_qc=REQUESTED_QC,
            limit_price_cc=PRICE_CC,
            outcome_side=OutcomeSide.YES,
        )
        full_fee = estimate_fee_micros(
            price_cc=PRICE_CC, qty_qc=REQUESTED_QC
        ).fee_micros
        verdict = engine.evaluate(
            intent, state, book=None, now_ms=0, estimated_fees_micros=full_fee
        )

        if verdict.approved_qty_qc < intent.qty_qc:
            assert verdict.estimated_fees_micros < full_fee
        assert verdict.estimated_fees_micros <= full_fee


class TestAChargeIsNotAPaddedEstimate:
    """The safety margin belongs to the decision, not to the bill.

    When the series coefficient is unknown the estimate is inflated by 50%, so
    an unknown fee cannot make a marginal trade look profitable. Charging that
    padded number claims the exchange took more than its published schedule
    says, and makes paper results worse than the reality they exist to predict
    -- the direction that quietly rejects strategies that would have worked.
    """

    def test_the_realised_fee_drops_the_margin_an_estimate_adds(self):
        estimated = estimate_fee_micros(price_cc=PRICE_CC, qty_qc=APPROVED_QC).fee_micros
        charged = realised_fee_micros(price_cc=PRICE_CC, qty_qc=APPROVED_QC)

        assert charged < estimated
        assert charged == pytest.approx(estimated / float(INEXACT_SAFETY_MARGIN), rel=0.01)

    def test_a_known_schedule_charges_exactly_what_it_estimates(self):
        """With nothing guessed at there is nothing to pad, so the two agree."""
        from kat.kalshi.fees import FeeSchedule

        schedule = FeeSchedule.from_series("quadratic", 0.07)
        assert schedule.exact

        estimated = estimate_fee_micros(
            price_cc=PRICE_CC, qty_qc=APPROVED_QC, schedule=schedule
        ).fee_micros
        charged = realised_fee_micros(
            price_cc=PRICE_CC, qty_qc=APPROVED_QC, schedule=schedule
        )
        assert charged == estimated

    async def test_the_paper_fill_charges_the_unpadded_fee(self):
        engine = _engine()
        order = await engine.submit(
            _verdict(approved_qc=APPROVED_QC, estimated_fees_micros=366_700)
        )
        padded = estimate_fee_micros(
            price_cc=order.avg_fill_price_cc, qty_qc=order.filled_qty_qc
        ).fee_micros

        assert order.fees_micros < padded
