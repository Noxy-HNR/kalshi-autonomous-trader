"""Fees are subtracted once.

``OrderIntent.expected_value_micros`` is already net: strategies build it from
``Costing.net_micros``, which is gross minus a *round-trip* fee. The event risk
engine took it for a gross figure and subtracted its own entry-fee estimate on
top, so every trade was charged twice -- once for two legs, once more for one --
and marginal-but-profitable trades were refused.

That failure is invisible in operation. A refusal for "insufficient expected
value" looks exactly the same whether the arithmetic was right or wrong, so
nothing downstream could have caught it; only reading the two sides together
does. These tests are what stop it coming back.
"""

from __future__ import annotations

from kat.common.enums import MarketType, OutcomeSide
from kat.config.settings import EventRiskSettings
from kat.marketdata.book import OrderBook
from kat.risk.event_risk import EventAccountState, EventRiskEngine
from kat.risk.types import OrderIntent

NOW_MS = 1_700_000_000_000
FLOOR_MICROS = 5_000  # $0.005, the shipped minimum


def _book() -> OrderBook:
    return OrderBook.from_event_contract_levels(
        "EV-TEST",
        yes_levels=[(1150, 50_000), (1050, 50_000)],
        no_levels=[(8750, 50_000), (8650, 50_000)],
    )


def _intent(expected_value_micros: int) -> OrderIntent:
    return OrderIntent(
        market_type=MarketType.EVENT,
        ticker="EV-TEST",
        qty_qc=1_000,
        limit_price_cc=1250,
        outcome_side=OutcomeSide.YES,
        strategy_id="test",
        expected_value_micros=expected_value_micros,
    )


def _engine() -> tuple[EventRiskEngine, EventAccountState]:
    state = EventAccountState(balance_micros=200_000_000, updated_at_ms=NOW_MS)
    state.peak_equity_micros = state.equity_micros
    return EventRiskEngine(EventRiskSettings()), state


def _evaluate(ev_micros: int, fees_micros: int):
    engine, state = _engine()
    return engine.evaluate(
        _intent(ev_micros),
        state,
        book=_book(),
        now_ms=NOW_MS,
        estimated_fees_micros=fees_micros,
    )


def test_an_intent_just_above_the_floor_is_approved() -> None:
    """The exact case double-charging used to reject.

    EV sits a little above the floor, and the engine's own fee estimate is
    larger than the margin. Subtracting it again would push the trade under.
    """
    verdict = _evaluate(ev_micros=FLOOR_MICROS + 1_000, fees_micros=50_000)
    assert verdict.approved, (
        f"a positive-EV trade was refused: {verdict.reason_code} -- {verdict.reason}"
    )


def test_an_intent_below_the_floor_is_still_refused() -> None:
    """The check must still bite. Removing a double charge is not removing the check."""
    verdict = _evaluate(ev_micros=FLOOR_MICROS - 1, fees_micros=0)
    assert not verdict.approved
    assert verdict.reason_code == "INSUFFICIENT_EXPECTED_VALUE"


def test_a_negative_expected_value_is_refused() -> None:
    verdict = _evaluate(ev_micros=-1_000, fees_micros=0)
    assert not verdict.approved
    assert verdict.reason_code == "INSUFFICIENT_EXPECTED_VALUE"


def test_the_verdict_does_not_depend_on_the_engines_fee_estimate() -> None:
    """The decisive property.

    ``expected_value_micros`` already accounts for fees, so the engine's
    separate estimate must not move this decision at all. If a larger fee
    estimate can flip an approval to a refusal, fees are being counted twice.
    """
    ev = FLOOR_MICROS + 1_000
    decisions = {_evaluate(ev, fees).approved for fees in (0, 10_000, 500_000, 10_000_000)}
    assert decisions == {True}, "the engine's fee estimate is still being applied to EV"


def test_the_reported_observation_is_the_net_value_itself() -> None:
    """What the check reports must match what it compared, or an operator
    reading a refusal is debugging the wrong number."""
    ev = FLOOR_MICROS + 2_000
    verdict = _evaluate(ev, fees_micros=123_456)
    check = next(c for c in verdict.checks if c.name == "min_expected_value")
    assert check.observed == "0.007000"


def test_an_intent_without_an_expected_value_skips_the_check() -> None:
    """Not every producer supplies one; absence must not read as zero."""
    engine, state = _engine()
    verdict = engine.evaluate(
        _intent(0).__class__(
            market_type=MarketType.EVENT,
            ticker="EV-TEST",
            qty_qc=1_000,
            limit_price_cc=1250,
            outcome_side=OutcomeSide.YES,
            strategy_id="test",
        ),
        state,
        book=_book(),
        now_ms=NOW_MS,
        estimated_fees_micros=50_000,
    )
    assert not any(c.name == "min_expected_value" for c in verdict.checks)
